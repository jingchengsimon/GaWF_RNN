"""Checkpoint and resume behaviour for the Atari DQN entry point.

These tests exercise the recovery path that lets a preempted Slurm job continue
instead of restarting from step 0. They deliberately do not assert that an
interrupted run equals an uninterrupted one: the ALE environment and the
recurrent state are re-initialised on resume by design, so each resume injects
one artificial episode boundary. What must hold is that the resume path is
itself deterministic and that it never mixes two trajectories into one result.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import train_atari_dqn
from train_atari_dqn import CHECKPOINT_FILENAME, REPLAY_SUBDIR, build_arg_parser, train

try:  # ALE is not installed on every development machine.
    import ale_py  # noqa: F401

    ATARI_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    ATARI_AVAILABLE = False


def _args(save_dir: str, **overrides):
    """A deliberately tiny run: CPU, ANN core, short horizon, small replay."""
    parser = build_arg_parser()
    argv = [
        "--env_id",
        "ALE/Pong-v5",
        "--model_type",
        "ann",
        "--hidden_size",
        "32",
        "--encoder_feature_dim",
        "32",
        "--frame_stack",
        "1",
        "--frame_skip",
        "4",
        "--total_timesteps",
        "400",
        "--buffer_size",
        "400",
        "--learning_starts",
        "100",
        "--batch_size",
        "8",
        "--train_frequency",
        "4",
        "--target_network_frequency",
        "100",
        "--log_interval",
        "100",
        "--device",
        "cpu",
        # train() resolves this before validating a resume, so fix it here to
        # keep saved and live args comparable.
        "--feedback_mode",
        "none",
        "--save_dir",
        save_dir,
    ]
    args = parser.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _history_steps(save_dir: str) -> list[int]:
    path = os.path.join(save_dir, "metrics_history.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as stream:
        return [int(json.loads(line)["global_step"]) for line in stream if line.strip()]


@contextlib.contextmanager
def _preempt_at_step(step: int):
    """Raise SIGUSR1 exactly once when training reaches ``step``.

    Wall-clock timing would make the test flaky, so the signal is driven off the
    step counter through the per-step epsilon call. This exercises the same
    handler Slurm triggers on preemption.
    """
    original = train_atari_dqn._linear_epsilon
    fired = False

    def patched(args, global_step):
        nonlocal fired
        if not fired and global_step >= step:
            fired = True
            signal.raise_signal(signal.SIGUSR1)
        return original(args, global_step)

    train_atari_dqn._linear_epsilon = patched
    try:
        yield
    finally:
        train_atari_dqn._linear_epsilon = original


class AtariCheckpointDefaultsTest(unittest.TestCase):
    """The recovery machinery must be inert unless explicitly requested."""

    def test_resume_and_auto_resume_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, resume_from="/nonexistent.pth", auto_resume=True)
            with self.assertRaises(ValueError):
                train(args)

    def test_negative_checkpoint_interval_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, checkpoint_interval_steps=-1)
            with self.assertRaises(ValueError):
                train(args)

    def test_resume_requires_mmap_backing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, CHECKPOINT_FILENAME)
            args = _args(tmp, resume_from=checkpoint_path, replay_backing="memory")
            torch.save(
                {
                    "format_version": train_atari_dqn.CHECKPOINT_FORMAT_VERSION,
                    "args": vars(args),
                    "learning_rate": float(args.learning_rate),
                },
                checkpoint_path,
            )
            with self.assertRaises(ValueError) as caught:
                train(args)
            self.assertIn("mmap", str(caught.exception))

    def test_protocol_mismatch_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, CHECKPOINT_FILENAME)
            saved = _args(tmp, hidden_size=32)
            torch.save(
                {
                    "format_version": train_atari_dqn.CHECKPOINT_FORMAT_VERSION,
                    "args": vars(saved),
                    "learning_rate": float(saved.learning_rate),
                },
                checkpoint_path,
            )
            resumed = _args(
                tmp,
                hidden_size=64,  # different capacity
                resume_from=checkpoint_path,
                replay_backing="mmap",
            )
            with self.assertRaises(ValueError) as caught:
                train(resumed)
            self.assertIn("hidden_size", str(caught.exception))

    def test_stale_format_version_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, CHECKPOINT_FILENAME)
            saved = _args(tmp)
            torch.save(
                {
                    "format_version": 999,
                    "args": vars(saved),
                    "learning_rate": float(saved.learning_rate),
                },
                checkpoint_path,
            )
            resumed = _args(
                tmp, resume_from=checkpoint_path, replay_backing="mmap"
            )
            with self.assertRaises(ValueError) as caught:
                train(resumed)
            self.assertIn("format_version", str(caught.exception))


@unittest.skipUnless(ATARI_AVAILABLE, "ALE/Gymnasium Atari not installed")
class AtariCheckpointRoundTripTest(unittest.TestCase):
    def test_plain_run_writes_no_checkpoint_or_replay_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = train(_args(tmp))
            self.assertEqual(metrics["global_step"], 400)
            self.assertFalse(os.path.exists(os.path.join(tmp, CHECKPOINT_FILENAME)))
            self.assertFalse(os.path.exists(os.path.join(tmp, REPLAY_SUBDIR)))
            self.assertEqual(metrics["resume_count"], 0)
            self.assertEqual(metrics["replay_backing"], "memory")

    def test_preempted_run_checkpoints_then_resumes_to_completion(self) -> None:
        """The real recovery path: signal, save, exit, restart, finish."""
        with tempfile.TemporaryDirectory() as tmp:
            partial = _args(
                tmp,
                total_timesteps=400,
                checkpoint_interval_steps=100,
                replay_backing="mmap",
            )
            with _preempt_at_step(200):
                outcome = train(partial)

            self.assertEqual(outcome["status"], "preempted")
            self.assertLess(outcome["global_step"], 400)
            checkpoint_path = os.path.join(tmp, CHECKPOINT_FILENAME)
            self.assertTrue(os.path.isfile(checkpoint_path))
            # Replay must survive a preemption; that is the whole point.
            self.assertTrue(os.path.isdir(os.path.join(tmp, REPLAY_SUBDIR)))
            self.assertFalse(os.path.exists(os.path.join(tmp, "metrics.json")))
            interrupted_at = outcome["global_step"]

            resumed = _args(
                tmp,
                total_timesteps=400,
                checkpoint_interval_steps=100,
                replay_backing="mmap",
                auto_resume=True,
            )
            metrics = train(resumed)

            self.assertEqual(metrics["global_step"], 400)
            self.assertEqual(metrics["resume_count"], 1)
            self.assertEqual(metrics["resumed_at_steps"], [interrupted_at])
            self.assertEqual(metrics["replay_backing"], "mmap")
            steps = _history_steps(tmp)
            self.assertEqual(steps, sorted(steps))
            self.assertEqual(len(steps), len(set(steps)))
            # Successful completion reclaims the ~28 GB-scale replay directory
            # and drops the scaffolding checkpoint, leaving exactly one .pth as
            # the launcher validator expects.
            self.assertFalse(os.path.exists(os.path.join(tmp, REPLAY_SUBDIR)))
            self.assertFalse(os.path.exists(checkpoint_path))
            self.assertEqual(
                len([name for name in os.listdir(tmp) if name.endswith(".pth")]), 1
            )

    def test_resume_is_deterministic(self) -> None:
        """Resuming twice from one checkpoint must produce identical weights."""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            partial = _args(
                source,
                total_timesteps=400,
                checkpoint_interval_steps=100,
                replay_backing="mmap",
            )
            with _preempt_at_step(200):
                train(partial)

            weights = []
            for attempt in range(2):
                target = os.path.join(tmp, f"resume{attempt}")
                shutil.copytree(source, target)
                args = _args(
                    target,
                    total_timesteps=400,
                    checkpoint_interval_steps=100,
                    replay_backing="mmap",
                    auto_resume=True,
                )
                metrics = train(args)
                self.assertEqual(metrics["global_step"], 400)
                weights.append(torch.load(metrics["checkpoint"], map_location="cpu"))

            self.assertEqual(weights[0].keys(), weights[1].keys())
            for key in weights[0]:
                self.assertTrue(
                    torch.equal(weights[0][key], weights[1][key]),
                    f"resume diverged on parameter {key}",
                )

    def test_protocol_change_blocks_resume_of_a_real_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            partial = _args(
                tmp,
                total_timesteps=400,
                checkpoint_interval_steps=100,
                replay_backing="mmap",
            )
            with _preempt_at_step(200):
                train(partial)

            # total_timesteps defines the protocol, so extending a run mid-flight
            # must be refused rather than silently lengthening it.
            extended = _args(
                tmp,
                total_timesteps=800,
                checkpoint_interval_steps=100,
                replay_backing="mmap",
                auto_resume=True,
            )
            with self.assertRaises(ValueError) as caught:
                train(extended)
            self.assertIn("total_timesteps", str(caught.exception))

    def test_checkpointing_refuses_to_append_to_foreign_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train(_args(tmp))  # leaves metrics.json without a checkpoint
            args = _args(tmp, checkpoint_interval_steps=100, replay_backing="mmap")
            with self.assertRaises(FileExistsError):
                train(args)


if __name__ == "__main__":
    unittest.main()
