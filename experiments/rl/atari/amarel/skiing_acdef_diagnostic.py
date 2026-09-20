"""Run one independent seed1 Skiing variant with a smoke gate and offline BF16 evaluations.

Inputs are an immutable source weights directory and exact output parents. Outputs are retained
smoke/formal training bundles and fixed-seed JSON/NPZ audits at each 500k model snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

VARIANTS = {
    "a": dict(exploration_steps=500000, end_epsilon=.01, learning_rate=1e-4,
              learning_rate_decay_step=1000000, seq_len=16),
    "c": dict(exploration_steps=1000000, end_epsilon=.05, learning_rate=1e-4,
              learning_rate_decay_step=1000000, seq_len=16),
    "d": dict(exploration_steps=500000, end_epsilon=.01, learning_rate=1e-4,
              learning_rate_decay_step=3000000, seq_len=16),
    "e": dict(exploration_steps=500000, end_epsilon=.01, learning_rate=3e-5,
              learning_rate_decay_step=1000000, seq_len=16),
    "f": dict(exploration_steps=500000, end_epsilon=.01, learning_rate=1e-4,
              learning_rate_decay_step=1000000, seq_len=64),
}
HIDDEN = {"lstm": 373, "gru": 458, "gawf": 604}
STEPS = tuple(range(500000, 4000001, 500000))
LOG = logging.getLogger(__name__)


def training_command(model: str, variant: str, source: Path, result: Path,
                     smoke: bool) -> list[str]:
    """Build the fixed protocol; variants only change their declared parameters."""
    params = dict(env_id="ALE/Skiing-v5", action_space_mode="full18",
                  atari_env_protocol="skiing-stall-actionfix-v1", model_type=model,
                  feedback_mode="qvalues" if model == "gawf" else "none",
                  num_layers=3, hidden_size=HIDDEN[model], gawf_feedback_lr_scale=1.0,
                  frame_skip=4, frame_stack=4, flicker_prob=0.0, num_envs=1, seed=1,
                  total_timesteps=25000 if smoke else 4000000, gamma=.999,
                  learning_rate_decay_per_task_steps=0, learning_rate_decay_scale=.1,
                  learning_starts=2000 if smoke else 20000, learning_starts_per_task=0,
                  start_epsilon=1.0, replay_backing="mmap",
                  buffer_size=25000 if smoke else 500000,
                  checkpoint_interval_steps=10000 if smoke else 50000,
                  batch_size=32, sequences_per_batch=8, train_frequency=4,
                  target_network_frequency=1000, max_grad_norm=10.0,
                  amp_dtype="bfloat16", device="cuda", result_suffix=result.name,
                  save_dir=str(result), **VARIANTS[variant])
    args = [sys.executable, "-B", "run_task.py", "atari-dqn"]
    for key, value in params.items():
        args += [f"--{key}", str(value)]
    args += ["--no_reward_clip", "--keep_replay_on_success", "--allow_tf32",
             "--cudnn_benchmark", "--fused_optimizer"]
    if not smoke:
        args += ["--diagnostic_checkpoint_steps", *map(str, STEPS)]
    checkpoint = result / "checkpoint.pth"
    if checkpoint.exists():
        args += ["--resume_from", str(checkpoint)]
    else:
        if result.exists() and any(result.iterdir()):
            raise RuntimeError(f"Non-empty training leaf without checkpoint: {result}")
        args += ["--init_weights_from", str(source / model / "model.pth")]
    return args


def validate_training(result: Path, model: str, variant: str, smoke: bool) -> dict:
    """Require complete finite training and protocol-specific model artifacts."""
    metrics = json.loads((result / "metrics.json").read_text())
    expected = dict(global_step=25000 if smoke else 4000000, model_type=model,
                    hidden_size=HIDDEN[model], num_layers=3, num_actions=18,
                    gamma=.999, reward_clip=False, amp_dtype="bfloat16",
                    atari_env_protocol="skiing-stall-actionfix-v1",
                    **{k: v for k, v in VARIANTS[variant].items() if k != "end_epsilon"})
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"{result}: {key}={metrics.get(key)!r}, expected {value}")
    config = VARIANTS[variant]
    expected_epsilon = 1 + (config["end_epsilon"] - 1) * min(
        1, expected["global_step"] / config["exploration_steps"])
    assert math.isclose(metrics["epsilon"], expected_epsilon)
    for key in ("loss", "q_values_mean", "episodic_return_100"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"Non-finite {key}: {result}")
    feedback = "qvalues" if model == "gawf" else "none"
    checkpoint = result / f"dqn_{model}_{feedback}_L3_ALE_Skiing-v5.pth"
    assert checkpoint.stat().st_size > 0
    assert (result / "metrics_history.jsonl").stat().st_size > 0
    return metrics


def child_command(command: list[str], log: Path, resumable: bool) -> bool:
    """Forward Slurm's warning to training; return False after a resumable pause."""
    with log.open("a") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        warned = False

        def pause(signum: int, frame: object) -> None:
            nonlocal warned
            warned = True
            process.send_signal(signal.SIGUSR1 if resumable else signal.SIGTERM)

        previous = {s: signal.signal(s, pause) for s in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            rc = process.wait()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
    if rc and not (warned and not resumable):
        raise RuntimeError(f"Command exit {rc}; see {log}")
    return not warned


def requeue() -> None:
    """Recover the exact current array task without creating a duplicate writer."""
    job = os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    subprocess.run(["scontrol", "requeue", job], check=True)


def audit_statistics(summary_path: Path, trace_path: Path) -> dict:
    """Verify trace and summarize ties across distinct legal groups and action fractions."""
    import numpy as np

    summary = json.loads(summary_path.read_text())
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == summary["trace_sha256"]
    assert summary["amp_dtype"] == "bfloat16"
    with np.load(trace_path) as data:
        q = data["q_values"].astype(np.float64)
        assert np.isfinite(q).all() and q.shape[1] == 18
        mapping = np.asarray(summary["canonical_action_mapping"])
        legal = np.stack([q[:, mapping == a].max(1) for a in range(9)], axis=1)
        gaps = np.sort(legal, axis=1)[:, -1] - np.sort(legal, axis=1)[:, -2]
        return dict(raw_return_mean=summary["mean_return"], stall_rate=summary["stall_rate"],
                    legal_top_q_tie_rate=float((gaps == 0).mean()),
                    canonical_top_q_tie_rate=float(((q == q.max(1)[:, None]).sum(1) > 1).mean()),
                    all18_equal_rate=float((q.max(1) == q.min(1)).mean()),
                    legal_top_margin_median=float(np.median(gaps)),
                    canonical_action_fraction=(np.bincount(data["action"], minlength=18)
                                               / len(q)).tolist(),
                    legal_action_fraction=(np.bincount(data["legal_action"], minlength=9)
                                           / len(q)).tolist(),
                    trace_sha256=summary["trace_sha256"],
                    source_checkpoint_sha256=summary["source_checkpoint_sha256"],
                    episodes=summary["num_episodes"], eval_seed=summary["eval_seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=HIDDEN, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    unit = f"{args.model}_{args.variant}_seed1"
    artifact = args.artifacts / unit
    artifact.mkdir(parents=True, exist_ok=True)
    for smoke in (True, False):
        result = args.results / ("smoke" if smoke else "formal") / unit
        if not (result / "metrics.json").exists():
            command = training_command(args.model, args.variant, args.source, result, smoke)
            (artifact / f"command_{'smoke' if smoke else 'formal'}.json").write_text(
                json.dumps(command, indent=2) + "\n")
            LOG.info("Training %s", result)
            child_command(command, artifact / f"{'smoke' if smoke else 'formal'}.log", True)
            if not (result / "metrics.json").exists():
                assert (result / "checkpoint.pth").stat().st_size > 0
                requeue()
                return
        validate_training(result, args.model, args.variant, smoke)
        if smoke:
            # Explicit user protection of checkpoints/results overrides temporary-smoke cleanup.
            (artifact / "smoke_accepted.json").write_text(json.dumps(dict(
                result=str(result), budget=25000, artifacts_retained=True)) + "\n")
    for step in STEPS:
        checkpoint = result / "diagnostic_checkpoints" / f"model_step{step}.pth"
        assert checkpoint.stat().st_size > 0
        leaf = args.results / "evaluations" / unit / f"step{step}"
        leaf.mkdir(parents=True, exist_ok=True)
        summary, trace = leaf / "summary.json", leaf / "step_trace.npz"
        if not summary.exists():
            # New attempt leaves preserve any interrupted partial outputs.
            attempt = 0
            while (leaf / f"attempt{attempt}").exists():
                attempt += 1
            attempt_leaf = leaf / f"attempt{attempt}"
            attempt_leaf.mkdir()
            attempt_summary = attempt_leaf / "summary.json"
            attempt_trace = attempt_leaf / "step_trace.npz"
            command = [sys.executable, "-B", "-m",
                       "utils.analysis.rl.atari.evaluate_skiing_behavior",
                       "--metrics_path", str(result / "metrics.json"),
                       "--checkpoint", str(checkpoint), "--summary_path", str(attempt_summary),
                       "--trace_path", str(attempt_trace), "--num_episodes", "20",
                       "--eval_seed", "20260904", "--device", "cuda", "--amp_dtype", "bfloat16"]
            if not child_command(command, artifact / f"eval_step{step}.log", False):
                requeue()
                return
            audit_statistics(attempt_summary, attempt_trace)
            # Relative links select a verified attempt without replacing any output.
            trace.symlink_to(attempt_trace.relative_to(leaf))
            summary.symlink_to(attempt_summary.relative_to(leaf))
        stats = audit_statistics(summary, trace)
        stats["training_step"] = step
        stats["raw_return_definition"] = "unclipped wrapper reward including existing stall adjustment"
        stats["source_metrics_training_step"] = 4000000
        (leaf / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (artifact / "done").write_text("training 4M and eight 20-episode BF16 audits complete\n")


if __name__ == "__main__":
    main()
