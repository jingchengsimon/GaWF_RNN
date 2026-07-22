"""Smoke tests for the Atari replay buffer (no Gymnasium/ALE required)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_replay import AtariReplayBuffer

OBS_SHAPE = (2, 4, 4)


def _fill(
    buffer: AtariReplayBuffer,
    n_steps: int,
    dones=(),
    resets=(),
    task_ids: tuple[int, ...] | list[int] | None = None,
) -> None:
    """Add ``n_steps`` rows whose obs encode the global step index mod 256."""
    for step in range(n_steps):
        obs = np.full((1, *OBS_SHAPE), step % 256, dtype=np.uint8)
        buffer.add(
            obs=obs,
            actions=np.array([step % 5]),
            rewards=np.array([float(step)], dtype=np.float32),
            dones=np.array([1 if step in dones else 0]),
            resets=np.array([1 if step in resets else 0]),
            task_ids=None if task_ids is None else np.array([task_ids[step]]),
        )


class AtariReplayBufferSmokeTest(unittest.TestCase):
    def test_transition_successor_after_wraparound(self) -> None:
        buffer = AtariReplayBuffer(
            buffer_size=10, num_envs=1, obs_shape=OBS_SHAPE, device="cpu", seed=0
        )
        _fill(buffer, 25)
        self.assertEqual(buffer.size, 10)
        batch = buffer.sample_transitions(64)
        obs_vals = batch.obs[:, 0, 0, 0].long()
        next_vals = batch.next_obs[:, 0, 0, 0].long()
        self.assertTrue(torch_all_equal(next_vals, (obs_vals + 1) % 256))
        # Newest stored step (24) has no successor and must never be a base.
        self.assertFalse((obs_vals == 24).any().item())

    def test_sequence_contiguity_and_wraparound(self) -> None:
        buffer = AtariReplayBuffer(
            buffer_size=10, num_envs=1, obs_shape=OBS_SHAPE, device="cpu", seed=1
        )
        _fill(buffer, 27, dones={20, 23}, resets={21, 24})
        seq = buffer.sample_sequences(batch_size=32, seq_len=6)
        obs_vals = seq.obs[:, :, 0, 0, 0].long()
        diffs = (obs_vals[:, 1:] - obs_vals[:, :-1]) % 256
        self.assertTrue((diffs == 1).all().item())
        # Oldest retained step is 17, so some sampled window must cross the
        # physical head boundary (steps 19->20 wrap from slot 9 to slot 0).
        self.assertTrue((obs_vals[:, 0] < 20).any().item())
        self.assertTrue((seq.prev_dones[:, 0] == 1.0).all().item())
        self.assertTrue(seq.has_internal_reset)
        self.assertTrue((seq.loss_mask >= 0).all().item())
        for row in range(obs_vals.shape[0]):
            for col in range(obs_vals.shape[1]):
                step = int(obs_vals[row, col])
                is_autoreset = step in {21, 24}
                expected_state_reset = step in {21, 22, 24, 25}
                self.assertEqual(float(seq.loss_mask[row, col]), 0.0 if is_autoreset else 1.0)
                if col > 0:
                    self.assertEqual(
                        float(seq.prev_dones[row, col]),
                        1.0 if expected_state_reset else 0.0,
                    )
                elif not expected_state_reset:
                    # Step 0 is forced to 1.0 regardless of the stored flag.
                    self.assertEqual(float(seq.prev_dones[row, col]), 1.0)

    def test_not_full_index_ranges(self) -> None:
        buffer = AtariReplayBuffer(
            buffer_size=100, num_envs=1, obs_shape=OBS_SHAPE, device="cpu", seed=2
        )
        _fill(buffer, 5)
        self.assertEqual(buffer.size, 5)
        batch = buffer.sample_transitions(32)
        self.assertTrue((batch.obs[:, 0, 0, 0].long() <= 3).all().item())
        seq = buffer.sample_sequences(batch_size=16, seq_len=3)
        self.assertFalse(seq.has_internal_reset)
        self.assertTrue((seq.obs[:, 0, 0, 0, 0].long() <= 1).all().item())
        with self.assertRaises(ValueError):
            buffer.sample_sequences(batch_size=1, seq_len=5)

    def test_reset_rows_excluded_from_transition_bases(self) -> None:
        buffer = AtariReplayBuffer(
            buffer_size=50, num_envs=1, obs_shape=OBS_SHAPE, device="cpu", seed=3
        )
        _fill(buffer, 30, dones={9, 19}, resets={10, 20})
        batch = buffer.sample_transitions(256)
        obs_vals = batch.obs[:, 0, 0, 0].long()
        self.assertFalse((obs_vals == 10).any().item())
        self.assertFalse((obs_vals == 20).any().item())
        done_bases = batch.dones == 1.0
        self.assertTrue(done_bases.any().item())

    def test_task_balanced_transition_sampling_ignores_task_frequency(self) -> None:
        task_ids = [0] * 100 + [1] * 20
        buffer = AtariReplayBuffer(
            buffer_size=120,
            num_envs=1,
            obs_shape=OBS_SHAPE,
            device="cpu",
            seed=4,
            num_tasks=2,
        )
        _fill(buffer, 120, task_ids=task_ids)
        batch = buffer.sample_transitions(64)
        counts = np.bincount(batch.task_ids.numpy(), minlength=2)
        np.testing.assert_array_equal(counts, np.array([32, 32]))

    def test_global_uniform_transition_sampling_remains_available(self) -> None:
        task_ids = [0] * 100 + [1] * 20
        buffer = AtariReplayBuffer(
            buffer_size=120,
            num_envs=1,
            obs_shape=OBS_SHAPE,
            device="cpu",
            seed=5,
            num_tasks=2,
            sampling_mode="global_uniform",
        )
        _fill(buffer, 120, task_ids=task_ids)
        batch = buffer.sample_transitions(4096)
        task_zero_fraction = float((batch.task_ids == 0).float().mean())
        self.assertGreater(task_zero_fraction, 0.75)

    def test_task_balanced_sequence_windows_are_task_pure(self) -> None:
        task_ids = [0] * 40 + [1] * 40
        buffer = AtariReplayBuffer(
            buffer_size=80,
            num_envs=1,
            obs_shape=OBS_SHAPE,
            device="cpu",
            seed=6,
            num_tasks=2,
        )
        _fill(buffer, 80, dones={38}, resets={39}, task_ids=task_ids)
        seq = buffer.sample_sequences(batch_size=32, seq_len=6)
        sequence_tasks = []
        for row in range(seq.task_ids.shape[0]):
            valid_task_ids = seq.task_ids[row][seq.loss_mask[row].bool()]
            self.assertEqual(valid_task_ids.unique().numel(), 1)
            sequence_tasks.append(int(valid_task_ids[0]))
        counts = np.bincount(sequence_tasks, minlength=2)
        np.testing.assert_array_equal(counts, np.array([16, 16]))


class AtariReplayMemmapTest(unittest.TestCase):
    """The mmap backing exists so a preempted run resumes on identical samples."""

    def _build(self, storage_dir: str | None, reuse_existing: bool = False):
        return AtariReplayBuffer(
            buffer_size=40,
            num_envs=1,
            obs_shape=OBS_SHAPE,
            device="cpu",
            seed=11,
            num_tasks=1,
            sampling_mode="global_uniform",
            storage_dir=storage_dir,
            reuse_existing=reuse_existing,
        )

    def test_memory_backing_writes_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = self._build(None)
            _fill(buffer, 20)
            self.assertIsNone(buffer.storage_dir)
            self.assertEqual(os.listdir(tmp), [])

    def test_reopened_storage_resumes_identical_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "replay")
            original = self._build(storage)
            _fill(original, 30)
            state = original.state_dict()
            original.flush()
            reference = original.sample_transitions(16)

            reopened = self._build(storage, reuse_existing=True)
            reopened.load_state_dict(state)
            self.assertEqual(reopened.size, original.size)
            resumed = reopened.sample_transitions(16)

            # Same stored frames and same sampler RNG position.
            np.testing.assert_array_equal(
                reference.obs.numpy(), resumed.obs.numpy()
            )
            np.testing.assert_array_equal(
                reference.next_obs.numpy(), resumed.next_obs.numpy()
            )
            np.testing.assert_array_equal(
                reference.rewards.numpy(), resumed.rewards.numpy()
            )

    def test_appending_after_resume_continues_at_saved_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "replay")
            original = self._build(storage)
            _fill(original, 12)
            state = original.state_dict()
            original.flush()

            reopened = self._build(storage, reuse_existing=True)
            reopened.load_state_dict(state)
            self.assertEqual(reopened.size, 12)
            _fill(reopened, 5)
            self.assertEqual(reopened.size, 17)
            batch = reopened.sample_transitions(64)
            self.assertTrue(bool((batch.obs[:, 0, 0, 0] < 17).all()))

    def test_geometry_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "replay")
            _fill(self._build(storage), 8)
            with self.assertRaises(ValueError):
                AtariReplayBuffer(
                    buffer_size=40,
                    num_envs=1,
                    obs_shape=(3, 4, 4),  # different observation geometry
                    device="cpu",
                    seed=11,
                    sampling_mode="global_uniform",
                    storage_dir=storage,
                    reuse_existing=True,
                )

    def test_state_dict_from_other_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = self._build(os.path.join(tmp, "replay"))
            state = buffer.state_dict()
            state["capacity"] = state["capacity"] + 1
            with self.assertRaises(ValueError):
                buffer.load_state_dict(state)

    def test_reuse_without_existing_storage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self._build(os.path.join(tmp, "absent"), reuse_existing=True)

    def test_reuse_requires_storage_dir(self) -> None:
        with self.assertRaises(ValueError):
            self._build(None, reuse_existing=True)


def torch_all_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a.float(), torch.as_tensor(b).float()))


if __name__ == "__main__":
    unittest.main()
