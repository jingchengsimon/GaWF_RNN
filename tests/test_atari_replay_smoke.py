"""Smoke tests for the Atari replay buffer (no Gymnasium/ALE required)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_replay import AtariReplayBuffer

OBS_SHAPE = (2, 4, 4)


def _fill(buffer: AtariReplayBuffer, n_steps: int, dones=(), resets=()) -> None:
    """Add ``n_steps`` rows whose obs encode the global step index mod 256."""
    for step in range(n_steps):
        obs = np.full((1, *OBS_SHAPE), step % 256, dtype=np.uint8)
        buffer.add(
            obs=obs,
            actions=np.array([step % 5]),
            rewards=np.array([float(step)], dtype=np.float32),
            dones=np.array([1 if step in dones else 0]),
            resets=np.array([1 if step in resets else 0]),
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
        self.assertTrue((seq.loss_mask >= 0).all().item())
        for row in range(obs_vals.shape[0]):
            for col in range(obs_vals.shape[1]):
                step = int(obs_vals[row, col])
                expected_reset = 1.0 if step in {21, 24} else 0.0
                self.assertEqual(float(seq.loss_mask[row, col]), 1.0 - expected_reset)
                if col > 0:
                    self.assertEqual(float(seq.prev_dones[row, col]), expected_reset)
                elif expected_reset == 0.0:
                    # Step 0 is forced to 1.0 regardless of the stored flag.
                    self.assertEqual(float(seq.prev_dones[row, col]), 1.0)
        self.assertTrue(
            torch_all_equal(seq.prev_dones[:, 1:], 1.0 - seq.loss_mask[:, 1:])
        )

    def test_not_full_index_ranges(self) -> None:
        buffer = AtariReplayBuffer(
            buffer_size=100, num_envs=1, obs_shape=OBS_SHAPE, device="cpu", seed=2
        )
        _fill(buffer, 5)
        self.assertEqual(buffer.size, 5)
        batch = buffer.sample_transitions(32)
        self.assertTrue((batch.obs[:, 0, 0, 0].long() <= 3).all().item())
        seq = buffer.sample_sequences(batch_size=16, seq_len=3)
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


def torch_all_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a.float(), torch.as_tensor(b).float()))


if __name__ == "__main__":
    unittest.main()
