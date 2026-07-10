"""Replay buffers for Atari DQN experiments.

Frames are stored once per step as uint8 and successors are derived by index,
so a 1M-step buffer with 4 stacked 84x84 frames costs ~28 GB RAM. Gymnasium
>=1.0 vector envs use NEXT_STEP autoreset: the step after a terminal one
returns the reset observation with the chosen action ignored. Such rows are
recorded with ``resets=1`` and excluded from TD losses (rejected as transition
bases; masked out of sequence losses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TransitionBatch:
    """Single-step transitions for classic DQN updates."""

    obs: torch.Tensor  # (B, C, H, W) uint8
    actions: torch.Tensor  # (B,) long
    rewards: torch.Tensor  # (B,) float32
    dones: torch.Tensor  # (B,) float32
    next_obs: torch.Tensor  # (B, C, H, W) uint8


@dataclass(frozen=True)
class SequenceBatch:
    """Contiguous windows of length L+1 for DRQN-style updates.

    ``prev_dones`` forces 1.0 at window step 0 (zero-state unroll start), while
    ``loss_mask`` only zeroes autoreset rows; the two must stay distinct.
    """

    obs: torch.Tensor  # (B, L+1, C, H, W) uint8
    actions: torch.Tensor  # (B, L+1) long
    rewards: torch.Tensor  # (B, L+1) float32
    dones: torch.Tensor  # (B, L+1) float32
    prev_dones: torch.Tensor  # (B, L+1) float32
    loss_mask: torch.Tensor  # (B, L+1) float32


class AtariReplayBuffer:
    """Circular per-env replay storage with transition and sequence sampling."""

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, int, int],
        device: torch.device | str,
        seed: int,
    ) -> None:
        if buffer_size < num_envs:
            raise ValueError("buffer_size must be at least num_envs")
        self.num_envs = int(num_envs)
        self.capacity = int(buffer_size) // self.num_envs
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._full = False

        self._obs = np.zeros((self.capacity, self.num_envs, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity, self.num_envs), dtype=np.int64)
        self._rewards = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self._dones = np.zeros((self.capacity, self.num_envs), dtype=np.uint8)
        self._resets = np.zeros((self.capacity, self.num_envs), dtype=np.uint8)

    @property
    def size(self) -> int:
        """Number of filled slots per env column."""
        return self.capacity if self._full else self._pos

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        resets: np.ndarray,
    ) -> None:
        obs = np.asarray(obs)
        if obs.shape != (self.num_envs, *self.obs_shape):
            raise ValueError(
                f"obs must have shape {(self.num_envs, *self.obs_shape)}, got {obs.shape}"
            )
        self._obs[self._pos] = obs
        self._actions[self._pos] = np.asarray(actions).reshape(self.num_envs)
        self._rewards[self._pos] = np.asarray(rewards).reshape(self.num_envs)
        self._dones[self._pos] = np.asarray(dones).reshape(self.num_envs).astype(np.uint8)
        self._resets[self._pos] = np.asarray(resets).reshape(self.num_envs).astype(np.uint8)
        self._pos += 1
        if self._pos == self.capacity:
            self._pos = 0
            self._full = True

    def _physical(self, logical: np.ndarray) -> np.ndarray:
        return (self._pos - self.size + logical) % self.capacity

    def sample_transitions(self, batch_size: int) -> TransitionBatch:
        """Sample iid transitions; the newest slot has no successor and autoreset
        rows are invalid bases, so both are rejected and redrawn."""
        if self.size < 2:
            raise ValueError("Need at least 2 stored steps to sample transitions")
        logical = self._rng.integers(0, self.size - 1, size=batch_size)
        env_idx = self._rng.integers(0, self.num_envs, size=batch_size)
        phys = self._physical(logical)
        invalid = self._resets[phys, env_idx].astype(bool)
        while invalid.any():
            redraw = self._rng.integers(0, self.size - 1, size=int(invalid.sum()))
            logical[invalid] = redraw
            phys = self._physical(logical)
            invalid = self._resets[phys, env_idx].astype(bool)
        next_phys = self._physical(logical + 1)

        return TransitionBatch(
            obs=torch.as_tensor(self._obs[phys, env_idx], device=self.device),
            actions=torch.as_tensor(self._actions[phys, env_idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[phys, env_idx], device=self.device),
            dones=torch.as_tensor(
                self._dones[phys, env_idx].astype(np.float32), device=self.device
            ),
            next_obs=torch.as_tensor(self._obs[next_phys, env_idx], device=self.device),
        )

    def sample_sequences(self, batch_size: int, seq_len: int) -> SequenceBatch:
        """Sample contiguous windows of ``seq_len + 1`` steps within one env.

        Episode boundaries inside a window are kept; the caller resets recurrent
        state via ``prev_dones`` and drops autoreset rows via ``loss_mask``.
        """
        window = seq_len + 1
        if self.size < window:
            raise ValueError(
                f"Need at least {window} stored steps to sample sequences, have {self.size}"
            )
        starts = self._rng.integers(0, self.size - window + 1, size=batch_size)
        env_idx = self._rng.integers(0, self.num_envs, size=batch_size)
        logical = starts[:, None] + np.arange(window)[None, :]
        phys = self._physical(logical)
        env_col = env_idx[:, None]

        resets = self._resets[phys, env_col].astype(np.float32)
        prev_dones = resets.copy()
        prev_dones[:, 0] = 1.0
        loss_mask = 1.0 - resets

        return SequenceBatch(
            obs=torch.as_tensor(self._obs[phys, env_col], device=self.device),
            actions=torch.as_tensor(self._actions[phys, env_col], device=self.device),
            rewards=torch.as_tensor(self._rewards[phys, env_col], device=self.device),
            dones=torch.as_tensor(
                self._dones[phys, env_col].astype(np.float32), device=self.device
            ),
            prev_dones=torch.as_tensor(prev_dones, device=self.device),
            loss_mask=torch.as_tensor(loss_mask, device=self.device),
        )
