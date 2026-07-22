"""Replay buffers for Atari DQN experiments.

Frames are stored once per step as uint8 and successors are derived by index,
so a 1M-step buffer with 4 stacked 84x84 frames costs ~28 GB. Gymnasium
>=1.0 vector envs use NEXT_STEP autoreset: the step after a terminal one
returns the reset observation with the chosen action ignored. Such rows are
recorded with ``resets=1`` and excluded from TD losses (rejected as transition
bases; masked out of sequence losses). Task ids are sampling metadata only and
are never model inputs.

Storage is anonymous RAM by default. Passing ``storage_dir`` backs the six
arrays with ``np.memmap`` files instead, so a preempted run can reopen the exact
same replay contents with ``reuse_existing=True`` and resume from a checkpoint
that only carries the small position/RNG metadata. That path costs disk instead
of RAM; the caller is responsible for the quota budget and for deleting the
directory once the run completes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


REPLAY_SAMPLING_MODES = ("task_balanced", "global_uniform")

REPLAY_META_FILENAME = "meta.json"
REPLAY_STATE_FORMAT_VERSION = 1

# (attribute, filename, dtype, whether the array carries the observation shape)
_REPLAY_ARRAY_SPECS: tuple[tuple[str, str, Any, bool], ...] = (
    ("_obs", "obs.dat", np.uint8, True),
    ("_actions", "actions.dat", np.int64, False),
    ("_rewards", "rewards.dat", np.float32, False),
    ("_dones", "dones.dat", np.uint8, False),
    ("_resets", "resets.dat", np.uint8, False),
    ("_task_ids", "task_ids.dat", np.int16, False),
)


@dataclass(frozen=True)
class TransitionBatch:
    """Single-step transitions for classic DQN updates."""

    obs: torch.Tensor  # (B, C, H, W) uint8
    actions: torch.Tensor  # (B,) long
    rewards: torch.Tensor  # (B,) float32
    dones: torch.Tensor  # (B,) float32
    next_obs: torch.Tensor  # (B, C, H, W) uint8
    task_ids: torch.Tensor  # (B,) long; sampling/loss metadata only


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
    task_ids: torch.Tensor  # (B, L+1) long; sampling/loss metadata only
    has_internal_reset: bool  # CPU-side reset check for fused recurrent scans


def _assert_replay_geometry_matches(
    stored: dict[str, Any],
    expected: dict[str, Any],
    *,
    source: str,
) -> None:
    """Reject any replay storage or state whose shape contract differs."""
    mismatches = []
    for key, expected_value in expected.items():
        stored_value = stored.get(key)
        if key == "obs_shape":
            stored_value = list(stored_value) if stored_value is not None else None
            expected_value = list(expected_value)
        if stored_value != expected_value:
            mismatches.append(f"{key}: stored={stored_value!r} expected={expected_value!r}")
    if mismatches:
        raise ValueError(
            f"Replay geometry mismatch from {source}: " + "; ".join(mismatches)
        )


class AtariReplayBuffer:
    """Circular per-env replay storage with transition and sequence sampling."""

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_shape: tuple[int, int, int],
        device: torch.device | str,
        seed: int,
        num_tasks: int = 1,
        sampling_mode: str = "task_balanced",
        storage_dir: str | None = None,
        reuse_existing: bool = False,
    ) -> None:
        if buffer_size < num_envs:
            raise ValueError("buffer_size must be at least num_envs")
        self.num_envs = int(num_envs)
        self.capacity = int(buffer_size) // self.num_envs
        self.obs_shape = tuple(int(dim) for dim in obs_shape)
        self.device = torch.device(device)
        self.num_tasks = int(num_tasks)
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be at least 1")
        if sampling_mode not in REPLAY_SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode must be one of {REPLAY_SAMPLING_MODES}, got {sampling_mode!r}"
            )
        if reuse_existing and storage_dir is None:
            raise ValueError("reuse_existing requires storage_dir")
        self.sampling_mode = sampling_mode
        self.storage_dir = storage_dir
        self._rng = np.random.default_rng(seed)
        self._pos = 0
        self._full = False

        if storage_dir is None:
            self._allocate_in_memory()
        else:
            self._allocate_memmap(storage_dir, reuse_existing=reuse_existing)
        self._stored_task_counts = np.zeros(self.num_tasks, dtype=np.int64)

    def _array_shape(self, carries_obs_shape: bool) -> tuple[int, ...]:
        if carries_obs_shape:
            return (self.capacity, self.num_envs, *self.obs_shape)
        return (self.capacity, self.num_envs)

    def _allocate_in_memory(self) -> None:
        for attribute, _filename, dtype, carries_obs_shape in _REPLAY_ARRAY_SPECS:
            setattr(
                self,
                attribute,
                np.zeros(self._array_shape(carries_obs_shape), dtype=dtype),
            )

    def _geometry(self) -> dict[str, Any]:
        return {
            "format_version": REPLAY_STATE_FORMAT_VERSION,
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "obs_shape": list(self.obs_shape),
            "num_tasks": self.num_tasks,
            "sampling_mode": self.sampling_mode,
        }

    def _allocate_memmap(self, storage_dir: str, *, reuse_existing: bool) -> None:
        meta_path = os.path.join(storage_dir, REPLAY_META_FILENAME)
        geometry = self._geometry()
        if reuse_existing:
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Cannot reuse replay storage without {meta_path}"
                )
            with open(meta_path, "r", encoding="utf-8") as stream:
                stored = json.load(stream)
            _assert_replay_geometry_matches(stored, geometry, source=meta_path)
            mode = "r+"
        else:
            os.makedirs(storage_dir, exist_ok=True)
            mode = "w+"

        for attribute, filename, dtype, carries_obs_shape in _REPLAY_ARRAY_SPECS:
            path = os.path.join(storage_dir, filename)
            if reuse_existing and not os.path.isfile(path):
                raise FileNotFoundError(f"Missing replay storage file: {path}")
            setattr(
                self,
                attribute,
                np.memmap(
                    path,
                    dtype=dtype,
                    mode=mode,
                    shape=self._array_shape(carries_obs_shape),
                ),
            )

        if not reuse_existing:
            with open(meta_path, "w", encoding="utf-8") as stream:
                json.dump(geometry, stream, indent=2, sort_keys=True)

    def flush(self) -> None:
        """Persist memmap pages so a checkpoint never outruns the stored data.

        No-op for the in-memory backing.
        """
        if self.storage_dir is None:
            return
        for attribute, _filename, _dtype, _carries_obs_shape in _REPLAY_ARRAY_SPECS:
            getattr(self, attribute).flush()

    def close(self) -> None:
        """Flush and drop the memmap handles so the files can be removed.

        The buffer is unusable afterwards; call this only once the run's results
        are written.
        """
        if self.storage_dir is None:
            return
        self.flush()
        for attribute, _filename, _dtype, _carries_obs_shape in _REPLAY_ARRAY_SPECS:
            setattr(self, attribute, None)

    def state_dict(self) -> dict[str, Any]:
        """Small resumable metadata; the samples themselves live in the memmaps."""
        state = self._geometry()
        state.update(
            {
                "pos": int(self._pos),
                "full": bool(self._full),
                "stored_task_counts": self._stored_task_counts.tolist(),
                "rng_state": self._rng.bit_generator.state,
                "storage_dir": self.storage_dir,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore position, task counts, and the sampler RNG.

        Geometry must match exactly; a mismatch means the checkpoint and the
        storage describe different experiments and resuming would silently mix
        them.
        """
        _assert_replay_geometry_matches(state, self._geometry(), source="checkpoint")
        pos = int(state["pos"])
        if not 0 <= pos < self.capacity:
            raise ValueError(f"Replay pos={pos} outside capacity={self.capacity}")
        stored_task_counts = np.asarray(state["stored_task_counts"], dtype=np.int64)
        if stored_task_counts.shape != (self.num_tasks,):
            raise ValueError(
                "stored_task_counts shape "
                f"{stored_task_counts.shape} does not match num_tasks={self.num_tasks}"
            )
        self._pos = pos
        self._full = bool(state["full"])
        self._stored_task_counts = stored_task_counts
        self._rng.bit_generator.state = state["rng_state"]

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
        task_ids: np.ndarray | None = None,
    ) -> None:
        obs = np.asarray(obs)
        if obs.shape != (self.num_envs, *self.obs_shape):
            raise ValueError(
                f"obs must have shape {(self.num_envs, *self.obs_shape)}, got {obs.shape}"
            )
        if task_ids is None:
            task_ids = np.zeros(self.num_envs, dtype=np.int16)
        task_ids = np.asarray(task_ids).reshape(self.num_envs).astype(np.int16)
        if np.any(task_ids < 0) or np.any(task_ids >= self.num_tasks):
            raise ValueError(f"task_ids must be in [0, {self.num_tasks - 1}]")
        if self._full:
            old_task_ids = self._task_ids[self._pos]
            self._stored_task_counts -= np.bincount(
                old_task_ids, minlength=self.num_tasks
            )

        self._obs[self._pos] = obs
        self._actions[self._pos] = np.asarray(actions).reshape(self.num_envs)
        self._rewards[self._pos] = np.asarray(rewards).reshape(self.num_envs)
        self._dones[self._pos] = np.asarray(dones).reshape(self.num_envs).astype(np.uint8)
        self._resets[self._pos] = np.asarray(resets).reshape(self.num_envs).astype(np.uint8)
        self._task_ids[self._pos] = task_ids
        self._stored_task_counts += np.bincount(task_ids, minlength=self.num_tasks)
        self._pos += 1
        if self._pos == self.capacity:
            self._pos = 0
            self._full = True

    def _physical(self, logical: np.ndarray) -> np.ndarray:
        return (self._pos - self.size + logical) % self.capacity

    def _balanced_task_targets(self, batch_size: int) -> np.ndarray:
        if batch_size < self.num_tasks:
            raise ValueError(
                f"batch_size={batch_size} must be at least num_tasks={self.num_tasks}"
            )
        missing = np.flatnonzero(self._stored_task_counts < 2)
        if missing.size:
            raise ValueError(f"Not enough replay rows for tasks {missing.tolist()}")
        repeats = (batch_size + self.num_tasks - 1) // self.num_tasks
        targets = np.tile(np.arange(self.num_tasks, dtype=np.int16), repeats)[:batch_size]
        self._rng.shuffle(targets)
        return targets

    def _sample_transition_indices(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logical = self._rng.integers(0, self.size - 1, size=batch_size)
        env_idx = self._rng.integers(0, self.num_envs, size=batch_size)
        targets = (
            self._balanced_task_targets(batch_size)
            if self.sampling_mode == "task_balanced"
            else np.full(batch_size, -1, dtype=np.int16)
        )
        for _ in range(10_000):
            phys = self._physical(logical)
            invalid = self._resets[phys, env_idx].astype(bool)
            if self.sampling_mode == "task_balanced":
                invalid |= self._task_ids[phys, env_idx] != targets
            if not invalid.any():
                return logical, env_idx, phys
            redraw_count = int(invalid.sum())
            logical[invalid] = self._rng.integers(0, self.size - 1, size=redraw_count)
            env_idx[invalid] = self._rng.integers(0, self.num_envs, size=redraw_count)
        raise RuntimeError("Could not sample valid replay transitions after 10,000 redraws")

    def sample_transitions(self, batch_size: int) -> TransitionBatch:
        """Sample iid transitions; the newest slot has no successor and autoreset
        rows are invalid bases, so both are rejected and redrawn."""
        if self.size < 2:
            raise ValueError("Need at least 2 stored steps to sample transitions")
        logical, env_idx, phys = self._sample_transition_indices(batch_size)
        next_phys = self._physical(logical + 1)

        return TransitionBatch(
            obs=torch.as_tensor(self._obs[phys, env_idx], device=self.device),
            actions=torch.as_tensor(self._actions[phys, env_idx], device=self.device),
            rewards=torch.as_tensor(self._rewards[phys, env_idx], device=self.device),
            dones=torch.as_tensor(
                self._dones[phys, env_idx].astype(np.float32), device=self.device
            ),
            next_obs=torch.as_tensor(self._obs[next_phys, env_idx], device=self.device),
            task_ids=torch.as_tensor(
                self._task_ids[phys, env_idx].astype(np.int64), device=self.device
            ),
        )

    def _sample_sequence_indices(
        self,
        batch_size: int,
        window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        starts = self._rng.integers(0, self.size - window + 1, size=batch_size)
        env_idx = self._rng.integers(0, self.num_envs, size=batch_size)
        if self.sampling_mode == "global_uniform":
            return starts, env_idx

        targets = self._balanced_task_targets(batch_size)
        offsets = np.arange(window)[None, :]
        for _ in range(10_000):
            logical = starts[:, None] + offsets
            phys = self._physical(logical)
            env_col = env_idx[:, None]
            resets = self._resets[phys, env_col].astype(bool)
            task_ids = self._task_ids[phys, env_col]
            valid_rows = ~resets
            wrong_task = np.any(valid_rows & (task_ids != targets[:, None]), axis=1)
            no_loss_row = ~np.any(valid_rows[:, :-1], axis=1)
            invalid = wrong_task | no_loss_row
            if not invalid.any():
                return starts, env_idx
            redraw_count = int(invalid.sum())
            starts[invalid] = self._rng.integers(
                0, self.size - window + 1, size=redraw_count
            )
            env_idx[invalid] = self._rng.integers(
                0, self.num_envs, size=redraw_count
            )
        raise RuntimeError("Could not sample task-pure replay sequences after 10,000 redraws")

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
        starts, env_idx = self._sample_sequence_indices(batch_size, window)
        logical = starts[:, None] + np.arange(window)[None, :]
        phys = self._physical(logical)
        env_col = env_idx[:, None]

        resets = self._resets[phys, env_col].astype(np.float32)
        prev_dones = resets.copy()
        # NEXT_STEP autoreset first returns one invalid row containing the old
        # terminal observation. Reset again on the following valid row so that
        # this ignored observation cannot leak state into the new episode/task.
        prev_dones[:, 1:] = np.maximum(prev_dones[:, 1:], resets[:, :-1])
        prev_dones[:, 0] = 1.0
        loss_mask = 1.0 - resets
        has_internal_reset = bool(np.any(prev_dones[:, 1:] != 0))

        return SequenceBatch(
            obs=torch.as_tensor(self._obs[phys, env_col], device=self.device),
            actions=torch.as_tensor(self._actions[phys, env_col], device=self.device),
            rewards=torch.as_tensor(self._rewards[phys, env_col], device=self.device),
            dones=torch.as_tensor(
                self._dones[phys, env_col].astype(np.float32), device=self.device
            ),
            prev_dones=torch.as_tensor(prev_dones, device=self.device),
            loss_mask=torch.as_tensor(loss_mask, device=self.device),
            task_ids=torch.as_tensor(
                self._task_ids[phys, env_col].astype(np.int64), device=self.device
            ),
            has_internal_reset=has_internal_reset,
        )
