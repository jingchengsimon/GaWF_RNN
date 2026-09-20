"""Validate fixed-task asynchronous Atari collection without requiring ALE ROMs."""

from __future__ import annotations

import pytest

from utils.training.atari.atari_envs import multitask_scheduler_states
from utils.training.train_scripts.atari_dqn import (
    _cadence_crossings,
    _resolve_task_config,
    _validate_actor_workers,
    build_arg_parser,
)


def _five_task_actor_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--env_ids",
            "ALE/Pong-v5",
            "ALE/Breakout-v5",
            "ALE/Assault-v5",
            "ALE/Seaquest-v5",
            "ALE/Riverraid-v5",
            "--action_space_mode",
            "full18",
            "--actor_workers",
            "5",
            "--num_envs",
            "5",
            "--replay_layout",
            "per_task",
            "--total_timesteps",
            "200000",
        ]
    )
    env_ids, _ = _resolve_task_config(args)
    return args, env_ids


def test_fixed_task_async_actor_protocol_requires_one_actor_per_task() -> None:
    args, env_ids = _five_task_actor_args()
    _validate_actor_workers(args, env_ids)

    args.num_envs = 1
    with pytest.raises(ValueError, match="--num_envs 5"):
        _validate_actor_workers(args, env_ids)


def test_vector_collection_preserves_four_transition_update_cadence() -> None:
    assert _cadence_crossings(0, 5, 4) == 1
    assert _cadence_crossings(5, 10, 4) == 1
    assert _cadence_crossings(15, 20, 4) == 2
    assert _cadence_crossings(995, 1000, 1000) == 1


def test_actor_workers_reject_non_divisible_budget() -> None:
    args, env_ids = _five_task_actor_args()
    args.total_timesteps = 200_001
    with pytest.raises(ValueError, match="divisible"):
        _validate_actor_workers(args, env_ids)


def test_async_scheduler_state_uses_vector_call() -> None:
    class _AsyncLikeVectorEnv:
        def call(self, name: str):
            assert name == "task_scheduler_state"
            return (
                {"mode": "fixed_task_actor", "task_idx": 0, "task_steps": 12},
                {"mode": "fixed_task_actor", "task_idx": 1, "task_steps": 13},
            )

    states = multitask_scheduler_states(_AsyncLikeVectorEnv())
    assert states[0]["task_idx"] == 0
    assert states[1]["task_steps"] == 13
