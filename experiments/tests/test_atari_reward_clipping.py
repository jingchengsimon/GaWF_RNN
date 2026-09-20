"""Regression tests for Atari DQN reward-clipping protocol selection."""

from __future__ import annotations

import torch

from utils.training.checkpointing import validate_resume_protocol
from utils.training.train_scripts.atari_dqn import (
    RESUME_ARG_KEYS,
    _normalize_legacy_resume_args,
    _td_rewards,
    build_arg_parser,
)


def test_td_rewards_preserve_or_clip_large_signed_values() -> None:
    rewards = torch.tensor([-26_000.0, -6.0, 0.0, 7.0])
    bootstrap = torch.zeros_like(rewards)
    clipped_target = _td_rewards(rewards, True) + 0.999 * bootstrap
    raw_target = _td_rewards(rewards, False) + 0.999 * bootstrap
    assert clipped_target.tolist() == [-1.0, -1.0, 0.0, 1.0]
    assert raw_target.tolist() == [-26_000.0, -6.0, 0.0, 7.0]


def test_legacy_resume_defaults_to_clipped_rewards_and_rejects_new_protocol() -> None:
    args = build_arg_parser().parse_args([])
    saved_args = {key: getattr(args, key) for key in RESUME_ARG_KEYS if key != "reward_clip"}
    saved_args.setdefault("total_timesteps", args.total_timesteps)
    saved_args.setdefault("learning_rate", args.learning_rate)
    checkpoint = {"format_version": 1, "args": saved_args, "learning_rate": args.learning_rate}
    _normalize_legacy_resume_args(checkpoint["args"])
    validate_resume_protocol(
        checkpoint, args, RESUME_ARG_KEYS, expected_format_version=1, learning_rate=args.learning_rate
    )
    raw_args = build_arg_parser().parse_args(["--no_reward_clip"])
    try:
        validate_resume_protocol(
            checkpoint,
            raw_args,
            RESUME_ARG_KEYS,
            expected_format_version=1,
            learning_rate=raw_args.learning_rate,
        )
    except ValueError as error:
        assert "reward_clip" in str(error)
    else:
        raise AssertionError("A legacy clipped checkpoint must reject raw-reward resume")
