"""Train Atari DQN models: classic CNN-DQN and a GaWF-gated recurrent variant.

This entry point mirrors ``utils.training.train_scripts.atari_a2c`` conventions but stays
decoupled from it. The classic branch uses iid single-transition replay; the GaWF branch uses
DRQN-style sequence replay, unrolling from zero state with detached previous
Q-values as gate feedback.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.training.atari.atari_stage_profile import AtariStageProfile
from utils.training.atari.atari_dqn_models import AtariQNetwork, AtariQNetworkState
from utils.training.atari.atari_envs import (
    ATARI_ENV_PROTOCOLS,
    ATARI_PILOT_ENVS,
    ATARI_TASK_SCHEDULES,
    make_fixed_multitask_async_vector_atari_env,
    make_multitask_vector_atari_env,
    make_vector_atari_env,
    multitask_scheduler_states,
)
from utils.training.atari.atari_replay import (
    REPLAY_SAMPLING_MODES,
    AtariReplayBuffer,
    PerTaskAtariReplayBuffer,
)
from utils.training.atari.atari_train_acceleration import (
    AtariAcceleration,
    configure_atari_acceleration,
)
from utils.training.atari.atari_train_utils import (
    ensure_dir,
    obs_to_tensor,
    save_json,
    select_device,
    set_atari_seed,
    to_channel_first_obs,
)

ReplayBuffer = AtariReplayBuffer | PerTaskAtariReplayBuffer
from utils.training.recurrent_cores import configure_gawf_feedback_acceleration
from utils.training.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
    reconcile_history,
    restore_rng_state,
    rng_state,
    validate_resume_protocol,
)


def _json_safe(value: Any) -> Any:
    """Replace unavailable non-finite metric scalars with JSON ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Atari DQN models")
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--env_id", type=str, default="ALE/Pong-v5", choices=ATARI_PILOT_ENVS
    )
    env_group.add_argument(
        "--env_ids",
        type=str,
        nargs="+",
        default=None,
        choices=ATARI_PILOT_ENVS,
        help="Phase0 task list. Multiple games are selected only at episode boundaries.",
    )
    parser.add_argument(
        "--action_space_mode",
        type=str,
        default="auto",
        choices=["auto", "minimal", "full18"],
        help="auto preserves minimal actions for one game and uses canonical 18 for multi-task.",
    )
    parser.add_argument(
        "--task_schedule",
        type=str,
        default="transition_balanced",
        choices=ATARI_TASK_SCHEDULES,
        help="Phase0 episode-boundary scheduler; default balances collected transitions.",
    )
    parser.add_argument(
        "--atari_env_protocol",
        choices=ATARI_ENV_PROTOCOLS,
        default="baseline",
        help="Versioned Atari environment boundary and action-mapping protocol.",
    )
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn"])
    parser.add_argument(
        "--model_type",
        type=str,
        default="ann",
        choices=["ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba"],
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default=None,
        choices=["none", "qvalues"],
        help="GaWF gate feedback source; defaults to 'qvalues' for gawf, 'none' otherwise.",
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gawf_feedback_lr_scale", type=float, default=1.0)
    parser.add_argument("--encoder_feature_dim", type=int, default=512)
    parser.add_argument("--core_dropout", type=float, default=0.0)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="ALE frames advanced per environment step; Pong DQN defaults to 1.",
    )
    parser.add_argument(
        "--flicker_prob",
        type=float,
        default=0.0,
        help="Per-timestep probability of blanking the whole screen "
        "(Flickering-Atari POMDP, Hausknecht & Stone 2015). 0 disables it.",
    )
    # S5/Mamba core sizing (parameter-matched to the LSTM anchor by default; see
    # experiments/atari/atari_ssm_param_match.py). For Mamba, --ssm_state_size
    # maps to d_state.
    parser.add_argument("--ssm_d_model", type=int, default=256)
    parser.add_argument("--ssm_state_size", type=int, default=128)
    parser.add_argument("--ssm_num_layers", type=int, default=1)
    parser.add_argument(
        "--ssm_context_len",
        type=int,
        default=None,
        help="Rolling-window length for S5/Mamba online stepping; defaults to --seq_len.",
    )
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--actor_workers",
        type=int,
        default=0,
        choices=[0, 5],
        help=(
            "0 keeps synchronous episode-balanced collection. 5 uses one spawn-safe "
            "asynchronous actor per task; it requires exactly five tasks, --num_envs 5, "
            "and --replay_layout per_task."
        ),
    )
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument(
        "--replay_layout",
        type=str,
        default="shared",
        choices=["shared", "per_task"],
        help="Use one shared replay or one independent replay partition per task.",
    )
    parser.add_argument(
        "--replay_sampling",
        type=str,
        default="task_balanced",
        choices=REPLAY_SAMPLING_MODES,
        help="Replay sampler. Task ids are sampling/loss metadata and never model inputs.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_decay_step", type=int, default=0)
    parser.add_argument("--learning_rate_decay_scale", type=float, default=1.0)
    parser.add_argument(
        "--learning_rate_decay_per_task_steps",
        type=int,
        default=0,
        help=(
            "For multi-task runs, decay the shared optimizer only after every task reaches "
            "this many environment steps. 0 uses --learning_rate_decay_step."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--no_reward_clip",
        action="store_false",
        dest="reward_clip",
        help="Use raw environment rewards in TD targets instead of the historical [-1, 1] clip.",
    )
    parser.set_defaults(reward_clip=True)
    parser.add_argument("--learning_starts", type=int, default=20_000)
    parser.add_argument(
        "--learning_starts_per_task",
        type=int,
        default=0,
        help=(
            "For multi-task runs, delay updates until every task has this many valid "
            "environment steps. 0 preserves the historical global-only gate."
        ),
    )
    parser.add_argument("--start_epsilon", type=float, default=1.0)
    parser.add_argument("--end_epsilon", type=float, default=0.01)
    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=None,
        help="Fixed global steps for linear epsilon decay; defaults to 500000.",
    )
    parser.add_argument(
        "--exploration_fraction",
        type=float,
        default=None,
        help="Legacy alternative to --exploration_steps; only use for historical reproduction.",
    )
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--sequences_per_batch", type=int, default=8)
    parser.add_argument("--target_network_frequency", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--double_dqn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--result_suffix", type=str, default="atari_dqn")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--profile_stages", action="store_true",
                        help="Capture a bounded post-warmup CPU/CUDA trace and stage timings.")
    parser.add_argument(
        "--record_timing",
        action="store_true",
        help=(
            "Record host-wall environment/replay I/O and optimizer-update timing in final metrics. "
            "This is for performance benchmarks and does not change training behaviour."
        ),
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",
        choices=["none", "bfloat16", "float16"],
        help="CUDA autocast dtype; BF16 is recommended on Amarel L40S GPUs.",
    )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    parser.add_argument(
        "--fused_optimizer",
        action="store_true",
        help="Use CUDA fused Adam for supported non-S5 models.",
    )
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--checkpoint_interval_steps",
        type=int,
        default=0,
        help=(
            "Atomically save a resumable checkpoint every N environment steps. "
            "0 disables checkpointing and preserves the historical single-save behaviour."
        ),
    )
    parser.add_argument(
        "--diagnostic_checkpoint_steps",
        type=int,
        nargs="*",
        default=[],
        help="Immutable model-only snapshots for offline evaluation; does not save replay again.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume model, optimizer, replay position, counters, and RNG from a checkpoint.",
    )
    parser.add_argument(
        "--init_weights_from",
        default=None,
        help="Initialize only model weights from a completed final state_dict.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Resume from <save_dir>/checkpoint.pth when it exists.",
    )
    parser.add_argument(
        "--allow_total_timesteps_extension",
        action="store_true",
        help=(
            "Allow a resumable checkpoint target to increase while every other protocol "
            "field remains identical."
        ),
    )
    parser.add_argument(
        "--replay_backing",
        type=str,
        default="memory",
        choices=["memory", "mmap"],
        help=(
            "mmap backs replay with files under <save_dir>/replay so a preempted run "
            "resumes with the exact same transitions; memory keeps the historical "
            "anonymous-RAM allocation."
        ),
    )
    parser.add_argument(
        "--keep_replay_on_success",
        action="store_true",
        help="Keep <save_dir>/replay after a completed run instead of reclaiming the space.",
    )
    return parser


# Every field that defines the scientific protocol. A resume that disagrees on
# any of these would splice two different experiments into one result directory.
RESUME_ARG_KEYS = (
    "env_id",
    "env_ids",
    "action_space_mode",
    "task_schedule",
    "atari_env_protocol",
    "algo",
    "model_type",
    "feedback_mode",
    "hidden_size",
    "num_layers",
    "gawf_feedback_lr_scale",
    "encoder_feature_dim",
    "core_dropout",
    "frame_stack",
    "frame_skip",
    "flicker_prob",
    "ssm_d_model",
    "ssm_state_size",
    "ssm_num_layers",
    "ssm_context_len",
    "total_timesteps",
    "num_envs",
    "actor_workers",
    "buffer_size",
    "replay_layout",
    "replay_sampling",
    "learning_rate",
    "learning_rate_decay_step",
    "learning_rate_decay_scale",
    "learning_rate_decay_per_task_steps",
    "gamma",
    "reward_clip",
    "learning_starts",
    "learning_starts_per_task",
    "start_epsilon",
    "end_epsilon",
    "exploration_steps",
    "exploration_fraction",
    "train_frequency",
    "batch_size",
    "seq_len",
    "sequences_per_batch",
    "target_network_frequency",
    "max_grad_norm",
    "double_dqn",
    "seed",
)

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.pth"
REPLAY_SUBDIR = "replay"


SequenceForward = Callable[
    [torch.Tensor, torch.Tensor, AtariQNetworkState | None, bool | None],
    tuple[torch.Tensor, AtariQNetworkState | None],
]


def _resume_validation_keys(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> tuple[tuple[str, ...], int | None]:
    """Return protocol keys and the original budget for an explicit extension resume."""

    previous_origin = checkpoint.get("extended_from_total_timesteps")
    if not args.allow_total_timesteps_extension:
        return RESUME_ARG_KEYS, int(previous_origin) if previous_origin is not None else None
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict) or "total_timesteps" not in saved_args:
        raise ValueError("Checkpoint is missing saved total_timesteps")
    saved_total = int(saved_args["total_timesteps"])
    requested_total = int(args.total_timesteps)
    if requested_total < saved_total:
        raise ValueError(
            "A total-timestep extension cannot reduce the checkpoint target: "
            f"saved={saved_total} requested={requested_total}"
        )
    if requested_total == saved_total and previous_origin is None:
        raise ValueError(
            "--allow_total_timesteps_extension requires a larger target on first use"
        )
    origin = int(previous_origin) if previous_origin is not None else saved_total
    return tuple(key for key in RESUME_ARG_KEYS if key != "total_timesteps"), origin


def _normalize_legacy_resume_args(saved_args: dict[str, Any]) -> None:
    """Fill saved Atari protocol fields that were absent before their CLI existed."""
    saved_args.setdefault("atari_env_protocol", "baseline")
    saved_args.setdefault("reward_clip", True)
    saved_args.setdefault("actor_workers", 0)
    # Checkpoints created before the per-task gate retain historical semantics.
    saved_args.setdefault("learning_starts_per_task", 0)
    # Historical checkpoints parameterized epsilon decay as a fraction.
    if "exploration_steps" not in saved_args:
        fraction = saved_args.get("exploration_fraction")
        if fraction is not None:
            saved_args["exploration_steps"] = int(
                float(fraction) * int(saved_args["total_timesteps"])
            )


def _step_with_sequence_forward(
    forward_sequence: SequenceForward,
    obs: torch.Tensor,
    prev_done: torch.Tensor,
    state: AtariQNetworkState | None = None,
) -> tuple[torch.Tensor, AtariQNetworkState | None]:
    q_values, next_state = forward_sequence(
        obs.unsqueeze(1),
        prev_done.view(-1, 1),
        state,
        False,
    )
    return q_values[:, 0, :], next_state


def _resolve_feedback_mode(args: argparse.Namespace) -> str:
    if args.feedback_mode is not None:
        return args.feedback_mode
    return "qvalues" if args.model_type == "gawf" else "none"


def _build_atari_optimizer(
    model: torch.nn.Module,
    *,
    model_type: str,
    learning_rate: float,
    gawf_feedback_lr_scale: float,
    use_fused_optimizer: bool,
) -> optim.Optimizer:
    """Build Adam with a non-fused compatibility path for S5."""

    if model_type == "s5":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            fused=False,
        )

    parameter_groups: list[torch.nn.Parameter] | list[dict[str, object]]
    if model_type != "gawf":
        parameter_groups = list(model.parameters())
    else:
        gate_params = [
            param
            for name, param in model.named_parameters()
            if name.startswith("core.U") or name.startswith("core.V")
        ]
        gate_ids = {id(param) for param in gate_params}
        base_params = [param for param in model.parameters() if id(param) not in gate_ids]
        parameter_groups = [
            {"params": base_params, "lr": learning_rate},
            {
                "params": gate_params,
                "lr": learning_rate * gawf_feedback_lr_scale,
                "weight_decay": 0.0,
            },
        ]
    adam_kwargs = {"fused": True} if use_fused_optimizer else {}
    return optim.Adam(
        parameter_groups,
        lr=learning_rate,
        **adam_kwargs,
    )


def _supports_fused_adam_params(model: torch.nn.Module) -> bool:
    """Return whether every trainable parameter is real floating point."""

    return all(param.is_floating_point() for param in model.parameters())


def _checkpoint_sha256(path: Path) -> str:
    """Return the SHA256 digest of one immutable initialization checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_initial_weights(
    model: AtariQNetwork,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, str]:
    """Load only a completed model state_dict and return immutable provenance."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Initial model weights not found: {path}")
    state_dict = torch.load(path, map_location=device)
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f"Expected a non-empty final model state_dict: {path}")
    if "model" in state_dict or "optimizer" in state_dict or "replay" in state_dict:
        raise ValueError(
            "--init_weights_from requires the completed final model state_dict, "
            f"not a resumable training checkpoint: {path}"
        )
    filtered = {
        str(key): value
        for key, value in state_dict.items()
        if not str(key).endswith("prev_feedback")
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    logger = logging.getLogger("train_atari_dqn")
    logger.info(
        "initialized model weights=%s missing_keys=%s unexpected_keys=%s",
        path,
        incompatible.missing_keys,
        incompatible.unexpected_keys,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Initial checkpoint is incompatible with the requested model: {path}")
    return {
        "mode": "weights_only",
        "source_checkpoint": str(path),
        "source_checkpoint_name": path.name,
        "source_checkpoint_sha256": _checkpoint_sha256(path),
    }


def _resolve_task_config(args: argparse.Namespace) -> tuple[tuple[str, ...], str]:
    env_ids = tuple(args.env_ids) if args.env_ids is not None else (args.env_id,)
    if len(set(env_ids)) != len(env_ids):
        raise ValueError("env_ids must be unique")
    action_space_mode = args.action_space_mode
    if action_space_mode == "auto":
        action_space_mode = "full18" if len(env_ids) > 1 else "minimal"
    if len(env_ids) > 1 and action_space_mode != "full18":
        raise ValueError("Multi-task Atari requires --action_space_mode full18")
    return env_ids, action_space_mode


def _validate_actor_workers(args: argparse.Namespace, env_ids: tuple[str, ...]) -> None:
    """Reject actor settings that would interleave recurrent task trajectories."""
    if args.num_envs < 1:
        raise ValueError("--num_envs must be positive")
    if args.actor_workers == 0:
        return
    if args.actor_workers != 5:
        raise ValueError("Unsupported --actor_workers value")
    if len(env_ids) != 5:
        raise ValueError("--actor_workers 5 requires exactly five multi-task environments")
    if args.num_envs != args.actor_workers:
        raise ValueError("--actor_workers 5 requires --num_envs 5")
    if args.replay_layout != "per_task":
        raise ValueError("--actor_workers 5 requires --replay_layout per_task")
    if args.total_timesteps % args.num_envs:
        raise ValueError("--actor_workers 5 requires total_timesteps divisible by --num_envs")


def _cadence_crossings(previous_step: int, current_step: int, frequency: int) -> int:
    """Count periodic boundaries crossed by one vectorized collection batch."""
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if current_step < previous_step:
        raise ValueError("current_step must not precede previous_step")
    return current_step // frequency - previous_step // frequency


# Mirrors train_atari._extract_episode_returns (kept private there; entry
# points stay decoupled).
def _extract_episode_returns(infos) -> list[float]:
    returns: list[float] = []
    if not isinstance(infos, dict):
        return returns

    episode = infos.get("episode")
    episode_mask = infos.get("_episode")
    if isinstance(episode, dict) and "r" in episode:
        raw_returns = np.asarray(episode["r"]).reshape(-1)
        if episode_mask is None:
            mask = np.ones(raw_returns.shape, dtype=bool)
        else:
            mask = np.asarray(episode_mask).reshape(-1).astype(bool)
        returns.extend(float(value) for value, keep in zip(raw_returns, mask) if keep)

    final_infos = infos.get("final_info")
    if final_infos is None:
        return returns
    for final_info in final_infos:
        if final_info and "episode" in final_info:
            episode_return = np.asarray(final_info["episode"]["r"]).reshape(-1)[0]
            returns.append(float(episode_return))
    return returns


def _extract_episode_records(
    infos: Any,
    default_env_id: str | None = None,
) -> list[tuple[str, float, int, str | None]]:
    """Extract per-environment return, length, and end reason records."""
    records: list[tuple[str, float, int, str | None]] = []
    if not isinstance(infos, dict):
        return records

    episode = infos.get("episode")
    if isinstance(episode, dict) and "r" in episode:
        raw_returns = np.asarray(episode["r"]).reshape(-1)
        mask_value = infos.get("_episode")
        mask = (
            np.ones(raw_returns.shape, dtype=bool)
            if mask_value is None
            else np.asarray(mask_value).reshape(-1).astype(bool)
        )
        env_values = np.asarray(
            infos.get(
                "env_id",
                np.full(raw_returns.shape, default_env_id or "unknown", dtype=object),
            )
        ).reshape(-1)
        raw_lengths = np.asarray(
            episode.get("l", np.zeros(raw_returns.shape, dtype=np.int64))
        ).reshape(-1)
        reason_values = np.asarray(
            infos.get("end_reason", np.full(raw_returns.shape, None, dtype=object)),
            dtype=object,
        ).reshape(-1)
        reason_mask_value = infos.get("_end_reason")
        reason_mask = (
            np.ones(reason_values.shape, dtype=bool)
            if reason_mask_value is None
            else np.asarray(reason_mask_value).reshape(-1).astype(bool)
        )
        for index, (episode_return, keep) in enumerate(zip(raw_returns, mask)):
            if keep:
                env_id = str(env_values[index]) if index < env_values.size else "unknown"
                length = int(raw_lengths[index]) if index < raw_lengths.size else 0
                raw_reason = reason_values[index] if index < reason_values.size else None
                reason = (
                    str(raw_reason)
                    if raw_reason is not None and reason_mask[index]
                    else None
                )
                records.append((env_id, float(episode_return), length, reason))

    final_infos = infos.get("final_info")
    if final_infos is None:
        return records
    for final_info in final_infos:
        if final_info and "episode" in final_info:
            episode_return = np.asarray(final_info["episode"]["r"]).reshape(-1)[0]
            episode_length = np.asarray(final_info["episode"].get("l", [0])).reshape(-1)[0]
            records.append(
                (
                    str(final_info.get("env_id", default_env_id or "unknown")),
                    float(episode_return),
                    int(episode_length),
                    (
                        str(final_info["end_reason"])
                        if "end_reason" in final_info
                        else None
                    ),
                )
            )
    return records


def _extract_step_env_ids(infos: Any, num_envs: int) -> list[str | None]:
    """Return the task associated with each vector slot's current transition."""
    result: list[str | None] = [None] * num_envs
    if not isinstance(infos, dict) or "env_id" not in infos:
        return result
    values = np.asarray(infos["env_id"], dtype=object).reshape(-1)
    mask_value = infos.get("_env_id")
    mask = (
        np.ones(values.shape, dtype=bool)
        if mask_value is None
        else np.asarray(mask_value).reshape(-1).astype(bool)
    )
    for index, (value, keep) in enumerate(zip(values, mask)):
        if index < num_envs and keep and value is not None:
            result[index] = str(value)
    return result


def _extract_step_end_reasons(infos: Any, num_envs: int) -> list[str | None]:
    """Return any end reason associated with each current vector transition."""
    result: list[str | None] = [None] * num_envs
    if not isinstance(infos, dict) or "end_reason" not in infos:
        return result
    values = np.asarray(infos["end_reason"], dtype=object).reshape(-1)
    mask_value = infos.get("_end_reason")
    mask = (
        np.ones(values.shape, dtype=bool)
        if mask_value is None
        else np.asarray(mask_value).reshape(-1).astype(bool)
    )
    for index, (value, keep) in enumerate(zip(values, mask)):
        if index < num_envs and keep:
            result[index] = str(value)
    return result


def _resolve_exploration_steps(args: argparse.Namespace) -> None:
    """Resolve the fixed default or explicit legacy fraction to one epsilon-decay duration."""

    if args.exploration_steps is not None and args.exploration_fraction is not None:
        raise ValueError("--exploration_steps and --exploration_fraction are mutually exclusive")
    if args.exploration_fraction is not None:
        if not 0 < args.exploration_fraction <= 1:
            raise ValueError("--exploration_fraction must be in (0, 1]")
        args.exploration_steps = int(args.exploration_fraction * args.total_timesteps)
    elif args.exploration_steps is None:
        args.exploration_steps = 500_000
    if args.exploration_steps <= 0:
        raise ValueError("--exploration_steps must be positive")


def _linear_epsilon(args: argparse.Namespace, global_step: int) -> float:
    exploration_steps = getattr(args, "exploration_steps", None)
    if exploration_steps is None:
        # MiniGrid reuses this helper and retains its own fraction-based CLI.
        exploration_steps = float(args.exploration_fraction) * args.total_timesteps
    decay_steps = max(1.0, float(exploration_steps))
    slope = (args.end_epsilon - args.start_epsilon) / decay_steps
    return max(args.end_epsilon, args.start_epsilon + slope * global_step)


def _learning_ready(
    args: argparse.Namespace,
    global_step: int,
    environment_steps: dict[str, int],
) -> bool:
    """Return whether both global and per-task replay warm-up gates are satisfied."""
    if global_step < args.learning_starts:
        return False
    if args.learning_starts_per_task <= 0:
        return True
    return bool(environment_steps) and min(environment_steps.values()) >= int(
        args.learning_starts_per_task
    )


def _learning_rate_decay_ready(
    args: argparse.Namespace,
    global_step: int,
    environment_steps: dict[str, int],
) -> bool:
    """Return whether the configured global or per-task LR decay threshold is met."""
    if args.learning_rate_decay_per_task_steps > 0:
        return bool(environment_steps) and min(environment_steps.values()) >= int(
            args.learning_rate_decay_per_task_steps
        )
    return args.learning_rate_decay_step > 0 and global_step >= args.learning_rate_decay_step


class _PreemptionWatcher:
    """Turn Slurm's preemption signals into a checkpoint-then-exit request.

    The handler only flips a flag; the training loop performs the actual save at
    a point where model, optimizer, and replay are mutually consistent. Slurm
    sends SIGTERM on preemption and can be asked for an early SIGUSR1 warning
    via ``--signal=USR1@600``.
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def install(self, logger: logging.Logger) -> None:
        for signal_number in (signal.SIGTERM, signal.SIGUSR1):
            try:
                self._previous[signal_number] = signal.signal(
                    signal_number, self._handle
                )
            except (ValueError, OSError):  # Not on the main thread, or unsupported.
                logger.warning("Could not install handler for signal %s", signal_number)

    def restore(self) -> None:
        for signal_number, handler in self._previous.items():
            try:
                signal.signal(signal_number, handler)
            except (ValueError, OSError):
                pass
        self._previous.clear()

    def _handle(self, signal_number: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signal_number).name


def _release_replay_storage(
    *,
    buffer: ReplayBuffer,
    replay_dir: str,
    keep: bool,
    logger: logging.Logger,
) -> None:
    """Reclaim the mmap replay directory once the run has produced its results.

    A completed fs4/stack4 run leaves ~28 GB behind, which would exhaust the
    shared /scratch quota within a few seeds.
    """
    if buffer.storage_dir is None or not os.path.isdir(replay_dir):
        return
    if keep:
        logger.info("keeping replay storage at %s", replay_dir)
        return
    buffer.close()
    shutil.rmtree(replay_dir, ignore_errors=True)
    logger.info("released replay storage at %s", replay_dir)


def _checkpoint_payload(
    *,
    model: AtariQNetwork,
    target_net: AtariQNetwork,
    optimizer: optim.Optimizer,
    scaler: Any,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    global_step: int,
    elapsed_seconds: float,
    rolling_returns: list[float],
    rolling_returns_by_env: dict[str, list[float]],
    rolling_lengths_by_env: dict[str, list[int]],
    episode_counts: dict[str, int],
    end_reason_counts_by_env: dict[str, dict[str, int]],
    environment_steps: dict[str, int],
    task_scheduler_states: tuple[dict[str, Any], ...] | None,
    learning_started_at_step: int | None,
    learning_rate_decay_applied: bool,
    last_loss: float,
    last_q_mean: float,
    resume_count: int,
    resumed_at_steps: list[int],
    initialization: dict[str, str] | None,
    extended_from_total_timesteps: int | None,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "replay": buffer.state_dict(),
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
        "rolling_returns": list(rolling_returns),
        "rolling_returns_by_env": {
            env_id: list(returns) for env_id, returns in rolling_returns_by_env.items()
        },
        "rolling_lengths_by_env": {
            env_id: list(lengths) for env_id, lengths in rolling_lengths_by_env.items()
        },
        "episode_counts": dict(episode_counts),
        "end_reason_counts_by_env": {
            env_id: dict(counts) for env_id, counts in end_reason_counts_by_env.items()
        },
        "environment_steps": dict(environment_steps),
        "task_scheduler_states": task_scheduler_states,
        "learning_started_at_step": learning_started_at_step,
        "learning_rate_decay_applied": bool(learning_rate_decay_applied),
        "last_loss": float(last_loss),
        "last_q_mean": float(last_q_mean),
        "resume_count": resume_count,
        "resumed_at_steps": list(resumed_at_steps),
        "initialization": initialization,
        "extended_from_total_timesteps": extended_from_total_timesteps,
        "rng_state": rng_state(),
        "learning_rate": float(args.learning_rate),
        "args": vars(args),
        # ALE internals and recurrent hidden state remain fresh on resume. Multi-task
        # collection restores only its scheduler counts/cursor so balancing continues.
        "environment_state_restored": False,
        "recurrent_state_restored": False,
    }


def _next_state_reset_flags(dones: np.ndarray, autoreset_rows: np.ndarray) -> np.ndarray:
    """Reset on terminal rows and again after NEXT_STEP's ignored autoreset row."""
    return np.logical_or(dones, autoreset_rows).astype(np.uint8)


def _replay_boundary_flags(
    terminated: np.ndarray,
    truncated: np.ndarray,
    end_reasons: list[str | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Return episode-reset and TD-bootstrap-stop flags for collected transitions."""
    episode_ends = np.logical_or(terminated, truncated).astype(np.uint8)
    stalled = np.asarray([reason == "stalled" for reason in end_reasons], dtype=bool)
    if np.any(stalled & np.asarray(terminated, dtype=bool)):
        raise RuntimeError("A stalled Skiing boundary must truncate, not terminate")
    if np.any(stalled & ~np.asarray(truncated, dtype=bool)):
        raise RuntimeError("A stalled Skiing boundary is missing truncated=True")
    bootstrap_stops = episode_ends.copy()
    bootstrap_stops[stalled] = 0
    return episode_ends, bootstrap_stops


def _aggregate_td_loss(
    elementwise_loss: torch.Tensor,
    task_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    buffer: ReplayBuffer,
) -> torch.Tensor:
    """Aggregate TD errors globally or as an equal-weight mean across tasks."""
    if buffer.sampling_mode == "global_uniform":
        return (elementwise_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    task_losses = []
    for task_id in range(buffer.num_tasks):
        task_mask = loss_mask * (task_ids == task_id).to(loss_mask.dtype)
        task_losses.append(
            (elementwise_loss * task_mask).sum() / task_mask.sum().clamp(min=1.0)
        )
    return torch.stack(task_losses).mean()


def _td_rewards(rewards: torch.Tensor, reward_clip: bool) -> torch.Tensor:
    """Apply the configured reward transform before constructing TD targets."""
    return rewards.clamp(-1.0, 1.0) if reward_clip else rewards


def _dqn_transition_loss(
    model_forward: SequenceForward,
    target_forward: SequenceForward,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = buffer.sample_transitions(args.batch_size)
    zeros = torch.zeros(batch.actions.shape[0], device=device)
    q_all, _ = _step_with_sequence_forward(model_forward, batch.obs, zeros)
    q_taken = q_all.gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next_target, _ = _step_with_sequence_forward(target_forward, batch.next_obs, zeros)
        if args.double_dqn:
            q_next_online, _ = _step_with_sequence_forward(
                model_forward, batch.next_obs, zeros
            )
            greedy = q_next_online.argmax(dim=1, keepdim=True)
            q_next = q_next_target.gather(1, greedy).squeeze(1)
        else:
            q_next = q_next_target.max(dim=1).values
        td_target = _td_rewards(batch.rewards, args.reward_clip) + args.gamma * (
            1.0 - batch.dones
        ) * q_next
    loss = _aggregate_td_loss(
        F.smooth_l1_loss(q_taken, td_target, reduction="none"),
        batch.task_ids,
        torch.ones_like(q_taken),
        buffer,
    )
    return loss, q_taken.detach().mean()


def _drqn_sequence_loss(
    model_forward: SequenceForward,
    target_forward: SequenceForward,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.profiler.record_function("aim3/replay_sample") if getattr(
        args, "_profile_window_active", False
    ) else nullcontext():
        seq = buffer.sample_sequences(args.sequences_per_batch, args.seq_len)
    n_steps = args.seq_len
    q_online, _ = model_forward(
        seq.obs,
        seq.prev_dones,
        None,
        seq.has_internal_reset,
    )
    q_taken = (
        q_online[:, :n_steps]
        .gather(-1, seq.actions[:, :n_steps].unsqueeze(-1))
        .squeeze(-1)
    )
    with torch.no_grad():
        # The target network unrolls the same window from zero state with its
        # own previous Q-values as gate feedback: bootstrap targets follow the
        # frozen network's recurrent dynamics, as in DRQN.
        q_target, _ = target_forward(
            seq.obs,
            seq.prev_dones,
            None,
            seq.has_internal_reset,
        )
        if args.double_dqn:
            greedy = q_online[:, 1:].argmax(-1, keepdim=True)
            q_next = q_target[:, 1:].gather(-1, greedy).squeeze(-1)
        else:
            q_next = q_target[:, 1:].max(-1).values
        td_target = (
            _td_rewards(seq.rewards[:, :n_steps], args.reward_clip)
            + args.gamma * (1.0 - seq.dones[:, :n_steps]) * q_next
        )
    mask = seq.loss_mask[:, :n_steps]
    loss = _aggregate_td_loss(
        F.smooth_l1_loss(q_taken, td_target, reduction="none"),
        seq.task_ids[:, :n_steps],
        mask,
        buffer,
    )
    q_mean = (q_taken.detach() * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, q_mean


def _materialize_training_stats(
    loss: torch.Tensor | None,
    q_mean: torch.Tensor | None,
) -> tuple[float, float]:
    """Copy logging-only scalars to CPU in one synchronization."""

    if loss is None or q_mean is None:
        return float("nan"), float("nan")
    values = torch.stack((loss.detach().float(), q_mean.detach().float())).cpu().tolist()
    return float(values[0]), float(values[1])


def train(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_atari_dqn")
    args.feedback_mode = _resolve_feedback_mode(args)
    _resolve_exploration_steps(args)
    env_ids, action_space_mode = _resolve_task_config(args)
    _validate_actor_workers(args, env_ids)
    if args.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {args.num_layers}")
    if args.frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {args.frame_skip}")
    if args.gawf_feedback_lr_scale <= 0:
        raise ValueError("gawf_feedback_lr_scale must be > 0")
    if (
        args.learning_rate_decay_step < 0
        or args.learning_rate_decay_per_task_steps < 0
        or args.learning_rate_decay_scale <= 0
    ):
        raise ValueError("learning-rate decay arguments must be non-negative/positive")
    if args.learning_starts < 0 or args.learning_starts_per_task < 0:
        raise ValueError("learning-start thresholds must be non-negative")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("checkpoint_interval_steps must be non-negative")
    if any(step <= 0 or step > args.total_timesteps for step in args.diagnostic_checkpoint_steps):
        raise ValueError("diagnostic checkpoint steps must be within (0, total_timesteps]")
    if args.resume_from and args.auto_resume:
        raise ValueError("--resume_from and --auto_resume are mutually exclusive")
    if args.init_weights_from and (args.resume_from or args.auto_resume):
        raise ValueError(
            "--init_weights_from is mutually exclusive with --resume_from/--auto_resume"
        )
    set_atari_seed(args.seed)
    device = select_device(args.device)
    acceleration = AtariAcceleration(
        device=device,
        amp_dtype_name=args.amp_dtype,
        allow_tf32=args.allow_tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )
    configure_atari_acceleration(acceleration, logger)
    save_dir = args.save_dir or os.path.join(
        "results", "data", "rl", "atari", "runs", args.result_suffix
    )
    ensure_dir(save_dir)
    video_dir = os.path.join(save_dir, "videos")
    history_path = os.path.join(save_dir, "metrics_history.jsonl")
    checkpoint_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    diagnostic_checkpoint_dir = os.path.join(save_dir, "diagnostic_checkpoints")
    replay_dir = os.path.join(save_dir, REPLAY_SUBDIR)

    resume_path = args.resume_from
    if args.auto_resume and os.path.isfile(checkpoint_path):
        resume_path = checkpoint_path
    resume_checkpoint = None
    extended_from_total_timesteps: int | None = None
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = load_checkpoint(resume_path, device)
        saved_args = resume_checkpoint.get("args")
        if isinstance(saved_args, dict):
            _normalize_legacy_resume_args(saved_args)
        resume_keys, extended_from_total_timesteps = _resume_validation_keys(
            resume_checkpoint, args
        )
        validate_resume_protocol(
            resume_checkpoint,
            args,
            resume_keys,
            expected_format_version=CHECKPOINT_FORMAT_VERSION,
            learning_rate=float(args.learning_rate),
        )
        if args.replay_backing != "mmap":
            raise ValueError(
                "Resuming requires --replay_backing mmap; the memory backing keeps no "
                "transitions across processes"
            )
    elif args.checkpoint_interval_steps > 0 and (
        os.path.exists(history_path)
        or os.path.exists(os.path.join(save_dir, "metrics.json"))
        or os.path.exists(checkpoint_path)
    ):
        # A checkpointing run that finds leftovers it was not asked to resume
        # would append a second trajectory to the same history. Refuse instead
        # of silently mixing two runs.
        raise FileExistsError(
            "Refusing to start a checkpointing run over existing results without "
            f"--resume_from/--auto_resume: {save_dir}"
        )

    is_multitask = len(env_ids) > 1
    restored_scheduler_states: tuple[dict[str, Any], ...] | None = None
    scheduler_state_restored = False
    if is_multitask and resume_checkpoint is not None:
        saved_scheduler_states = resume_checkpoint.get("task_scheduler_states")
        if saved_scheduler_states is not None:
            restored_scheduler_states = tuple(dict(state) for state in saved_scheduler_states)
            scheduler_state_restored = True
        else:
            logger.warning(
                "resume checkpoint predates task-scheduler persistence; collection "
                "balancing restarts while replay and aggregate counters resume"
            )
    if is_multitask:
        if args.capture_video:
            raise ValueError("Phase0 multi-task training does not support --capture_video")
        if args.actor_workers:
            envs = make_fixed_multitask_async_vector_atari_env(
                env_ids=env_ids,
                seed=args.seed,
                frame_stack=args.frame_stack,
                frame_skip=args.frame_skip,
                flicker_prob=args.flicker_prob,
                scheduler_states=restored_scheduler_states,
                atari_env_protocol=args.atari_env_protocol,
            )
        else:
            envs = make_multitask_vector_atari_env(
                env_ids=env_ids,
                seed=args.seed,
                num_envs=args.num_envs,
                frame_stack=args.frame_stack,
                frame_skip=args.frame_skip,
                flicker_prob=args.flicker_prob,
                task_schedule=args.task_schedule,
                scheduler_states=restored_scheduler_states,
                atari_env_protocol=args.atari_env_protocol,
            )
    else:
        envs = make_vector_atari_env(
            env_id=env_ids[0],
            seed=args.seed,
            num_envs=args.num_envs,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            flicker_prob=args.flicker_prob,
            capture_video=args.capture_video,
            video_dir=video_dir,
            full_action_space=action_space_mode == "full18",
            atari_env_protocol=args.atari_env_protocol,
        )
    try:
        assert envs.single_action_space.__class__.__name__ == "Discrete"
        num_actions = int(envs.single_action_space.n)
        obs_np, _info = envs.reset(seed=args.seed)
        current_obs_np = to_channel_first_obs(obs_np)
        next_obs = torch.as_tensor(current_obs_np, device=device)
        input_channels = int(next_obs.shape[1])

        ssm_context_len = args.ssm_context_len if args.ssm_context_len else args.seq_len
        model_kwargs = dict(
            num_actions=num_actions,
            input_channels=input_channels,
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            encoder_feature_dim=args.encoder_feature_dim,
            core_dropout=args.core_dropout,
            feedback_mode=args.feedback_mode,
            ssm_d_model=args.ssm_d_model,
            ssm_state_size=args.ssm_state_size,
            ssm_num_layers=args.ssm_num_layers,
            ssm_context_len=ssm_context_len,
            num_layers=args.num_layers,
        )
        model = AtariQNetwork(**model_kwargs).to(device)
        target_net = AtariQNetwork(**model_kwargs).to(device)
        initialization = (
            _load_initial_weights(model, args.init_weights_from, device)
            if args.init_weights_from
            else None
        )
        target_net.load_state_dict(model.state_dict())
        target_net.eval()
        target_net.requires_grad_(False)
        compiled_gawf_cores = sum(
            configure_gawf_feedback_acceleration(
                network,
                enabled=acceleration.compile_model,
                compile_mode=acceleration.compile_mode,
            )
            for network in (model, target_net)
        )
        if compiled_gawf_cores:
            logger.info(
                "Compiled %d shared GaWF feedback/gate core(s) with mode=%s",
                compiled_gawf_cores,
                acceleration.compile_mode,
            )
        fused_requested = args.fused_optimizer and device.type == "cuda"
        use_fused_optimizer = (
            fused_requested
            and args.model_type != "s5"
            and _supports_fused_adam_params(model)
        )
        if fused_requested and not use_fused_optimizer:
            logger.info(
                "Fused Adam disabled for model=%s; using non-fused Adam",
                args.model_type,
            )
        optimizer = _build_atari_optimizer(
            model,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            gawf_feedback_lr_scale=args.gawf_feedback_lr_scale,
            use_fused_optimizer=use_fused_optimizer,
        )
        optimizer_name = "adam"
        logger.info("Optimizer=%s fused=%s", optimizer_name, use_fused_optimizer)
        scaler = acceleration.build_grad_scaler()
        if compiled_gawf_cores:
            model_forward = model.forward_sequence
            target_forward = target_net.forward_sequence
        else:
            model_forward = acceleration.compile_callable(model.forward_sequence)
            target_forward = acceleration.compile_callable(target_net.forward_sequence)

        replay_kwargs = dict(
            obs_shape=tuple(next_obs.shape[1:]),
            device=device,
            seed=args.seed,
            sampling_mode=args.replay_sampling,
            storage_dir=replay_dir if args.replay_backing == "mmap" else None,
            reuse_existing=resume_checkpoint is not None,
        )
        if args.replay_layout == "per_task":
            buffer = PerTaskAtariReplayBuffer(
                buffer_size_per_task=args.buffer_size,
                num_envs=args.num_envs,
                num_tasks=len(env_ids),
                **replay_kwargs,
            )
        else:
            buffer = AtariReplayBuffer(
                buffer_size=args.buffer_size,
                num_envs=args.num_envs,
                num_tasks=len(env_ids),
                **replay_kwargs,
            )

        state = None
        next_done = torch.ones(args.num_envs, device=device)
        prev_done_np = np.zeros(args.num_envs, dtype=np.uint8)
        global_step = 0
        start_time = time.time()
        elapsed_before_resume = 0.0
        resume_count = 0
        resumed_at_steps: list[int] = []
        rolling_returns: list[float] = []
        rolling_returns_by_env: dict[str, list[float]] = {env_id: [] for env_id in env_ids}
        rolling_lengths_by_env: dict[str, list[int]] = {env_id: [] for env_id in env_ids}
        episode_counts = {env_id: 0 for env_id in env_ids}
        end_reason_counts_by_env: dict[str, dict[str, int]] = {
            env_id: {} for env_id in env_ids
        }
        environment_steps = {env_id: 0 for env_id in env_ids}
        env_id_to_task = {env_id: task_id for task_id, env_id in enumerate(env_ids)}
        last_loss_tensor: torch.Tensor | None = None
        last_q_mean_tensor: torch.Tensor | None = None
        restored_last_loss = float("nan")
        restored_last_q_mean = float("nan")
        learning_started_at_step: int | None = None
        learning_rate_decay_applied = False
        final_metrics: dict[str, Any] = {}
        timing = {
            "environment_seconds": 0.0,
            "replay_io_seconds": 0.0,
            "inference_seconds": 0.0,
            "optimization_seconds": 0.0,
            "optimizer_updates": 0,
        }

        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model"])
            target_net.load_state_dict(resume_checkpoint["target_net"])
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scaler.load_state_dict(resume_checkpoint["scaler"])
            buffer.load_state_dict(resume_checkpoint["replay"])
            global_step = int(resume_checkpoint["global_step"])
            elapsed_before_resume = float(resume_checkpoint.get("elapsed_seconds", 0.0))
            rolling_returns = [
                float(value) for value in resume_checkpoint.get("rolling_returns", [])
            ]
            for env_id, returns in resume_checkpoint.get(
                "rolling_returns_by_env", {}
            ).items():
                if env_id in rolling_returns_by_env:
                    rolling_returns_by_env[env_id] = [float(value) for value in returns]
            for env_id, lengths in resume_checkpoint.get(
                "rolling_lengths_by_env", {}
            ).items():
                if env_id in rolling_lengths_by_env:
                    rolling_lengths_by_env[env_id] = [int(value) for value in lengths]
            for env_id, count in resume_checkpoint.get("episode_counts", {}).items():
                if env_id in episode_counts:
                    episode_counts[env_id] = int(count)
            for env_id, counts in resume_checkpoint.get(
                "end_reason_counts_by_env", {}
            ).items():
                if env_id in end_reason_counts_by_env:
                    end_reason_counts_by_env[env_id] = {
                        str(reason): int(count) for reason, count in counts.items()
                    }
            for env_id, steps in resume_checkpoint.get("environment_steps", {}).items():
                if env_id in environment_steps:
                    environment_steps[env_id] = int(steps)
            restored_last_loss = float(resume_checkpoint.get("last_loss", float("nan")))
            restored_last_q_mean = float(
                resume_checkpoint.get("last_q_mean", float("nan"))
            )
            saved_learning_started = resume_checkpoint.get("learning_started_at_step")
            learning_started_at_step = (
                int(saved_learning_started) if saved_learning_started is not None else None
            )
            learning_rate_decay_applied = bool(
                resume_checkpoint.get("learning_rate_decay_applied", False)
            )
            resume_count = int(resume_checkpoint.get("resume_count", 0)) + 1
            resumed_at_steps = [
                int(step) for step in resume_checkpoint.get("resumed_at_steps", [])
            ]
            resumed_at_steps.append(global_step)
            initialization = resume_checkpoint.get("initialization")
            restore_rng_state(resume_checkpoint["rng_state"])
            history_archive = reconcile_history(history_path, global_step)
            logger.info(
                "resumed checkpoint=%s step=%d replay_size=%d resume_count=%d "
                "history_archive=%s env_state=fresh_reset recurrent_state=zero",
                resume_path,
                global_step,
                buffer.size,
                resume_count,
                history_archive,
            )

        preemption = _PreemptionWatcher()
        if args.checkpoint_interval_steps > 0:
            preemption.install(logger)

        def write_checkpoint() -> None:
            """Flush replay first so the saved position never outruns the data."""
            last_loss_value, last_q_mean_value = _materialize_training_stats(
                last_loss_tensor,
                last_q_mean_tensor,
            )
            if last_loss_tensor is None:
                last_loss_value = restored_last_loss
            if last_q_mean_tensor is None:
                last_q_mean_value = restored_last_q_mean
            buffer.flush()
            atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    target_net=target_net,
                    optimizer=optimizer,
                    scaler=scaler,
                    buffer=buffer,
                    args=args,
                    global_step=global_step,
                    elapsed_seconds=elapsed_before_resume + (time.time() - start_time),
                    rolling_returns=rolling_returns,
                    rolling_returns_by_env=rolling_returns_by_env,
                    rolling_lengths_by_env=rolling_lengths_by_env,
                    episode_counts=episode_counts,
                    end_reason_counts_by_env=end_reason_counts_by_env,
                    environment_steps=environment_steps,
                    task_scheduler_states=(
                        multitask_scheduler_states(envs) if is_multitask else None
                    ),
                    learning_started_at_step=learning_started_at_step,
                    learning_rate_decay_applied=learning_rate_decay_applied,
                    last_loss=last_loss_value,
                    last_q_mean=last_q_mean_value,
                    resume_count=resume_count,
                    resumed_at_steps=resumed_at_steps,
                    initialization=initialization,
                    extended_from_total_timesteps=extended_from_total_timesteps,
                ),
                checkpoint_path,
            )

        stage_profile = (AtariStageProfile(Path(save_dir) / "profile", device)
                         if getattr(args, "profile_stages", False) else None)
        preempted = False
        while global_step < args.total_timesteps:
            if preemption.requested:
                logger.info(
                    "preemption signal=%s at step=%d; checkpointing and exiting",
                    preemption.signal_name,
                    global_step,
                )
                write_checkpoint()
                preempted = True
                break
            previous_global_step = global_step
            global_step += args.num_envs
            if stage_profile is not None:
                stage_profile.begin(
                    global_step, _learning_ready(args, global_step, environment_steps)
                )
                args._profile_window_active = stage_profile.active
            epsilon = _linear_epsilon(args, global_step)

            # Always advance the model step so the GaWF recurrent state and
            # prev-Q feedback evolve identically whether or not the epsilon
            # coin picks a random action.
            inference_start = time.perf_counter() if args.record_timing else 0.0
            with stage_profile.phase("inference") if stage_profile else nullcontext():
                with torch.no_grad(), acceleration.autocast():
                    q_values, state = _step_with_sequence_forward(
                        model_forward, next_obs, next_done, state
                    )
                greedy_action = q_values.argmax(dim=-1).cpu().numpy()
                random_action = np.random.randint(0, num_actions, size=args.num_envs)
                explore = np.random.random(size=args.num_envs) < epsilon
                action_np = np.where(explore, random_action, greedy_action)
            if args.record_timing:
                timing["inference_seconds"] += time.perf_counter() - inference_start

            environment_start = time.perf_counter() if args.record_timing else 0.0
            with stage_profile.phase("environment") if stage_profile else nullcontext():
                next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)
            if args.record_timing:
                timing["environment_seconds"] += time.perf_counter() - environment_start
            end_reasons = _extract_step_end_reasons(infos, args.num_envs)
            episode_end_np, bootstrap_stop_np = _replay_boundary_flags(
                terminated_np,
                truncated_np,
                end_reasons,
            )
            # A stall is an artificial time limit: reset the episode/recurrent
            # state, but keep TD bootstrap from its final observation.
            # NEXT_STEP autoreset consumes one ignored action after a terminal
            # row. Reset before that invalid row and again before the first
            # valid observation of the newly selected episode/task.
            state_reset_np = _next_state_reset_flags(episode_end_np, prev_done_np)

            step_env_ids = _extract_step_env_ids(infos, args.num_envs)
            if stage_profile is not None and stage_profile.active:
                for task in step_env_ids:
                    key = str(task)
                    stage_profile.tasks[key] = stage_profile.tasks.get(key, 0) + 1
            if not is_multitask:
                step_env_ids = [env_ids[0]] * args.num_envs
            task_ids_np = np.zeros(args.num_envs, dtype=np.int16)
            for slot, env_id in enumerate(step_env_ids):
                if is_multitask and env_id not in env_id_to_task:
                    raise RuntimeError(
                        f"Missing or unknown env_id for multi-task replay slot {slot}: {env_id!r}"
                    )
                if env_id in env_id_to_task:
                    task_ids_np[slot] = env_id_to_task[env_id]
                if env_id in environment_steps and not prev_done_np[slot]:
                    environment_steps[env_id] += 1

            replay_io_start = time.perf_counter() if args.record_timing else 0.0
            with stage_profile.phase("replay_write") if stage_profile else nullcontext():
                buffer.add(
                    obs=current_obs_np,
                    actions=action_np,
                    rewards=np.asarray(reward_np, dtype=np.float32),
                    dones=bootstrap_stop_np,
                    resets=prev_done_np,
                    task_ids=task_ids_np,
                )
            if args.record_timing:
                timing["replay_io_seconds"] += time.perf_counter() - replay_io_start
            prev_done_np = episode_end_np
            current_obs_np = to_channel_first_obs(next_obs_np)
            next_obs = torch.as_tensor(current_obs_np, device=device)
            next_done = torch.as_tensor(state_reset_np, device=device, dtype=torch.float32)

            rolling_returns.extend(_extract_episode_returns(infos))
            if len(rolling_returns) > 100:
                rolling_returns = rolling_returns[-100:]
            for env_id, episode_return, episode_length, end_reason in _extract_episode_records(
                infos,
                env_ids[0] if not is_multitask else None,
            ):
                if env_id not in rolling_returns_by_env:
                    continue
                rolling_returns_by_env[env_id].append(episode_return)
                rolling_returns_by_env[env_id] = rolling_returns_by_env[env_id][-100:]
                rolling_lengths_by_env[env_id].append(episode_length)
                rolling_lengths_by_env[env_id] = rolling_lengths_by_env[env_id][-100:]
                episode_counts[env_id] += 1
                reason = end_reason or "natural"
                end_reason_counts_by_env[env_id][reason] = (
                    end_reason_counts_by_env[env_id].get(reason, 0) + 1
                )

            optimizer_updates_due = _cadence_crossings(
                previous_global_step,
                global_step,
                args.train_frequency,
            )
            if _learning_ready(args, global_step, environment_steps) and optimizer_updates_due:
                if learning_started_at_step is None:
                    learning_started_at_step = global_step
                    logger.info(
                        "learning started at step=%d per_task_steps=%s",
                        global_step,
                        environment_steps,
                    )
                if not learning_rate_decay_applied and _learning_rate_decay_ready(
                    args, global_step, environment_steps
                ):
                    for group in optimizer.param_groups:
                        group["lr"] *= args.learning_rate_decay_scale
                    learning_rate_decay_applied = True
                    logger.info(
                        "applied learning-rate decay at global_step=%d min_task_steps=%d",
                        global_step,
                        min(environment_steps.values()) if environment_steps else 0,
                    )
                for _ in range(optimizer_updates_due):
                    optimization_start = time.perf_counter() if args.record_timing else 0.0
                    with stage_profile.phase("optimization") if stage_profile else nullcontext():
                        with acceleration.autocast():
                            if model.is_recurrent:
                                loss, q_mean = _drqn_sequence_loss(
                                    model_forward, target_forward, buffer, args, device
                                )
                            else:
                                loss, q_mean = _dqn_transition_loss(
                                    model_forward, target_forward, buffer, args, device
                                )
                        optimizer.zero_grad(set_to_none=True)
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    if args.record_timing:
                        timing["optimization_seconds"] += time.perf_counter() - optimization_start
                    timing["optimizer_updates"] += 1
                    last_loss_tensor = loss.detach()
                    last_q_mean_tensor = q_mean.detach()

            if _cadence_crossings(
                previous_global_step, global_step, args.target_network_frequency
            ):
                target_net.load_state_dict(model.state_dict())

            if global_step % args.log_interval == 0:
                fps = int(
                    global_step
                    / max(elapsed_before_resume + (time.time() - start_time), 1e-6)
                )
                last_loss, last_q_mean = _materialize_training_stats(
                    last_loss_tensor,
                    last_q_mean_tensor,
                )
                rolling_return = (
                    float(np.mean(rolling_returns)) if rolling_returns else float("nan")
                )
                logger.info(
                    "step=%d/%d return100=%.3f eps=%.3f fps=%d loss=%.5f q_mean=%.3f",
                    global_step,
                    args.total_timesteps,
                    rolling_return,
                    epsilon,
                    fps,
                    last_loss,
                    last_q_mean,
                )
                per_env_history = {
                    env_id: {
                        "episodic_return_100": (
                            float(np.mean(returns)) if returns else float("nan")
                        ),
                        "episodes": episode_counts[env_id],
                        "environment_steps": environment_steps[env_id],
                        "episode_length_100": (
                            float(np.mean(rolling_lengths_by_env[env_id]))
                            if rolling_lengths_by_env[env_id]
                            else float("nan")
                        ),
                        "end_reason_counts": dict(end_reason_counts_by_env[env_id]),
                    }
                    for env_id, returns in rolling_returns_by_env.items()
                }
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            _json_safe({
                                "global_step": global_step,
                                "episodic_return_100": rolling_return,
                                "epsilon": epsilon,
                                "loss": last_loss,
                                "q_values_mean": last_q_mean,
                                "fps": fps,
                                "wall_time_s": elapsed_before_resume
                                + (time.time() - start_time),
                                "per_env": per_env_history,
                            })
                        )
                        + "\n"
                    )

            if (
                args.checkpoint_interval_steps > 0
                and global_step % args.checkpoint_interval_steps == 0
            ):
                write_checkpoint()
            if global_step in args.diagnostic_checkpoint_steps:
                ensure_dir(diagnostic_checkpoint_dir)
                snapshot_path = os.path.join(
                    diagnostic_checkpoint_dir, f"model_step{global_step}.pth"
                )
                atomic_torch_save(model.state_dict(), snapshot_path)
                logger.info("saved diagnostic model snapshot=%s", snapshot_path)

        preemption.restore()
        if preempted:
            logger.info(
                "exiting after preemption checkpoint at step=%d/%d",
                global_step,
                args.total_timesteps,
            )
            return {
                "status": "preempted",
                "global_step": global_step,
                "total_timesteps": args.total_timesteps,
                "checkpoint": checkpoint_path,
                "resume_count": resume_count,
            }

        if stage_profile is not None:
            stage_profile.close(global_step)
        fps = int(
            global_step / max(elapsed_before_resume + (time.time() - start_time), 1e-6)
        )
        last_loss, last_q_mean = _materialize_training_stats(
            last_loss_tensor,
            last_q_mean_tensor,
        )
        rolling_return = float(np.mean(rolling_returns)) if rolling_returns else float("nan")
        elapsed_seconds = elapsed_before_resume + (time.time() - start_time)
        timing_metrics = None
        if args.record_timing:
            environment_seconds = float(timing["environment_seconds"])
            replay_io_seconds = float(timing["replay_io_seconds"])
            inference_seconds = float(timing["inference_seconds"])
            optimization_seconds = float(timing["optimization_seconds"])
            optimizer_updates = int(timing["optimizer_updates"])
            timing_metrics = {
                "host_wall_elapsed_seconds": elapsed_seconds,
                "environment_seconds": environment_seconds,
                "replay_io_seconds": replay_io_seconds,
                "inference_seconds": inference_seconds,
                "optimization_seconds": optimization_seconds,
                "optimizer_updates": optimizer_updates,
                "environment_io_ms_per_step": 1_000 * (
                    environment_seconds + replay_io_seconds
                ) / max(global_step, 1),
                "optimization_ms_per_update": 1_000 * optimization_seconds / max(
                    optimizer_updates, 1
                ),
                "optimization_ms_per_step": 1_000 * optimization_seconds / max(global_step, 1),
            }
        per_env_metrics = {
            env_id: {
                "episodic_return_100": (
                    float(np.mean(returns)) if returns else float("nan")
                ),
                "episodes": episode_counts[env_id],
                "environment_steps": environment_steps[env_id],
                "episode_length_100": (
                    float(np.mean(rolling_lengths_by_env[env_id]))
                    if rolling_lengths_by_env[env_id]
                    else float("nan")
                ),
                "end_reason_counts": dict(end_reason_counts_by_env[env_id]),
                "stall_rate": (
                    end_reason_counts_by_env[env_id].get("stalled", 0)
                    / episode_counts[env_id]
                    if episode_counts[env_id]
                    else 0.0
                ),
            }
            for env_id, returns in rolling_returns_by_env.items()
        }
        final_metrics = {
            "env_id": env_ids[0] if not is_multitask else None,
            "env_ids": list(env_ids),
            "multitask": is_multitask,
            "action_space_mode": action_space_mode,
            "num_actions": num_actions,
            "atari_env_protocol": args.atari_env_protocol,
            "action_mapping_protocol": (
                "single_canonical_full18"
                if args.atari_env_protocol == "skiing-stall-actionfix-v1"
                else "baseline"
            ),
            "skiing_progress_signal": (
                "ale_ram_course_object_y_86_94_change"
                if args.atari_env_protocol == "skiing-stall-actionfix-v1"
                else None
            ),
            "skiing_stall_steps": (
                450 if args.atari_env_protocol == "skiing-stall-actionfix-v1" else None
            ),
            "skiing_stall_return_floor": (
                -30_000.0
                if args.atari_env_protocol == "skiing-stall-actionfix-v1"
                else None
            ),
            "stalled_truncation_bootstrap": (
                True if args.atari_env_protocol == "skiing-stall-actionfix-v1" else None
            ),
            "task_schedule": args.task_schedule if is_multitask else None,
            "collection_mode": (
                "fixed_task_async_actors" if args.actor_workers else "episode_balanced_sync"
            ),
            "actor_workers": args.actor_workers,
            "replay_sampling": args.replay_sampling,
            "replay_layout": args.replay_layout,
            "buffer_size": args.buffer_size,
            "buffer_size_per_task": (
                args.buffer_size if args.replay_layout == "per_task" else None
            ),
            "total_replay_capacity": (
                args.buffer_size * len(env_ids)
                if args.replay_layout == "per_task"
                else args.buffer_size
            ),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "sequences_per_batch": args.sequences_per_batch,
            "learning_rate": args.learning_rate,
            "reward_clip": args.reward_clip,
            "gamma": args.gamma,
            "learning_rate_decay_step": args.learning_rate_decay_step,
            "learning_rate_decay_scale": args.learning_rate_decay_scale,
            "learning_rate_decay_per_task_steps": args.learning_rate_decay_per_task_steps,
            "learning_rate_decay_applied": learning_rate_decay_applied,
            "effective_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "learning_starts": args.learning_starts,
            "learning_starts_per_task": args.learning_starts_per_task,
            "exploration_steps": args.exploration_steps,
            "exploration_fraction": args.exploration_fraction,
            "learning_started_at_step": learning_started_at_step,
            "task_scheduler_state_restored": scheduler_state_restored,
            "task_scheduler_states": (
                multitask_scheduler_states(envs) if is_multitask else None
            ),
            "replay_remainder_cursor": buffer.state_dict()["remainder_cursor"],
            "algo": args.algo,
            "model_type": args.model_type,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "encoder_feature_dim": args.encoder_feature_dim,
            "core_readout_params": int(
                sum(p.numel() for p in (model.core or model.proj).parameters())
            ),
            "total_param_count": int(sum(p.numel() for p in model.parameters())),
            "gawf_feedback_lr_scale": (
                args.gawf_feedback_lr_scale if args.model_type == "gawf" else None
            ),
            "feedback_mode": args.feedback_mode,
            "frame_stack": args.frame_stack,
            "frame_skip": args.frame_skip,
            "raw_ale_frames": global_step * args.frame_skip,
            "flicker_prob": args.flicker_prob,
            "global_step": global_step,
            "optimizer_updates": int(timing["optimizer_updates"]),
            "seed": args.seed,
            "timing": timing_metrics,
            "episodic_return_100": rolling_return,
            "fps": fps,
            "loss": last_loss,
            "q_values_mean": last_q_mean,
            "epsilon": _linear_epsilon(args, global_step),
            "amp_dtype": acceleration.amp_dtype_name,
            "allow_tf32": acceleration.allow_tf32,
            "cudnn_benchmark": acceleration.cudnn_benchmark,
            "optimizer": optimizer_name,
            "fused_optimizer": use_fused_optimizer,
            "compile_model": acceleration.compile_model,
            "compile_mode": acceleration.compile_mode,
            "per_env": per_env_metrics,
            # Interruption provenance: a resumed run restarts the env and the
            # recurrent state, so these fields belong in the result, not the log.
            "replay_backing": args.replay_backing,
            "checkpoint_interval_steps": args.checkpoint_interval_steps,
            "resume_count": resume_count,
            "resumed_at_steps": resumed_at_steps,
            "initialization": initialization,
            "extended_from_total_timesteps": extended_from_total_timesteps,
        }
        layer_suffix = f"_L{args.num_layers}" if args.num_layers > 1 else ""
        env_tag = "__".join(env_id.replace("/", "_") for env_id in env_ids)
        ckpt_name = (
            f"{args.algo}_{args.model_type}_{args.feedback_mode}{layer_suffix}_{env_tag}.pth"
        )
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        final_metrics["checkpoint"] = ckpt_path
        save_json(os.path.join(save_dir, "metrics.json"), _json_safe(final_metrics))
        # The resumable checkpoint is scaffolding: once metrics.json and the
        # final model exist the run is complete, and leaving it behind would
        # both waste space and break the "exactly one .pth" result contract.
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        _release_replay_storage(
            buffer=buffer,
            replay_dir=replay_dir,
            keep=args.keep_replay_on_success,
            logger=logger,
        )
        return final_metrics
    finally:
        envs.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
