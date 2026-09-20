"""Audit greedy Skiing behavior without rendering video.

Inputs are one completed ``skiing-stall-actionfix-v1`` metrics file and final
model ``state_dict``. Outputs are a JSON episode summary and a compressed NPZ
step trace containing actions, rewards, RAM course markers, and Q-values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.analysis.rl.atari.evaluate_dqn_video import (
    autocast_context,
    build_model,
    load_checkpoint,
    load_metrics,
    training_seed_from_metrics,
)
from utils.training.atari.atari_envs import (
    SKIING_PROGRESS_RAM_SLICE,
    make_atari_env,
)
from utils.training.atari.atari_train_utils import (
    select_device,
    set_atari_seed,
    to_channel_first_obs,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--trace_path", required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=20260904)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument(
        "--amp_dtype",
        choices=["none", "bfloat16", "float16"],
        default="bfloat16",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mapping(env: Any) -> tuple[int, ...]:
    current = env
    while current is not None:
        mapping = vars(current).get("canonical_action_mapping")
        if mapping is not None:
            return tuple(int(value) for value in mapping)
        current = vars(current).get("env")
    raise RuntimeError("Canonical 18-action mapping wrapper is absent")


def _course_marker(env: Any) -> np.ndarray:
    ram = np.asarray(env.unwrapped.ale.getRAM(), dtype=np.uint8)
    marker = ram[SKIING_PROGRESS_RAM_SLICE].copy()
    if marker.size != SKIING_PROGRESS_RAM_SLICE.stop - SKIING_PROGRESS_RAM_SLICE.start:
        raise RuntimeError("Skiing RAM does not expose course-object y slots 86:94")
    return marker


def _episode_summary(
    episode: int,
    rewards: list[float],
    actions: list[int],
    legal_actions: list[int],
    progress_changed: list[bool],
    terminated: bool,
    truncated: bool,
    end_reason: str,
    final_info: dict[str, Any],
) -> dict[str, Any]:
    no_progress = 0
    max_no_progress = 0
    for changed in progress_changed:
        no_progress = 0 if changed else no_progress + 1
        max_no_progress = max(max_no_progress, no_progress)
    return {
        "episode": episode,
        "return": float(sum(rewards)),
        "length": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "end_reason": end_reason,
        "course_progress_events": int(sum(progress_changed)),
        "max_consecutive_no_progress_steps": max_no_progress,
        "final_consecutive_no_progress_steps": no_progress,
        "canonical_action_counts": np.bincount(actions, minlength=18).astype(int).tolist(),
        "legal_action_counts": np.bincount(legal_actions, minlength=9).astype(int).tolist(),
        "wrapper_stall_steps": final_info.get("stall_steps"),
        "wrapper_course_progress_events": final_info.get("course_progress_events"),
        "stall_reward_adjustment": final_info.get("stall_reward_adjustment"),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed greedy suite and write summary and step-level trace files."""
    if args.num_episodes < 1:
        raise ValueError("--num_episodes must be positive")
    metrics_path = Path(args.metrics_path).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    summary_path = Path(args.summary_path).resolve()
    trace_path = Path(args.trace_path).resolve()
    for output in (summary_path, trace_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    metrics, env_id = load_metrics(metrics_path, None)
    if env_id != "ALE/Skiing-v5":
        raise ValueError(f"Behavior audit requires ALE/Skiing-v5, got {env_id}")
    if metrics.get("atari_env_protocol") != "skiing-stall-actionfix-v1":
        raise ValueError("Behavior audit requires skiing-stall-actionfix-v1")

    set_atari_seed(args.eval_seed)
    device = select_device(args.device)
    model = build_model(metrics, device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    env = make_atari_env(
        env_id=env_id,
        seed=args.eval_seed,
        idx=0,
        frame_stack=int(metrics["frame_stack"]),
        frame_skip=int(metrics["frame_skip"]),
        flicker_prob=float(metrics["flicker_prob"]),
        full_action_space=True,
        render_mode=None,
        atari_env_protocol="skiing-stall-actionfix-v1",
    )()
    mapping = _canonical_mapping(env)

    episode_summaries: list[dict[str, Any]] = []
    episode_index: list[int] = []
    step_index: list[int] = []
    actions: list[int] = []
    legal_actions: list[int] = []
    rewards: list[float] = []
    q_values: list[np.ndarray] = []
    course_markers: list[np.ndarray] = []
    progress_changed: list[bool] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    try:
        if int(env.action_space.n) != 18 or len(mapping) != 18:
            raise RuntimeError(f"Expected one canonical 18-action wrapper, got {mapping}")
        for episode in range(args.num_episodes):
            obs, _ = env.reset(seed=args.eval_seed + episode)
            previous_marker = _course_marker(env)
            state = None
            previous_done = torch.ones(1, device=device)
            local_rewards: list[float] = []
            local_actions: list[int] = []
            local_legal_actions: list[int] = []
            local_progress: list[bool] = []
            terminated = truncated = False
            final_info: dict[str, Any] = {}
            step = 0
            while not (terminated or truncated):
                obs_batch = to_channel_first_obs(np.expand_dims(np.asarray(obs), axis=0))
                obs_tensor = torch.as_tensor(obs_batch, device=device)
                with torch.no_grad(), autocast_context(device, args.amp_dtype):
                    values, state = model.step(obs_tensor, previous_done, state)
                values_cpu = values[0].float().cpu().numpy()
                action = int(values_cpu.argmax())
                obs, reward, terminated, truncated, final_info = env.step(action)
                marker = _course_marker(env)
                changed = bool(np.any(marker != previous_marker))
                legal_action = mapping[action]

                episode_index.append(episode)
                step_index.append(step)
                actions.append(action)
                legal_actions.append(legal_action)
                rewards.append(float(reward))
                q_values.append(values_cpu)
                course_markers.append(marker)
                progress_changed.append(changed)
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                local_rewards.append(float(reward))
                local_actions.append(action)
                local_legal_actions.append(legal_action)
                local_progress.append(changed)

                previous_marker = marker
                previous_done = torch.zeros(1, device=device)
                step += 1
            end_reason = str(final_info.get("end_reason", "natural"))
            episode_summaries.append(
                _episode_summary(
                    episode,
                    local_rewards,
                    local_actions,
                    local_legal_actions,
                    local_progress,
                    terminated,
                    truncated,
                    end_reason,
                    final_info,
                )
            )
    finally:
        env.close()

    trace_tmp = trace_path.with_name(f".{trace_path.name}.tmp.npz")
    np.savez_compressed(
        trace_tmp,
        episode=np.asarray(episode_index, dtype=np.int16),
        step=np.asarray(step_index, dtype=np.int32),
        action=np.asarray(actions, dtype=np.uint8),
        legal_action=np.asarray(legal_actions, dtype=np.uint8),
        reward=np.asarray(rewards, dtype=np.float32),
        q_values=np.asarray(q_values, dtype=np.float32),
        course_marker=np.asarray(course_markers, dtype=np.uint8),
        course_progress_changed=np.asarray(progress_changed, dtype=np.bool_),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
    )
    trace_tmp.replace(trace_path)

    returns = np.asarray([item["return"] for item in episode_summaries], dtype=np.float64)
    lengths = np.asarray([item["length"] for item in episode_summaries], dtype=np.float64)
    stalled = [item["end_reason"] == "stalled" for item in episode_summaries]
    summary = {
        "schema_version": 1,
        "env_id": env_id,
        "model_type": metrics["model_type"],
        "feedback_mode": metrics["feedback_mode"],
        "num_layers": int(metrics["num_layers"]),
        "hidden_size": int(metrics["hidden_size"]),
        "training_seed": training_seed_from_metrics(metrics, metrics_path),
        "eval_seed": int(args.eval_seed),
        "num_episodes": int(args.num_episodes),
        "policy": "greedy",
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "atari_env_protocol": metrics["atari_env_protocol"],
        "canonical_action_mapping": list(mapping),
        "stall_rate": float(np.mean(stalled)),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_episode_length": float(lengths.mean()),
        "episodes": episode_summaries,
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
    }
    summary_tmp = summary_path.with_name(f".{summary_path.name}.tmp")
    summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_tmp.replace(summary_path)
    return summary


def main() -> None:
    """Run the behavior audit."""
    summary = evaluate(parse_args())
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
