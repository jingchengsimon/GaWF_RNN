"""Render greedy Atari DQN evaluation episodes and retain the best video.

Inputs are a final training ``metrics.json`` and its checkpoint. The evaluator
reconstructs the exact saved network and strict Atari observation protocol,
records several deterministic greedy-policy episodes, and copies the
highest-return episode to the requested MP4 path. A companion JSON records the
source checkpoint, training/evaluation seeds, all episode returns, selected
episode, protocol fields, and output path.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, ContextManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import cv2

from utils.atari_dqn_models import AtariQNetwork
from utils.atari_envs import make_atari_env
from utils.atari_train_utils import select_device, set_atari_seed, to_channel_first_obs
from utils_anal.anal_paths import output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--eval_seed", type=int, default=20260718)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    """Load and validate the strict single-task Atari metadata contract."""
    metrics = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "env_id": "ALE/Pong-v5",
        "multitask": False,
        "action_space_mode": "minimal",
        "num_actions": 6,
        "model_type": "gawf",
        "feedback_mode": "qvalues",
        "frame_skip": 4,
        "frame_stack": 4,
        "flicker_prob": 0.0,
    }
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported or mismatched Atari video protocol: {mismatches}")
    if int(metrics.get("num_layers", 0)) not in {1, 2}:
        raise ValueError(f"Expected one or two GaWF layers, got {metrics.get('num_layers')}")
    if int(metrics.get("global_step", 0)) < 1:
        raise ValueError("metrics.json does not describe a completed training run")
    return metrics


def build_model(metrics: dict[str, Any], device: torch.device) -> AtariQNetwork:
    """Construct the Atari Q-network encoded by final metrics."""
    return AtariQNetwork(
        num_actions=int(metrics["num_actions"]),
        input_channels=int(metrics["frame_stack"]),
        model_type="gawf",
        hidden_size=int(metrics["hidden_size"]),
        encoder_feature_dim=int(metrics.get("encoder_feature_dim", 512)),
        feedback_mode="qvalues",
        num_layers=int(metrics["num_layers"]),
    ).to(device)


def load_checkpoint(model: AtariQNetwork, path: Path, device: torch.device) -> None:
    """Load one final checkpoint while reporting compatibility evidence."""
    state_dict = torch.load(path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict mapping in {path}")
    state_dict = {
        key: value for key, value in state_dict.items() if not key.endswith("prev_feedback")
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    print("missing_keys:", incompatible.missing_keys)
    print("unexpected_keys:", incompatible.unexpected_keys)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint is incompatible with reconstructed model: {path}")


def autocast_context(device: torch.device, amp_dtype: str) -> ContextManager[Any]:
    """Return the requested CUDA autocast context or an eager no-op context."""
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def open_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Create a validated OpenCV MP4 writer matching one RGB render frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected an RGB render frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not open MP4 writer: {path}")
    return writer


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Record greedy episodes, retain the highest-return MP4, and save metadata."""
    if args.num_episodes < 1:
        raise ValueError("--num_episodes must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    metrics_path = Path(args.metrics_path).resolve()
    metrics = load_metrics(metrics_path)
    checkpoint = Path(args.checkpoint or metrics["checkpoint"]).resolve()
    training_seed = int(metrics_path.parent.name.rsplit("seed", 1)[-1])
    env_slug = "".join(
        character.lower() if character.isalnum() else "_" for character in metrics["env_id"]
    ).strip("_")
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else output_dir("G_behaviour", "evaluate_atari_dqn_video", "figs")
        / f"{env_slug}_seed{training_seed}.mp4"
    )
    metadata_path = (
        Path(args.metadata_path).resolve()
        if args.metadata_path
        else output_dir("G_behaviour", "evaluate_atari_dqn_video", "data")
        / f"{env_slug}_seed{training_seed}.json"
    )
    if f"seed{training_seed}" not in output_path.name:
        raise ValueError("Output filename must include the selected training seed")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing final Atari video artifact")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_video_dir = output_path.parent / f"raw_episodes_eval{args.eval_seed}_{os.getpid()}"
    raw_video_dir.mkdir(parents=True, exist_ok=False)

    set_atari_seed(args.eval_seed)
    device = select_device(args.device)
    model = build_model(metrics, device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    env = make_atari_env(
        env_id=str(metrics["env_id"]),
        seed=args.eval_seed,
        idx=0,
        frame_stack=int(metrics["frame_stack"]),
        frame_skip=int(metrics["frame_skip"]),
        flicker_prob=float(metrics["flicker_prob"]),
        capture_video=False,
        full_action_space=False,
        render_mode="rgb_array",
    )()
    returns: list[float] = []
    episode_lengths: list[int] = []
    videos: list[Path] = []
    try:
        if int(env.action_space.n) != int(metrics["num_actions"]):
            raise RuntimeError(
                f"Action count mismatch: env={env.action_space.n}, metrics={metrics['num_actions']}"
            )
        for episode in range(args.num_episodes):
            obs, _info = env.reset(seed=args.eval_seed + episode)
            episode_video = raw_video_dir / f"{output_path.stem}-episode-{episode}.mp4"
            writer: cv2.VideoWriter | None = None
            state = None
            prev_done = torch.ones(1, device=device)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False
            try:
                while not (terminated or truncated):
                    obs_batch = to_channel_first_obs(np.expand_dims(np.asarray(obs), axis=0))
                    obs_tensor = torch.as_tensor(obs_batch, device=device)
                    with torch.no_grad(), autocast_context(device, args.amp_dtype):
                        q_values, state = model.step(obs_tensor, prev_done, state)
                    action = int(q_values.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, _info = env.step(action)
                    frame = np.asarray(env.render(), dtype=np.uint8)
                    if writer is None:
                        writer = open_video_writer(episode_video, frame, args.fps)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    prev_done = torch.zeros(1, device=device)
                    episode_return += float(reward)
                    episode_length += 1
            finally:
                if writer is not None:
                    writer.release()
            if writer is None or not episode_video.is_file() or episode_video.stat().st_size <= 0:
                raise RuntimeError(f"Episode did not produce a valid MP4: {episode_video}")
            videos.append(episode_video)
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            print(
                f"episode={episode} return={episode_return:.1f} "
                f"environment_steps={episode_length}"
            )
    finally:
        env.close()

    if len(videos) != args.num_episodes:
        raise RuntimeError(
            f"Expected {args.num_episodes} recorded videos, found {len(videos)} in {raw_video_dir}"
        )
    best_episode = int(np.argmax(np.asarray(returns, dtype=np.float64)))
    source_video = videos[best_episode]
    shutil.copy2(source_video, output_path)
    if output_path.stat().st_size <= 0:
        raise RuntimeError(f"Generated empty video: {output_path}")

    metadata = {
        "env_id": metrics["env_id"],
        "model_type": metrics["model_type"],
        "feedback_mode": metrics["feedback_mode"],
        "num_layers": int(metrics["num_layers"]),
        "frame_skip": int(metrics["frame_skip"]),
        "frame_stack": int(metrics["frame_stack"]),
        "action_space_mode": metrics["action_space_mode"],
        "num_actions": int(metrics["num_actions"]),
        "training_seed": training_seed,
        "eval_seed": int(args.eval_seed),
        "num_episodes": int(args.num_episodes),
        "fps": float(args.fps),
        "episode_returns": returns,
        "episode_lengths": episode_lengths,
        "best_episode": best_episode,
        "best_return": returns[best_episode],
        "source_metrics": str(metrics_path),
        "source_checkpoint": str(checkpoint),
        "source_video": str(source_video),
        "output_video": str(output_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Run the greedy-video evaluator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    metadata = evaluate(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
