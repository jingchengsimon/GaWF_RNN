"""Check Gymnasium/ALE Atari environment availability and observation shape."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_envs import ATARI_PILOT_ENVS, make_vector_atari_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", choices=ATARI_PILOT_ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_stack", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envs = make_vector_atari_env(args.env_id, args.seed, num_envs=1, frame_stack=args.frame_stack)
    try:
        obs, info = envs.reset(seed=args.seed)
        print(f"env_id={args.env_id}")
        print(f"action_space={envs.single_action_space}")
        print(f"observation_shape={obs.shape} dtype={obs.dtype}")
        print(f"reset_info_keys={list(info.keys())}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
