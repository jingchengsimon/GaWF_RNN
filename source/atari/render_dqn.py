"""Render a trained Atari DQN-family agent playing, as an mp4.

Loads a checkpoint saved by ``train_atari_dqn.py`` (``results/train_data/<suffix>/
dqn_<model>_<feedback>_<env>.pth``), runs the greedy policy (epsilon=0) with the
recurrent state carried across steps, captures the full-resolution RGB frames via
``env.render()``, and encodes them to mp4 with imageio.

Recurrent-core sizing (hidden_size / ssm_d_model) is read from the param-match
JSON so it matches the trained checkpoint; override on the CLI if needed. Runs on
CPU by default (single-episode inference is cheap).

Example:
    python -m source.atari.render_dqn \
        --result_suffix atari_dqn_pong_fs4_stack1_gawf_seed42 --model_type gawf \
        --frame_skip 4 --frame_stack 1 --num_episodes 1 \
        --output results/videos/gawf_seed42.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from utils.atari_dqn_models import AtariQNetwork
from utils.atari_train_utils import to_channel_first_obs


def _sizing_kwargs(model_type: str, args, num_actions: int) -> dict:
    """hidden_size / ssm_* for the recurrent cores, from JSON or CLI override."""
    if model_type == "ann":
        return {}
    match = {}
    if os.path.isfile(args.param_match_json):
        match = json.load(open(args.param_match_json)).get("matched", {}).get(model_type, {})
    if model_type in ("rnn", "gru", "lstm", "gawf"):
        hidden = args.hidden_size or match.get("hidden_size")
        if hidden is None:
            raise SystemExit(f"hidden_size unknown for {model_type}; pass --hidden_size")
        return {"hidden_size": int(hidden)}
    # s5 / mamba
    d_model = args.ssm_d_model or match.get("d_model")
    state_size = args.ssm_state_size or match.get("state_size", 128)
    if d_model is None:
        raise SystemExit(f"ssm_d_model unknown for {model_type}; pass --ssm_d_model")
    return {
        "ssm_d_model": int(d_model),
        "ssm_state_size": int(state_size),
        "ssm_context_len": args.seq_len,
    }


def _find_checkpoint(save_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(save_dir, "*.pth")))
    if not ckpts:
        raise SystemExit(f"No .pth checkpoint under {save_dir}")
    return ckpts[0]


def _make_render_env(
    env_id: str,
    seed: int,
    flicker_prob: float,
    frame_stack: int,
    frame_skip: int,
):
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="rgb_array",
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    if flicker_prob > 0:
        rng = np.random.default_rng(seed)

        class _Flicker(gym.ObservationWrapper):
            def observation(self, obs):
                return np.zeros_like(np.asarray(obs)) if rng.random() < flicker_prob else obs

        env = _Flicker(env)
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    env.action_space.seed(seed)
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a trained Atari DQN agent to mp4")
    p.add_argument(
        "--result_suffix",
        required=True,
        help="Run dir under results/train_data holding the checkpoint.",
    )
    p.add_argument("--data_root", default="results/train_data")
    p.add_argument(
        "--model_type", required=True, choices=["ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba"]
    )
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--env_id", default="ALE/Pong-v5")
    p.add_argument("--frame_stack", type=int, default=1)
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--flicker_prob", type=float, default=0.0)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--ssm_d_model", type=int, default=None)
    p.add_argument("--ssm_state_size", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--param_match_json", default="results/atari_param_match/atari_param_match.json")
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import imageio

    device = torch.device(args.device)
    save_dir = os.path.join(args.data_root, args.result_suffix)
    ckpt_path = _find_checkpoint(save_dir)

    env = _make_render_env(
        args.env_id,
        args.seed,
        args.flicker_prob,
        args.frame_stack,
        args.frame_skip,
    )
    num_actions = int(env.action_space.n)
    feedback_mode = "qvalues" if args.model_type == "gawf" else "none"

    model = AtariQNetwork(
        num_actions=num_actions,
        input_channels=args.frame_stack,
        model_type=args.model_type,
        feedback_mode=feedback_mode,
        num_layers=args.num_layers,
        **_sizing_kwargs(args.model_type, args, num_actions),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    frames, returns = [], []
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state, done, ep_ret, steps = None, False, 0.0, 0
        prev_done = torch.ones(1, device=device)
        while not done and steps < args.max_steps:
            obs_t = torch.as_tensor(to_channel_first_obs(obs[None]), device=device)
            with torch.no_grad():
                q, state = model.step(obs_t, prev_done, state=state)
            action = int(q.argmax(-1).item())
            obs, reward, term, trunc, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            ep_ret += float(reward)
            done = bool(term or trunc)
            prev_done = torch.zeros(1, device=device)
            steps += 1
        returns.append(ep_ret)
        print(f"episode {ep}: return={ep_ret:.1f} steps={steps}")
    env.close()

    out = args.output or os.path.join("results", "videos", f"{args.result_suffix}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264")
    print(f"wrote {out}  ({len(frames)} frames, mean return={np.mean(returns):.1f})")


if __name__ == "__main__":
    main()
