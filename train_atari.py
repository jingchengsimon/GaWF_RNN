"""Train Atari recurrent actor-critic models with synchronous A2C.

This entry point composes Atari-specific CNN encoder/heads with shared recurrent
cores from ``utils.recurrent_cores``. LSTM and GaWF models receive previous
action/reward inputs; GaWF can additionally gate recurrence with detached
previous policy/value outputs.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.atari_envs import ATARI_PILOT_ENVS, make_vector_atari_env
from utils.atari_task_models import AtariActorCritic
from utils.atari_train_utils import (
    compute_gae,
    ensure_dir,
    explained_variance,
    obs_to_tensor,
    save_json,
    select_device,
    set_atari_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Atari recurrent A2C models")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", choices=ATARI_PILOT_ENVS)
    parser.add_argument("--algo", type=str, default="a2c", choices=["a2c"])
    parser.add_argument("--model_type", type=str, default="gawf", choices=["lstm", "gawf"])
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default="none",
        choices=["none", "output"],
        help="GaWF gate feedback source; LSTM requires 'none'.",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gawf_feedback_lr_scale", type=float, default=1.0)
    parser.add_argument("--encoder_feature_dim", type=int, default=512)
    parser.add_argument("--core_dropout", type=float, default=0.0)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--result_suffix", type=str, default="atari_a2c")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1)
    return parser


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


def train(args: argparse.Namespace) -> dict[str, float | int | str | None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_atari")
    set_atari_seed(args.seed)
    device = select_device(args.device)
    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    video_dir = os.path.join(save_dir, "videos")

    envs = make_vector_atari_env(
        env_id=args.env_id,
        seed=args.seed,
        num_envs=args.num_envs,
        frame_stack=args.frame_stack,
        capture_video=args.capture_video,
        video_dir=video_dir,
    )
    try:
        assert envs.single_action_space.__class__.__name__ == "Discrete"
        num_actions = int(envs.single_action_space.n)
        obs_np, _info = envs.reset(seed=args.seed)
        next_obs = obs_to_tensor(obs_np, device)
        input_channels = int(next_obs.shape[1])
        model = AtariActorCritic(
            num_actions=num_actions,
            input_channels=input_channels,
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            encoder_feature_dim=args.encoder_feature_dim,
            core_dropout=args.core_dropout,
            feedback_mode=args.feedback_mode,
            num_layers=args.num_layers,
        ).to(device)
        if args.model_type == "gawf":
            gate_params = [
                p
                for n, p in model.named_parameters()
                if n.startswith("core.U") or n.startswith("core.V")
            ]
            gate_ids = {id(p) for p in gate_params}
            optimizer = optim.Adam(
                [
                    {"params": [p for p in model.parameters() if id(p) not in gate_ids]},
                    {
                        "params": gate_params,
                        "lr": args.learning_rate * args.gawf_feedback_lr_scale,
                        "weight_decay": 0.0,
                    },
                ],
                lr=args.learning_rate,
                eps=1e-5,
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

        batch_size = args.num_envs * args.num_steps
        num_updates = max(1, args.total_timesteps // batch_size)
        obs_buf = torch.zeros(
            (args.num_steps, args.num_envs, *next_obs.shape[1:]),
            device=device,
            dtype=next_obs.dtype,
        )
        actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
        prev_actions_buf = torch.zeros_like(actions_buf)
        rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
        prev_rewards_buf = torch.zeros_like(rewards_buf)
        dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
        prev_dones_buf = torch.zeros_like(dones_buf)
        values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

        next_action = torch.zeros(args.num_envs, device=device, dtype=torch.long)
        next_reward = torch.zeros(args.num_envs, device=device)
        next_done = torch.ones(args.num_envs, device=device)
        state = None
        global_step = 0
        start_time = time.time()
        rolling_returns: list[float] = []
        final_metrics: dict[str, float | int | str | None] = {}

        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            rollout_start_state = model.detach_state(state)
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                prev_actions_buf[step] = next_action
                prev_rewards_buf[step] = next_reward.clamp(-1.0, 1.0)
                prev_dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _entropy, value, _logits, state = model.act(
                        next_obs,
                        next_action,
                        next_reward,
                        next_done,
                        state=state,
                    )
                actions_buf[step] = action
                values_buf[step] = value

                next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(
                    action.cpu().numpy()
                )
                done_np = np.logical_or(terminated_np, truncated_np)
                reward = torch.as_tensor(reward_np, device=device, dtype=torch.float32)
                done = torch.as_tensor(done_np, device=device, dtype=torch.float32)
                rewards_buf[step] = reward
                dones_buf[step] = done
                next_obs = obs_to_tensor(next_obs_np, device)
                keep = 1.0 - done
                next_action = action.masked_fill(done.bool(), 0)
                next_reward = reward * keep
                next_done = done
                rolling_returns.extend(_extract_episode_returns(infos))
                if len(rolling_returns) > 100:
                    rolling_returns = rolling_returns[-100:]

            with torch.no_grad():
                _action, _logprob, _entropy, next_value, _logits, _next_state = model.act(
                    next_obs,
                    next_action,
                    next_reward,
                    next_done,
                    state=state,
                    deterministic=True,
                )
                advantages, returns = compute_gae(
                    rewards_buf,
                    dones_buf,
                    values_buf,
                    next_value,
                    next_done,
                    args.gamma,
                    args.gae_lambda,
                )

            b_obs = obs_buf.transpose(0, 1).contiguous()
            b_prev_actions = prev_actions_buf.transpose(0, 1).contiguous()
            b_prev_rewards = prev_rewards_buf.transpose(0, 1).contiguous()
            b_prev_dones = prev_dones_buf.transpose(0, 1).contiguous()
            b_actions = actions_buf.transpose(0, 1).contiguous()
            b_advantages = advantages.transpose(0, 1).reshape(-1)
            b_returns = returns.transpose(0, 1).reshape(-1)
            b_values = values_buf.transpose(0, 1).reshape(-1)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            new_logprobs, entropy, new_values, _logits = model.evaluate_actions_sequence(
                b_obs,
                b_prev_actions,
                b_prev_rewards,
                b_prev_dones,
                b_actions,
                state=rollout_start_state,
            )
            new_logprobs_flat = new_logprobs.reshape(-1)
            entropy_flat = entropy.reshape(-1)
            new_values_flat = new_values.reshape(-1)
            policy_loss = -(new_logprobs_flat * b_advantages).mean()
            value_loss = 0.5 * F.mse_loss(new_values_flat, b_returns)
            entropy_loss = entropy_flat.mean()
            loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            fps = int(global_step / max(time.time() - start_time, 1e-6))
            rolling_return = float(np.mean(rolling_returns)) if rolling_returns else float("nan")
            final_metrics = {
                "env_id": args.env_id,
                "algo": args.algo,
                "model_type": args.model_type,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "core_param_count": int(sum(p.numel() for p in model.core.parameters())),
                "core_readout_params": int(
                    sum(p.numel() for p in model.core.parameters())
                    + sum(p.numel() for p in model.head.parameters())
                ),
                "total_param_count": int(sum(p.numel() for p in model.parameters())),
                "gawf_feedback_lr_scale": (
                    args.gawf_feedback_lr_scale if args.model_type == "gawf" else None
                ),
                "feedback_mode": args.feedback_mode,
                "global_step": global_step,
                "episodic_return_100": rolling_return,
                "fps": fps,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_loss.detach().cpu()),
                "explained_variance": explained_variance(b_values, b_returns),
            }
            if update % args.log_interval == 0:
                logger.info(
                    "update=%d/%d step=%d return100=%.3f fps=%d pg=%.4f v=%.4f ent=%.4f",
                    update,
                    num_updates,
                    global_step,
                    rolling_return,
                    fps,
                    final_metrics["policy_loss"],
                    final_metrics["value_loss"],
                    final_metrics["entropy"],
                )

        layer_suffix = f"_L{args.num_layers}" if args.num_layers > 1 else ""
        ckpt_name = (
            f"{args.algo}_{args.model_type}_{args.feedback_mode}{layer_suffix}_"
            f"{args.env_id.replace('/', '_')}.pth"
        )
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        final_metrics["checkpoint"] = ckpt_path
        save_json(os.path.join(save_dir, "metrics.json"), final_metrics)
        return final_metrics
    finally:
        envs.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
