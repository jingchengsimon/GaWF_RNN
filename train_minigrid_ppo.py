"""Recurrent PPO for MiniGrid memory/navigation (BabyAI-style).

Sparse-reward MiniGrid tasks are poorly served by value-based DRQN (unstable,
collapses), so this uses on-policy PPO with a recurrent memory core -- the
approach BabyAI (Chevalier-Boisvert et al., 2019) and the MiniGrid community use.
The memory core is swappable across rnn/gru/lstm/gawf/s5/mamba (via
``MiniGridActorCritic``) so the six architectures can be compared under identical
PPO. GAE + clipped surrogate; success rate is the reported metric.

Rollout is (num_steps x num_envs); the PPO update replays the full-length
sequences from the rollout's start recurrent state (K epochs, full batch) so
recurrence is respected. Defaults follow BabyAI (gamma .99, gae_lambda .99,
4 PPO epochs, Adam 1e-4).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from utils.atari_train_utils import (
    compute_gae,
    ensure_dir,
    save_json,
    select_device,
    set_atari_seed,
)
from utils.minigrid_envs import MINIGRID_PILOT_ENVS, make_vector_minigrid_env
from utils.minigrid_models import MiniGridEncoder
from utils.minigrid_ppo_models import MiniGridActorCritic
from train_atari_dqn import _extract_episode_returns


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recurrent PPO on MiniGrid")
    p.add_argument(
        "--env_id", type=str, default="MiniGrid-MemoryS7-v0", choices=MINIGRID_PILOT_ENVS
    )
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo"])
    p.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["rnn", "gru", "lstm", "gawf", "s5", "mamba"],
    )
    p.add_argument(
        "--agent_view_size",
        type=int,
        default=None,
        help="Egocentric view size (odd, >=3). Small (e.g. 3) forces memory.",
    )
    p.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "cnn"])
    p.add_argument("--encoder_output_size", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--gawf_feedback_lr_scale", type=float, default=1.0)
    p.add_argument("--core_dropout", type=float, default=0.0)
    p.add_argument("--ssm_d_model", type=int, default=128)
    p.add_argument("--ssm_state_size", type=int, default=64)
    p.add_argument("--ssm_num_layers", type=int, default=1)
    # PPO / rollout
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=40)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.99)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--anneal_lr", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--result_suffix", type=str, default="minigrid_ppo")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--log_interval_updates", type=int, default=10)
    return p


def _mg_obs(obs) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs), dtype=np.uint8)


def train(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_minigrid_ppo")
    set_atari_seed(args.seed)
    device = select_device(args.device)
    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    history_path = os.path.join(save_dir, "metrics_history.jsonl")

    envs = make_vector_minigrid_env(
        args.env_id, seed=args.seed, num_envs=args.num_envs, agent_view_size=args.agent_view_size
    )
    try:
        num_actions = int(envs.single_action_space.n)
        obs_np, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(_mg_obs(obs_np), device=device)

        def encoder_factory():
            return MiniGridEncoder(
                output_size=args.encoder_output_size,
                encoder_type=args.encoder,
                grid_size=int(next_obs.shape[-1]),
                hidden_size=args.encoder_hidden,
            )

        model = MiniGridActorCritic(
            num_actions=num_actions,
            encoder=encoder_factory(),
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            core_dropout=args.core_dropout,
            ssm_d_model=args.ssm_d_model,
            ssm_state_size=args.ssm_state_size,
            ssm_num_layers=args.ssm_num_layers,
            ssm_context_len=args.num_steps,
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

        N, T = args.num_envs, args.num_steps
        obs_buf = torch.zeros((T, N, *next_obs.shape[1:]), device=device, dtype=next_obs.dtype)
        actions_buf = torch.zeros((T, N), device=device, dtype=torch.long)
        logprobs_buf = torch.zeros((T, N), device=device)
        rewards_buf = torch.zeros((T, N), device=device)
        dones_buf = torch.zeros((T, N), device=device)
        prev_dones_buf = torch.zeros((T, N), device=device)
        values_buf = torch.zeros((T, N), device=device)

        next_done = torch.ones(N, device=device)
        state = None
        global_step = 0
        start_time = time.time()
        recent_returns: list[float] = []
        batch_size = N * T
        num_updates = max(1, args.total_timesteps // batch_size)
        final_metrics: dict = {}

        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (
                    1.0 - (update - 1.0) / num_updates
                ) * args.learning_rate
            rollout_start_state = model.detach_state(state)

            for step in range(T):
                global_step += N
                obs_buf[step] = next_obs
                prev_dones_buf[step] = next_done
                with torch.no_grad():
                    action, logprob, _ent, value, state = model.act(
                        next_obs, next_done, state=state
                    )
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value

                obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term_np, trunc_np)
                rewards_buf[step] = torch.as_tensor(reward_np, device=device, dtype=torch.float32)
                dones_buf[step] = torch.as_tensor(done_np, device=device, dtype=torch.float32)
                next_obs = torch.as_tensor(_mg_obs(obs_np), device=device)
                next_done = dones_buf[step]
                recent_returns.extend(_extract_episode_returns(infos))
            if len(recent_returns) > 200:
                recent_returns = recent_returns[-200:]

            with torch.no_grad():
                _a, _lp, _e, next_value, _s = model.act(
                    next_obs, next_done, state=state, deterministic=True
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

            b_obs = obs_buf.transpose(0, 1).contiguous()  # (N,T,...)
            b_actions = actions_buf.transpose(0, 1).contiguous()
            b_prev_dones = prev_dones_buf.transpose(0, 1).contiguous()
            b_old_logp = logprobs_buf.transpose(0, 1).contiguous()
            b_adv = advantages.transpose(0, 1).contiguous()
            b_ret = returns.transpose(0, 1).contiguous()
            adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            last_pl = last_vl = last_ent = 0.0
            for _epoch in range(args.update_epochs):
                new_logp, entropy, new_v = model.evaluate_actions_sequence(
                    b_obs, b_prev_dones, b_actions, state=rollout_start_state
                )
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(new_v, b_ret)
                entropy_loss = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                last_pl = float(policy_loss.detach())
                last_vl = float(value_loss.detach())
                last_ent = float(entropy_loss.detach())

            if recent_returns:
                mean_ret = float(np.mean(recent_returns))
                success = float(np.mean([1.0 if r > 0 else 0.0 for r in recent_returns]))
            else:
                mean_ret = success = float("nan")

            if update % args.log_interval_updates == 0 or update == num_updates:
                fps = int(global_step / max(time.time() - start_time, 1e-6))
                logger.info(
                    "upd=%d/%d step=%d success=%.3f return=%.3f pl=%.4f vl=%.4f ent=%.3f fps=%d",
                    update,
                    num_updates,
                    global_step,
                    success,
                    mean_ret,
                    last_pl,
                    last_vl,
                    last_ent,
                    fps,
                )
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "update": update,
                                "success_rate": success,
                                "episodic_return_100": mean_ret,
                                "policy_loss": last_pl,
                                "value_loss": last_vl,
                                "entropy": last_ent,
                                "fps": fps,
                            }
                        )
                        + "\n"
                    )

        final_metrics = {
            "env_id": args.env_id,
            "algo": args.algo,
            "model_type": args.model_type,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "core_param_count": int(sum(p.numel() for p in model.core.parameters())),
            "core_readout_params": int(
                sum(p.numel() for p in model.core.parameters())
                + sum(p.numel() for p in model.policy.parameters())
                + sum(p.numel() for p in model.value.parameters())
            ),
            "total_param_count": int(sum(p.numel() for p in model.parameters())),
            "gawf_feedback_lr_scale": (
                args.gawf_feedback_lr_scale if args.model_type == "gawf" else None
            ),
            "encoder": args.encoder,
            "global_step": global_step,
            "success_rate": success,
            "episodic_return_100": mean_ret,
        }
        layer_suffix = f"_L{args.num_layers}" if args.num_layers > 1 else ""
        ckpt = os.path.join(
            save_dir, f"ppo_{args.model_type}{layer_suffix}_{args.env_id.replace('/', '_')}.pth"
        )
        torch.save(model.state_dict(), ckpt)
        final_metrics["checkpoint"] = ckpt
        save_json(os.path.join(save_dir, "metrics.json"), final_metrics)
        return final_metrics
    finally:
        envs.close()


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
