"""Train DRQN-family models on MiniGrid memory/navigation tasks.

Reuses the Atari DRQN core verbatim (recurrent cores, sequence replay, and the
DQN/DRQN losses from ``train_atari_dqn``); only the *encoder* and *environment*
change. The 7x7x3 symbolic partial view is encoded by ``MiniGridEncoder`` (mlp by
default) and fed to the same recurrent readout slot + linear Q-head. No frame
stacking: recurrence is the only source of memory.
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

from utils.atari_dqn_models import AtariQNetwork
from utils.atari_replay import AtariReplayBuffer
from utils.atari_train_utils import ensure_dir, save_json, select_device, set_atari_seed
from utils.minigrid_envs import MINIGRID_PILOT_ENVS, make_vector_minigrid_env
from utils.minigrid_models import MiniGridEncoder

# Reuse the DRQN core losses + schedules unchanged.
from train_atari_dqn import (
    _drqn_sequence_loss,
    _dqn_transition_loss,
    _extract_episode_returns,
    _linear_epsilon,
    _resolve_feedback_mode,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiniGrid DRQN models")
    parser.add_argument(
        "--env_id", type=str, default="MiniGrid-MemoryS7-v0", choices=MINIGRID_PILOT_ENVS
    )
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn"])
    parser.add_argument(
        "--model_type",
        type=str,
        default="ann",
        choices=["ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba"],
    )
    parser.add_argument("--feedback_mode", type=str, default=None, choices=["none", "qvalues"])
    # Encoder (pluggable; default mlp). output_size = recurrent input_size.
    parser.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--encoder_output_size", type=int, default=128)
    parser.add_argument("--encoder_hidden", type=int, default=128)
    # Recurrent-core sizing (MiniGrid scale; anchor ~128). Param-match separately.
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--gawf_feedback_lr_scale", type=float, default=1.0)
    parser.add_argument("--core_dropout", type=float, default=0.0)
    parser.add_argument("--ssm_d_model", type=int, default=128)
    parser.add_argument("--ssm_state_size", type=int, default=64)
    parser.add_argument("--ssm_num_layers", type=int, default=1)
    parser.add_argument("--ssm_context_len", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--start_epsilon", type=float, default=1.0)
    parser.add_argument("--end_epsilon", type=float, default=0.05)
    parser.add_argument("--exploration_fraction", type=float, default=0.30)
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--sequences_per_batch", type=int, default=8)
    parser.add_argument("--target_network_frequency", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--double_dqn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--result_suffix", type=str, default="minigrid_dqn")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=1000)
    return parser


def _mg_obs(obs) -> np.ndarray:
    """Vector MiniGrid obs is already (num_envs, 3, 7, 7) uint8 (channel-first)."""
    return np.ascontiguousarray(np.asarray(obs), dtype=np.uint8)


def train(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_minigrid_dqn")
    args.feedback_mode = _resolve_feedback_mode(args)
    if args.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {args.num_layers}")
    set_atari_seed(args.seed)
    device = select_device(args.device)
    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    history_path = os.path.join(save_dir, "metrics_history.jsonl")

    envs = make_vector_minigrid_env(args.env_id, seed=args.seed, num_envs=args.num_envs)
    try:
        assert envs.single_action_space.__class__.__name__ == "Discrete"
        num_actions = int(envs.single_action_space.n)
        obs_np, _ = envs.reset(seed=args.seed)
        current_obs_np = _mg_obs(obs_np)
        next_obs = torch.as_tensor(current_obs_np, device=device)

        ssm_context_len = args.ssm_context_len if args.ssm_context_len else args.seq_len

        def encoder_factory():
            return MiniGridEncoder(
                output_size=args.encoder_output_size,
                encoder_type=args.encoder,
                grid_size=int(next_obs.shape[-1]),
                hidden_size=args.encoder_hidden,
            )

        model_kwargs = dict(
            num_actions=num_actions,
            input_channels=int(next_obs.shape[1]),
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            encoder_feature_dim=args.encoder_output_size,
            core_dropout=args.core_dropout,
            feedback_mode=args.feedback_mode,
            ssm_d_model=args.ssm_d_model,
            ssm_state_size=args.ssm_state_size,
            ssm_num_layers=args.ssm_num_layers,
            ssm_context_len=ssm_context_len,
            num_layers=args.num_layers,
            encoder_factory=encoder_factory,
        )
        model = AtariQNetwork(**model_kwargs).to(device)
        target_net = AtariQNetwork(**model_kwargs).to(device)
        target_net.load_state_dict(model.state_dict())
        target_net.eval()
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
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        buffer = AtariReplayBuffer(
            buffer_size=args.buffer_size,
            num_envs=args.num_envs,
            obs_shape=tuple(next_obs.shape[1:]),
            device=device,
            seed=args.seed,
        )

        state = None
        prev_done_np = np.zeros(args.num_envs, dtype=np.uint8)
        next_done = torch.ones(args.num_envs, device=device)
        global_step = 0
        start_time = time.time()
        rolling_returns: list[float] = []
        last_loss = float("nan")
        last_q_mean = float("nan")

        while global_step < args.total_timesteps:
            global_step += args.num_envs
            epsilon = _linear_epsilon(args, global_step)

            with torch.no_grad():
                q_values, state = model.step(next_obs, next_done, state=state)
            greedy = q_values.argmax(dim=-1).cpu().numpy()
            explore = np.random.random(size=args.num_envs) < epsilon
            action_np = np.where(explore, np.random.randint(0, num_actions, args.num_envs), greedy)

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action_np)
            done_np = np.logical_or(term_np, trunc_np).astype(np.uint8)

            buffer.add(
                obs=current_obs_np,
                actions=action_np,
                rewards=np.asarray(reward_np, dtype=np.float32),
                dones=done_np,
                resets=prev_done_np,
            )
            prev_done_np = done_np
            current_obs_np = _mg_obs(next_obs_np)
            next_obs = torch.as_tensor(current_obs_np, device=device)
            next_done = torch.as_tensor(done_np, device=device, dtype=torch.float32)

            rolling_returns.extend(_extract_episode_returns(infos))
            if len(rolling_returns) > 100:
                rolling_returns = rolling_returns[-100:]

            if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
                if model.is_recurrent:
                    loss, q_mean = _drqn_sequence_loss(model, target_net, buffer, args, device)
                else:
                    loss, q_mean = _dqn_transition_loss(model, target_net, buffer, args, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                last_q_mean = q_mean

            if global_step % args.target_network_frequency == 0:
                target_net.load_state_dict(model.state_dict())

            if global_step % args.log_interval == 0:
                rolling_return = (
                    float(np.mean(rolling_returns)) if rolling_returns else float("nan")
                )
                fps = int(global_step / max(time.time() - start_time, 1e-6))
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
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "episodic_return_100": rolling_return,
                                "epsilon": epsilon,
                                "loss": last_loss,
                                "q_values_mean": last_q_mean,
                                "fps": fps,
                                "wall_time_s": time.time() - start_time,
                            }
                        )
                        + "\n"
                    )

        rolling_return = float(np.mean(rolling_returns)) if rolling_returns else float("nan")
        final_metrics = {
            "env_id": args.env_id,
            "algo": args.algo,
            "model_type": args.model_type,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "core_readout_params": int(
                sum(p.numel() for p in (model.core or model.proj).parameters())
            ),
            "total_param_count": int(sum(p.numel() for p in model.parameters())),
            "gawf_feedback_lr_scale": (
                args.gawf_feedback_lr_scale if args.model_type == "gawf" else None
            ),
            "feedback_mode": args.feedback_mode,
            "encoder": args.encoder,
            "global_step": global_step,
            "episodic_return_100": rolling_return,
            "loss": last_loss,
            "q_values_mean": last_q_mean,
        }
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
