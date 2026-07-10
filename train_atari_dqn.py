"""Train Atari DQN models: classic CNN-DQN and a GaWF-gated recurrent variant.

This entry point mirrors ``train_atari.py`` conventions but stays decoupled from
it. The classic branch uses iid single-transition replay; the GaWF branch uses
DRQN-style sequence replay, unrolling from zero state with detached previous
Q-values as gate feedback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.atari_dqn_models import AtariQNetwork
from utils.atari_envs import ATARI_PILOT_ENVS, make_vector_atari_env
from utils.atari_replay import AtariReplayBuffer
from utils.atari_train_utils import (
    ensure_dir,
    obs_to_tensor,
    save_json,
    select_device,
    set_atari_seed,
    to_channel_first_obs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Atari DQN models")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", choices=ATARI_PILOT_ENVS)
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn"])
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "rnn", "gru", "lstm", "gawf"],
    )
    parser.add_argument(
        "--feedback_mode",
        type=str,
        default=None,
        choices=["none", "qvalues"],
        help="GaWF gate feedback source; defaults to 'qvalues' for gawf, 'none' otherwise.",
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--encoder_feature_dim", type=int, default=512)
    parser.add_argument("--core_dropout", type=float, default=0.0)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_starts", type=int, default=20_000)
    parser.add_argument("--start_epsilon", type=float, default=1.0)
    parser.add_argument("--end_epsilon", type=float, default=0.01)
    parser.add_argument("--exploration_fraction", type=float, default=0.10)
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
    return parser


def _resolve_feedback_mode(args: argparse.Namespace) -> str:
    if args.feedback_mode is not None:
        return args.feedback_mode
    return "qvalues" if args.model_type == "gawf" else "none"


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


def _linear_epsilon(args: argparse.Namespace, global_step: int) -> float:
    decay_steps = max(1.0, args.exploration_fraction * args.total_timesteps)
    slope = (args.end_epsilon - args.start_epsilon) / decay_steps
    return max(args.end_epsilon, args.start_epsilon + slope * global_step)


def _dqn_transition_loss(
    model: AtariQNetwork,
    target_net: AtariQNetwork,
    buffer: AtariReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    batch = buffer.sample_transitions(args.batch_size)
    zeros = torch.zeros(batch.actions.shape[0], device=device)
    q_all, _ = model.step(batch.obs, zeros)
    q_taken = q_all.gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next_target, _ = target_net.step(batch.next_obs, zeros)
        if args.double_dqn:
            q_next_online, _ = model.step(batch.next_obs, zeros)
            greedy = q_next_online.argmax(dim=1, keepdim=True)
            q_next = q_next_target.gather(1, greedy).squeeze(1)
        else:
            q_next = q_next_target.max(dim=1).values
        td_target = batch.rewards.clamp(-1.0, 1.0) + args.gamma * (1.0 - batch.dones) * q_next
    loss = F.smooth_l1_loss(q_taken, td_target)
    return loss, float(q_taken.detach().mean().cpu())


def _drqn_sequence_loss(
    model: AtariQNetwork,
    target_net: AtariQNetwork,
    buffer: AtariReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    seq = buffer.sample_sequences(args.sequences_per_batch, args.seq_len)
    n_steps = args.seq_len
    q_online, _ = model.forward_sequence(seq.obs, seq.prev_dones)
    q_taken = q_online[:, :n_steps].gather(
        -1, seq.actions[:, :n_steps].unsqueeze(-1)
    ).squeeze(-1)
    with torch.no_grad():
        # The target network unrolls the same window from zero state with its
        # own previous Q-values as gate feedback: bootstrap targets follow the
        # frozen network's recurrent dynamics, as in DRQN.
        q_target, _ = target_net.forward_sequence(seq.obs, seq.prev_dones)
        if args.double_dqn:
            greedy = q_online[:, 1:].argmax(-1, keepdim=True)
            q_next = q_target[:, 1:].gather(-1, greedy).squeeze(-1)
        else:
            q_next = q_target[:, 1:].max(-1).values
        td_target = (
            seq.rewards[:, :n_steps].clamp(-1.0, 1.0)
            + args.gamma * (1.0 - seq.dones[:, :n_steps]) * q_next
        )
    mask = seq.loss_mask[:, :n_steps]
    loss = (
        F.smooth_l1_loss(q_taken, td_target, reduction="none") * mask
    ).sum() / mask.sum().clamp(min=1.0)
    return loss, float((q_taken.detach() * mask).sum().cpu() / mask.sum().clamp(min=1.0).cpu())


def train(args: argparse.Namespace) -> dict[str, float | int | str | None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_atari_dqn")
    args.feedback_mode = _resolve_feedback_mode(args)
    set_atari_seed(args.seed)
    device = select_device(args.device)
    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    video_dir = os.path.join(save_dir, "videos")
    history_path = os.path.join(save_dir, "metrics_history.jsonl")

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
        current_obs_np = to_channel_first_obs(obs_np)
        next_obs = torch.as_tensor(current_obs_np, device=device)
        input_channels = int(next_obs.shape[1])

        model_kwargs = dict(
            num_actions=num_actions,
            input_channels=input_channels,
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            encoder_feature_dim=args.encoder_feature_dim,
            core_dropout=args.core_dropout,
            feedback_mode=args.feedback_mode,
        )
        model = AtariQNetwork(**model_kwargs).to(device)
        target_net = AtariQNetwork(**model_kwargs).to(device)
        target_net.load_state_dict(model.state_dict())
        target_net.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        buffer = AtariReplayBuffer(
            buffer_size=args.buffer_size,
            num_envs=args.num_envs,
            obs_shape=tuple(next_obs.shape[1:]),
            device=device,
            seed=args.seed,
        )

        state = None
        next_done = torch.ones(args.num_envs, device=device)
        prev_done_np = np.zeros(args.num_envs, dtype=np.uint8)
        global_step = 0
        start_time = time.time()
        rolling_returns: list[float] = []
        last_loss = float("nan")
        last_q_mean = float("nan")
        final_metrics: dict[str, float | int | str | None] = {}

        while global_step < args.total_timesteps:
            global_step += args.num_envs
            epsilon = _linear_epsilon(args, global_step)

            # Always advance the model step so the GaWF recurrent state and
            # prev-Q feedback evolve identically whether or not the epsilon
            # coin picks a random action.
            with torch.no_grad():
                q_values, state = model.step(next_obs, next_done, state=state)
            greedy_action = q_values.argmax(dim=-1).cpu().numpy()
            random_action = np.random.randint(0, num_actions, size=args.num_envs)
            explore = np.random.random(size=args.num_envs) < epsilon
            action_np = np.where(explore, random_action, greedy_action)

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.uint8)

            buffer.add(
                obs=current_obs_np,
                actions=action_np,
                rewards=np.asarray(reward_np, dtype=np.float32),
                dones=done_np,
                resets=prev_done_np,
            )
            prev_done_np = done_np
            current_obs_np = to_channel_first_obs(next_obs_np)
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
                fps = int(global_step / max(time.time() - start_time, 1e-6))
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

        fps = int(global_step / max(time.time() - start_time, 1e-6))
        rolling_return = float(np.mean(rolling_returns)) if rolling_returns else float("nan")
        final_metrics = {
            "env_id": args.env_id,
            "algo": args.algo,
            "model_type": args.model_type,
            "feedback_mode": args.feedback_mode,
            "global_step": global_step,
            "episodic_return_100": rolling_return,
            "fps": fps,
            "loss": last_loss,
            "q_values_mean": last_q_mean,
            "epsilon": _linear_epsilon(args, global_step),
        }
        ckpt_name = (
            f"{args.algo}_{args.model_type}_{args.feedback_mode}_"
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
