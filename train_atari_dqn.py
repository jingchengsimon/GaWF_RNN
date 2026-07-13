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
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.atari_dqn_models import AtariQNetwork, AtariQNetworkState
from utils.atari_envs import (
    ATARI_PILOT_ENVS,
    make_multitask_vector_atari_env,
    make_vector_atari_env,
)
from utils.atari_replay import REPLAY_SAMPLING_MODES, AtariReplayBuffer
from utils.atari_train_acceleration import (
    AtariAcceleration,
    configure_atari_acceleration,
)
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
        help="Phase0 task list. Multiple games switch round-robin at episode boundaries.",
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
        default="round_robin",
        choices=["round_robin"],
        help="Phase0 episode-level game scheduler.",
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
    # experiments/generalization/atari_ssm_param_match.py). For Mamba, --ssm_state_size
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
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument(
        "--replay_sampling",
        type=str,
        default="task_balanced",
        choices=REPLAY_SAMPLING_MODES,
        help="Replay sampler. Task ids are sampling/loss metadata and never model inputs.",
    )
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
        help="Use CUDA fused Adam while preserving Adam hyperparameters.",
    )
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    return parser


SequenceForward = Callable[
    [torch.Tensor, torch.Tensor, AtariQNetworkState | None, bool | None],
    tuple[torch.Tensor, AtariQNetworkState | None],
]


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


def _extract_episode_records(infos: Any) -> list[tuple[str, float]]:
    """Extract ``(env_id, return)`` pairs from current Gymnasium vector formats."""
    records: list[tuple[str, float]] = []
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
            infos.get("env_id", np.full(raw_returns.shape, "unknown", dtype=object))
        ).reshape(-1)
        for index, (episode_return, keep) in enumerate(zip(raw_returns, mask)):
            if keep:
                env_id = str(env_values[index]) if index < env_values.size else "unknown"
                records.append((env_id, float(episode_return)))

    final_infos = infos.get("final_info")
    if final_infos is None:
        return records
    for final_info in final_infos:
        if final_info and "episode" in final_info:
            episode_return = np.asarray(final_info["episode"]["r"]).reshape(-1)[0]
            records.append((str(final_info.get("env_id", "unknown")), float(episode_return)))
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
        if index < num_envs and keep:
            result[index] = str(value)
    return result


def _linear_epsilon(args: argparse.Namespace, global_step: int) -> float:
    decay_steps = max(1.0, args.exploration_fraction * args.total_timesteps)
    slope = (args.end_epsilon - args.start_epsilon) / decay_steps
    return max(args.end_epsilon, args.start_epsilon + slope * global_step)


def _next_state_reset_flags(dones: np.ndarray, autoreset_rows: np.ndarray) -> np.ndarray:
    """Reset on terminal rows and again after NEXT_STEP's ignored autoreset row."""
    return np.logical_or(dones, autoreset_rows).astype(np.uint8)


def _aggregate_td_loss(
    elementwise_loss: torch.Tensor,
    task_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    buffer: AtariReplayBuffer,
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


def _dqn_transition_loss(
    model_forward: SequenceForward,
    target_forward: SequenceForward,
    buffer: AtariReplayBuffer,
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
        td_target = batch.rewards.clamp(-1.0, 1.0) + args.gamma * (1.0 - batch.dones) * q_next
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
    buffer: AtariReplayBuffer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
            seq.rewards[:, :n_steps].clamp(-1.0, 1.0)
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
    env_ids, action_space_mode = _resolve_task_config(args)
    if args.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {args.num_layers}")
    if args.frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {args.frame_skip}")
    if args.gawf_feedback_lr_scale <= 0:
        raise ValueError("gawf_feedback_lr_scale must be > 0")
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
    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    video_dir = os.path.join(save_dir, "videos")
    history_path = os.path.join(save_dir, "metrics_history.jsonl")

    is_multitask = len(env_ids) > 1
    if is_multitask:
        if args.capture_video:
            raise ValueError("Phase0 multi-task training does not support --capture_video")
        envs = make_multitask_vector_atari_env(
            env_ids=env_ids,
            seed=args.seed,
            num_envs=args.num_envs,
            frame_stack=args.frame_stack,
            frame_skip=args.frame_skip,
            flicker_prob=args.flicker_prob,
            task_schedule=args.task_schedule,
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
        target_net.load_state_dict(model.state_dict())
        target_net.eval()
        target_net.requires_grad_(False)
        use_fused_optimizer = args.fused_optimizer and device.type == "cuda"
        adam_kwargs = {"fused": True} if use_fused_optimizer else {}
        if args.model_type == "gawf":
            gate_params = [
                param
                for name, param in model.named_parameters()
                if name.startswith("core.U") or name.startswith("core.V")
            ]
            gate_ids = {id(param) for param in gate_params}
            base_params = [param for param in model.parameters() if id(param) not in gate_ids]
            optimizer = optim.Adam(
                [
                    {"params": base_params, "lr": args.learning_rate},
                    {
                        "params": gate_params,
                        "lr": args.learning_rate * args.gawf_feedback_lr_scale,
                        "weight_decay": 0.0,
                    },
                ],
                **adam_kwargs,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                **adam_kwargs,
            )
        scaler = acceleration.build_grad_scaler()
        model_forward = acceleration.compile_callable(model.forward_sequence)
        target_forward = acceleration.compile_callable(target_net.forward_sequence)

        buffer = AtariReplayBuffer(
            buffer_size=args.buffer_size,
            num_envs=args.num_envs,
            obs_shape=tuple(next_obs.shape[1:]),
            device=device,
            seed=args.seed,
            num_tasks=len(env_ids),
            sampling_mode=args.replay_sampling,
        )

        state = None
        next_done = torch.ones(args.num_envs, device=device)
        prev_done_np = np.zeros(args.num_envs, dtype=np.uint8)
        global_step = 0
        start_time = time.time()
        rolling_returns: list[float] = []
        rolling_returns_by_env: dict[str, list[float]] = {env_id: [] for env_id in env_ids}
        episode_counts = {env_id: 0 for env_id in env_ids}
        environment_steps = {env_id: 0 for env_id in env_ids}
        env_id_to_task = {env_id: task_id for task_id, env_id in enumerate(env_ids)}
        last_loss_tensor: torch.Tensor | None = None
        last_q_mean_tensor: torch.Tensor | None = None
        final_metrics: dict[str, Any] = {}

        while global_step < args.total_timesteps:
            global_step += args.num_envs
            epsilon = _linear_epsilon(args, global_step)

            # Always advance the model step so the GaWF recurrent state and
            # prev-Q feedback evolve identically whether or not the epsilon
            # coin picks a random action.
            with torch.no_grad(), acceleration.autocast():
                q_values, state = _step_with_sequence_forward(
                    model_forward, next_obs, next_done, state
                )
            greedy_action = q_values.argmax(dim=-1).cpu().numpy()
            random_action = np.random.randint(0, num_actions, size=args.num_envs)
            explore = np.random.random(size=args.num_envs) < epsilon
            action_np = np.where(explore, random_action, greedy_action)

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(action_np)
            done_np = np.logical_or(terminated_np, truncated_np).astype(np.uint8)
            # NEXT_STEP autoreset consumes one ignored action after a terminal
            # row. Reset before that invalid row and again before the first
            # valid observation of the newly selected episode/task.
            state_reset_np = _next_state_reset_flags(done_np, prev_done_np)

            step_env_ids = _extract_step_env_ids(infos, args.num_envs)
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

            buffer.add(
                obs=current_obs_np,
                actions=action_np,
                rewards=np.asarray(reward_np, dtype=np.float32),
                dones=done_np,
                resets=prev_done_np,
                task_ids=task_ids_np,
            )
            prev_done_np = done_np
            current_obs_np = to_channel_first_obs(next_obs_np)
            next_obs = torch.as_tensor(current_obs_np, device=device)
            next_done = torch.as_tensor(state_reset_np, device=device, dtype=torch.float32)

            rolling_returns.extend(_extract_episode_returns(infos))
            if len(rolling_returns) > 100:
                rolling_returns = rolling_returns[-100:]
            for env_id, episode_return in _extract_episode_records(infos):
                if env_id not in rolling_returns_by_env:
                    continue
                rolling_returns_by_env[env_id].append(episode_return)
                rolling_returns_by_env[env_id] = rolling_returns_by_env[env_id][-100:]
                episode_counts[env_id] += 1

            if global_step >= args.learning_starts and global_step % args.train_frequency == 0:
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
                last_loss_tensor = loss.detach()
                last_q_mean_tensor = q_mean.detach()

            if global_step % args.target_network_frequency == 0:
                target_net.load_state_dict(model.state_dict())

            if global_step % args.log_interval == 0:
                fps = int(global_step / max(time.time() - start_time, 1e-6))
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
                    }
                    for env_id, returns in rolling_returns_by_env.items()
                }
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
                                "per_env": per_env_history,
                            }
                        )
                        + "\n"
                    )

        fps = int(global_step / max(time.time() - start_time, 1e-6))
        last_loss, last_q_mean = _materialize_training_stats(
            last_loss_tensor,
            last_q_mean_tensor,
        )
        rolling_return = float(np.mean(rolling_returns)) if rolling_returns else float("nan")
        per_env_metrics = {
            env_id: {
                "episodic_return_100": (
                    float(np.mean(returns)) if returns else float("nan")
                ),
                "episodes": episode_counts[env_id],
                "environment_steps": environment_steps[env_id],
            }
            for env_id, returns in rolling_returns_by_env.items()
        }
        final_metrics = {
            "env_id": env_ids[0] if not is_multitask else None,
            "env_ids": list(env_ids),
            "multitask": is_multitask,
            "action_space_mode": action_space_mode,
            "num_actions": num_actions,
            "task_schedule": args.task_schedule if is_multitask else None,
            "replay_sampling": args.replay_sampling,
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
            "episodic_return_100": rolling_return,
            "fps": fps,
            "loss": last_loss,
            "q_values_mean": last_q_mean,
            "epsilon": _linear_epsilon(args, global_step),
            "amp_dtype": acceleration.amp_dtype_name,
            "allow_tf32": acceleration.allow_tf32,
            "cudnn_benchmark": acceleration.cudnn_benchmark,
            "fused_optimizer": use_fused_optimizer,
            "compile_model": acceleration.compile_model,
            "compile_mode": acceleration.compile_mode,
            "per_env": per_env_metrics,
        }
        layer_suffix = f"_L{args.num_layers}" if args.num_layers > 1 else ""
        env_tag = "__".join(env_id.replace("/", "_") for env_id in env_ids)
        ckpt_name = (
            f"{args.algo}_{args.model_type}_{args.feedback_mode}{layer_suffix}_{env_tag}.pth"
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
