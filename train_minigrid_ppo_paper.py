"""Paper-protocol recurrent PPO for MiniGrid RedBlueDoors and MemoryS7.

This is a separate reproduction entry point. It does not change the existing
MiniGrid PPO trainer or any Atari DQN code. The defaults reproduce the PPO2
protocol reported by Toro Icarte et al. (2020) while keeping the repository's
installed MiniGrid environment version explicit in the saved metadata.

Outputs (in ``--save_dir`` or ``results/train_data/<result_suffix>``):
- ``metrics_history.jsonl``  — periodic learning-curve records.
- ``metrics.json``           — final metrics and complete protocol metadata.
- ``checkpoint.pth``         — model and strict-Adam optimizer state.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utils.atari_train_utils import compute_gae, ensure_dir, save_json, select_device
from utils.atari_train_utils import set_atari_seed
from utils.minigrid_envs import make_vector_minigrid_env
from utils.minigrid_ppo_paper_models import (
    PAPER_MINIGRID_MODEL_TYPES,
    PaperMiniGridActorCritic,
)

PAPER_ENVS = (
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-MemoryS7-v0",
)
PAPER_LEARNING_RATES = {
    "MiniGrid-RedBlueDoors-8x8-v0": 1e-5,
    "MiniGrid-MemoryS7-v0": 1e-3,
}


def paper_learning_rate(env_id: str, model_type: str) -> float:
    # Only the paper LSTM MemoryS7 baseline uses 1e-3; core comparisons use 1e-5.
    if model_type == "paper_lstm" and env_id == "MiniGrid-MemoryS7-v0":
        return 1e-3
    return 1e-5


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI for the isolated paper-protocol trainer."""
    parser = argparse.ArgumentParser(description="Paper-aligned recurrent PPO on MiniGrid")
    parser.add_argument("--env_id", choices=PAPER_ENVS, default=PAPER_ENVS[0])
    parser.add_argument(
        "--model_type",
        choices=PAPER_MINIGRID_MODEL_TYPES,
        default="paper_lstm",
    )
    parser.add_argument("--agent_view_size", type=int, default=3)
    parser.add_argument("--encoder_hidden_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--core_dropout", type=float, default=0.0)
    parser.add_argument("--ssm_d_model", type=int, default=128)
    parser.add_argument("--ssm_state_size", type=int, default=64)
    parser.add_argument("--ssm_num_layers", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--result_suffix", default="minigrid_ppo_paper")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--log_interval_updates", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _minigrid_version() -> str:
    try:
        import minigrid

        return str(minigrid.__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _extract_episode_returns(infos: dict[str, Any]) -> list[float]:
    """Extract finalized episode returns from Gymnasium vector info."""
    returns: list[float] = []
    final_info = infos.get("final_info")
    if final_info is not None:
        for item in final_info:
            if item and "episode" in item:
                returns.append(float(np.asarray(item["episode"]["r"]).reshape(-1)[0]))
    episode = infos.get("episode")
    mask = infos.get("_episode")
    if episode is not None and mask is not None:
        values = np.asarray(episode["r"]).reshape(-1)
        returns.extend(float(value) for value, keep in zip(values, mask) if keep)
    return returns


def _gawf_parameters(model: PaperMiniGridActorCritic) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name in ("core.U", "core.V")
    ]


def _gawf_gradient_norm(model: PaperMiniGridActorCritic) -> float:
    squared = 0.0
    for parameter in _gawf_parameters(model):
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().item())
    return squared**0.5


def _validate_args(args: argparse.Namespace) -> None:
    if args.agent_view_size != 3:
        raise ValueError("Paper protocol requires --agent_view_size 3")
    if args.encoder_hidden_size != 128:
        raise ValueError("Paper protocol requires five 128-unit tanh encoder layers")
    if args.num_envs < args.num_minibatches:
        raise ValueError("num_envs must be at least num_minibatches")
    if args.num_envs % args.num_minibatches != 0:
        raise ValueError("num_envs must be divisible by num_minibatches")
    if args.num_steps != 128:
        raise ValueError("Paper protocol requires --num_steps 128")
    if args.update_epochs != 4:
        raise ValueError("Paper protocol requires --update_epochs 4")


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train one paper baseline or one recurrent-core replacement."""
    _validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_minigrid_ppo_paper")
    set_atari_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = select_device(args.device)
    learning_rate = (
        paper_learning_rate(args.env_id, args.model_type)
        if args.learning_rate is None
        else float(args.learning_rate)
    )

    save_dir = args.save_dir or os.path.join("results", "train_data", args.result_suffix)
    ensure_dir(save_dir)
    history_path = os.path.join(save_dir, "metrics_history.jsonl")
    metrics_path = os.path.join(save_dir, "metrics.json")
    if not args.overwrite and (os.path.exists(history_path) or os.path.exists(metrics_path)):
        raise FileExistsError(f"Refusing to overwrite existing result directory: {save_dir}")
    if args.overwrite:
        for path in (history_path, metrics_path):
            if os.path.exists(path):
                os.remove(path)

    envs = make_vector_minigrid_env(
        args.env_id,
        seed=args.seed,
        num_envs=args.num_envs,
        agent_view_size=args.agent_view_size,
    )
    try:
        num_actions = int(envs.single_action_space.n)
        obs_np, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(np.asarray(obs_np), device=device, dtype=torch.uint8)
        model = PaperMiniGridActorCritic(
            num_actions=num_actions,
            grid_size=int(next_obs.shape[-1]),
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            encoder_hidden_size=args.encoder_hidden_size,
            core_dropout=args.core_dropout,
            ssm_d_model=args.ssm_d_model,
            ssm_state_size=args.ssm_state_size,
            ssm_num_layers=args.ssm_num_layers,
            ssm_context_len=args.num_steps,
        ).to(device=device)

        # Strict PPO2-style Adam: no fused/foreach acceleration and one parameter group.
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=args.adam_eps,
            foreach=False,
            fused=False,
        )

        num_envs, num_steps = args.num_envs, args.num_steps
        obs_buffer = torch.zeros(
            (num_steps, num_envs, *next_obs.shape[1:]),
            device=device,
            dtype=torch.uint8,
        )
        actions_buffer = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
        old_logprobs_buffer = torch.zeros((num_steps, num_envs), device=device)
        rewards_buffer = torch.zeros((num_steps, num_envs), device=device)
        prev_dones_buffer = torch.zeros((num_steps, num_envs), device=device)
        values_buffer = torch.zeros((num_steps, num_envs), device=device)

        next_done = torch.ones(num_envs, device=device)
        state = None
        global_step = 0
        start_time = time.time()
        recent_returns: list[float] = []
        batch_size = num_envs * num_steps
        num_updates = max(1, math.ceil(args.total_timesteps / batch_size))
        gawf_grad_norm: float | None = None
        mean_return = success_rate = float("nan")
        last_policy_loss = last_value_loss = last_entropy = float("nan")

        logger.info(
            "protocol=paper_ppo2 env=%s model=%s seed=%d lr=%g steps=%d updates=%d",
            args.env_id,
            args.model_type,
            args.seed,
            learning_rate,
            args.total_timesteps,
            num_updates,
        )

        for update in range(1, num_updates + 1):
            rollout_start_state = model.detach_state(state)
            for step in range(num_steps):
                global_step += num_envs
                obs_buffer[step] = next_obs
                prev_dones_buffer[step] = next_done
                with torch.no_grad():
                    action, logprob, _entropy, value, state = model.act(
                        next_obs, next_done, state=state
                    )
                actions_buffer[step] = action
                old_logprobs_buffer[step] = logprob
                values_buffer[step] = value
                obs_np, reward_np, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                rewards_buffer[step] = torch.as_tensor(
                    reward_np, device=device, dtype=torch.float32
                )
                next_done = torch.as_tensor(
                    np.logical_or(terminated, truncated),
                    device=device,
                    dtype=torch.float32,
                )
                next_obs = torch.as_tensor(
                    np.ascontiguousarray(obs_np), device=device, dtype=torch.uint8
                )
                recent_returns.extend(_extract_episode_returns(infos))
            recent_returns = recent_returns[-100:]

            with torch.no_grad():
                _action, _logprob, _entropy, next_value, _next_state = model.act(
                    next_obs, next_done, state=state, deterministic=True
                )
                advantages, returns = compute_gae(
                    rewards_buffer,
                    prev_dones_buffer,
                    values_buffer,
                    next_value,
                    next_done,
                    args.gamma,
                    args.gae_lambda,
                )

            env_major_obs = obs_buffer.transpose(0, 1).contiguous()
            env_major_actions = actions_buffer.transpose(0, 1).contiguous()
            env_major_dones = prev_dones_buffer.transpose(0, 1).contiguous()
            env_major_old_logprobs = old_logprobs_buffer.transpose(0, 1).contiguous()
            env_major_old_values = values_buffer.transpose(0, 1).contiguous()
            env_major_advantages = advantages.transpose(0, 1).contiguous()
            env_major_returns = returns.transpose(0, 1).contiguous()

            env_order = np.arange(num_envs)
            envs_per_minibatch = num_envs // args.num_minibatches
            for _epoch in range(args.update_epochs):
                np.random.shuffle(env_order)
                for start in range(0, num_envs, envs_per_minibatch):
                    indices_np = env_order[start : start + envs_per_minibatch]
                    indices = torch.as_tensor(indices_np, device=device, dtype=torch.long)
                    minibatch_state = model.select_state(rollout_start_state, indices)
                    new_logprobs, entropy, new_values = model.evaluate_actions_sequence(
                        env_major_obs.index_select(0, indices),
                        env_major_dones.index_select(0, indices),
                        env_major_actions.index_select(0, indices),
                        state=minibatch_state,
                    )
                    old_logprobs = env_major_old_logprobs.index_select(0, indices)
                    old_values = env_major_old_values.index_select(0, indices)
                    minibatch_returns = env_major_returns.index_select(0, indices)
                    minibatch_advantages = env_major_advantages.index_select(0, indices)
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std(unbiased=False) + 1e-8)

                    ratio = (new_logprobs - old_logprobs).exp()
                    unclipped = ratio * minibatch_advantages
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                    ) * minibatch_advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    # PPO2 default cliprange_vf=-1: no value clipping.
                    value_loss = 0.5 * (new_values - minibatch_returns).pow(2).mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        - args.ent_coef * entropy_mean
                        + args.vf_coef * value_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.model_type == "gawf" and gawf_grad_norm is None:
                        gawf_grad_norm = _gawf_gradient_norm(model)
                        if not np.isfinite(gawf_grad_norm) or gawf_grad_norm <= 0.0:
                            raise RuntimeError(
                                "GaWF U/V received no gradient from action-logit feedback"
                            )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    last_policy_loss = float(policy_loss.detach())
                    last_value_loss = float(value_loss.detach())
                    last_entropy = float(entropy_mean.detach())

            if recent_returns:
                mean_return = float(np.mean(recent_returns))
                success_rate = float(np.mean(np.asarray(recent_returns) > 0.0))

            if update % args.log_interval_updates == 0 or update == num_updates:
                fps = int(global_step / max(time.time() - start_time, 1e-6))
                record = {
                    "global_step": global_step,
                    "update": update,
                    "success_rate": success_rate,
                    "episodic_return_100": mean_return,
                    "policy_loss": last_policy_loss,
                    "value_loss": last_value_loss,
                    "entropy": last_entropy,
                    "fps": fps,
                }
                logger.info(
                    "upd=%d/%d step=%d success=%.3f return=%.3f fps=%d",
                    update,
                    num_updates,
                    global_step,
                    success_rate,
                    mean_return,
                    fps,
                )
                with open(history_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")

        checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            },
            checkpoint_path,
        )
        metrics: dict[str, Any] = {
            "protocol": "toro_icarte_2020_openai_baselines_ppo2",
            "protocol_layer": (
                "paper_lstm" if args.model_type == "paper_lstm" else "core_comparison"
            ),
            "environment_implementation": "current_minigrid_separate_reproduction",
            "minigrid_version": _minigrid_version(),
            "env_id": args.env_id,
            "agent_view_size": args.agent_view_size,
            "model_type": args.model_type,
            "seed": args.seed,
            "requested_total_timesteps": args.total_timesteps,
            "global_step": global_step,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "value_clipping": False,
            "advantage_normalization": "per_recurrent_minibatch",
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "optimizer": "torch.optim.Adam",
            "learning_rate": learning_rate,
            "adam_eps": args.adam_eps,
            "adam_fused": False,
            "adam_foreach": False,
            "precision": "float32",
            "tf32": False,
            "encoder": "one_hot_flatten_5x128_tanh",
            "hidden_size": args.hidden_size,
            "feedback_mode": model.feedback_mode,
            "feedback_dim": model.feedback_dim,
            "feedback_detached": bool(args.model_type == "gawf"),
            "gawf_first_uv_grad_norm": gawf_grad_norm,
            "core_param_count": int(sum(p.numel() for p in model.core.parameters())),
            "total_param_count": int(sum(p.numel() for p in model.parameters())),
            "success_rate": success_rate,
            "episodic_return_100": mean_return,
            "checkpoint": checkpoint_path,
        }
        save_json(metrics_path, metrics)
        return metrics
    finally:
        envs.close()


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
