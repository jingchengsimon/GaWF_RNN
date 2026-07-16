"""Training utilities for Atari reinforcement-learning experiments.

The helpers in this module are task-level RL math and filesystem utilities. They
do not define model recurrence; recurrent computation lives in ``utils.recurrent_cores``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_atari_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


def to_channel_first_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[-1] in {1, 3, 4} and arr.shape[1] not in {1, 3, 4}:
        arr = np.transpose(arr, (0, 3, 1, 2))
    if arr.ndim != 4:
        raise ValueError(f"Expected vector Atari obs with 4 dims, got {arr.shape}")
    return np.ascontiguousarray(arr)


def obs_to_tensor(obs: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(to_channel_first_obs(obs), device=device)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates for rollout tensors shaped ``(T, N)``."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - next_done.float()
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1].float()
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_true_np = y_true.detach().float().cpu().numpy().reshape(-1)
    y_pred_np = y_pred.detach().float().cpu().numpy().reshape(-1)
    var_y = np.var(y_true_np)
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true_np - y_pred_np) / var_y)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
