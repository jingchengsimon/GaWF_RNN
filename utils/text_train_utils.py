"""Shared helpers for text benchmark training.

This module keeps optimizer construction, parameter counting, device selection,
and small-loader subsetting consistent across IMDB and SentiHood text tasks.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def resolve_results_root(
    explicit_results_dir: Optional[str] = None,
    legacy_project_root: Optional[str] = ".",
) -> str:
    """Resolve the physical ``results`` root with the repository-wide precedence."""
    if explicit_results_dir:
        return os.path.abspath(os.path.expanduser(explicit_results_dir))
    for env_name in ("AIM3_RESULTS_PATH", "FAW_RNN_RESULTS_PATH"):
        value = os.environ.get(env_name)
        if value:
            return os.path.abspath(os.path.expanduser(value))
    return os.path.abspath(os.path.join(legacy_project_root or ".", "results"))


def select_device(requested: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_core_params(model: nn.Module) -> int:
    """Trainable params excluding embedding and, by default, the shared head."""
    total = 0
    include_fc = bool(getattr(model, "include_fc_in_core_params", False))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("embedding."):
            continue
        if name.startswith("fc.") and not include_fc:
            continue
        total += p.numel()
    return total


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_name: str,
    gawf_feedback_lr_scale: float = 1.0,
):
    """Adam(W) with GaWF U/V excluded from weight decay."""
    decay, no_decay, gate = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".U") or name.endswith(".V") or "U_layers" in name or "V_layers" in name:
            gate.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    if gawf_feedback_lr_scale == 1.0:
        no_decay.extend(gate)
        gate = []
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if gate:
        groups.append(
            {
                "params": gate,
                "weight_decay": 0.0,
                "lr": lr * gawf_feedback_lr_scale,
            }
        )
    if optim_name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    return torch.optim.Adam(groups, lr=lr)


def maybe_subset(
    loader: DataLoader, max_samples: Optional[int], batch_size: int, shuffle: bool
) -> DataLoader:
    """Return a loader over the first ``max_samples`` examples for smoke tests."""
    if not max_samples or len(loader.dataset) <= max_samples:
        return loader
    subset = Subset(loader.dataset, list(range(max_samples)))
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
