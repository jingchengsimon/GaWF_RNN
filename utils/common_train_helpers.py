"""Cross-task training helpers shared by clutter, text, and future Atari code."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, seed: int) -> None:
    """Initialize a DataLoader worker with a deterministic seed."""
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
