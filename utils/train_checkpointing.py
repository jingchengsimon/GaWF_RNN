"""Shared helpers for resumable training entry points.

The atomic-save, RNG-capture, history-reconciliation, and protocol-validation
logic here is the pattern already proven by the paper-aligned MiniGrid PPO
runner. It lives in ``utils`` so Atari DQN can reuse it without duplicating the
subtle parts: replacing a checkpoint through a temporary file so preemption
cannot leave a half-written payload, and trimming log records newer than the
checkpoint so a resumed run never shows a regressed or duplicated step.

``train_minigrid_ppo_paper.py`` intentionally still carries its own private
copies while its recovery jobs are in flight; de-duplicating it is a follow-up.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    """Capture every RNG stream a training loop draws from."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the streams captured by :func:`rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def atomic_torch_save(payload: dict[str, Any], path: str) -> None:
    """Replace a checkpoint atomically so preemption cannot leave a partial file."""
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    """Load a trusted local training checkpoint across supported PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return checkpoint


def reconcile_history(history_path: str, global_step: int) -> str | None:
    """Archive and trim log records newer than the checkpoint before appending.

    A checkpoint is written less often than history rows, so a preempted run
    leaves rows past the restore point. Keeping them would make the resumed
    history non-monotonic in ``global_step``.
    """
    if not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as stream:
        lines = stream.readlines()
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["global_step"]) <= global_step:
            kept.append(line if line.endswith("\n") else line + "\n")
    if len(kept) == len(lines):
        return None
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    archive_path = f"{history_path}.pre_resume_{timestamp}"
    shutil.copy2(history_path, archive_path)
    temporary_path = f"{history_path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        stream.writelines(kept)
    os.replace(temporary_path, history_path)
    return archive_path


def validate_resume_protocol(
    checkpoint: dict[str, Any],
    args: Any,
    keys: Iterable[str],
    *,
    expected_format_version: int,
    learning_rate: float | None = None,
) -> None:
    """Reject continuation when the saved scientific protocol is incompatible.

    Resuming across a changed protocol would silently splice two different
    experiments into one result directory, so every mismatch is fatal.
    """
    if checkpoint.get("format_version") != expected_format_version:
        raise ValueError(
            "Checkpoint format_version "
            f"{checkpoint.get('format_version')!r} != {expected_format_version}"
        )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError("Checkpoint is missing saved arguments")
    mismatches = [
        key
        for key in keys
        if key not in saved_args or saved_args[key] != getattr(args, key)
    ]
    if learning_rate is not None:
        saved_learning_rate = float(checkpoint.get("learning_rate", float("nan")))
        if not math.isclose(saved_learning_rate, learning_rate, rel_tol=0.0, abs_tol=0.0):
            mismatches.append("learning_rate")
    if mismatches:
        raise ValueError(
            "Resume checkpoint protocol mismatch for: " + ", ".join(sorted(set(mismatches)))
        )
