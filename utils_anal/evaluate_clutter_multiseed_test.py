"""Evaluate completed Clutter multi-seed checkpoints on the standard test split.

Inputs are a campaign checkpoint directory and ``stimulus_reg-test-*`` data. The script writes
one CSV row per checkpoint with exact frame-weighted character/sector accuracies and
cross-entropy losses plus JSON metadata. Models are loaded through the canonical analysis helper,
and the test DataLoader uses the production uint8/device-cast/compact pipeline.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.clutter_data_pipeline import prepare_clutter_inputs
from utils.clutter_train_acceleration import run_forward_with_feedback
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device

MODEL_ORDER = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
UNIT_RE = re.compile(r"^(gawf|rnn|lstm|gru|mamba|s5)-seed(\d+)$")


def parse_args() -> argparse.Namespace:
    """Parse campaign, test data, output, and DataLoader settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--save_csv", required=True)
    parser.add_argument("--save_meta", required=True)
    parser.add_argument(
        "--seed_filter_csv",
        default=None,
        help="Optional model/seed CSV defining the exact checkpoint cohort.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    return parser.parse_args()


def collect_checkpoints(checkpoint_root: str) -> list[tuple[str, int, str, str]]:
    """Return sorted ``(model, seed, unit, checkpoint)`` entries for complete units."""

    entries: list[tuple[str, int, str, str]] = []
    root = Path(checkpoint_root).resolve()
    for unit_dir in root.iterdir():
        if not unit_dir.is_dir():
            continue
        match = UNIT_RE.fullmatch(unit_dir.name)
        if match is None:
            continue
        checkpoints = sorted(unit_dir.glob("*_model.pth"))
        if len(checkpoints) > 1:
            raise RuntimeError(
                f"Expected at most one checkpoint in {unit_dir}, got {len(checkpoints)}"
            )
        if checkpoints:
            entries.append(
                (match.group(1), int(match.group(2)), unit_dir.name, str(checkpoints[0]))
            )
    entries.sort(key=lambda row: (MODEL_ORDER.index(row[0]), row[1]))
    if not entries:
        raise RuntimeError(f"No completed checkpoints found under {root}")
    return entries


def load_seed_filter(csv_path: str | None) -> set[tuple[str, int]] | None:
    """Load an optional exact model/seed cohort from a CSV containing model and seed columns."""

    if csv_path is None:
        return None
    with Path(csv_path).resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "model" not in rows[0] or "seed" not in rows[0]:
        raise ValueError(f"Seed filter CSV must contain model and seed columns: {csv_path}")
    return {(row["model"].lower(), int(row["seed"])) for row in rows}


def filter_checkpoints(
    entries: list[tuple[str, int, str, str]],
    allowed_units: set[tuple[str, int]] | None,
) -> list[tuple[str, int, str, str]]:
    """Select the requested model/seed cohort and fail if any requested checkpoint is absent."""

    if allowed_units is None:
        return entries
    filtered = [entry for entry in entries if (entry[0], entry[1]) in allowed_units]
    found = {(entry[0], entry[1]) for entry in filtered}
    if found != allowed_units:
        missing = sorted(allowed_units - found)
        raise RuntimeError(f"Missing filtered model/seed checkpoints: {missing}")
    return filtered


def configure_mamba_cpu_reference(model: torch.nn.Module) -> bool:
    """Switch Mamba's optional CUDA kernels to its official PyTorch CPU reference path."""

    try:
        import mamba_ssm.modules.mamba_simple as mamba_simple
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    except ImportError:
        return False
    configured = False
    for module in model.modules():
        if module.__class__.__module__.startswith("mamba_ssm") and hasattr(
            module, "use_fast_path"
        ):
            module.use_fast_path = False
            configured = True
    if configured:
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan_ref
    return configured


def evaluate_checkpoint(
    checkpoint: str,
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    chan_num: int,
) -> dict[str, int | float]:
    """Compute exact frame-weighted test accuracy and cross-entropy loss."""

    char_correct = 0
    sector_correct = 0
    char_loss_sum = 0.0
    sector_loss_sum = 0.0
    n_frames = 0
    model.eval()
    use_feedback = True if bool(getattr(model, "is_gawf_model", False)) else None
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            inputs, labels = batch[0], batch[1]
            inputs = prepare_clutter_inputs(
                inputs,
                device=device,
                cast_mode="device",
                frame_layout="compact",
                chan_num=chan_num,
                non_blocking=bool(data_loader.pin_memory),
            )
            labels = labels.to(device=device, non_blocking=bool(data_loader.pin_memory))
            out_char, out_sector = run_forward_with_feedback(
                model,
                inputs,
                use_feedback=use_feedback,
            )
            true_char = labels[:, :, 0].long()
            true_sector = labels[:, :, 1].long()
            char_loss_sum += float(
                F.cross_entropy(
                    out_char.reshape(-1, out_char.shape[-1]),
                    true_char.reshape(-1),
                    reduction="sum",
                ).item()
            )
            sector_loss_sum += float(
                F.cross_entropy(
                    out_sector.reshape(-1, out_sector.shape[-1]),
                    true_sector.reshape(-1),
                    reduction="sum",
                ).item()
            )
            char_correct += int((out_char.argmax(dim=2) == true_char).sum().item())
            sector_correct += int((out_sector.argmax(dim=2) == true_sector).sum().item())
            n_frames += int(true_char.numel())
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(f"  {Path(checkpoint).parent.name}: batches={batch_index + 1}")
    if n_frames == 0:
        raise RuntimeError(f"Test loader emitted no frames for {checkpoint}")
    return {
        "n_frames": n_frames,
        "char_correct": char_correct,
        "sector_correct": sector_correct,
        "test_char_acc": 100.0 * char_correct / n_frames,
        "test_sector_acc": 100.0 * sector_correct / n_frames,
        "test_char_loss": char_loss_sum / n_frames,
        "test_sector_loss": sector_loss_sum / n_frames,
        "char_loss_sum": char_loss_sum,
        "sector_loss_sum": sector_loss_sum,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    """Write per-seed CSV and companion metadata JSON."""

    csv_path = Path(args.save_csv).resolve()
    meta_path = Path(args.save_meta).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model",
        "seed",
        "unit",
        "test_char_acc",
        "test_sector_acc",
        "test_char_loss",
        "test_sector_loss",
        "n_frames",
        "char_correct",
        "sector_correct",
        "char_loss_sum",
        "sector_loss_sum",
        "checkpoint",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = {model: sum(row["model"] == model for row in rows) for model in MODEL_ORDER}
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_split": "test",
        "dataset_suffix": args.data_suffix,
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "seed_filter_csv": (
            str(Path(args.seed_filter_csv).resolve()) if args.seed_filter_csv else None
        ),
        "data_dir": str(Path(args.data_dir).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device == "cuda" and args.num_workers > 0,
        "use_mmap": args.use_mmap,
        "input_cast_mode": "device",
        "frame_layout": "compact",
        "chan_num": args.chan_num,
        "aggregation": "exact frame-weighted accuracy and cross-entropy loss per checkpoint",
        "mamba_cpu_reference": args.device == "cpu"
        and any(row["model"] == "mamba" for row in rows),
        "completed_checkpoint_count": len(rows),
        "seed_counts": counts,
        "csv": str(csv_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Evaluate every completed checkpoint and save one reproducible result table."""

    args = parse_args()
    if args.num_workers < 0 or args.batch_size <= 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    torch.manual_seed(42)
    np.random.seed(42)

    test_ds, num_pos = build_test_dataset(args)
    test_ds.input_cast_mode = "device"
    test_ds.frame_layout = "compact"
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda" and args.num_workers > 0,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    allowed_units = load_seed_filter(args.seed_filter_csv)
    entries = filter_checkpoints(collect_checkpoints(args.checkpoint_root), allowed_units)
    print(f"Evaluating {len(entries)} completed checkpoints on test suffix {args.data_suffix}.")
    rows: list[dict[str, Any]] = []
    for index, (model_key, seed, unit, checkpoint) in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] {unit}")
        model = build_model_from_ckpt(
            checkpoint,
            num_pos=num_pos,
            device=device,
            chan_num=args.chan_num,
        )
        if device.type == "cpu" and model_key == "mamba":
            if not configure_mamba_cpu_reference(model):
                raise RuntimeError("Unable to configure Mamba's official CPU reference path")
            print("  Mamba CPU evaluation uses selective_scan_ref (pure PyTorch).")
        metrics = evaluate_checkpoint(checkpoint, model, loader, device, args.chan_num)
        rows.append(
            {
                "model": model_key,
                "seed": seed,
                "unit": unit,
                **metrics,
                "checkpoint": str(Path(checkpoint).resolve()),
            }
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_outputs(args, rows)
    print(f"Saved {len(rows)} test rows to {Path(args.save_csv).resolve()}")


if __name__ == "__main__":
    main()
