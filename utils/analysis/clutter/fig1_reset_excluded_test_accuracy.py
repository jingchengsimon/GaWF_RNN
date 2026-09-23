"""Evaluate Clutter checkpoints after excluding every rollout's initial frame.

``collect`` writes one compact accuracy JSON for one frozen checkpoint.  ``aggregate`` combines
the registered six-model, ten-seed set into the CSV consumed by the Figure 1 renderer.  Loss is
intentionally not evaluated or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.training.clutter.clutter_train_acceleration import run_forward_with_feedback


ORIGINAL_MODELS = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
FEEDBACK_CONTROL_MODELS = (
    "gawf_additive",
    "rnn_fb",
    "gru_fb",
    "lstm_fb",
    "rnn_fb_add",
    "gru_fb_add",
    "lstm_fb_add",
)
DYNAMIC_WEIGHT_MODELS = ("mlstm", "hyperlstm", "brims")
NONLINEARITY_ABLATION_MODELS = (
    "gawf_nowrap",
    "rnn_nowrap",
    "gru_nowrap",
    "lstm_nowrap",
    "mamba_nowrap",
    "s5_nowrap",
    "gawf_notanh",
    "rnn_notanh",
    "gawf_rnncore",
    "gawf_legacy",
)
MODELS = (
    ORIGINAL_MODELS
    + FEEDBACK_CONTROL_MODELS
    + DYNAMIC_WEIGHT_MODELS
    + NONLINEARITY_ABLATION_MODELS
)
RESULT_NAME = "reset_excluded_test_accuracy.json"


def parse_args() -> argparse.Namespace:
    """Parse one-checkpoint collection or six-model aggregation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--ckpt", required=True, type=Path)
    collect.add_argument("--model", required=True, choices=MODELS)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--output_dir", required=True, type=Path)
    collect.add_argument("--data_dir", required=True, type=Path)
    collect.add_argument("--data_suffix", default="40h-uint8")
    collect.add_argument("--sequence_length", type=int, default=32)
    collect.add_argument("--chan_num", type=int, default=2)
    collect.add_argument("--batch_size", type=int, default=256)
    collect.add_argument("--num_workers", type=int, default=2)
    collect.add_argument("--device", default="cuda")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--data_root", required=True, type=Path)
    aggregate.add_argument("--output_csv", required=True, type=Path)
    aggregate.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(ORIGINAL_MODELS),
        help="Models to aggregate; default preserves the original six-model output.",
    )
    return parser.parse_args()


def collect(args: argparse.Namespace) -> Path:
    """Evaluate one checkpoint on all non-initial frames of its fixed windows."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.sequence_length <= 1 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("sequence_length must exceed one; batch_size/workers must be nonnegative")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        data_suffix=args.data_suffix,
        use_mmap=True,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
        sequence_length=args.sequence_length,
    )
    dataset, num_pos = build_test_dataset(dataset_args)
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    char_correct = sector_correct = n_frames = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            inputs = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch[1].to(device=device, non_blocking=True)
            use_feedback = True if args.model == "gawf" else None
            char_logits, sector_logits = run_forward_with_feedback(model, inputs, use_feedback)
            valid = torch.ones(labels.shape[:2], device=device, dtype=torch.bool)
            valid[:, 0] = False
            char_correct += int(
                ((char_logits.argmax(dim=2) == labels[:, :, 0]) & valid).sum().item()
            )
            sector_correct += int(
                ((sector_logits.argmax(dim=2) == labels[:, :, 1]) & valid).sum().item()
            )
            n_frames += int(valid.sum().item())
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(
                    f"processed {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}"
                )
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / RESULT_NAME
    destination.write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "checkpoint": str(args.ckpt.resolve()),
                "data_suffix": args.data_suffix,
                "sequence_length": args.sequence_length,
                "excluded_timestep": 0,
                "n_windows": len(dataset),
                "n_frames": n_frames,
                "char_acc": 100.0 * char_correct / n_frames,
                "sector_acc": 100.0 * sector_correct / n_frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def aggregate(args: argparse.Namespace) -> Path:
    """Write the exact 60-row reset-excluded Figure 1 accuracy CSV."""

    rows: list[dict[str, object]] = []
    for model in args.models:
        for seed in range(1, 11):
            path = args.data_root / f"{model}-seed{seed:02d}" / RESULT_NAME
            if not path.is_file():
                raise FileNotFoundError(path)
            item = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "source": "test_reset_excluded",
                    "model": model,
                    "seed": seed,
                    "char_acc": item["char_acc"],
                    "sector_acc": item["sector_acc"],
                }
            )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return args.output_csv


def main() -> None:
    """Dispatch collection or aggregation."""

    args = parse_args()
    output = collect(args) if args.command == "collect" else aggregate(args)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
