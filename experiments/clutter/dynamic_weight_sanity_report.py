"""Validate one two-epoch dynamic-baseline sanity run and write a compact report."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse sanity-report inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("lstm", "mlstm", "hyperlstm", "brims"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--elapsed-seconds", required=True, type=float)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {number}")
    return number


def build_report(args: argparse.Namespace) -> dict[str, object]:
    """Require finite decreasing total train loss across the two completed epochs."""
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite sanity report: {args.output}")
    with args.history.open("rb") as handle:
        history = pickle.load(handle)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    actual_epochs = int(metrics["actual_epochs"])
    if actual_epochs != 2:
        raise ValueError(f"Expected exactly two completed epochs, got {actual_epochs}")

    char_losses = list(history["train_loss_char"][:actual_epochs])
    position_losses = list(history["train_loss_pos"][:actual_epochs])
    total_losses = [
        _finite_float(char_loss, f"char loss epoch {index + 1}")
        + _finite_float(position_loss, f"position loss epoch {index + 1}")
        for index, (char_loss, position_loss) in enumerate(zip(char_losses, position_losses))
    ]
    if total_losses[-1] >= total_losses[0]:
        raise ValueError(
            f"{args.model} total train loss did not decrease: {total_losses[0]} -> "
            f"{total_losses[-1]}"
        )
    elapsed_seconds = _finite_float(args.elapsed_seconds, "elapsed seconds")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed seconds must be positive")

    return {
        "model": args.model,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "actual_epochs": actual_epochs,
        "train_total_loss_by_epoch": total_losses,
        "loss_decreased": True,
        "all_finite": True,
        "elapsed_seconds": elapsed_seconds,
        "wall_clock_seconds_per_epoch": elapsed_seconds / actual_epochs,
        "parameter_count": int(metrics["total_param_count"]),
        "core_parameter_count": int(metrics["core_param_count"]),
        "learning_rate": float(metrics["lr"]),
        "weight_decay": float(metrics["weight_decay"]),
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
