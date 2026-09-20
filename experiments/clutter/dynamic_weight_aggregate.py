"""Aggregate the three-model, ten-seed dynamic-baseline Clutter campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev


MODELS = ("mlstm", "hyperlstm", "brims")
METRICS = (
    "best_val_acc_char",
    "best_val_acc_pos",
    "test_char_acc",
    "test_sector_acc",
)


def parse_args() -> argparse.Namespace:
    """Parse formal aggregation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-base", required=True, type=Path)
    parser.add_argument("--test-base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _one_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} below {directory}, found {len(matches)}")
    return matches[0]


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if len(values) != 10:
        raise ValueError(f"Expected ten seed values, got {len(values)}")
    return fmean(values), stdev(values) / math.sqrt(len(values))


def aggregate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect exact model/seed artifacts and summarize seeds as replicates."""
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite summary directory: {args.output_dir}")
    seed_rows: list[dict[str, object]] = []
    for model in MODELS:
        for seed in range(1, 11):
            leaf = f"{model}-seed{seed:02d}"
            metrics = json.loads(
                _one_file(args.run_base / leaf, "*_metrics.json").read_text(encoding="utf-8")
            )
            test = json.loads(
                (args.test_base / leaf / "reset_excluded_test_accuracy.json").read_text(
                    encoding="utf-8"
                )
            )
            if metrics["model_type"] != model or int(metrics["seed"]) != seed:
                raise ValueError(f"Training identity mismatch for {leaf}")
            if test["model"] != model or int(test["seed"]) != seed:
                raise ValueError(f"Test identity mismatch for {leaf}")
            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "total_param_count": int(metrics["total_param_count"]),
                    "core_param_count": int(metrics["core_param_count"]),
                    "best_val_acc_char": float(metrics["best_val_acc_char"]),
                    "best_val_acc_pos": float(metrics["best_val_acc_pos"]),
                    "test_char_acc": float(test["char_acc"]),
                    "test_sector_acc": float(test["sector_acc"]),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for model in MODELS:
        model_rows = [row for row in seed_rows if row["model"] == model]
        summary: dict[str, object] = {
            "model": model,
            "n": len(model_rows),
            "total_param_count": model_rows[0]["total_param_count"],
            "core_param_count": model_rows[0]["core_param_count"],
        }
        for metric in METRICS:
            mean, sem = _mean_sem([float(row[metric]) for row in model_rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sem"] = sem
        summary_rows.append(summary)
    return seed_rows, summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "| Model | n | Parameters | Validation char | Validation sector | Test char | Test sector |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        formatted = []
        for metric in METRICS:
            formatted.append(f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sem']:.3f}")
        lines.append(
            f"| {row['model']} | {row['n']} | {row['total_param_count']} | "
            + " | ".join(formatted)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    seed_rows, summary_rows = aggregate(args)
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "per_seed.csv", seed_rows)
    _write_csv(args.output_dir / "summary_mean_sem.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"n_per_model": 10, "models": summary_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "summary_mean_sem.md", summary_rows)
    (args.output_dir / ".complete").touch()
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
