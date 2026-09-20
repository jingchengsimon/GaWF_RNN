"""Summarize formal Clutter feedback controls with seed-level mean and SEM."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ORIGINAL_MODELS = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
FEEDBACK_MODELS = ("gawf_additive", "rnn_fb", "gru_fb", "lstm_fb")
MODEL_ORDER = ORIGINAL_MODELS + FEEDBACK_MODELS
SHUFFLE_CONDITIONS = ("baseline", "shuffle_digit", "shuffle_sector", "shuffle_all")


def parse_args() -> argparse.Namespace:
    """Parse formal accuracy, ablation, and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_test_csv", required=True, type=Path)
    parser.add_argument("--feedback_test_csv", required=True, type=Path)
    parser.add_argument("--gawf_ablation_dir", required=True, type=Path)
    parser.add_argument("--feedback_ablation_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def _mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,) or not np.isfinite(array).all():
        raise ValueError(f"Expected ten finite seed values, got shape={array.shape}")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def _load_test_rows(paths: tuple[Path, Path]) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"char_acc": [], "sector_acc": []}
    )
    seen: set[tuple[str, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model = row["model"]
                seed = int(row["seed"])
                key = model, seed
                if key in seen:
                    raise RuntimeError(f"Duplicate test row for {model} seed {seed}")
                seen.add(key)
                grouped[model]["char_acc"].append(float(row["char_acc"]))
                grouped[model]["sector_acc"].append(float(row["sector_acc"]))
    missing = [model for model in MODEL_ORDER if model not in grouped]
    if missing:
        raise RuntimeError(f"Missing formal test rows for models: {missing}")
    return grouped


def _load_ablation_model(root: Path, model: str) -> dict[str, dict[str, list[float]]]:
    grouped = {
        condition: {"char_acc": [], "sector_acc": []}
        for condition in SHUFFLE_CONDITIONS
    }
    for seed in range(1, 11):
        path = root / f"{model}-seed{seed:02d}" / "ablation_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("exclude_window_initial_frame") is not True:
            raise RuntimeError(f"Ablation does not exclude reset frames: {path}")
        if int(payload.get("sequence_length", -1)) != 512:
            raise RuntimeError(f"Expected sequence_length=512: {path}")
        for condition in SHUFFLE_CONDITIONS:
            metrics = payload["conditions"][condition]
            grouped[condition]["char_acc"].append(float(metrics["char_acc"]))
            grouped[condition]["sector_acc"].append(float(metrics["sector_acc"]))
    return grouped


def _summary_row(
    model: str,
    test_metrics: dict[str, list[float]],
    ablation: dict[str, dict[str, list[float]]] | None,
) -> dict[str, Any]:
    char_mean, char_sem = _mean_sem(test_metrics["char_acc"])
    sector_mean, sector_sem = _mean_sem(test_metrics["sector_acc"])
    row: dict[str, Any] = {
        "model": model,
        "n": 10,
        "test_char_mean": char_mean,
        "test_char_sem": char_sem,
        "test_sector_mean": sector_mean,
        "test_sector_sem": sector_sem,
    }
    if ablation is None:
        return row
    for condition in SHUFFLE_CONDITIONS:
        for metric in ("char_acc", "sector_acc"):
            mean, sem = _mean_sem(ablation[condition][metric])
            prefix = metric.removesuffix("_acc")
            row[f"{condition}_{prefix}_mean"] = mean
            row[f"{condition}_{prefix}_sem"] = sem
    for metric in ("char", "sector"):
        baseline = np.asarray(ablation["baseline"][f"{metric}_acc"], dtype=np.float64)
        shuffled = np.asarray(ablation["shuffle_all"][f"{metric}_acc"], dtype=np.float64)
        delta_mean, delta_sem = _mean_sem((baseline - shuffled).tolist())
        row[f"shuffle_all_{metric}_drop_mean"] = delta_mean
        row[f"shuffle_all_{metric}_drop_sem"] = delta_sem
    return row


def _format_mean_sem(mean: Any, sem: Any) -> str:
    if mean is None or sem is None:
        return "--"
    return f"{float(mean):.3f} ± {float(sem):.3f}"


def write_summary(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Validate ten seeds per model and write CSV, JSON, and Markdown summaries."""
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    test_rows = _load_test_rows((args.original_test_csv, args.feedback_test_csv))
    ablations = {"gawf": _load_ablation_model(args.gawf_ablation_dir, "gawf")}
    for model in FEEDBACK_MODELS:
        ablations[model] = _load_ablation_model(args.feedback_ablation_dir, model)
    rows = [
        _summary_row(model, test_rows[model], ablations.get(model)) for model in MODEL_ORDER
    ]

    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "feedback_control_summary_mean_sem.csv"
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / "feedback_control_summary_mean_sem.json"
    json_path.write_text(json.dumps({"n_seeds": 10, "models": rows}, indent=2) + "\n")

    markdown_path = args.output_dir / "feedback_control_summary_mean_sem.md"
    lines = [
        "| Model | Test digit accuracy | Test sector accuracy | "
        "Shuffle-all digit drop | Shuffle-all sector drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {char} | {sector} | {shuffle_char} | {shuffle_sector} |".format(
                model=row["model"],
                char=_format_mean_sem(row["test_char_mean"], row["test_char_sem"]),
                sector=_format_mean_sem(row["test_sector_mean"], row["test_sector_sem"]),
                shuffle_char=_format_mean_sem(
                    row.get("shuffle_all_char_drop_mean"),
                    row.get("shuffle_all_char_drop_sem"),
                ),
                shuffle_sector=_format_mean_sem(
                    row.get("shuffle_all_sector_drop_mean"),
                    row.get("shuffle_all_sector_drop_sem"),
                ),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, markdown_path


def main() -> None:
    """CLI entry point."""
    outputs = write_summary(parse_args())
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
