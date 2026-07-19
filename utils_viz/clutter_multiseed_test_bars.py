"""Plot Clutter multi-seed test metrics as compact grouped mean-plus-SD bar figures.

Reads the per-checkpoint CSV from ``evaluate_clutter_multiseed_test.py`` and writes the required
accuracy PNG/summary plus optional matching loss outputs. Bar colors exactly match
``fg_switch_offset_acc.py``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils_viz.fg_switch_offset_acc import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from utils_anal.anal_paths import output_dir


def parse_args() -> argparse.Namespace:
    """Parse input table and figure output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", required=True)
    figure_dir = output_dir("G_behaviour", "clutter_multiseed_test_bars", "figs")
    data_dir = output_dir("G_behaviour", "clutter_multiseed_test_bars", "data")
    parser.add_argument("--save_png", default=str(figure_dir / "test_accuracy_mean_sd.png"))
    parser.add_argument("--save_summary_csv", default=str(data_dir / "test_accuracy_mean_sd.csv"))
    parser.add_argument("--save_loss_png", default=None)
    parser.add_argument("--save_loss_summary_csv", default=None)
    parser.add_argument("--title", default="Clutter 40h multi-seed · test performance")
    return parser.parse_args()


def load_rows(path: str) -> dict[str, dict[str, list[float]]]:
    """Load per-seed test accuracies grouped by model."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"char": [], "sector": [], "char_loss": [], "sector_loss": []}
    )
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].lower()
            if model not in MODEL_ORDER:
                continue
            grouped[model]["char"].append(float(row["test_char_acc"]))
            grouped[model]["sector"].append(float(row["test_sector_acc"]))
            if row.get("test_char_loss") and row.get("test_sector_loss"):
                grouped[model]["char_loss"].append(float(row["test_char_loss"]))
                grouped[model]["sector_loss"].append(float(row["test_sector_loss"]))
    if not grouped:
        raise RuntimeError(f"No recognized model rows in {path}")
    return grouped


def stats(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation (zero for one seed)."""

    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def write_summary(
    path: str,
    models: list[str],
    grouped: dict[str, dict[str, list[float]]],
) -> None:
    """Write plotted mean, SD, and completed-seed counts."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "n_seeds",
                "test_char_mean",
                "test_char_sd",
                "test_sector_mean",
                "test_sector_sd",
            ]
        )
        for model in models:
            char_mean, char_sd = stats(grouped[model]["char"])
            sector_mean, sector_sd = stats(grouped[model]["sector"])
            writer.writerow(
                [model, len(grouped[model]["char"]), char_mean, char_sd, sector_mean, sector_sd]
            )


def write_loss_summary(
    path: str,
    models: list[str],
    grouped: dict[str, dict[str, list[float]]],
) -> None:
    """Write test-loss mean, sample SD, and completed-seed counts."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "n_seeds",
                "test_char_loss_mean",
                "test_char_loss_sd",
                "test_sector_loss_mean",
                "test_sector_loss_sd",
            ]
        )
        for model in models:
            char_mean, char_sd = stats(grouped[model]["char_loss"])
            sector_mean, sector_sd = stats(grouped[model]["sector_loss"])
            writer.writerow(
                [
                    model,
                    len(grouped[model]["char_loss"]),
                    char_mean,
                    char_sd,
                    sector_mean,
                    sector_sd,
                ]
            )


def plot_loss(
    output_path: str,
    title: str,
    models: list[str],
    grouped: dict[str, dict[str, list[float]]],
) -> None:
    """Render grouped test character/sector loss bars with per-seed points."""

    group_centers = np.arange(2, dtype=np.float64)
    width = 0.11
    model_offsets = (np.arange(len(models), dtype=np.float64) - (len(models) - 1) / 2.0) * width
    fig, axis = plt.subplots(figsize=(10.2, 5.0))
    rng = np.random.default_rng(0)
    for group_index, metric in enumerate(("char_loss", "sector_loss")):
        for model_index, model in enumerate(models):
            position = group_centers[group_index] + model_offsets[model_index]
            values = np.asarray(grouped[model][metric], dtype=np.float64)
            mean, error = stats(grouped[model][metric])
            axis.bar(
                position,
                mean,
                width,
                yerr=error,
                color=MODEL_COLORS[model],
                edgecolor="none",
                capsize=3,
                error_kw={
                    "elinewidth": 1.1,
                    "capthick": 1.1,
                    "ecolor": "#333333",
                },
            )
            jitter = rng.uniform(-width * 0.26, width * 0.26, size=values.size)
            axis.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=15,
                color="#333333",
                alpha=0.58,
                linewidths=0,
                zorder=3,
            )
    axis.set_title(f"{title} - test at validation-selected checkpoint (mean ± sample SD)")
    axis.set_xticks(group_centers, ["Character", "Sector"])
    axis.set_ylabel("Cross-entropy loss")
    axis.set_ylim(bottom=0.0)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], ec="none") for model in models
    ]
    axis.legend(
        legend_handles,
        [MODEL_LABELS[model] for model in models],
        frameon=False,
        ncol=len(models),
        loc="upper center",
        title="Model",
    )
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    """Render one compact character/sector test-accuracy comparison."""
    args = parse_args()
    if bool(args.save_loss_png) != bool(args.save_loss_summary_csv):
        raise ValueError("--save_loss_png and --save_loss_summary_csv must be provided together")
    grouped = load_rows(args.data_csv)
    models = [model for model in MODEL_ORDER if model in grouped]
    group_centers = np.arange(2, dtype=np.float64)
    width = 0.11
    model_offsets = (np.arange(len(models), dtype=np.float64) - (len(models) - 1) / 2.0) * width
    fig, axis = plt.subplots(figsize=(10.2, 5.0))
    rng = np.random.default_rng(0)
    for group_index, metric in enumerate(("char", "sector")):
        for model_index, model in enumerate(models):
            position = group_centers[group_index] + model_offsets[model_index]
            values = np.asarray(grouped[model][metric], dtype=np.float64)
            mean, error = stats(grouped[model][metric])
            axis.bar(
                position,
                mean,
                width,
                yerr=error,
                color=MODEL_COLORS[model],
                edgecolor="none",
                capsize=3,
                error_kw={
                    "elinewidth": 1.1,
                    "capthick": 1.1,
                    "ecolor": "#333333",
                },
            )
            jitter = rng.uniform(-width * 0.26, width * 0.26, size=values.size)
            axis.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=15,
                color="#333333",
                alpha=0.58,
                linewidths=0,
                zorder=3,
            )
    axis.set_title(f"{args.title} (mean ± sample SD)")
    axis.set_xticks(group_centers, ["Character", "Sector"])
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(70.0, 100.0)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], ec="none") for model in models
    ]
    axis.legend(
        legend_handles,
        [MODEL_LABELS[model] for model in models],
        frameon=False,
        ncol=len(models),
        loc="upper center",
        title="Model",
    )
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout()
    output = Path(args.save_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    write_summary(args.save_summary_csv, models, grouped)
    print(f"Saved figure: {output.resolve()}")
    if args.save_loss_png:
        for model in models:
            if len(grouped[model]["char_loss"]) != len(grouped[model]["char"]):
                raise RuntimeError(f"Missing test loss values for one or more {model} seeds")
        plot_loss(args.save_loss_png, args.title, models, grouped)
        write_loss_summary(args.save_loss_summary_csv, models, grouped)
        print(f"Saved loss figure: {Path(args.save_loss_png).resolve()}")


if __name__ == "__main__":
    main()
