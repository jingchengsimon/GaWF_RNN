"""Plot per-model multi-seed best accuracy as mean +/- sample SD bars.

The validation mode reads one training-history pickle per ``model-seed`` directory and
computes the maximum validation accuracy for each seed.  The test mode reads the
per-checkpoint test table produced by ``evaluate_clutter_multiseed_test.py``; those
test values are already from the checkpoint selected using validation and therefore
must not be re-selected by maximizing test accuracy.

Examples
--------
Validation diagnostic (available immediately after training)::

    python -m utils_viz.clutter_multiseed_best_acc_bars \
      --source validation --input_root /path/to/clutter-completed-metrics \
      --save_png results/train_figs/clutter/clutter_best6_multiseed_40h_ep150/best_acc_validation_mean_std.png \
      --save_summary_csv results/anal_data/clutter_best6_multiseed_40h_ep150/best_acc_validation_mean_std.csv

Final test result after the multi-seed test evaluator finishes::

    python -m utils_viz.clutter_multiseed_best_acc_bars \
      --source test --test_csv results/anal_data/clutter_multiseed_test/per_seed_test_accuracy.csv \
      --save_png results/anal_figs/clutter_multiseed_test/best_acc_test_mean_std.png \
      --save_summary_csv results/anal_data/clutter_multiseed_test/best_acc_test_mean_std.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils_viz.fg_switch_offset_acc import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("validation", "test"), required=True)
    parser.add_argument("--input_root", help="Directory containing model-seed training pickle directories")
    parser.add_argument("--test_csv", help="Per-seed test CSV from the checkpoint evaluator")
    parser.add_argument("--save_png", required=True)
    parser.add_argument("--save_summary_csv", required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def _model_seed(name: str) -> tuple[str, int] | None:
    if "-seed" not in name:
        return None
    model, seed_text = name.rsplit("-seed", 1)
    if model not in MODEL_ORDER:
        return None
    try:
        return model, int(seed_text)
    except ValueError:
        return None


def load_validation(root: str) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Return per-seed maxima of validation character and sector accuracy."""

    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {"char": [], "sector": []}
    )
    root_path = Path(root).expanduser().resolve()
    for unit_dir in sorted(root_path.glob("*-seed*")):
        parsed = _model_seed(unit_dir.name)
        if parsed is None:
            continue
        pkl_files = sorted(unit_dir.glob("*.pkl"))
        if not pkl_files:
            continue
        with pkl_files[0].open("rb") as handle:
            history = pickle.load(handle)
        model, seed = parsed
        char = np.asarray(history["val_acc_char"], dtype=np.float64)
        sector = np.asarray(history["val_acc_pos"], dtype=np.float64)
        grouped[model]["char"].append((seed, float(np.nanmax(char))))
        grouped[model]["sector"].append((seed, float(np.nanmax(sector))))
    if not grouped:
        raise RuntimeError(f"No completed training histories found under {root_path}")
    return grouped


def load_test(path: str) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Return per-seed test values from already validation-selected checkpoints."""

    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {"char": [], "sector": []}
    )
    with Path(path).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].lower()
            if model not in MODEL_ORDER:
                continue
            seed = int(row["seed"])
            grouped[model]["char"].append((seed, float(row["test_char_acc"])))
            grouped[model]["sector"].append((seed, float(row["test_sector_acc"])))
    if not grouped:
        raise RuntimeError(f"No recognized test rows found in {path}")
    return grouped


def _mean_sd(values: list[tuple[int, float]]) -> tuple[float, float]:
    array = np.asarray([value for _, value in values], dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def write_summary(path: str, source: str, models: list[str], grouped: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "model", "seed", "char_acc", "sector_acc"])
        for model in models:
            chars = dict(grouped[model]["char"])
            sectors = dict(grouped[model]["sector"])
            for seed in sorted(chars):
                writer.writerow([source, model, seed, chars[seed], sectors[seed]])


def _style_axis(axis: plt.Axes) -> None:
    """Use the project figure style without top/bottom frame borders."""
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)


def plot(source: str, grouped: dict[str, dict[str, list[tuple[int, float]]]], save_png: str, title: str | None) -> None:
    models = [model for model in MODEL_ORDER if model in grouped]
    if title is None:
        title = "Clutter best-6 multi-seed accuracy"
    if source == "test":
        group_centers = np.arange(2, dtype=np.float64)
        width = 0.11
        model_offsets = (
            np.arange(len(models), dtype=np.float64) - (len(models) - 1) / 2.0
        ) * width
        fig, axis = plt.subplots(figsize=(10.2, 5.0))
        rng = np.random.default_rng(0)
        for group_index, metric in enumerate(("char", "sector")):
            for model_index, model in enumerate(models):
                position = group_centers[group_index] + model_offsets[model_index]
                values = np.asarray(
                    [value for _, value in grouped[model][metric]], dtype=np.float64
                )
                mean, error = _mean_sd(grouped[model][metric])
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
        axis.set_title(f"{title} · test at validation-selected checkpoint (mean ± sample SD)")
        axis.set_xticks(group_centers, ["Character", "Sector"])
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(70.0, 100.0)
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[model], ec="none")
            for model in models
        ]
        axis.legend(
            legend_handles,
            [MODEL_LABELS[model] for model in models],
            frameon=False,
            ncol=len(models),
            loc="upper center",
            title="Model",
        )
        _style_axis(axis)
        fig.tight_layout()
    else:
        x = np.arange(len(models), dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
        rng = np.random.default_rng(0)
        for axis, (metric, label) in zip(axes, (("char", "Character accuracy"), ("sector", "Sector accuracy"))):
            means = np.asarray([_mean_sd(grouped[model][metric])[0] for model in models])
            errors = np.asarray([_mean_sd(grouped[model][metric])[1] for model in models])
            bars = axis.bar(x, means, yerr=errors, color=[MODEL_COLORS[model] for model in models], edgecolor="none", capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#333333"})
            for index, (model, bar) in enumerate(zip(models, bars)):
                values = np.asarray([value for _, value in grouped[model][metric]], dtype=np.float64)
                jitter = rng.uniform(-0.16, 0.16, size=values.size)
                axis.scatter(np.full(values.size, x[index]) + jitter, values, s=16, color="#333333", alpha=0.55, linewidths=0, zorder=3)
                mean, sd = _mean_sd(grouped[model][metric])
                axis.text(bar.get_x() + bar.get_width() / 2, min(mean + sd + 1.8, 102.0), f"{mean:.1f} ± {sd:.1f}\nn={len(values)}", ha="center", va="bottom", fontsize=8)
            axis.set_title(label)
            axis.set_xticks(x, [MODEL_LABELS[model] for model in models])
            axis.set_xlabel("Model")
            axis.set_ylim(0.0, 105.0)
            axis.set_ylabel("Accuracy (%)")
            _style_axis(axis)
        fig.suptitle(f"{title} · validation best per seed (mean ± sample SD)")
        fig.tight_layout()
    output = Path(save_png).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.source == "validation":
        if not args.input_root:
            raise SystemExit("--input_root is required for --source validation")
        grouped = load_validation(args.input_root)
    else:
        if not args.test_csv:
            raise SystemExit("--test_csv is required for --source test")
        grouped = load_test(args.test_csv)
    models = [model for model in MODEL_ORDER if model in grouped]
    write_summary(args.save_summary_csv, args.source, models, grouped)
    plot(args.source, grouped, args.save_png, args.title)
    print(f"Saved figure: {Path(args.save_png).expanduser().resolve()}")
    print(f"Saved per-seed summary: {Path(args.save_summary_csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
