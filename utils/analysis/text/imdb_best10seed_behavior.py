"""Render the 2x4 behaviour comparison for the IMDB ten-seed five-model campaign.

The layout follows the retained best-six 2x4 summary: one row per quantity (accuracy, loss) and four
columns with the same roles — validation/test performance, validation dynamics, training dynamics,
and one distribution panel. The IMDB series has no target switch, so the fourth column reports the
generalization gap at the best-validation epoch instead of a shuffle ablation. The existing
hyperparameter-grid figure is left untouched.

Inputs
- ``--series-root``: ``imdb_5model_best10seed`` with ``<model>/seed_NN/`` metric JSON and history
  pickle files, as produced by the two-GPU SJC campaign.

Outputs
- ``results/data/analysis/G_behaviour/<script>/``: NPZ bundle, CSV summary,
  ``key_results.json``.
- ``results/figs/G_behaviour/<script>/``: PNG and PDF.
- curated copy under ``results/save/`` unless ``--no_save_copy`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir  # noqa: E402

SCRIPT_NAME = Path(__file__).stem
CATEGORY = "G_behaviour"
MODEL_ORDER = ("gawf", "rnn", "lstm", "gru", "gawf_logits")
MODEL_LABELS = {
    "gawf": "GaWF",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "gawf_logits": "GaWF-Logits",
}
MODEL_COLORS = {
    "gawf": "#4C78A8",
    "rnn": "#F58518",
    "lstm": "#54A24B",
    "gru": "#E45756",
    "gawf_logits": "#B279A2",
}
EXPECTED_EPOCHS = 50
EXPECTED_SEEDS = 10
DEFAULT_SERIES_ROOT = (
    PROJECT_ROOT / "results" / "save_data" / "imdb" / "imdb_5model_best10seed_series"
)


def parse_args() -> argparse.Namespace:
    """Parse the IMDB series root and the figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, default=DEFAULT_SERIES_ROOT)
    parser.add_argument(
        "--save-pdf",
        type=Path,
        default=PROJECT_ROOT / "results" / "save" / "imdb_best10seed_behavior_2x4.pdf",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=PROJECT_ROOT / "results" / "save" / "imdb_best10seed_behavior_2x4.png",
    )
    parser.add_argument("--no_save_copy", action="store_true")
    return parser.parse_args()


def _load_unit(directory: Path) -> dict[str, Any]:
    """Load one seed's metrics JSON and per-epoch history."""

    metrics_paths = sorted(directory.glob("*_metrics.json"))
    history_paths = sorted(directory.glob("*.pkl"))
    if len(metrics_paths) != 1 or len(history_paths) != 1:
        raise RuntimeError(f"Expected exactly one metrics JSON and one pickle in {directory}")
    metrics = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
    with history_paths[0].open("rb") as stream:
        history = pickle.load(stream)
    for key in ("train_acc", "val_acc", "train_loss", "val_loss"):
        values = np.asarray(history.get(key), dtype=np.float64)
        if values.shape != (EXPECTED_EPOCHS,):
            raise RuntimeError(f"{history_paths[0]} has {key} shaped {values.shape}")
        history[key] = values
    return {"metrics": metrics, "history": history, "path": metrics_paths[0]}


def load_series(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load every completed seed of the five-model IMDB series."""

    if not root.is_dir():
        raise RuntimeError(f"Missing IMDB series root: {root}")
    series: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in MODEL_ORDER}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if model_dir.name not in MODEL_ORDER:
            raise RuntimeError(f"Unrecognized IMDB model directory: {model_dir.name}")
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            seed = int(seed_dir.name.split("_")[-1])
            unit = _load_unit(seed_dir)
            if int(unit["metrics"]["seed"]) != seed:
                raise RuntimeError(f"Seed mismatch in {seed_dir}")
            series[model_dir.name][seed] = unit
    incomplete = {
        model: sorted(seeds) for model, seeds in series.items() if len(seeds) != EXPECTED_SEEDS
    }
    if incomplete:
        raise RuntimeError(f"Expected {EXPECTED_SEEDS} seeds per model, got {incomplete}")
    return series


def _stack(series: dict[str, Any], key: str) -> np.ndarray:
    """Stack one per-epoch history key over the seeds of a model, ordered by seed."""

    seeds = sorted(series)
    return np.stack([series[seed]["history"][key] for seed in seeds])


def _seed_values(series: dict[str, Any], key: str) -> np.ndarray:
    """Collect one scalar metric over the seeds of a model, ordered by seed."""

    seeds = sorted(series)
    return np.asarray([float(series[seed]["metrics"][key]) for seed in seeds], dtype=np.float64)


def _bars_with_seeds(
    axis: plt.Axes,
    values: dict[str, np.ndarray],
    *,
    percent: bool,
    show_xticks: bool,
    ylabel: str | None = None,
) -> None:
    """Draw one bar per model with individual seed dots and a SEM error bar."""

    scale = 100.0 if percent else 1.0
    for index, model in enumerate(MODEL_ORDER):
        samples = np.asarray(values[model], dtype=np.float64) * scale
        mean = samples.mean()
        sem = samples.std(ddof=1) / np.sqrt(samples.size)
        axis.bar(index, mean, width=0.66, color=MODEL_COLORS[model], alpha=0.9, zorder=2)
        axis.errorbar(index, mean, yerr=sem, color="#333333", capsize=2.6, linewidth=0.9, zorder=4)
        jitter = np.linspace(-0.16, 0.16, samples.size)
        axis.scatter(index + jitter, samples, s=9, color="#555555", alpha=0.75, zorder=3)
    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    stacked = np.concatenate([values[model] * scale for model in MODEL_ORDER])
    span = float(stacked.max() - stacked.min())
    axis.set_ylim(float(stacked.min() - 0.45 * span), float(stacked.max() + 0.15 * span))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_curves(
    axis: plt.Axes,
    series: dict[str, Any],
    key: str,
    *,
    percent: bool,
    show_xlabel: bool,
    ylabel: str | None = None,
) -> None:
    """Draw mean plus SEM epoch curves for one history key."""

    epochs = np.arange(1, EXPECTED_EPOCHS + 1)
    for model in MODEL_ORDER:
        stacked = _stack(series[model], key) * (100.0 if percent else 1.0)
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        axis.plot(epochs, mean, color=MODEL_COLORS[model], linewidth=1.6, zorder=3)
        axis.fill_between(
            epochs, mean - sem, mean + sem, color=MODEL_COLORS[model], alpha=0.20, zorder=2
        )
    axis.set_xlim(1, EXPECTED_EPOCHS)
    axis.set_xlabel("Epoch" if show_xlabel else "", fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _generalization_gaps(series: dict[str, Any], row: str) -> np.ndarray:
    """Return the per-seed gap between training and validation at the best-validation epoch."""

    gaps = []
    for seed in sorted(series):
        unit = series[seed]
        best = int(unit["metrics"]["best_epoch_val_acc_1based"]) - 1
        history = unit["history"]
        if row == "accuracy":
            gaps.append(
                (history["train_acc"][best] - history["val_acc"][best]) * 100.0
            )
        else:
            gaps.append(history["val_loss"][best] - history["train_loss"][best])
    return np.asarray(gaps, dtype=np.float64)


def _plot_gaps(
    axis: plt.Axes,
    series: dict[str, Any],
    row: str,
    *,
    show_xticks: bool,
    ylabel: str | None = None,
) -> None:
    """Draw the generalization-gap distribution per model."""

    for index, model in enumerate(MODEL_ORDER):
        samples = _generalization_gaps(series[model], row)
        mean = samples.mean()
        sem = samples.std(ddof=1) / np.sqrt(samples.size)
        axis.bar(index, mean, width=0.66, color=MODEL_COLORS[model], alpha=0.9, zorder=2)
        axis.errorbar(index, mean, yerr=sem, color="#333333", capsize=2.6, linewidth=0.9, zorder=4)
        jitter = np.linspace(-0.16, 0.16, samples.size)
        axis.scatter(index + jitter, samples, s=9, color="#555555", alpha=0.75, zorder=3)
    axis.axhline(0.0, color="#999999", linewidth=0.8, zorder=1)
    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def render_figure(
    series: dict[str, Any], output_png: Path, output_pdf: Path
) -> None:
    """Render the IMDB ten-seed 2x4 behaviour comparison."""

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(11.8, 6.0))
        rows = ("accuracy", "loss")
        for row_index, row in enumerate(rows):
            percent = row == "accuracy"
            _bars_with_seeds(
                axes[row_index][0],
                {
                    model: _seed_values(series[model], "test_acc_at_best")
                    if percent
                    else _seed_values(series[model], "test_loss_at_best")
                    for model in MODEL_ORDER
                },
                percent=percent,
                show_xticks=row_index == 1,
            )
            _plot_curves(
                axes[row_index][1],
                series,
                "val_acc" if percent else "val_loss",
                percent=percent,
                show_xlabel=row_index == 1,
            )
            _plot_curves(
                axes[row_index][2],
                series,
                "train_acc" if percent else "train_loss",
                percent=percent,
                show_xlabel=row_index == 1,
            )
            _plot_gaps(
                axes[row_index][3],
                series,
                row,
                show_xticks=row_index == 1,
            )
        titles = (
            "A  Validation-selected test performance",
            "B  Validation dynamics",
            "C  Training dynamics",
            "D  Generalization gap\n(at best-validation epoch)",
        )
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, pad=12, fontweight="semibold")

        figure.subplots_adjust(
            left=0.062, right=0.99, bottom=0.20, top=0.83, hspace=0.36, wspace=0.30
        )
        row_centers = [
            float(np.mean([axis.get_position().y0 + axis.get_position().height / 2 for axis in row]))
            for row in axes
        ]
        for label, center in zip(("Accuracy (%)", "Loss"), row_centers):
            figure.text(0.012, center, label, rotation=90, va="center", fontsize=10)
        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.5) for model in MODEL_ORDER
        ]
        figure.legend(
            handles,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(MODEL_ORDER),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.6,
        )
        figure.text(
            0.5,
            0.02,
            "IMDB ten-seed series, 50 epochs, seeds 1-10, batch 64; A: test split evaluated once from"
            " each unit's best-validation checkpoint;\nB/C: mean $\\pm$ SEM of the validation and"
            " training histories; D: training minus validation at the best-validation epoch."
        ,
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        plt.close(figure)


def write_structured_outputs(series: dict[str, Any], data_dir: Path) -> dict[str, Path]:
    """Write the NPZ bundle, CSV summary, and key numerical results."""

    data_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "model_order": np.asarray(MODEL_ORDER),
        "epochs": np.arange(1, EXPECTED_EPOCHS + 1, dtype=np.int64),
    }
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        seeds = sorted(series[model])
        arrays[f"{model}__seeds"] = np.asarray(seeds, dtype=np.int64)
        arrays[f"{model}__test_acc"] = _seed_values(series[model], "test_acc_at_best").astype(
            np.float32
        )
        arrays[f"{model}__test_loss"] = _seed_values(series[model], "test_loss_at_best").astype(
            np.float32
        )
        for key in ("train_acc", "val_acc", "train_loss", "val_loss"):
            arrays[f"{model}__{key}"] = _stack(series[model], key).astype(np.float32)
        arrays[f"{model}__accuracy_gap"] = _generalization_gaps(series[model], "accuracy").astype(
            np.float32
        )
        arrays[f"{model}__loss_gap"] = _generalization_gaps(series[model], "loss").astype(
            np.float32
        )
        row: dict[str, Any] = {
            "model": model,
            "n_seeds": len(seeds),
            "core_param_count": float(series[model][seeds[0]]["metrics"]["core_param_count"]),
            "total_param_count": float(series[model][seeds[0]]["metrics"]["total_param_count"]),
        }
        for name, values in (
            ("test_acc", _seed_values(series[model], "test_acc_at_best")),
            ("test_loss", _seed_values(series[model], "test_loss_at_best")),
            ("accuracy_gap", _generalization_gaps(series[model], "accuracy")),
            ("loss_gap", _generalization_gaps(series[model], "loss")),
        ):
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_sem"] = float(values.std(ddof=1) / np.sqrt(values.size))
            row[f"{name}_min"] = float(values.min())
            row[f"{name}_max"] = float(values.max())
        rows.append(row)
    npz_path = data_dir / "imdb_best10seed_behavior_2x4.npz"
    np.savez_compressed(npz_path, **arrays)
    csv_path = data_dir / "imdb_best10seed_behavior_2x4.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    key_results = {
        f"{row['model']}.{key}": float(value)
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    }
    key_path = data_dir / "key_results.json"
    key_path.write_text(json.dumps(key_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"npz": npz_path, "csv": csv_path, "key": key_path}


def main() -> None:
    """Load the completed IMDB series, aggregate, and render the 2x4 comparison."""

    args = parse_args()
    series = load_series(args.series_root)
    data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
    fig_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    written = write_structured_outputs(series, data_dir)
    png_path = fig_dir / "imdb_best10seed_behavior_2x4.png"
    pdf_path = fig_dir / "imdb_best10seed_behavior_2x4.pdf"
    render_figure(series, png_path, pdf_path)
    if not args.no_save_copy:
        render_figure(series, args.save_png, args.save_pdf)
    for model in MODEL_ORDER:
        values = _seed_values(series[model], "test_acc_at_best")
        losses = _seed_values(series[model], "test_loss_at_best")
        print(
            f"{MODEL_LABELS[model]:>12} n={values.size:>2}"
            f" test acc {values.mean() * 100:.2f} ± {values.std(ddof=1) / np.sqrt(values.size) * 100:.2f}"
            f" | test loss {losses.mean():.4f} ± {losses.std(ddof=1) / np.sqrt(losses.size):.4f}"
        )
    for name, path in written.items():
        print(f"{name}: {path}")
    print(f"figure: {png_path}")
    print(f"figure: {pdf_path}")
    if not args.no_save_copy:
        print(f"curated: {args.save_png}")
        print(f"curated: {args.save_pdf}")


if __name__ == "__main__":
    main()
