"""Render a 2-by-4 behavioural comparison for the IMDB five-model full-50 grid.

Inputs are the 80 per-configuration metrics JSON files produced by
``experiments/text/imdb_5model_full50_grid.py``.  The validation-selected configuration for
each model drives the test bars and epoch curves; all 16 ``lr x weight_decay`` configurations
per model are retained as descriptive grid distributions.  The script writes PNG/PDF figures,
CSV/JSON summaries, and an NPZ bundle under the canonical ``G_behaviour`` analysis roots.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.analysis.anal_paths import output_dir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
CATEGORY = "G_behaviour"
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "results" / "data" / "text" / "imdb" / "imdb_5model_full50_grid"
)
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
EXPECTED_CONFIGS_PER_MODEL = 16
EXPECTED_EPOCHS = 50
EXPECTED_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the input grid and optional figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    return parser.parse_args()


def _finite_float(record: dict[str, Any], key: str, path: Path) -> float:
    """Return one required finite scalar from a metrics record."""

    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key!r} in {path}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in {path}: {value}")
    return value


def _finite_history(record: dict[str, Any], key: str, path: Path) -> np.ndarray:
    """Return one required 50-epoch float64 history."""

    values = np.asarray(record.get(key), dtype=np.float64)
    if values.shape != (EXPECTED_EPOCHS,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Expected finite {key} shaped ({EXPECTED_EPOCHS},) in {path}, got {values.shape}"
        )
    return values


def load_trials(input_dir: Path) -> list[dict[str, Any]]:
    """Load and validate the exact 80-task single-seed full-50 grid."""

    paths = sorted(input_dir.glob("task_*/*_metrics.json"))
    expected_total = len(MODEL_ORDER) * EXPECTED_CONFIGS_PER_MODEL
    if len(paths) != expected_total:
        raise FileNotFoundError(
            f"Expected {expected_total} metrics JSON files under {input_dir}, found {len(paths)}"
        )

    trials: list[dict[str, Any]] = []
    seen_configs: set[tuple[str, float, float]] = set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        model = str(record.get("model_type"))
        if model not in MODEL_ORDER:
            raise ValueError(f"Unexpected model_type={model!r} in {path}")
        if record.get("dataset") != "imdb":
            raise ValueError(f"Unexpected dataset in {path}: {record.get('dataset')!r}")
        if int(record.get("seed", -1)) != EXPECTED_SEED:
            raise ValueError(f"Expected seed {EXPECTED_SEED} in {path}")
        if int(record.get("actual_epochs", -1)) != EXPECTED_EPOCHS:
            raise ValueError(f"Expected {EXPECTED_EPOCHS} actual epochs in {path}")
        if record.get("stopped_by_patience") is not False:
            raise ValueError(f"Unexpected early stopping in {path}")

        lr = _finite_float(record, "lr", path)
        weight_decay = _finite_float(record, "weight_decay", path)
        config_key = (model, lr, weight_decay)
        if config_key in seen_configs:
            raise ValueError(f"Duplicate model/lr/weight_decay configuration: {config_key}")
        seen_configs.add(config_key)

        trial = {
            "model": model,
            "hidden": int(record["hidden_size"]),
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": int(record["seed"]),
            "core_param_count": int(record["core_param_count"]),
            "total_param_count": int(record["total_param_count"]),
            "val_acc_at_best": _finite_float(record, "val_acc_at_best", path),
            "test_acc_at_best": _finite_float(record, "test_acc_at_best", path),
            "test_loss_at_best": _finite_float(record, "test_loss_at_best", path),
            "best_epoch": int(record["best_epoch_val_acc_1based"]),
            "train_loss": _finite_history(record, "train_loss", path),
            "train_acc": _finite_history(record, "train_acc", path),
            "val_loss": _finite_history(record, "val_loss", path),
            "val_acc": _finite_history(record, "val_acc", path),
            "metrics_path": str(path.resolve()),
        }
        trials.append(trial)

    counts = {model: sum(t["model"] == model for t in trials) for model in MODEL_ORDER}
    if set(counts.values()) != {EXPECTED_CONFIGS_PER_MODEL}:
        raise ValueError(f"Expected {EXPECTED_CONFIGS_PER_MODEL} configs per model, got {counts}")
    return trials


def select_by_validation(trials: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select one configuration per model by highest validation accuracy."""

    selected: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        candidates = [trial for trial in trials if trial["model"] == model]
        if len(candidates) != EXPECTED_CONFIGS_PER_MODEL:
            raise ValueError(f"Expected 16 candidates for {model}, found {len(candidates)}")
        selected[model] = max(candidates, key=lambda row: float(row["val_acc_at_best"]))
    return selected


def _style_axis(axis: plt.Axes) -> None:
    """Apply the clean axis style used by the reference behaviour figure."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)


def _padded_limits(
    values: Iterable[float], minimum_span: float, lower_floor: float
) -> tuple[float, float]:
    """Return compact finite limits with a small display margin."""

    array = np.asarray(list(values), dtype=np.float64)
    low = float(np.min(array))
    high = float(np.max(array))
    span = max(high - low, minimum_span)
    return max(lower_floor, low - 0.16 * span), high + 0.18 * span


def _plot_selected_bars(
    axis: plt.Axes,
    selected: dict[str, dict[str, Any]],
    metric: str,
    *,
    percent: bool,
    show_xticks: bool,
) -> None:
    """Plot validation-selected test performance without pseudo-replicate error bars."""

    positions = np.arange(len(MODEL_ORDER), dtype=np.float64)
    scale = 100.0 if percent else 1.0
    values = np.asarray([selected[m][metric] for m in MODEL_ORDER], dtype=np.float64) * scale
    axis.bar(
        positions,
        values,
        width=0.72,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="none",
    )
    for x, value in zip(positions, values):
        label = f"{value:.1f}" if percent else f"{value:.3f}"
        axis.annotate(
            label,
            (x, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.3,
        )
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    if not show_xticks:
        axis.tick_params(axis="x", labelbottom=False)
    axis.set_xlim(-0.65, len(MODEL_ORDER) - 0.35)
    axis.set_ylim(*_padded_limits(values, 3.0 if percent else 0.08, 0.0))
    _style_axis(axis)


def _plot_selected_histories(
    axis: plt.Axes,
    selected: dict[str, dict[str, Any]],
    metric: str,
    *,
    percent: bool,
    show_xlabel: bool,
) -> None:
    """Plot one selected 50-epoch trajectory per model and mark its best validation epoch."""

    epochs = np.arange(1, EXPECTED_EPOCHS + 1, dtype=np.int64)
    scale = 100.0 if percent else 1.0
    all_values: list[float] = []
    for model in MODEL_ORDER:
        values = np.asarray(selected[model][metric], dtype=np.float64) * scale
        all_values.extend(values.tolist())
        axis.plot(epochs, values, color=MODEL_COLORS[model], linewidth=1.7)
        if metric.startswith("val_"):
            best_index = int(selected[model]["best_epoch"]) - 1
            axis.scatter(
                epochs[best_index],
                values[best_index],
                s=22,
                marker="*",
                color=MODEL_COLORS[model],
                edgecolor="#222222",
                linewidth=0.45,
                zorder=4,
            )
    axis.set_xlim(1, EXPECTED_EPOCHS)
    axis.set_xticks((1, 10, 20, 30, 40, 50))
    if show_xlabel:
        axis.set_xlabel("Epoch")
    else:
        axis.tick_params(axis="x", labelbottom=False)
    floor = 45.0 if percent else 0.0
    axis.set_ylim(*_padded_limits(all_values, 8.0 if percent else 0.2, floor))
    _style_axis(axis)


def _plot_grid_distribution(
    axis: plt.Axes,
    trials: Sequence[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    metric: str,
    *,
    percent: bool,
    show_xticks: bool,
) -> None:
    """Plot all hyperparameter configurations descriptively, with the selected trial starred."""

    positions = np.arange(len(MODEL_ORDER), dtype=np.float64)
    scale = 100.0 if percent else 1.0
    groups = [
        np.asarray([t[metric] for t in trials if t["model"] == model], dtype=np.float64) * scale
        for model in MODEL_ORDER
    ]
    boxes = axis.boxplot(
        groups,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#666666", "linewidth": 0.8},
        capprops={"color": "#666666", "linewidth": 0.8},
    )
    rng = np.random.default_rng(42)
    for index, (model, values, box) in enumerate(zip(MODEL_ORDER, groups, boxes["boxes"])):
        box.set_facecolor(MODEL_COLORS[model])
        box.set_alpha(0.32)
        box.set_edgecolor(MODEL_COLORS[model])
        jitter = rng.uniform(-0.20, 0.20, size=values.size)
        axis.scatter(
            positions[index] + jitter,
            values,
            s=12,
            color=MODEL_COLORS[model],
            alpha=0.62,
            edgecolor="none",
            zorder=3,
        )
        selected_value = float(selected[model][metric]) * scale
        axis.scatter(
            positions[index],
            selected_value,
            s=42,
            marker="*",
            color=MODEL_COLORS[model],
            edgecolor="#111111",
            linewidth=0.55,
            zorder=5,
        )
    axis.set_xticks(
        positions,
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        rotation=35,
        ha="right",
    )
    if not show_xticks:
        axis.tick_params(axis="x", labelbottom=False)
    axis.set_xlim(-0.65, len(MODEL_ORDER) - 0.35)
    axis.set_ylim(*_padded_limits(np.concatenate(groups), 5.0 if percent else 0.1, 0.0))
    _style_axis(axis)


def _plain_trial_row(trial: dict[str, Any], selected: bool) -> dict[str, Any]:
    """Return a CSV/JSON-safe scalar summary for one configuration."""

    return {
        "model": trial["model"],
        "hidden": trial["hidden"],
        "lr": trial["lr"],
        "weight_decay": trial["weight_decay"],
        "seed": trial["seed"],
        "val_acc_at_best": trial["val_acc_at_best"],
        "test_acc_at_best": trial["test_acc_at_best"],
        "test_loss_at_best": trial["test_loss_at_best"],
        "best_epoch": trial["best_epoch"],
        "core_param_count": trial["core_param_count"],
        "total_param_count": trial["total_param_count"],
        "selected_by_validation": selected,
        "metrics_path": trial["metrics_path"],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a non-empty list of homogeneous dictionaries."""

    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_structured_outputs(
    data_dir: Path,
    trials: Sequence[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> None:
    """Write all-trial, selected-trial, and curve data used by the figure."""

    selected_paths = {trial["metrics_path"] for trial in selected.values()}
    trial_rows = [
        _plain_trial_row(trial, trial["metrics_path"] in selected_paths) for trial in trials
    ]
    selected_rows = [_plain_trial_row(selected[model], True) for model in MODEL_ORDER]
    _write_csv(data_dir / "imdb_full50_trials.csv", trial_rows)
    _write_csv(data_dir / "imdb_full50_selected.csv", selected_rows)

    summary = {
        "selection": "highest val_acc_at_best within model",
        "seed": EXPECTED_SEED,
        "epochs": EXPECTED_EPOCHS,
        "configs_per_model": EXPECTED_CONFIGS_PER_MODEL,
        "models": list(MODEL_ORDER),
        "selected": {model: _plain_trial_row(selected[model], True) for model in MODEL_ORDER},
        "inference_note": "Single seed; lr x weight_decay configurations are not replicates.",
    }
    (data_dir / "imdb_full50_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    np.savez_compressed(
        data_dir / "imdb_full50_behavior.npz",
        model_order=np.asarray(MODEL_ORDER),
        epochs=np.arange(1, EXPECTED_EPOCHS + 1, dtype=np.int64),
        selected_test_acc=np.asarray(
            [selected[m]["test_acc_at_best"] for m in MODEL_ORDER], dtype=np.float32
        ),
        selected_test_loss=np.asarray(
            [selected[m]["test_loss_at_best"] for m in MODEL_ORDER], dtype=np.float32
        ),
        selected_val_acc=np.stack(
            [selected[m]["val_acc"] for m in MODEL_ORDER]
        ).astype(np.float32),
        selected_val_loss=np.stack(
            [selected[m]["val_loss"] for m in MODEL_ORDER]
        ).astype(np.float32),
        selected_train_acc=np.stack(
            [selected[m]["train_acc"] for m in MODEL_ORDER]
        ).astype(np.float32),
        selected_train_loss=np.stack(
            [selected[m]["train_loss"] for m in MODEL_ORDER]
        ).astype(np.float32),
        grid_test_acc=np.stack(
            [
                [t["test_acc_at_best"] for t in trials if t["model"] == model]
                for model in MODEL_ORDER
            ]
        ).astype(np.float32),
        grid_test_loss=np.stack(
            [
                [t["test_loss_at_best"] for t in trials if t["model"] == model]
                for model in MODEL_ORDER
            ]
        ).astype(np.float32),
    )


def render_figure(
    output_png: Path,
    output_pdf: Path,
    trials: Sequence[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
) -> None:
    """Render the reference-inspired 2-by-4 comparison to PNG and vector PDF."""

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
        _plot_selected_bars(
            axes[0, 0], selected, "test_acc_at_best", percent=True, show_xticks=False
        )
        _plot_selected_bars(
            axes[1, 0], selected, "test_loss_at_best", percent=False, show_xticks=True
        )
        _plot_selected_histories(
            axes[0, 1], selected, "val_acc", percent=True, show_xlabel=False
        )
        _plot_selected_histories(
            axes[1, 1], selected, "val_loss", percent=False, show_xlabel=True
        )
        _plot_selected_histories(
            axes[0, 2], selected, "train_acc", percent=True, show_xlabel=False
        )
        _plot_selected_histories(
            axes[1, 2], selected, "train_loss", percent=False, show_xlabel=True
        )
        _plot_grid_distribution(
            axes[0, 3],
            trials,
            selected,
            "test_acc_at_best",
            percent=True,
            show_xticks=False,
        )
        _plot_grid_distribution(
            axes[1, 3],
            trials,
            selected,
            "test_loss_at_best",
            percent=False,
            show_xticks=True,
        )

        titles = (
            "A  Validation-selected\ntest performance",
            "B  Validation dynamics\n(selected configuration)",
            "C  Training dynamics\n(selected configuration)",
            "D  Hyperparameter grid\n(16 configurations/model)",
        )
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, pad=13, fontweight="semibold")

        figure.subplots_adjust(
            left=0.065, right=0.99, bottom=0.22, top=0.82, hspace=0.43, wspace=0.32
        )

        row_centers = [
            float(
                np.mean(
                    [axis.get_position().y0 + axis.get_position().height / 2 for axis in row]
                )
            )
            for row in axes
        ]
        figure.text(0.012, row_centers[0], "Accuracy (%)", rotation=90, va="center", fontsize=10)
        figure.text(0.012, row_centers[1], "Loss", rotation=90, va="center", fontsize=10)

        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.5)
            for model in MODEL_ORDER
        ]
        figure.legend(
            handles,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(MODEL_ORDER),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.6,
        )
        figure.text(
            0.5,
            0.018,
            "IMDB, seed 42. A-C: highest validation-accuracy configuration per model; "
            "D: dots are lr x weight-decay configurations, not independent replicates. "
            "Stars mark the validation-selected configuration.",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        plt.close(figure)


def main() -> None:
    """Validate the grid, export structured data, and render the comparison figure."""

    args = parse_args()
    figure_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    data_dir = output_dir(CATEGORY, SCRIPT_NAME, "data")
    output_png = args.output_png or figure_dir / "imdb_5model_full50_behavior_2x4.png"
    output_pdf = args.output_pdf or figure_dir / "imdb_5model_full50_behavior_2x4.pdf"

    trials = load_trials(args.input_dir)
    selected = select_by_validation(trials)
    write_structured_outputs(data_dir, trials, selected)
    render_figure(output_png, output_pdf, trials, selected)
    print(f"Validated {len(trials)} trials ({EXPECTED_CONFIGS_PER_MODEL} per model).")
    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")
    print(f"Saved structured outputs under {data_dir}")


if __name__ == "__main__":
    main()
