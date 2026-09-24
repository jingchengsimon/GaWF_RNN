"""Family-grouped 14-model behaviour figure for the CM-MNIST nonlinearity-placement ablation.

Inputs: per-seed reset-excluded test accuracy (best6 baseline CSV plus the ablation collect JSONs),
per-seed training histories, and per-seed switch-recovery exports with their provenance metadata.

Computation: seed-level mean and SEM per model and readout, with models grouped into families
(GaWF, RNN, LSTM, GRU, Mamba, S5); every ablation variant is compared against its wrapped best6
baseline inside the same family.

Outputs: a CSV table (float32) under the G_behaviour analysis data root and a 2x3 PNG/PDF whose
panels are reset-excluded test accuracy, validation loss, and switch recovery for the sector (row 0)
and char (row 1) readouts. The visual conventions reuse ``Fig1_best6_multiseed_shuffle_2x4_seq512``:
the same MODEL_COLORS/MODEL_MARKERS palette, bar/error-bar/seed-point styling, curve widths, axis
styling (top/right spines hidden), and subplot spacing helpers from ``clutter_multiseed_summary`` and
``fig1_multiseed_summary``. One colour is used per family; variants within a family are separated by
hatch (bars) and linestyle (curves).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir  # noqa: E402
from utils.analysis.clutter.clutter_multiseed_summary import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    _mean_sem,
    _style_axis,
)
from utils.analysis.clutter.fg_switch_offset_acc import (  # noqa: E402
    MODEL_MARKERS,
    select_key_recovery_ticks,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points  # noqa: E402

CATEGORY = "G_behaviour"
SCRIPT_NAME = "ablation_nowrap_behavior"
FIGURE_STEM = "ablation_14model_behavior_2x3"

# Family -> variants, in campaign order. The family key owns the colour.
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gawf", ("gawf", "gawf_nowrap", "gawf_notanh")),
    ("rnn", ("rnn", "rnn_nowrap", "rnn_notanh")),
    ("lstm", ("lstm", "lstm_nowrap")),
    ("gru", ("gru", "gru_nowrap")),
    ("mamba", ("mamba", "mamba_nowrap")),
    ("s5", ("s5", "s5_nowrap")),
)
VARIANT_SUFFIX = {"": "wrap", "_nowrap": "no wrap", "_notanh": "no tanh"}
VARIANT_HATCH = {"": None, "_nowrap": "/", "_notanh": "///"}
VARIANT_LINESTYLE = {"": "-", "_nowrap": "--", "_notanh": ":"}
BAR_WIDTH = 0.72
FAMILY_GAP = 0.55


def all_variants() -> list[str]:
    """Return every model key of the 14-model comparison in family order."""

    return [variant for _, variants in FAMILIES for variant in variants]


def variant_suffix(model: str) -> str:
    """Return the variant suffix ('', '_nowrap', '_notanh') of one model key."""

    for suffix in ("_nowrap", "_notanh"):
        if model.endswith(suffix):
            return suffix
    return ""


def family_of(model: str) -> str:
    """Return the family key that owns one model's colour."""

    for family, variants in FAMILIES:
        if model in variants:
            return family
    raise KeyError(f"Unknown model key {model!r}")


def model_label(model: str) -> str:
    """Return the rotated tick label for one bar: family name plus variant."""

    return f"{MODEL_LABELS[family_of(model)]} {VARIANT_SUFFIX[variant_suffix(model)]}"


def parse_args() -> argparse.Namespace:
    """Parse the input locations of the 14-model comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation_collect_dir", type=Path, default=Path("tmp/abl_mirror/collect"))
    parser.add_argument("--ablation_runs_dir", type=Path, default=Path("tmp/abl_mirror/runs"))
    parser.add_argument(
        "--ablation_recovery_dir", type=Path, default=Path("tmp/abl_mirror/recovery")
    )
    parser.add_argument(
        "--baseline_test_csv",
        type=Path,
        default=Path(
            "results/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final"
            "/reset_excluded_test_accuracy_10seed.csv"
        ),
    )
    parser.add_argument(
        "--baseline_history_dir",
        type=Path,
        default=Path("results/save_data/fig1/validation_loss_histories"),
    )
    parser.add_argument(
        "--baseline_recovery_dir",
        type=Path,
        default=Path(
            "results/data/analysis/fig1_target_switch_recovery_resetexcluded_6model_10seed_v4"
        ),
    )
    return parser.parse_args()


def load_test_metrics(
    baseline_csv: Path, collect_dir: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Load reset-excluded test accuracy per model and readout (5 or 10 seeds)."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"char": [], "sector": []})
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].lower()
            if model not in all_variants():
                continue
            grouped[model]["char"].append(float(row["char_acc"]))
            grouped[model]["sector"].append(float(row["sector_acc"]))
    for path in sorted(glob.glob(str(collect_dir / "*" / "reset_excluded_test_accuracy.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = str(payload["model"])
        if model not in all_variants():
            continue
        grouped[model]["char"].append(float(payload["char_acc"]))
        grouped[model]["sector"].append(float(payload["sector_acc"]))
    metrics = {
        model: {metric: np.asarray(values, dtype=np.float64) for metric, values in per.items()}
        for model, per in grouped.items()
        if per["char"]
    }
    if not metrics:
        raise RuntimeError("No reset-excluded test accuracy inputs were found.")
    return metrics


def _history_paths(root: Path) -> dict[str, list[Path]]:
    """Map ``model-seedNN`` directory names to history pickles for either directory layout."""

    out: dict[str, list[Path]] = defaultdict(list)
    for pattern in (root / "*" / "*.pkl", root / "*" / "*" / "*.pkl"):
        for path in sorted(glob.glob(str(pattern))):
            out[Path(path).parent.name].append(Path(path))
    return out


def load_losses(
    baseline_history_dir: Path, ablation_runs_dir: Path
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Aggregate per-seed validation-loss curves into mean and SEM per model and readout."""

    units: dict[str, list[Path]] = defaultdict(list)
    for root in (baseline_history_dir, ablation_runs_dir):
        for unit, paths in _history_paths(root).items():
            units[unit].extend(paths)
    stacked: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for unit, paths in units.items():
        if "-seed" not in unit:
            continue
        model = unit.rpartition("-seed")[0]
        if model not in all_variants():
            continue
        for path in paths:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            for metric, key in (("char", "val_loss_char"), ("sector", "val_loss_pos")):
                if key in payload:
                    stacked[model][metric].append(np.asarray(payload[key], dtype=np.float64))
    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for model, per_metric in stacked.items():
        if not per_metric["char"]:
            continue
        result[model] = {}
        for metric, arrays in per_metric.items():
            lengths = {array.size for array in arrays}
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent {metric} loss lengths for {model}: {sorted(lengths)}"
                )
            block = np.stack(arrays, axis=0)
            sem = (
                block.std(axis=0, ddof=1) / np.sqrt(block.shape[0])
                if block.shape[0] > 1
                else np.zeros(block.shape[1])
            )
            result[model][metric] = (block.mean(axis=0), sem)
    if not result:
        raise RuntimeError("No validation-loss histories were found.")
    return result


def load_recovery(
    baseline_recovery_dir: Path, ablation_recovery_dir: Path
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Load switch-recovery curves, enforcing the reference reset-excluded provenance flag."""

    offsets: np.ndarray | None = None
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for root in (baseline_recovery_dir, ablation_recovery_dir):
        for filename in sorted(glob.glob(str(root / "*" / "fg_switch_offset_acc_*.npz"))):
            model = Path(filename).parent.name.rpartition("-seed")[0]
            if model not in all_variants():
                continue
            meta_path = Path(filename).with_name(
                Path(filename).name.replace("_acc_", "_meta_", 1).replace(".npz", ".json")
            )
            if not meta_path.is_file():
                raise RuntimeError(f"Missing recovery provenance metadata for {filename}")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("exclude_window_initial_frame") is not True:
                raise RuntimeError(f"Recovery input does not exclude rollout t=0: {meta_path}")
            with np.load(filename) as payload:
                current = payload["offset_order"].astype(np.int64)
                if offsets is None:
                    offsets = current
                elif not np.array_equal(offsets, current):
                    raise RuntimeError(f"Mismatched recovery offsets in {filename}")
                grouped[model]["char"].append(payload["char_acc"].astype(np.float64))
                grouped[model]["sector"].append(payload["sector_acc"].astype(np.float64))
    if offsets is None or not grouped:
        raise RuntimeError("No switch-recovery exports were found.")
    curves = {
        model: {metric: np.stack(values) for metric, values in per.items()}
        for model, per in grouped.items()
    }
    return offsets, curves


def family_positions(models: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return bar positions, rotated tick labels, and the family colour of every bar."""

    positions: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    cursor = 0.0
    for family, variants in FAMILIES:
        present = [variant for variant in variants if variant in models]
        for variant in present:
            positions.append(cursor)
            labels.append(model_label(variant))
            colors.append(MODEL_COLORS[family])
            cursor += 1.0
        if present:
            cursor += FAMILY_GAP
    return np.asarray(positions, dtype=np.float64), labels, colors


def plot_test_panel(
    axis: plt.Axes,
    metrics: dict[str, dict[str, np.ndarray]],
    metric: str,
    show_xticks: bool,
) -> None:
    """Plot family-grouped reset-excluded test-accuracy bars for one readout."""

    models = [model for model in all_variants() if model in metrics]
    positions, labels, colors = family_positions(models)
    rng = np.random.default_rng(0)
    for index, model in enumerate(models):
        values = metrics[model][metric]
        mean, sem = _mean_sem(values)
        axis.bar(
            positions[index],
            mean,
            width=BAR_WIDTH,
            yerr=sem,
            color=colors[index],
            edgecolor="none",
            hatch=VARIANT_HATCH[variant_suffix(model)],
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        add_seed_points(
            axis,
            np.asarray([positions[index]]),
            values[:, None],
            bar_width=BAR_WIDTH,
            show=True,
            rng=rng,
        )
    axis.set_xticks(positions, labels, rotation=40, ha="right")
    axis.tick_params(axis="x", labelsize=7)
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    stacked = np.concatenate([metrics[model][metric] for model in models])
    step = 8.0
    low = float(np.floor((stacked.min() - 2.0) / step) * step)
    high = float(np.ceil((stacked.max() + 2.0) / step) * step)
    axis.set_ylim(low, high)
    axis.set_yticks(np.arange(low, high + 0.5 * step, step))
    _style_axis(axis)


def plot_loss_panel(
    axis: plt.Axes,
    losses: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    metric: str,
) -> None:
    """Plot validation-loss curves with the reference line width and band styling."""

    epochs = np.arange(1, 151, dtype=np.float64)
    for model in all_variants():
        if model not in losses or metric not in losses[model]:
            continue
        mean, sem = losses[model][metric]
        color = MODEL_COLORS[family_of(model)]
        axis.plot(
            epochs,
            mean,
            color=color,
            linewidth=1.8,
            linestyle=VARIANT_LINESTYLE[variant_suffix(model)],
            zorder=2,
        )
        axis.fill_between(
            epochs, mean - sem, mean + sem, color=color, alpha=0.55, linewidth=0, zorder=1
        )
    axis.set_xlim(0.0, 150.0)
    axis.set_xticks((0, 50, 100, 150))
    if metric == "char":
        axis.set_ylim(0.3, 2.2)
        axis.set_yticks((0.3, 1.0, 1.7))
    else:
        axis.set_ylim(0.15, 1.4)
        axis.set_yticks((0.2, 0.6, 1.0, 1.4))
    _style_axis(axis)


def plot_recovery_panel(
    axis: plt.Axes,
    offsets: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> None:
    """Plot switch-recovery curves with the reference marker and band conventions."""

    selected_indices, selected_labels = select_key_recovery_ticks(offsets)
    x = np.arange(offsets.size, dtype=np.int64)
    for model in all_variants():
        if model not in curves:
            continue
        values = curves[model][metric]
        mean = values.mean(axis=0)
        sem = (
            values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        color = MODEL_COLORS[family_of(model)]
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=1.8,
            linestyle=VARIANT_LINESTYLE[variant_suffix(model)],
            marker=MODEL_MARKERS[family_of(model)],
            markevery=selected_indices.tolist(),
            markersize=3.8,
        )
        if np.any(sem):
            axis.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.55, linewidth=0)
    axis.set_xticks(selected_indices, selected_labels)
    tick_labels = axis.get_xticklabels()
    tick_labels[1].set_ha("right")
    tick_labels[2].set_ha("left")
    for tick_label in tick_labels:
        tick_label.set_rotation(30)
        tick_label.set_ha("right")
        tick_label.set_rotation_mode("anchor")
    axis.set_ylim(0.0, 100.0)
    axis.set_yticks((0, 20, 40, 60, 80, 100))
    _style_axis(axis)


def main() -> None:
    """Build the 14-model family-grouped 2x3 figure and its structured table."""

    args = parse_args()
    metrics = load_test_metrics(args.baseline_test_csv, args.ablation_collect_dir)
    losses = load_losses(args.baseline_history_dir, args.ablation_runs_dir)
    offsets, curves = load_recovery(args.baseline_recovery_dir, args.ablation_recovery_dir)

    data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
    fig_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = [["model", "family", "variant", "seed", "char_acc", "sector_acc"]]
    for model in all_variants():
        if model not in metrics:
            continue
        for seed, (char, sector) in enumerate(
            zip(metrics[model]["char"], metrics[model]["sector"]), start=1
        ):
            rows.append(
                [model, family_of(model), VARIANT_SUFFIX[variant_suffix(model)], seed, char, sector]
            )
    with (data_dir / "ablation_14model_test_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerows(rows)

    figure, axes = plt.subplots(
        2, 3, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.7, 1.0, 1.0]}
    )
    for row, metric in enumerate(("sector", "char")):
        plot_test_panel(axes[row, 0], metrics, metric, show_xticks=row == 1)
        plot_loss_panel(axes[row, 1], losses, metric)
        plot_recovery_panel(axes[row, 2], offsets, curves, metric)
        if row == 0:
            for column in range(3):
                axes[row, column].tick_params(axis="x", labelbottom=False)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 2].set_xlabel("Frame relative to target switch")

    # Reserve the top band for the shared legends before measuring panel positions, so the column
    # titles and row labels are placed against the final geometry (the reference figure's layout).
    figure.subplots_adjust(left=0.058, right=0.995, bottom=0.22, top=0.70, hspace=0.34, wspace=0.26)

    # Shared top legend: family colours, plus the hatch convention for the ablation variants.
    colour_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[family], linewidth=2.5, label=MODEL_LABELS[family])
        for family, _ in FAMILIES
    ]
    hatch_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#BBBBBB",
            edgecolor="black",
            linewidth=0.5,
            hatch=VARIANT_HATCH[suffix],
            label=VARIANT_SUFFIX[suffix],
        )
        for suffix in ("", "_nowrap", "_notanh")
    ]
    legend_colours = figure.legend(
        handles=colour_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(FAMILIES),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )
    figure.add_artist(legend_colours)
    figure.legend(
        handles=hatch_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.6,
    )
    for column, title in enumerate(
        ("Test accuracy", "Validation loss", "Target switch recovery\n(mean ± SEM)")
    ):
        position = axes[0, column].get_position()
        figure.text(
            position.x0 + position.width / 2,
            position.y1 + 0.02,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for row, row_label in enumerate(("Location", "Identity")):
        position = axes[row, 0].get_position()
        figure.text(
            0.012,
            position.y0 + position.height / 2,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
        )
    png_path = fig_dir / f"{FIGURE_STEM}.png"
    pdf_path = fig_dir / f"{FIGURE_STEM}.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)

    plotted = [model for model in all_variants() if model in metrics]
    counts = ", ".join(f"{model}:{metrics[model]['char'].size}" for model in plotted)
    print(f"models plotted ({len(plotted)}): {plotted}")
    print(f"seed counts: {counts}")
    print(f"recovery models: {[m for m in plotted if m in curves]}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
