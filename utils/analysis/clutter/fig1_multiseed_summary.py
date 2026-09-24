"""Merged 2x4 summary: the best6 multiseed 2x3 panels plus a GaWF shuffle-ablation column.

Columns 0-2 reuse the exact panels of ``clutter_multiseed_summary`` (test accuracy, validation
loss, target-switch recovery) for the Location (row 0) and Identity (row 1) readouts. Column 3 adds
the GaWF feedback shuffle ablation: three bars (Baseline, Shuffle digit, Shuffle sector) per
readout, with the baseline taken from the same multiseed test CSV as column 0. The two shuffle
panels share one y-axis. All four columns are equal width and 30% narrower than the 2x3 columns.

The original 2x3 summary and the standalone shuffle figure are left untouched; this writes a new
``best6_multiseed_shuffle_2x4`` PNG/PDF pair alongside them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.clutter.clutter_multiseed_summary import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _plot_recovery_axis,
    _plot_test_axis,
    _style_axis,
    load_recovery_curves,
    load_test_metrics,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points  # noqa: E402

# Digit/sector bar colours align with core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
DIGIT_COLOR = "#E76F51"
SECTOR_COLOR = "#264653"
SAVE_DATA_ROOT = PROJECT_ROOT / "results" / "save_data"
FIG_DIR = PROJECT_ROOT / "results" / "train_figs" / "clutter" / "clutter_best6_multiseed_40h_ep150"
TRAIN_DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "data"
    / "clutter"
    / "seed_search"
    / "clutter_best6_multiseed_40h_ep150"
)


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return the seed-level mean and SEM for one plotted bar."""

    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=SAVE_DATA_ROOT / "fig1" / "test_accuracy_summary" / "best_acc_test_mean_std.csv",
    )
    parser.add_argument("--loss_png", type=Path, default=FIG_DIR / "loss_mean_std.png")
    parser.add_argument(
        "--train_data_dir",
        type=Path,
        default=SAVE_DATA_ROOT / "fig1" / "validation_loss_histories",
        help="Per-seed training pickle directory used for a fresh validation-loss redraw.",
    )
    parser.add_argument(
        "--recovery_dir",
        type=Path,
        default=SAVE_DATA_ROOT / "fig1" / "target_switch_recovery",
    )
    parser.add_argument(
        "--ablation_dir", type=Path, default=SAVE_DATA_ROOT / "fig2" / "gawf_shuffle_ablation"
    )
    parser.add_argument(
        "--shuffle_anova_long_csv",
        type=Path,
        default=None,
        help="Use reset-excluded Figure 4 shuffle accuracies for the ablation column.",
    )
    parser.add_argument(
        "--ablation_baseline_source",
        choices=("canonical_test", "ablation"),
        default="canonical_test",
        help="Use the canonical test CSV or ablation metrics for the GaWF ablation baseline.",
    )
    parser.add_argument(
        "--output_png", type=Path, default=FIG_DIR / "best6_multiseed_shuffle_2x4.png"
    )
    parser.add_argument("--output_pdf", type=Path, default=None)
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each 10-seed bar.",
    )
    parser.add_argument(
        "--axis_profile",
        choices=("notanh_mixed", "jointbalanced_512"),
        default="notanh_mixed",
        help=(
            "Axis layout for the current mixed-protocol figure or the all-joint-balanced "
            "512-rollout comparison."
        ),
    )
    return parser.parse_args()


def _conditions_from_ablation(
    ablation_dir: Path,
    conditions_to_load: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Return selected per-seed condition arrays for both readouts."""

    files = sorted(glob.glob(str(ablation_dir / "gawf-seed*" / "ablation_metrics.json")))
    if len(files) != 10:
        raise FileNotFoundError(
            "Expected ten gawf-seed*/ablation_metrics.json under "
            f"{ablation_dir}, found {len(files)}."
        )
    collected: dict[str, dict[str, list[float]]] = {}
    for path in files:
        record = json.load(open(path))
        if record.get("exclude_window_initial_frame") is not True:
            raise RuntimeError(
                f"Ablation input includes or does not document rollout t=0: {path}"
            )
        conditions = record["conditions"]
        for cond in conditions_to_load:
            collected.setdefault(cond, {"char_acc": [], "sector_acc": []})
            for key in ("char_acc", "sector_acc"):
                collected[cond][key].append(float(conditions[cond][key]))
    return {c: {k: np.asarray(v) for k, v in d.items()} for c, d in collected.items()}


def _conditions_from_shuffle_anova(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read the three Figure 4 reset-excluded shuffle accuracy vectors."""

    collected: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["object"] != "hidden_activation":
                continue
            collected.setdefault(row["condition"], {"char_acc": [], "sector_acc": []})
            collected[row["condition"]]["char_acc"].append(float(row["digit_acc"]))
            collected[row["condition"]]["sector_acc"].append(float(row["sector_acc"]))
    if set(collected) != {"baseline", "shuffle_digit", "shuffle_sector"} or any(
        len(metrics["char_acc"]) != 10 for metrics in collected.values()
    ):
        raise ValueError(f"Expected ten Figure 4 rows per condition in {path}")
    return {
        condition: {key: np.asarray(values) for key, values in metrics.items()}
        for condition, metrics in collected.items()
    }


def _plot_shuffle_axis(
    axis: plt.Axes,
    baseline: np.ndarray,
    shuffle_digit: np.ndarray,
    shuffle_sector: np.ndarray,
    color: str,
    show_xticks: bool,
    y_min: float,
    y_max: float,
    y_step: float,
    *,
    show_seed_points: bool = True,
) -> None:
    """Plot one readout's GaWF shuffle-ablation bars (Baseline, Shuffle digit, Shuffle sector)."""

    conds = (
        ("Base", baseline),
        ("Shuf.\ndigit", shuffle_digit),
        ("Shuf.\nsector", shuffle_sector),
    )
    positions = np.arange(len(conds), dtype=np.float64)
    rng = np.random.default_rng(0)
    for index, (_label, values) in enumerate(conds):
        mean, sem = _mean_sem(values)
        axis.bar(
            positions[index],
            mean,
            width=0.72,
            yerr=sem,
            color=color,
            edgecolor="none",
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
        )
        add_seed_points(
            axis,
            np.asarray([positions[index]]),
            values[:, None],
            bar_width=0.72,
            show=show_seed_points,
            rng=rng,
        )
    axis.set_xticks(positions, [label for label, _ in conds])
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    axis.set_ylim(y_min, y_max)
    axis.set_yticks(np.arange(y_min, y_max + 0.001, y_step))
    _style_axis(axis)


def _load_validation_losses(
    train_data_dir: Path, metric: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Aggregate the saved per-seed validation-loss arrays without raster cropping."""

    key = "val_loss_char" if metric == "char" else "val_loss_pos"
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in MODEL_ORDER:
        paths = sorted(train_data_dir.glob(f"{model}-seed*/*.pkl"))
        arrays = []
        for path in paths:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if key in payload:
                arrays.append(np.asarray(payload[key], dtype=np.float64))
        if len(arrays) != 10:
            raise FileNotFoundError(
                f"Expected ten {key} arrays for {model} under {train_data_dir}, "
                f"found {len(arrays)}."
            )
        lengths = {array.size for array in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent {key} lengths for {model}: {sorted(lengths)}")
        stacked = np.stack(arrays, axis=0)
        result[model] = (
            np.mean(stacked, axis=0),
            np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0]),
        )
    return result


def _plot_validation_loss_axis(
    axis: plt.Axes,
    losses: dict[str, tuple[np.ndarray, np.ndarray]],
    metric: str,
    show_xlabel: bool,
    show_xticks: bool,
) -> None:
    """Draw validation loss directly from per-seed arrays, preserving the source semantics."""

    epochs = np.arange(1, 151, dtype=np.float64)
    for model in MODEL_ORDER:
        mean, sem = losses[model]
        axis.plot(epochs, mean, color=MODEL_COLORS[model], linewidth=1.8, zorder=2)
        axis.fill_between(
            epochs,
            mean - sem,
            mean + sem,
            color=MODEL_COLORS[model],
            alpha=0.55,
            linewidth=0,
            zorder=1,
        )
    axis.set_xlim(0.0, 150.0)
    if metric == "char":
        axis.set_ylim(0.3, 1.2)
        axis.set_yticks((0.3, 0.6, 0.9, 1.2))
    else:
        axis.set_ylim(0.15, 0.45)
        axis.set_yticks((0.15, 0.25, 0.35, 0.45))
    axis.set_xticks((0, 50, 100, 150))
    if show_xlabel:
        axis.set_xlabel("Epoch")
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    _style_axis(axis)


def _set_compact_test_axis(
    axis: plt.Axes,
    metrics: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> None:
    """Fit all seed points with three or four evenly spaced ticks."""

    values = np.concatenate([model_values[metric] for model_values in metrics.values()])
    for step in (2.0, 4.0, 6.0):
        lower = step * np.floor(values.min() / step)
        upper = step * np.ceil(values.max() / step)
        ticks = np.arange(lower, upper + 0.001, step)
        if 3 <= ticks.size <= 4:
            break
    else:
        step = 6.0
        lower = step * np.floor(values.min() / step)
        upper = step * np.ceil(values.max() / step)
        ticks = np.arange(lower, upper + 0.001, step)
    axis.set_ylim(lower - 0.5, upper + 0.5)
    axis.set_yticks(ticks)


def main() -> None:
    args = parse_args()
    test_metrics = load_test_metrics(args.test_csv)
    recovery_offsets, recovery_curves = load_recovery_curves(args.recovery_dir)
    validation_losses = {
        "char": _load_validation_losses(args.train_data_dir, "char"),
        "sector": _load_validation_losses(args.train_data_dir, "sector"),
    }
    ablation = (
        _conditions_from_shuffle_anova(args.shuffle_anova_long_csv)
        if args.shuffle_anova_long_csv is not None
        else _conditions_from_ablation(
            args.ablation_dir, ("baseline", "shuffle_digit", "shuffle_sector")
        )
    )
    if "gawf" not in test_metrics:
        raise RuntimeError("Multiseed test metrics must include gawf for the shuffle baseline.")

    with plt.rc_context(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        # Draw at the final ICLR text width so publication fonts remain 7--8 pt without scaling.
        fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.4))
        _plot_test_axis(
            axes[0, 0],
            test_metrics,
            "sector",
            show_xticks=False,
            show_seed_points=args.show_seed_points,
        )
        _plot_test_axis(
            axes[1, 0],
            test_metrics,
            "char",
            show_xticks=True,
            show_seed_points=args.show_seed_points,
        )
        if args.axis_profile == "notanh_mixed":
            axes[0, 0].set_ylim(86.0, 94.5)
            axes[0, 0].set_yticks((86.0, 90.0, 94.0))
            axes[1, 0].set_ylim(68.0, 87.0)
            axes[1, 0].set_yticks((70.0, 78.0, 86.0))
        else:
            _set_compact_test_axis(axes[0, 0], test_metrics, "sector")
            _set_compact_test_axis(axes[1, 0], test_metrics, "char")
        _plot_validation_loss_axis(
            axes[0, 1],
            validation_losses["sector"],
            "sector",
            show_xlabel=False,
            show_xticks=False,
        )
        _plot_validation_loss_axis(
            axes[1, 1], validation_losses["char"], "char", show_xlabel=True, show_xticks=True
        )
        _plot_recovery_axis(
            axes[0, 2],
            recovery_offsets,
            recovery_curves,
            "sector",
            show_xlabel=False,
            show_xticks=False,
        )
        _plot_recovery_axis(
            axes[1, 2],
            recovery_offsets,
            recovery_curves,
            "char",
            show_xlabel=True,
            show_xticks=True,
        )
        axes[0, 2].set_yticks((0, 50, 100))
        axes[1, 2].set_yticks((0, 50, 100))
        use_shuffle_baseline = (
            args.shuffle_anova_long_csv is not None or args.ablation_baseline_source == "ablation"
        )
        ablation_baseline = ablation["baseline"] if use_shuffle_baseline else test_metrics["gawf"]
        baseline_char_key = "char_acc" if use_shuffle_baseline else "char"
        baseline_sector_key = "sector_acc" if use_shuffle_baseline else "sector"
        _plot_shuffle_axis(
            axes[0, 3],
            ablation_baseline[baseline_sector_key],
            ablation["shuffle_digit"]["sector_acc"],
            ablation["shuffle_sector"]["sector_acc"],
            SECTOR_COLOR,
            show_xticks=False,
            y_min=75.0,
            y_max=95.0,
            y_step=4.0,
            show_seed_points=args.show_seed_points,
        )
        _plot_shuffle_axis(
            axes[1, 3],
            ablation_baseline[baseline_char_key],
            ablation["shuffle_digit"]["char_acc"],
            ablation["shuffle_sector"]["char_acc"],
            DIGIT_COLOR,
            show_xticks=True,
            y_min=70.0,
            y_max=90.0,
            y_step=4.0,
            show_seed_points=args.show_seed_points,
        )
        # The requested readout-specific ranges: location on top, identity on the bottom.
        axes[0, 3].set_ylim(40.0, 85.0)
        axes[0, 3].set_yticks((45.0, 65.0, 85.0))
        axes[1, 3].set_ylim(35.0, 75.0)
        axes[1, 3].set_yticks((35.0, 55.0, 75.0))

        # The 30% narrower columns crowd the recovery tick labels; rotate that column's bottom
        # x labels so pre10/switch/post4/post10 no longer overlap. Only this merged figure is
        # affected -- the shared helper and the original 2x3 keep their horizontal labels.
        for tick_label in axes[1, 2].get_xticklabels():
            tick_label.set_rotation(30)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")

        fig.subplots_adjust(
            left=0.085, right=0.995, bottom=0.23, top=0.76, hspace=0.42, wspace=0.38
        )
        column_centers = [
            np.mean(
                [
                    axes[row, column].get_position().x0 + axes[row, column].get_position().width / 2
                    for row in range(2)
                ]
            )
            for column in range(4)
        ]
        title_y = max(axes[0, column].get_position().y1 for column in range(4)) + 0.035
        for x, title in zip(
            column_centers,
            (
                "Test accuracy",
                "Validation loss",
                "Target switch recovery\n(mean ± SEM)",
                "GaWF shuffle\nablation",
            ),
        ):
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=8)
        for label, axis in zip("ABCD", axes[0]):
            position = axis.get_position()
            fig.text(
                position.x0 - 0.012,
                title_y,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        row_centers = [
            np.mean(
                [
                    axes[row, column].get_position().y0
                    + axes[row, column].get_position().height / 2
                    for column in range(4)
                ]
            )
            for row in range(2)
        ]
        for y, label in zip(row_centers, ("Location", "Identity")):
            fig.text(0.025, y, label, rotation=90, ha="center", va="center", fontsize=8)

        models = [model for model in MODEL_ORDER if model in test_metrics]
        handles = [Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.2) for model in models]
        fig.legend(
            handles,
            [MODEL_LABELS[model] for model in models],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(models),
            fontsize=6.5,
            handlelength=1.4,
            columnspacing=1.0,
        )
        output_pdf = (
            args.output_pdf if args.output_pdf is not None else args.output_png.with_suffix(".pdf")
        )
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_png, dpi=300)
        fig.savefig(output_pdf)
        plt.close(fig)
    print(f"Saved {args.output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
