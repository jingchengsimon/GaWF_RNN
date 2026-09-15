"""Create a compact 2-by-3 summary of the Clutter best-6 multi-seed results.

Inputs are the validation-selected per-seed test CSV, the rendered validation-loss figure, and
foreground target-switch accuracy exports.  The loss panel uses the existing validated loss
render because the historical per-seed training pickle files are not retained locally.  Outputs
are a development PNG in the corresponding ``results/figs`` folder and an official PDF in
the configured publication-figure directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from utils.training.paths import publication_figures_dir
from utils.analysis.clutter.fg_switch_offset_acc import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    _kind_and_tag,
    _model_key,
    select_key_recovery_ticks,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "results" / "train_figs" / "clutter" / "clutter_best6_multiseed_40h_ep150"
)
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "anal_data"
    / "G_behaviour"
    / "clutter_multiseed_best_acc_bars"
    / "clutter_best6_multiseed_40h_ep150"
)
DEFAULT_RECOVERY_DIR = (
    PROJECT_ROOT
    / "results"
    / "anal_data"
    / "G_behaviour"
    / "export_fg_switch_offset_acc"
    / "fg_switch_offset_acc_clutter_best_jointswitch_balanced_10digit_unique_sector_covered"
    / "fg10"
)


def parse_args() -> argparse.Namespace:
    """Parse source and destination paths for the combined result figure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "best_acc_test_mean_std.csv",
    )
    parser.add_argument(
        "--loss_png",
        type=Path,
        default=DEFAULT_RESULT_DIR / "loss_mean_std.png",
    )
    parser.add_argument("--recovery_dir", type=Path, default=DEFAULT_RECOVERY_DIR)
    parser.add_argument(
        "--output_png",
        type=Path,
        default=DEFAULT_RESULT_DIR / "best6_multiseed_summary_2x3.png",
    )
    parser.add_argument("--output_pdf", type=Path, default=None)
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each 10-seed bar.",
    )
    parser.add_argument(
        "--publication_fig_dir",
        type=Path,
        default=None,
        help=(
            "Opt-in extra PDF destination. When omitted, the PDF is written only next to its "
            "PNG in the local results tree (no automatic 6-Writing sync)."
        ),
    )
    return parser.parse_args()


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return mean and cross-seed SEM, treating a singleton as zero spread."""

    sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return float(np.mean(values)), sem


def load_test_metrics(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load validation-selected test character and sector accuracy by model and seed."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"char": [], "sector": []})
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].lower()
            if model not in MODEL_ORDER:
                continue
            grouped[model]["char"].append(float(row["char_acc"]))
            grouped[model]["sector"].append(float(row["sector_acc"]))
    if not grouped:
        raise RuntimeError(f"No recognized multi-seed test metrics found in {path}")
    counts = {
        f"{model}/{metric}": len(values)
        for model, metrics in grouped.items()
        for metric, values in metrics.items()
    }
    if any(count != 10 for count in counts.values()):
        raise RuntimeError(f"Expected exactly ten test seeds per model/readout, got {counts}.")
    return {
        model: {metric: np.asarray(values, dtype=np.float64) for metric, values in metrics.items()}
        for model, metrics in grouped.items()
    }


def load_recovery_curves(path: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Load reset-excluded, aligned ten-seed foreground-switch curves grouped by model."""

    paths = sorted(glob.glob(str(path / "**" / "fg_switch_offset_acc_*.npz"), recursive=True))
    if not paths:
        raise RuntimeError(f"No foreground switch exports found in {path}")
    offsets: np.ndarray | None = None
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"char": [], "sector": []}
    )
    for filename in paths:
        _, tag = _kind_and_tag(filename)
        model = _model_key(tag)
        if model not in MODEL_ORDER:
            continue
        metadata_path = Path(filename).with_name(
            Path(filename).name.replace("_acc_", "_meta_", 1).replace(".npz", ".json")
        )
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing recovery provenance metadata for {filename}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("exclude_window_initial_frame") is not True:
            raise RuntimeError(
                f"Recovery input includes or does not document rollout t=0: {metadata_path}"
            )
        with np.load(filename) as payload:
            current_offsets = payload["offset_order"].astype(np.int64)
            if offsets is None:
                offsets = current_offsets
            elif not np.array_equal(offsets, current_offsets):
                raise RuntimeError(f"Mismatched recovery offsets in {filename}")
            grouped[model]["char"].append(payload["char_acc"].astype(np.float64))
            grouped[model]["sector"].append(payload["sector_acc"].astype(np.float64))
    if offsets is None or not grouped:
        raise RuntimeError(f"No recognized recovery curves found in {path}")
    curves = {
        model: {metric: np.stack(values) for metric, values in metrics.items()}
        for model, metrics in grouped.items()
    }
    seed_counts = {model: values["char"].shape[0] for model, values in curves.items()}
    if any(count != 10 for count in seed_counts.values()) or len(set(seed_counts.values())) != 1:
        raise RuntimeError(f"Expected exactly ten recovery seeds per model, got {seed_counts}.")
    return offsets, curves


def _recovery_mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return offset-wise mean and cross-seed SEM for exactly ten seeds."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 10:
        raise ValueError(f"Expected recovery curves shaped (10, offsets), got {values.shape}.")
    mean = values.mean(axis=0)
    return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _plot_test_axis(
    axis: plt.Axes,
    metrics: dict[str, dict[str, np.ndarray]],
    metric: str,
    show_xticks: bool,
    *,
    show_seed_points: bool = True,
) -> None:
    """Plot one character or sector validation-selected test bar panel."""

    models = [model for model in MODEL_ORDER if model in metrics]
    positions = np.arange(len(models), dtype=np.float64)
    rng = np.random.default_rng(0)
    for index, model in enumerate(models):
        values = metrics[model][metric]
        mean, sem = _mean_sem(values)
        axis.bar(
            positions[index],
            mean,
            width=0.72,
            yerr=sem,
            color=MODEL_COLORS[model],
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
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models], rotation=40, ha="right")
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    if metric == "char":
        axis.set_ylim(70.0, 90.0)
        axis.set_yticks(np.arange(70.0, 90.1, 5.0))
    elif metric == "sector":
        axis.set_ylim(85.0, 95.0)
        axis.set_yticks(np.arange(85.0, 95.1, 2.0))
    else:
        raise ValueError(f"Unknown test metric {metric!r}")
    _style_axis(axis)


def _loss_crop(path: Path, metric: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract one rendered loss panel at the requested display range from the source PNG."""

    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != (610, 1861):
        raise ValueError(
            f"Expected the canonical 1861x610 loss source, "
            f"got {image.shape[1]}x{image.shape[0]}"
        )
    if metric == "char":
        # Use the complete source data rectangle. Cropping only its lower half and remapping it
        # to the full range introduces a vertical numerical shift in compact summary panels.
        return image[94:539, 102:906], (0.3, 1.2)
    if metric == "sector":
        # The sector panel begins farther right; use the same complete data rectangle.
        return image[94:539, 1029:1833], (0.1, 0.5)
    raise ValueError(f"Unknown loss metric {metric!r}")


def _plot_loss_axis(
    axis: plt.Axes,
    loss_png: Path,
    metric: str,
    show_xlabel: bool,
    show_xticks: bool,
) -> None:
    """Place the validated loss render in a compact summary panel."""

    crop, limits = _loss_crop(loss_png, metric)
    displayed_bottom = 0.3 if metric == "char" else 0.15
    axis.imshow(
        crop,
        extent=(0.0, 150.0, displayed_bottom, limits[1]),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    axis.set_xlim(0.0, 150.0)
    axis.set_ylim(*limits)
    axis.set_xticks((0, 50, 100, 150))
    axis.set_yticks(np.arange(limits[0], limits[1] + 0.001, 0.1 if metric == "sector" else 0.3))
    if show_xlabel:
        axis.set_xlabel("Epoch")
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    if metric == "sector":
        # The source raster stops at y=0.15.  Extend its vertical grid to the requested y=0.1.
        axis.vlines(
            (0, 50, 100, 150),
            0.1,
            0.15,
            colors="0.75",
            linewidth=0.7,
            alpha=0.35,
            zorder=1,
        )
    axis.set_axisbelow(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_recovery_axis(
    axis: plt.Axes,
    offsets: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],
    metric: str,
    show_xlabel: bool,
    show_xticks: bool,
) -> None:
    """Plot one character or sector target-switch recovery panel."""

    selected_indices, selected_labels = select_key_recovery_ticks(offsets)
    x = np.arange(offsets.size, dtype=np.int64)
    for model in MODEL_ORDER:
        if model not in curves:
            continue
        mean, sem = _recovery_mean_sem(curves[model][metric])
        axis.plot(
            x,
            mean,
            color=MODEL_COLORS[model],
            linewidth=1.8,
            marker=MODEL_MARKERS[model],
            markevery=selected_indices.tolist(),
            markersize=3.8,
        )
        if np.any(sem):
            axis.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=MODEL_COLORS[model],
                alpha=0.55,
                linewidth=0,
            )
    axis.set_xticks(selected_indices, selected_labels)
    tick_labels = axis.get_xticklabels()
    tick_labels[1].set_ha("right")
    tick_labels[2].set_ha("left")
    if show_xlabel:
        axis.set_xlabel("Frame relative to target switch")
    if not show_xticks:
        axis.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    axis.set_ylim(0.0, 100.0)
    _style_axis(axis)


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared borderless style to all six panels."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)


def plot_summary(
    test_metrics: dict[str, dict[str, np.ndarray]],
    loss_png: Path,
    recovery_offsets: np.ndarray,
    recovery_curves: dict[str, dict[str, np.ndarray]],
    output_png: Path,
    output_pdf: Path | None,
) -> None:
    """Render the requested two-row, three-column Clutter result summary."""

    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.9))
        _plot_test_axis(
            axes[0, 0], test_metrics, "char", show_xticks=False,
            show_seed_points=args.show_seed_points,
        )
        _plot_test_axis(
            axes[1, 0], test_metrics, "sector", show_xticks=True,
            show_seed_points=args.show_seed_points,
        )
        _plot_loss_axis(
            axes[0, 1], loss_png, "char", show_xlabel=False, show_xticks=False
        )
        _plot_loss_axis(
            axes[1, 1], loss_png, "sector", show_xlabel=True, show_xticks=True
        )
        _plot_recovery_axis(
            axes[0, 2], recovery_offsets, recovery_curves, "char", show_xlabel=False,
            show_xticks=False,
        )
        _plot_recovery_axis(
            axes[1, 2], recovery_offsets, recovery_curves, "sector", show_xlabel=True,
            show_xticks=True,
        )

        fig.subplots_adjust(
            left=0.08,
            right=0.995,
            bottom=0.12,
            top=0.81,
            hspace=0.40,
            wspace=0.18,
        )
        column_centers = [
            np.mean(
                [
                    axes[row, column].get_position().x0
                    + axes[row, column].get_position().width / 2
                    for row in range(2)
                ]
            )
            for column in range(3)
        ]
        title_y = max(axes[0, column].get_position().y1 for column in range(3)) + 0.05
        for x, title in zip(
            column_centers,
            ("Test accuracy", "Validation loss", "Target switch recovery"),
        ):
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=15)
        row_centers = [
            np.mean(
                [
                    axes[row, column].get_position().y0
                    + axes[row, column].get_position().height / 2
                    for column in range(3)
                ]
            )
            for row in range(2)
        ]
        fig.text(
            0.042,
            row_centers[0],
            "Digit",
            rotation=90,
            ha="center",
            va="center",
            fontsize=15,
        )
        fig.text(
            0.042,
            row_centers[1],
            "Sector",
            rotation=90,
            ha="center",
            va="center",
            fontsize=15,
        )

        models = [model for model in MODEL_ORDER if model in test_metrics]
        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.2)
            for model in models
        ]
        fig.legend(
            handles,
            [MODEL_LABELS[model] for model in models],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=len(models),
            fontsize=13,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        if output_pdf is not None:
            fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def main() -> None:
    """Load the selected best-6 results and save their combined visual summary."""

    args = parse_args()
    # Workflow policy: the PDF sits next to the PNG in the local results tree by default.
    # ``--publication_fig_dir`` is an opt-in extra copy only (no automatic 6-Writing sync).
    publication_dir = (
        publication_figures_dir(args.publication_fig_dir, create=True)
        if args.publication_fig_dir is not None
        else None
    )
    output_pdf = (
        args.output_pdf if args.output_pdf is not None else args.output_png.with_suffix(".pdf")
    )
    test_metrics = load_test_metrics(args.test_csv)
    recovery_offsets, recovery_curves = load_recovery_curves(args.recovery_dir)
    plot_summary(
        test_metrics,
        args.loss_png,
        recovery_offsets,
        recovery_curves,
        args.output_png,
        output_pdf,
    )
    print(f"Saved {args.output_png}")
    print(f"Saved {output_pdf}")
    if publication_dir is not None:
        publication_pdf = publication_dir / "best6_multiseed_summary_2x3.pdf"
        shutil.copyfile(output_pdf, publication_pdf)
        print(f"Saved {publication_pdf}")


if __name__ == "__main__":
    main()
