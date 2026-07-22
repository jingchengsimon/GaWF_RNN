"""Render compact GaWF gate-distribution summaries from saved histogram statistics.

Inputs are the pooled/sign/context/digit histogram ``.npz`` files and their pooled metadata.
Outputs are a poster-style 2-by-4 PNG/PDF summary (input and recurrent gates by four views) and
an additional all-gate pooled distribution that combines input and recurrent gate entries. Both
figures are stored in the ``A_raw_gate`` figure category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from utils.publication_paths import publication_figures_dir
from utils_anal.anal_paths import PROJECT_ROOT, output_dir


RAW_DATA_DIR = PROJECT_ROOT / "results" / "anal_data" / "gawf_gate_audit"
SUMMARY_FIG_DIR = output_dir("A_raw_gate", "gawf_gate_histogram_summary", "figs")
ALL_GATE_FIG_DIR = output_dir("A_raw_gate", "gawf_gate_histogram_summary", "figs")
ZOOM_XLIM = (0.48, 0.52)


def parse_args() -> argparse.Namespace:
    """Parse histogram input and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--digit_data_dir",
        type=Path,
        default=output_dir("B_gate_by_context", "gawf_gate_digit_distribution", "data"),
    )
    parser.add_argument("--summary_fig_dir", type=Path, default=SUMMARY_FIG_DIR)
    parser.add_argument("--all_gate_fig_dir", type=Path, default=ALL_GATE_FIG_DIR)
    parser.add_argument("--publication_fig_dir", type=Path, default=None)
    return parser.parse_args()


def _probability_percent(counts: np.ndarray) -> np.ndarray:
    """Return the probability mass in each histogram bin as percentages."""

    counts_float = np.asarray(counts, dtype=np.float64)
    total = counts_float.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("Histogram counts must have a positive total")
    return 100.0 * counts_float / total


def _histogram_mean_median(
    counts: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate mean and median gate values from one or more binned populations."""

    counts_float = np.asarray(counts, dtype=np.float64)
    if counts_float.ndim == 1:
        counts_float = counts_float[None, :]
    total = counts_float.sum(axis=1)
    if np.any(total <= 0.0):
        raise ValueError("Histogram counts must have a positive total")
    centers = (edges[:-1] + edges[1:]) / 2.0
    mean = (counts_float * centers[None, :]).sum(axis=1) / total
    cumulative = np.cumsum(counts_float, axis=1)
    median_indices = np.argmax(cumulative >= total[:, None] / 2.0, axis=1)
    median = centers[median_indices]
    return mean, median


def _add_mean_median_lines(
    axis: plt.Axes,
    means: np.ndarray,
    medians: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Add color-matched mean and median lines without duplicating legend entries."""

    for mean, median, color in zip(np.ravel(means), np.ravel(medians), colors):
        axis.axvline(
            float(mean),
            color=color,
            linewidth=0.9,
            alpha=0.85,
            linestyle="-",
            label="_nolegend_",
        )
        axis.axvline(
            float(median),
            color=color,
            linewidth=1.1,
            alpha=0.85,
            linestyle=":",
            label="_nolegend_",
        )


def _reference_legend_handles() -> list[Line2D]:
    """Return compact generic handles for the mean/median reference lines."""

    return [
        Line2D([], [], color="black", linewidth=1.1, linestyle="-", label="Mean"),
        Line2D([], [], color="black", linewidth=1.1, linestyle=":", label="Median"),
    ]


def _add_central_zoom_inset(
    axis: plt.Axes,
    centers: np.ndarray,
    probability: np.ndarray,
    *,
    color: str,
    bounds: tuple[float, float, float, float],
    label: str | None = None,
    reference_lines: tuple[tuple[float, str, str], ...] = (),
) -> plt.Axes:
    """Add a compact 0.48--0.52 gate-value zoom for a single probability curve."""

    inset = axis.inset_axes(bounds)
    inset.plot(centers, probability, color=color, linewidth=1.0)
    for value, line_color, line_style in reference_lines:
        inset.axvline(value, color=line_color, linestyle=line_style, linewidth=0.7)
    central = probability[(centers >= ZOOM_XLIM[0]) & (centers <= ZOOM_XLIM[1])]
    central_maximum = float(central.max()) if central.size else 0.0
    inset.set_xlim(*ZOOM_XLIM)
    inset.set_ylim(0.0, max(central_maximum * 1.1, 0.1))
    inset.set_xticks(np.asarray([0.48, 0.50, 0.52]))
    inset.set_xticklabels([".48", ".50", ".52"])
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=6, length=2, pad=1)
    inset.grid(axis="y", alpha=0.2, linewidth=0.4)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    if label is not None:
        inset.text(
            0.04,
            0.92,
            label,
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
        )
    return inset


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load all arrays from one compact histogram archive."""

    if not path.is_file():
        raise FileNotFoundError(f"Histogram statistics not found: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def _load_metadata(path: Path) -> dict[str, object]:
    """Load pooled histogram metadata."""

    if not path.is_file():
        raise FileNotFoundError(f"Histogram metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _style_axis(axis: plt.Axes, *, show_x_labels: bool) -> None:
    """Apply the poster axis style to one histogram panel."""

    axis.set_xlim(-0.05, 1.05)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelbottom=show_x_labels)


def _add_pooled_panel(
    axis: plt.Axes,
    kind: str,
    raw_arrays: dict[str, np.ndarray],
    raw_metadata: dict[str, object],
) -> float:
    """Draw one pooled input/recurrent panel and return its density maximum."""

    edges = raw_arrays["gate_edges"].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = raw_arrays[f"hist_{kind}_all"]
    probability = _probability_percent(counts)
    axis.plot(
        centers, probability, color="#2b6cb0", linewidth=1.6, label="Probability (%)"
    )
    stats = raw_metadata["distribution"][kind]
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1.1, label="0.5")
    axis.axvline(
        float(stats["mean"]), color="#d53f8c", linewidth=1.2, label="Mean"
    )
    axis.axvline(
        float(stats["median"]), color="#38a169", linestyle=":", linewidth=1.5, label="Median"
    )
    _add_central_zoom_inset(
        axis,
        centers,
        probability,
        color="#2b6cb0",
        bounds=(0.44, 0.10, 0.50, 0.32),
        reference_lines=(
            (0.5, "black", "--"),
            (float(stats["mean"]), "#d53f8c", "-"),
            (float(stats["median"]), "#38a169", ":"),
        ),
    )
    return float(probability.max())


def _add_weight_sign_panel(
    axis: plt.Axes,
    kind: str,
    raw_arrays: dict[str, np.ndarray],
) -> float:
    """Draw one weight-sign split panel and return its density maximum."""

    edges = raw_arrays["gate_edges"].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = raw_arrays[f"hist_{kind}_sign"]
    probabilities = _probability_percent(counts)
    axis.plot(centers, probabilities[0], color="#c53030", linewidth=1.25, label="W > 0")
    axis.plot(centers, probabilities[1], color="#2b6cb0", linewidth=1.25, label="W < 0")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="0.5")
    means, medians = _histogram_mean_median(counts, edges)
    _add_mean_median_lines(
        axis,
        means,
        medians,
        np.asarray(["#c53030", "#2b6cb0"], dtype=object),
    )
    _add_central_zoom_inset(
        axis,
        centers,
        probabilities[0],
        color="#c53030",
        bounds=(0.08, 0.10, 0.38, 0.32),
        label="W > 0",
        reference_lines=(
            (0.5, "black", "--"),
            (float(means[0]), "#c53030", "-"),
            (float(medians[0]), "#c53030", ":"),
        ),
    )
    _add_central_zoom_inset(
        axis,
        centers,
        probabilities[1],
        color="#2b6cb0",
        bounds=(0.54, 0.10, 0.38, 0.32),
        label="W < 0",
        reference_lines=(
            (0.5, "black", "--"),
            (float(means[1]), "#2b6cb0", "-"),
            (float(medians[1]), "#2b6cb0", ":"),
        ),
    )
    return float(probabilities.max())


def _add_context_panel(
    axis: plt.Axes,
    kind: str,
    raw_arrays: dict[str, np.ndarray],
    colors: np.ndarray,
) -> float:
    """Draw one sector-conditioned panel and return its density maximum."""

    edges = raw_arrays["gate_edges"].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = raw_arrays[f"hist_{kind}_context"]
    probabilities = _probability_percent(counts)
    for sector, probability in enumerate(probabilities):
        axis.plot(centers, probability, color=colors[sector], linewidth=1.0, label=str(sector))
    means, medians = _histogram_mean_median(counts, edges)
    _add_mean_median_lines(axis, means, medians, colors)
    return float(probabilities.max())


def _add_digit_panel(
    axis: plt.Axes,
    kind: str,
    digit_arrays: dict[str, np.ndarray],
    colors: np.ndarray,
) -> float:
    """Draw one digit-conditioned panel and return its density maximum."""

    edges = digit_arrays["gate_edges"].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = digit_arrays[f"hist_{kind}_digit"]
    probabilities = _probability_percent(counts)
    for digit, probability in enumerate(probabilities):
        axis.plot(centers, probability, color=colors[digit], linewidth=1.0, label=str(digit))
    means, medians = _histogram_mean_median(counts, edges)
    _add_mean_median_lines(axis, means, medians, colors)
    return float(probabilities.max())


def plot_histogram_summary(
    raw_arrays: dict[str, np.ndarray],
    raw_metadata: dict[str, object],
    digit_arrays: dict[str, np.ndarray],
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    """Render the four-by-two histogram summary with larger inset-capable panels."""

    kinds = ("input", "recurrent")
    row_titles = (
        "Pooled gate distribution",
        "Weight-sign split",
        "Foreground sector",
        "Foreground digit",
    )
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 10,
        }
    ):
        fig, axes = plt.subplots(4, 2, figsize=(12.2, 15.0), sharex="col")
        sector_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 9))
        digit_colors = plt.get_cmap("tab10")(np.arange(10))
        for row, _ in enumerate(row_titles):
            for column, kind in enumerate(kinds):
                axis = axes[row, column]
                if row == 0:
                    panel_maximum = _add_pooled_panel(
                        axis, kind, raw_arrays, raw_metadata
                    )
                elif row == 1:
                    panel_maximum = _add_weight_sign_panel(axis, kind, raw_arrays)
                elif row == 2:
                    panel_maximum = _add_context_panel(
                        axis, kind, raw_arrays, sector_colors
                    )
                else:
                    panel_maximum = _add_digit_panel(axis, kind, digit_arrays, digit_colors)
                _style_axis(axis, show_x_labels=row == len(row_titles) - 1)
                axis.set_ylim(0.0, panel_maximum * 1.05)

        for axis in axes[:-1, :].flat:
            axis.tick_params(axis="x", labelbottom=False)
        for axis in axes[-1, :]:
            axis.set_xlabel("")

        fig.subplots_adjust(
            left=0.18,
            right=0.99,
            bottom=0.075,
            top=0.93,
            hspace=0.32,
            wspace=0.20,
        )
        title_y = max(axis.get_position().y1 for axis in axes[0]) + 0.012
        column_centers = [
            axes[0, column].get_position().x0 + axes[0, column].get_position().width / 2
            for column in range(2)
        ]
        for x, title in zip(column_centers, ("Input gate", "Recurrent gate")):
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=16)

        row_centers = [
            axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
            for row in range(4)
        ]
        for y, title in zip(row_centers, row_titles):
            fig.text(0.10, y, title, rotation=90, ha="center", va="center", fontsize=14)
        fig.text(
            0.030,
            np.mean(row_centers),
            "Probability (%)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.text(0.60, 0.028, "Gate value", ha="center", va="center", fontsize=16)

        axes[0, 0].legend(frameon=False, loc="upper right", ncol=2, handlelength=1.4)
        sign_handles, sign_labels = axes[1, 0].get_legend_handles_labels()
        sign_handles.extend(_reference_legend_handles())
        sign_labels.extend(["Mean", "Median"])
        axes[1, 0].legend(
            sign_handles,
            sign_labels,
            frameon=False,
            loc="upper right",
            ncol=2,
            handlelength=1.4,
        )
        context_handles, context_labels = axes[2, 0].get_legend_handles_labels()
        context_handles.extend(_reference_legend_handles())
        context_labels.extend(["Mean", "Median"])
        axes[2, 0].legend(
            context_handles,
            context_labels,
            title="Sector",
            title_fontsize=10,
            frameon=False,
            loc="upper right",
            ncol=3,
            columnspacing=0.7,
            handlelength=1.2,
            handletextpad=0.3,
        )
        digit_handles, digit_labels = axes[3, 0].get_legend_handles_labels()
        digit_handles.extend(_reference_legend_handles())
        digit_labels.extend(["Mean", "Median"])
        axes[3, 0].legend(
            digit_handles,
            digit_labels,
            title="Digit",
            title_fontsize=10,
            frameon=False,
            loc="upper right",
            ncol=5,
            columnspacing=0.7,
            handlelength=1.2,
            handletextpad=0.3,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        if output_pdf is not None:
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return output_png


def plot_all_gate_distribution(
    raw_arrays: dict[str, np.ndarray],
    output_png: Path,
    output_pdf: Path | None = None,
) -> Path:
    """Render one pooled distribution after combining input and recurrent gate entries."""

    edges = raw_arrays["gate_edges"].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = raw_arrays["hist_input_all"].astype(np.int64) + raw_arrays[
        "hist_recurrent_all"
    ].astype(np.int64)
    probability = _probability_percent(counts)
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):
        fig, axis = plt.subplots(figsize=(6.8, 4.8))
        axis.plot(
            centers,
            probability,
            color="#2b6cb0",
            linewidth=1.7,
            label="Probability (%)",
        )
        axis.set_xlabel("Gate value")
        axis.set_ylabel("Probability (%)")
        axis.set_xlim(-0.05, 1.05)
        probability_max = float(probability.max())
        axis.set_ylim(-0.05 * probability_max, probability_max * 1.05)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        _add_central_zoom_inset(
            axis,
            centers,
            probability,
            color="#2b6cb0",
            bounds=(0.46, 0.48, 0.43, 0.35),
        )
        axis.set_title(
            "GaWF gate distribution",
            fontsize=15,
            pad=12,
        )
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.16, top=0.85)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        if output_pdf is not None:
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return output_png


def main() -> None:
    """Load saved histogram statistics and write both requested figures."""

    args = parse_args()
    raw_arrays = _load_npz(args.raw_data_dir / "gawf_gate_distribution_stats.npz")
    raw_metadata = _load_metadata(args.raw_data_dir / "gawf_gate_distribution_meta.json")
    digit_arrays = _load_npz(args.digit_data_dir / "gawf_gate_digit_stats.npz")
    if not np.array_equal(raw_arrays["gate_edges"], digit_arrays["gate_edges"]):
        raise RuntimeError("Pooled, sector, and digit histograms must use identical gate bins")

    publication_dir = (
        publication_figures_dir(args.publication_fig_dir, create=True)
        if args.publication_fig_dir is not None
        else None
    )
    summary_png = args.summary_fig_dir / "gawf_gate_histogram_summary_2x4.png"
    all_gate_png = args.all_gate_fig_dir / "01_pooled_all_gate_histogram.png"
    # The 2-by-4 summary is PNG-only. The all-gate panel keeps its local PDF; a publication
    # copy is written only when an explicit ``--publication_fig_dir`` is supplied.
    all_gate_pdf = args.all_gate_fig_dir / "01_pooled_all_gate_histogram.pdf"
    plot_histogram_summary(raw_arrays, raw_metadata, digit_arrays, summary_png)
    plot_all_gate_distribution(raw_arrays, all_gate_png, all_gate_pdf)
    if publication_dir is not None:
        publication_all_gate_pdf = publication_dir / "01_pooled_all_gate_histogram.pdf"
        plot_all_gate_distribution(raw_arrays, all_gate_png, publication_all_gate_pdf)
    print(f"Saved {summary_png}")
    print(f"Saved {all_gate_png}")
    print(f"Saved {all_gate_pdf}")
    if publication_dir is not None:
        print(f"Saved {publication_all_gate_pdf}")


if __name__ == "__main__":
    main()
