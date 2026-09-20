"""Render the three-part Sector input-gate Figure 6 from retained ten-seed summaries.

The PDF contains the encoder Sector spatial maps, reset-excluded sequential input-gate delta
maps, plus matching- and other-source sign-versus-|W| binned curves. It reads compact saved
arrays and does not reuse raster figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    NEG_COLOR,
    POS_COLOR,
    binned_mean_curve,
    quantile_bin_edges,
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
ENCODER_SUMMARY = (
    PROJECT_ROOT
    / "results/data/analysis/fig6_encoder_sector_patterns_gawf_10seed/final"
    / "encoder_sector_patterns_10seed_summary.npz"
)
GATE_ROOT = PROJECT_ROOT / "results/save_data/fig6"
SUPPLE2_ROOT = (
    PROJECT_ROOT / "results/data/analysis"
    / "supple2_input_gate_sign_magnitude_9sector_reset_excluded_10seed"
)
SAVE_FIGURE = PROJECT_ROOT / "results/save/Fig6_overall_sector_input_gate_1x3_10seed.pdf"
NUM_SECTORS = 9
SOURCE_GROUPS = ("sector0_sources", "other_sources")


def parse_args() -> argparse.Namespace:
    """Parse retained-summary paths and the one publication PDF destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder_summary", type=Path, default=ENCODER_SUMMARY)
    parser.add_argument("--gate_root", type=Path, default=GATE_ROOT)
    parser.add_argument("--supple2_root", type=Path, default=SUPPLE2_ROOT)
    parser.add_argument("--figure", type=Path, default=SAVE_FIGURE)
    return parser.parse_args()


def _load_encoder_maps(path: Path) -> np.ndarray:
    """Return the retained ten-seed mean encoder Sector maps."""

    with np.load(path, allow_pickle=False) as arrays:
        maps = np.asarray(arrays["spatial_maps"], dtype=np.float64)
    if maps.shape != (9, 6, 6) or not np.isfinite(maps).all():
        raise RuntimeError(f"Expected finite encoder maps (9, 6, 6) in {path}, got {maps.shape}.")
    return maps


def _load_gate_delta_maps(root: Path) -> np.ndarray:
    """Average the reset-excluded point maps across exactly ten training seeds."""

    paths = sorted(root.glob("seed*/sector_gate_mean_sequential_equal_n.npz"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten Fig6 gate files in {root}, found {len(paths)}.")
    maps = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            value = np.asarray(arrays["point_excluded"], dtype=np.float64)
        if value.shape != (9, 6, 6) or not np.isfinite(value).all():
            raise RuntimeError(f"Expected finite point-excluded maps (9, 6, 6) in {path}.")
        maps.append(value)
    mean_gate = np.mean(np.stack(maps), axis=0)
    return mean_gate - mean_gate.mean(axis=0, keepdims=True)


def _spatial_sector_indices(input_size: int) -> list[np.ndarray]:
    """Map flattened 32-by-6-by-6 encoder features to the nine coarse sectors."""

    if input_size != 32 * 6 * 6:
        raise RuntimeError(f"Expected 1152 encoder sources, got {input_size}.")
    layout = np.arange(input_size, dtype=np.int64).reshape(32, 6, 6)
    result = []
    for sector in range(NUM_SECTORS):
        row, column = divmod(sector, 3)
        result.append(
            layout[:, row * 2 : row * 2 + 2, column * 2 : column * 2 + 2].reshape(-1)
        )
    return result


def _source_arrays(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Pool the frozen all-sector input-gate arrays without importing the GPU collector."""

    paths = sorted(root.glob("seed*/input_gate_sign_magnitude_9sector.npz"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten input-gate seed files in {root}, found {len(paths)}.")
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        group: {"absW": [], "delta_gate": [], "signpos": []} for group in SOURCE_GROUPS
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            weight = np.asarray(arrays["weight"], dtype=np.float64)
            means = np.asarray(arrays["sector_gate_mean"], dtype=np.float64)
        if means.shape != (NUM_SECTORS, *weight.shape):
            raise RuntimeError(f"Invalid nine-sector data in {path}: {means.shape}.")
        grand = means.mean(axis=0)
        indices = _spatial_sector_indices(weight.shape[1])
        for sector in range(NUM_SECTORS):
            matching = np.zeros(weight.shape[1], dtype=bool)
            matching[indices[sector]] = True
            for group, source_mask in zip(SOURCE_GROUPS, (matching, ~matching)):
                selected_weight = weight[:, source_mask]
                selected_delta = means[sector][:, source_mask] - grand[:, source_mask]
                keep = selected_weight != 0.0
                pooled[group]["absW"].append(np.abs(selected_weight[keep]))
                pooled[group]["delta_gate"].append(selected_delta[keep])
                pooled[group]["signpos"].append(selected_weight[keep] > 0.0)
    return {
        group: {name: np.concatenate(values) for name, values in arrays.items()}
        for group, arrays in pooled.items()
    }


def _source_curves(
    source_arrays: dict[str, dict[str, np.ndarray]], group: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive- and negative-weight nine-bin curves for one Supple2 source group."""

    abs_weight = source_arrays[group]["absW"]
    delta_gate = source_arrays[group]["delta_gate"]
    sign_positive = source_arrays[group]["signpos"]
    edges = quantile_bin_edges(abs_weight)
    curves = []
    for select in (sign_positive, ~sign_positive):
        center, mean, sem, count = binned_mean_curve(abs_weight[select], delta_gate[select], edges)
        curves.append(np.stack((center, mean, sem, count.astype(np.float64))))
    return curves[0], curves[1]


def _zero_weight_baseline(positive: np.ndarray, negative: np.ndarray) -> float:
    """Estimate the shared ``|W|=0`` level by linear extrapolation from each curve's first bins."""

    intercepts = []
    for curve in (positive, negative):
        center, mean, _sem, count = curve
        valid = np.flatnonzero(count > 0)
        if valid.size < 2:
            raise RuntimeError("Need two non-empty bins to estimate the |W|=0 baseline.")
        first, second = valid[:2]
        slope = (mean[second] - mean[first]) / (center[second] - center[first])
        intercepts.append(float(mean[first] - slope * center[first]))
    return float(np.mean(intercepts))


def _draw_map_grid(
    figure: plt.Figure,
    grid: matplotlib.gridspec.SubplotSpec,
    maps: np.ndarray,
    *,
    cmap: str,
    norm: matplotlib.colors.Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    block_title: str,
) -> None:
    """Draw one Fig6-style 3-by-3 spatial map block with its own colorbar."""

    inner = grid.subgridspec(3, 4, width_ratios=(1, 1, 1, 0.10), wspace=0.08, hspace=0.18)
    image = None
    for sector in range(9):
        axis = figure.add_subplot(inner[sector // 3, sector % 3])
        image = axis.pcolormesh(
            maps[sector],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            edgecolors="face",
            linewidth=0.01,
            antialiased=False,
            rasterized=False,
            snap=True,
        )
        axis.set_xlim(0, 6)
        axis.set_ylim(6, 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    colorbar = figure.colorbar(image, cax=figure.add_subplot(inner[:, 3]))
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label, fontsize=7, labelpad=2)
    colorbar.ax.tick_params(labelsize=6, length=2, pad=1)
    position = grid.get_position(figure)
    figure.text(
        (position.x0 + position.x1) / 2,
        position.y1 + 0.045,
        block_title,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def _draw_curves(
    axis: plt.Axes,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    baseline: float,
    y_label: str,
    show_xaxis: bool,
) -> None:
    """Draw only the retained Supple2 binned mean plus SEM curves, without point clouds."""

    curves = (
        (positive, POS_COLOR, r"$w^{\mathrm{in}}_{+}$"),
        (negative, NEG_COLOR, r"$w^{\mathrm{in}}_{-}$"),
    )
    for curve, color, label in curves:
        center, mean, sem, count = curve
        valid = count > 0
        axis.errorbar(
            center[valid],
            mean[valid],
            yerr=sem[valid],
            color=color,
            linewidth=1.1,
            marker="o",
            markersize=2.5,
            capsize=1.5,
            label=label,
        )
    axis.set_xlim(0.0, 1.08 * max(np.nanmax(positive[0]), np.nanmax(negative[0])))
    y_low, y_high = baseline - 0.15, baseline + 0.15
    axis.set_ylim(y_low, y_high)
    tick_start = np.ceil(y_low * 10.0 - 1e-12) / 10.0
    axis.set_yticks(np.arange(tick_start, y_high + 1e-12, 0.1))
    axis.set_xlabel(r"$|w^{\mathrm{in}}|$" if show_xaxis else "", fontsize=7, labelpad=1)
    axis.set_ylabel(y_label, fontsize=7, labelpad=0)
    axis.tick_params(labelsize=6, length=2)
    axis.tick_params(axis="x", labelbottom=show_xaxis)
    axis.spines[["top", "right"]].set_visible(False)


def render(
    encoder_maps: np.ndarray,
    gate_delta_maps: np.ndarray,
    matching_positive: np.ndarray,
    matching_negative: np.ndarray,
    other_positive: np.ndarray,
    other_negative: np.ndarray,
    matching_baseline: float,
    other_baseline: float,
    destinations: tuple[Path, ...],
) -> None:
    """Render the title-free Figure 6 three-panel layout at every requested PDF path."""

    with plt.rc_context({"font.size": 7, "axes.titlesize": 8}):
        figure = plt.figure(figsize=(5.5, 2.0))
        map_grid = figure.add_gridspec(
            1,
            2,
            left=0.035,
            right=0.605,
            bottom=0.20,
            top=0.82,
            wspace=0.415,
        )
        _draw_map_grid(
            figure,
            map_grid[0, 0],
            encoder_maps,
            cmap="Reds",
            vmin=float(encoder_maps.min()),
            vmax=float(encoder_maps.max()),
            block_title="Encoder activation",
        )
        limit = float(np.abs(gate_delta_maps).max())
        _draw_map_grid(
            figure,
            map_grid[0, 1],
            gate_delta_maps,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            block_title=r"$\Delta g^{\mathrm{in}}$",
        )
        curve_grid = figure.add_gridspec(
            2, 1, left=0.743, right=0.979, bottom=0.20, top=0.82, hspace=0.58
        )
        top_curve_axis = figure.add_subplot(curve_grid[0, 0])
        curve_axes = (
            top_curve_axis,
            figure.add_subplot(curve_grid[1, 0], sharex=top_curve_axis),
        )
        _draw_curves(
            curve_axes[0],
            matching_positive,
            matching_negative,
            baseline=matching_baseline,
            y_label=r"$\Delta g^{\mathrm{in}}$",
            show_xaxis=False,
        )
        _draw_curves(
            curve_axes[1],
            other_positive,
            other_negative,
            baseline=other_baseline,
            y_label=r"$\Delta g^{\mathrm{in}}$",
            show_xaxis=True,
        )
        curve_axes[0].set_title("Location-matched", fontsize=7, pad=2)
        curve_axes[1].set_title("Other", fontsize=7, pad=2)
        figure.align_ylabels(curve_axes)
        handles, labels = curve_axes[0].get_legend_handles_labels()
        right_position = curve_axes[0].get_position()
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=((right_position.x0 + right_position.x1) / 2, 0.89),
            ncol=2,
            fontsize=5.7,
            handlelength=1.0,
            columnspacing=0.8,
        )
        block_positions = (
            map_grid[0, 0].get_position(figure),
            map_grid[0, 1].get_position(figure),
            right_position,
        )
        for label, position in zip("ABC", block_positions):
            figure.text(
                position.x0 - 0.018,
                0.865,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination)
        plt.close(figure)


def main() -> None:
    """Load retained summaries, save a compact numeric record, and render the final PDF."""

    args = parse_args()
    encoder_maps = _load_encoder_maps(args.encoder_summary)
    gate_delta_maps = _load_gate_delta_maps(args.gate_root)
    source_arrays = _source_arrays(args.supple2_root)
    matching_positive, matching_negative = _source_curves(
        source_arrays, "sector0_sources"
    )
    other_positive, other_negative = _source_curves(source_arrays, "other_sources")
    matching_baseline = _zero_weight_baseline(matching_positive, matching_negative)
    other_baseline = _zero_weight_baseline(other_positive, other_negative)
    data_dir = output_dir("B_gate_by_context", SCRIPT_NAME, "data")
    figure_dir = output_dir("B_gate_by_context", SCRIPT_NAME, "figs")
    np.savez_compressed(
        data_dir / "fig6_overall_sector_input_gate_1x3_10seed.npz",
        encoder_spatial_maps=encoder_maps.astype(np.float32),
        gate_delta_maps=gate_delta_maps.astype(np.float32),
        matching_positive_curve=matching_positive.astype(np.float32),
        matching_negative_curve=matching_negative.astype(np.float32),
        other_positive_curve=other_positive.astype(np.float32),
        other_negative_curve=other_negative.astype(np.float32),
        matching_weight_zero_baseline=np.float32(matching_baseline),
        other_weight_zero_baseline=np.float32(other_baseline),
    )
    (data_dir / "fig6_overall_sector_input_gate_1x3_10seed.json").write_text(
        json.dumps(
            {
                "training_seeds": 10,
                "panels": [
                    "mean encoder activation by Sector",
                    "sequential reset-excluded delta input-gate mean by Sector",
                    "matching- and other-source delta input-gate versus |W|, binned mean plus SEM",
                ],
                "supple2_removed": ["scatter", "shared-|W| shading", "n/gap annotation", "title"],
                "curve_y_axis": {
                    "definition": "raw delta gate; limits centered on |W|=0 baseline",
                    "half_range": 0.15,
                    "tick_step": 0.1,
                    "matching_weight_zero_baseline": matching_baseline,
                    "matching_limits": [matching_baseline - 0.15, matching_baseline + 0.15],
                    "other_weight_zero_baseline": other_baseline,
                    "other_limits": [other_baseline - 0.15, other_baseline + 0.15],
                },
                "inputs": {
                    "encoder_summary": str(args.encoder_summary),
                    "gate_root": str(args.gate_root),
                    "supple2_root": str(args.supple2_root),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = figure_dir / args.figure.name
    render(
        encoder_maps,
        gate_delta_maps,
        matching_positive,
        matching_negative,
        other_positive,
        other_negative,
        matching_baseline,
        other_baseline,
        (canonical, args.figure),
    )
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
