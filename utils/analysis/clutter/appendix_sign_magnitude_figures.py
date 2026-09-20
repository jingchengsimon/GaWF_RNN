"""Render ICLR-width Appendix sign-versus-magnitude figures from retained numeric data.

Inputs are the ten reset-excluded input-gate nine-sector NPZ files and the ten compact
recurrent-gate condition-mean NPZ files. Outputs are one two-panel input-gate PDF and one
two-by-four recurrent-gate PDF in the requested figure directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.analysis.clutter.fig7_recurrent_gate_multiseed import (  # noqa: E402
    RESULT_NAME as RECURRENT_RESULT_NAME,
    _pooled_records,
)
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (  # noqa: E402
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    binned_mean_curve,
    compute_overlap_band,
    quantile_bin_edges,
)
from utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector import (  # noqa: E402
    ALL_SECTORS_RESULT_NAME,
    GROUPS as INPUT_GROUPS,
    NUM_SECTORS,
    _source_group_masks,
)


CurveSummary = dict[str, object]
MAX_SCATTER_PER_SIGN = 900


def parse_args() -> argparse.Namespace:
    """Parse local structured-data roots and the output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--recurrent-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _summarize_arrays(
    abs_weight: np.ndarray,
    values: np.ndarray,
    sign_positive: np.ndarray,
    seed_index: np.ndarray,
    *,
    rng: np.random.Generator,
) -> CurveSummary:
    """Reduce rows to seed-level curves, overlap range, and display-only samples."""

    abs_weight = np.asarray(abs_weight, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sign_positive = np.asarray(sign_positive, dtype=bool)
    seed_index = np.asarray(seed_index, dtype=np.int64)
    if not (
        abs_weight.shape == values.shape == sign_positive.shape == seed_index.shape
    ) or abs_weight.size == 0:
        raise ValueError("Sign-versus-magnitude arrays must be nonempty and shape-aligned.")
    seeds = np.unique(seed_index)
    if not np.array_equal(seeds, np.arange(1, 11)):
        raise RuntimeError(f"Expected seed indices 1--10, got {seeds.tolist()}.")
    edges = quantile_bin_edges(abs_weight)
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    signed_weights: dict[str, np.ndarray] = {}
    for sign, label in ((True, "positive"), (False, "negative")):
        selected_weight = abs_weight[sign_positive == sign]
        selected_values = values[sign_positive == sign]
        selected_seeds = seed_index[sign_positive == sign]
        seed_centers = []
        seed_means = []
        for seed in seeds:
            in_seed = selected_seeds == seed
            centers, means, _row_sems, counts = binned_mean_curve(
                selected_weight[in_seed], selected_values[in_seed], edges
            )
            centers[counts == 0] = np.nan
            means[counts == 0] = np.nan
            seed_centers.append(centers)
            seed_means.append(means)
        center_matrix = np.stack(seed_centers)
        mean_matrix = np.stack(seed_means)
        valid_seed_counts = np.sum(np.isfinite(mean_matrix), axis=0)
        valid = valid_seed_counts > 0
        curve_centers = np.nanmean(center_matrix, axis=0)
        curve_means = np.nanmean(mean_matrix, axis=0)
        curve_sems = np.nanstd(mean_matrix, axis=0, ddof=1) / np.sqrt(valid_seed_counts)
        curves[label] = (
            curve_centers[valid],
            curve_means[valid],
            curve_sems[valid],
        )
        signed_weights[label] = selected_weight
        sample_count = min(MAX_SCATTER_PER_SIGN, selected_weight.size)
        indices = rng.choice(selected_weight.size, size=sample_count, replace=False)
        samples[label] = (selected_weight[indices], selected_values[indices])
    overlap = compute_overlap_band(
        signed_weights["positive"], signed_weights["negative"]
    )
    return {"curves": curves, "samples": samples, "overlap": overlap}


def _input_group_arrays(
    data_root: Path, group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool one input-source group across ten seeds and all nine sectors."""

    paths = sorted(data_root.glob(f"seed*/{ALL_SECTORS_RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten input-gate seed files, found {len(paths)}.")
    weight_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    sign_parts: list[np.ndarray] = []
    seed_parts: list[np.ndarray] = []
    for seed, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as arrays:
            weight = np.asarray(arrays["weight"], dtype=np.float32)
            sector_means = np.asarray(arrays["sector_gate_mean"], dtype=np.float32)
        if sector_means.shape != (NUM_SECTORS, *weight.shape):
            raise RuntimeError(f"Invalid input-gate array shape in {path}: {sector_means.shape}")
        grand_mean = sector_means.mean(axis=0)
        for sector in range(NUM_SECTORS):
            source_mask = _source_group_masks(weight.shape[1], sector)[group]
            selected_weight = weight[:, source_mask]
            selected_delta = sector_means[sector][:, source_mask] - grand_mean[:, source_mask]
            keep = selected_weight != 0.0
            retained_weight = selected_weight[keep]
            weight_parts.append(np.abs(retained_weight))
            value_parts.append(selected_delta[keep])
            sign_parts.append(retained_weight > 0.0)
            seed_parts.append(np.full(retained_weight.shape, seed, dtype=np.int8))
    return (
        np.concatenate(weight_parts),
        np.concatenate(value_parts),
        np.concatenate(sign_parts),
        np.concatenate(seed_parts),
    )


def _load_input_summaries(data_root: Path) -> dict[str, CurveSummary]:
    """Load and reduce the two input-source groups."""

    rng = np.random.default_rng(20260916)
    result: dict[str, CurveSummary] = {}
    for group in INPUT_GROUPS:
        result[group] = _summarize_arrays(*_input_group_arrays(data_root, group), rng=rng)
    return result


def _load_recurrent_summaries(data_root: Path) -> dict[str, dict[str, CurveSummary]]:
    """Load and reduce Digit/Sector by TT/TR/RT/RR recurrent-gate records."""

    paths = sorted(data_root.glob(f"seed*/compact/{RECURRENT_RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten recurrent-gate seed files, found {len(paths)}.")
    parts = {
        variable: {
            group: {"weight": [], "value": [], "sign": [], "seed": []}
            for group in GROUP_NAMES
        }
        for variable in ("digit", "sector")
    }
    for seed, path in enumerate(paths, start=1):
        with np.load(path, allow_pickle=False) as arrays:
            for variable in ("digit", "sector"):
                records = _pooled_records(arrays, variable)
                for group in GROUP_NAMES:
                    frame = records[group]
                    parts[variable][group]["weight"].append(
                        frame["absW"].to_numpy(dtype=np.float32)
                    )
                    parts[variable][group]["value"].append(
                        frame["delta_of"].to_numpy(dtype=np.float32)
                    )
                    parts[variable][group]["sign"].append(
                        frame["signpos"].to_numpy(dtype=bool)
                    )
                    parts[variable][group]["seed"].append(
                        np.full(len(frame), seed, dtype=np.int8)
                    )
    rng = np.random.default_rng(20260916)
    return {
        variable: {
            group: _summarize_arrays(
                np.concatenate(values["weight"]),
                np.concatenate(values["value"]),
                np.concatenate(values["sign"]),
                np.concatenate(values["seed"]),
                rng=rng,
            )
            for group, values in groups.items()
        }
        for variable, groups in parts.items()
    }


def _curve_bounds(summaries: list[CurveSummary]) -> tuple[float, float]:
    """Return a zero-inclusive y range with padding around all displayed curves."""

    lower = 0.0
    upper = 0.0
    for summary in summaries:
        curves = summary["curves"]
        assert isinstance(curves, dict)
        for centers, means, sems in curves.values():
            del centers
            lower = min(lower, float(np.min(means - sems)))
            upper = max(upper, float(np.max(means + sems)))
    span = max(upper - lower, 0.02)
    return lower - 0.10 * span, upper + 0.10 * span


def _draw_curve_panel(
    axis: plt.Axes,
    summary: CurveSummary,
    *,
    show_x_label: bool,
    weight_superscript: str,
) -> None:
    """Draw low-opacity connection samples and the retained nine-bin curves."""

    samples = summary["samples"]
    curves = summary["curves"]
    overlap = summary["overlap"]
    assert isinstance(samples, dict) and isinstance(curves, dict) and isinstance(overlap, dict)
    for label, color in (("positive", POS_COLOR), ("negative", NEG_COLOR)):
        sample_x, sample_y = samples[label]
        axis.scatter(
            sample_x,
            sample_y,
            s=2.0,
            color=color,
            alpha=0.10,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        centers, means, sems = curves[label]
        axis.errorbar(
            centers,
            means,
            yerr=sems,
            color=color,
            linewidth=1.0,
            marker="o",
            markersize=2.2,
            capsize=1.2,
            capthick=0.6,
            zorder=3,
        )
    low = float(overlap["overlap_low"])
    high = float(overlap["overlap_high"])
    if np.isfinite(low) and np.isfinite(high) and high > low:
        axis.axvspan(low, high, color="#d9d9d9", alpha=0.55, zorder=0)
    axis.axhline(0.0, color="#777777", linewidth=0.55, linestyle="--", zorder=2)
    curve_max = max(float(np.max(curves[label][0])) for label in ("positive", "negative"))
    axis.set_xlim(0.0, 1.08 * curve_max)
    axis.set_xlabel(
        rf"$|w^{{\mathrm{{{weight_superscript}}}}}|$" if show_x_label else ""
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.2, width=0.6)


def _legend_handles(superscript: str) -> list[Line2D]:
    """Return shared sign-curve legend handles."""

    return [
        Line2D(
            [0],
            [0],
            color=POS_COLOR,
            marker="o",
            markersize=3,
            label=rf"$w^{{\mathrm{{{superscript}}}}}_{{+}}$",
        ),
        Line2D(
            [0],
            [0],
            color=NEG_COLOR,
            marker="o",
            markersize=3,
            label=rf"$w^{{\mathrm{{{superscript}}}}}_{{-}}$",
        ),
    ]


def render_input_figure(summaries: dict[str, CurveSummary], output_path: Path) -> None:
    """Render the two-panel ICLR-width input-gate supplementary figure."""

    with plt.rc_context(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))
        titles = {
            "sector0_sources": "Location-matched sources",
            "other_sources": "Other sources",
        }
        for axis, group in zip(axes, INPUT_GROUPS):
            _draw_curve_panel(
                axis,
                summaries[group],
                show_x_label=True,
                weight_superscript="in",
            )
            axis.set_title(titles[group])
            axis.set_ylim(*_curve_bounds([summaries[group]]))
        axes[0].set_ylabel(r"$\Delta g^{\mathrm{in}}$")
        figure.legend(
            handles=_legend_handles("in"),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            handlelength=1.4,
        )
        figure.subplots_adjust(left=0.11, right=0.995, bottom=0.20, top=0.78, wspace=0.27)
        label_y = max(axis.get_position().y1 for axis in axes) + 0.035
        for label, axis in zip("AB", axes):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.012,
                label_y,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)


def render_recurrent_figure(
    summaries: dict[str, dict[str, CurveSummary]], output_path: Path
) -> None:
    """Render Digit and Sector rows by four recurrent connection groups."""

    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(5.35, 3.25), sharey="row")
        variables = ("digit", "sector")
        group_titles = {
            "TT": r"$T\!\to\!T$",
            "TR": r"$T\!\to\!R$",
            "RT": r"$R\!\to\!T$",
            "RR": r"$R\!\to\!R$",
        }
        for row, variable in enumerate(variables):
            row_bounds = _curve_bounds([summaries[variable][group] for group in GROUP_NAMES])
            for column, group in enumerate(GROUP_NAMES):
                axis = axes[row, column]
                _draw_curve_panel(
                    axis,
                    summaries[variable][group],
                    show_x_label=row == 1,
                    weight_superscript="rec",
                )
                axis.set_ylim(*row_bounds)
                if row == 0:
                    axis.set_title(group_titles[group])
                if column == 0:
                    axis.set_ylabel(
                        ("Digit\n" if variable == "digit" else "Sector\n")
                        + r"$\Delta g^{\mathrm{rec}}$"
                    )
                if row == 0:
                    axis.tick_params(labelbottom=False)
        figure.legend(
            handles=_legend_handles("rec"),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.998),
            handlelength=1.4,
        )
        figure.subplots_adjust(
            left=0.10,
            right=0.995,
            bottom=0.14,
            top=0.86,
            hspace=0.18,
            wspace=0.26,
        )
        label_y = max(axis.get_position().y1 for axis in axes[0]) + 0.025
        for label, axis in zip("ABCD", axes[0]):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.012,
                label_y,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)


def main() -> None:
    """Load retained numeric outputs and render both requested PDFs."""

    args = parse_args()
    input_output = (
        args.output_dir / "Supple2_input_gate_sign_vs_mag_sector_delta_zoom_10seed_9sector.pdf"
    )
    recurrent_output = (
        args.output_dir / "Supple3_rec_gate_sign_vs_mag_digit_sector_delta_2x4_10seed.pdf"
    )
    render_input_figure(_load_input_summaries(args.input_root), input_output)
    render_recurrent_figure(_load_recurrent_summaries(args.recurrent_root), recurrent_output)
    print(f"Saved {input_output}")
    print(f"Saved {recurrent_output}")


if __name__ == "__main__":
    main()
