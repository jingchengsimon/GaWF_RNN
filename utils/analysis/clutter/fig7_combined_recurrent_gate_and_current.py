"""Render the combined ICLR recurrent-gate and gate-dependent-current Figure 7.

The top row contains sign-split and balanced Digit and Sector delta-g bar panels plus the
stacked Digit/Sector T-to-T sign-magnitude curves. The bottom row contains the connection-
normalized Digit and Sector gate-dependent-current bars. All panels are regenerated from
the compact Figure 7 arrays and Figure 8 long tables; outputs are one development PNG and
one 5.5-inch-wide manuscript PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils.analysis.anal_paths import output_dir  # noqa: E402
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (  # noqa: E402
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    binned_mean_curve,
    compute_overlap_band,
    quantile_bin_edges,
)
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import (  # noqa: E402
    _condition_means,
    overlap_band_sign_stats,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_NAME = Path(__file__).stem
FIG8_ROOT = PROJECT_ROOT / "results/save_data/fig8/recurrent_current/connection"
OUTPUT_STEM = "Fig7_recurrent_gate_disinhibition_and_current_2x3_10seed"
DATA_ROOT = PROJECT_ROOT / "results/data/analysis/fig7_recurrent_gate_10seed"
VARIABLES = ("digit", "sector")
CURVE_GROUP = "TT"
CURRENT_COLORS = {"excitatory": "#c53030", "inhibitory": "#2b6cb0", "total": "#1a202c"}
POSTER_RC = {
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _compact_paths(data_root: Path) -> list[Path]:
    """Return the frozen ten-seed recurrent-gate compact files."""

    return sorted(data_root.glob("seed*/compact/recurrent_gate_condition_means.npz"))


def _group_masks(tuned: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return source-to-destination tuned/remainder masks."""

    remainder = ~tuned
    return {
        "TT": (tuned, tuned),
        "TR": (tuned, remainder),
        "RT": (remainder, tuned),
        "RR": (remainder, remainder),
    }


def _pooled_records(arrays: np.lib.npyio.NpzFile, family: str) -> dict[str, pd.DataFrame]:
    """Build plotting records from one frozen compact archive."""

    weight = np.asarray(arrays["weight"], dtype=np.float64)
    means = np.asarray(arrays[f"{family}_gate_mean"], dtype=np.float64)
    tuned = np.asarray(arrays[f"{family}_tuned"], dtype=bool)
    grand = means.mean(axis=0)
    hidden_size = weight.shape[0]
    rows: dict[str, list[pd.DataFrame]] = {group: [] for group in GROUP_NAMES}
    for context in range(means.shape[0]):
        for group, (source, destination) in _group_masks(tuned[context]).items():
            mask = destination[:, None] & source[None, :] & (weight != 0.0)
            dst, src = np.where(mask)
            values = weight[dst, src]
            rows[group].append(
                pd.DataFrame(
                    {
                        "absW": np.abs(values),
                        "delta_of": means[context, dst, src] - grand[dst, src],
                        "signpos": (values > 0.0).astype(np.int64),
                        "context": context,
                        "conn": dst.astype(np.int64) * hidden_size + src,
                    }
                )
            )
    return {group: pd.concat(records, ignore_index=True) for group, records in rows.items()}


def _cross_seed_stats(records: list[dict[str, dict]]) -> dict[str, dict]:
    """Aggregate per-seed overlap-band sign means."""

    result: dict[str, dict] = {}
    for group in GROUP_NAMES:
        result[group] = {}
        for sign in ("+", "-"):
            values = np.asarray([record[group][sign]["mean"] for record in records])
            result[group][sign] = {
                "mean": float(values.mean()),
                "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
                "values": values.tolist(),
            }
    return result


def _load_inputs(
    data_root: Path,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Load the frozen ten-seed gate inputs without importing the GPU collector."""

    paths = _compact_paths(data_root)
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten compact files in {data_root}, found {len(paths)}.")
    per_seed: dict[str, list[dict[str, dict]]] = {family: [] for family in VARIABLES}
    pooled: dict[str, dict[str, list[pd.DataFrame]]] = {
        family: {group: [] for group in GROUP_NAMES} for family in VARIABLES
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for family in VARIABLES:
                records = _pooled_records(arrays, family)
                overlap = {
                    group: compute_overlap_band(
                        frame.loc[frame["signpos"] == 1, "absW"].to_numpy(),
                        frame.loc[frame["signpos"] == 0, "absW"].to_numpy(),
                    )
                    for group, frame in records.items()
                }
                per_seed[family].append(
                    overlap_band_sign_stats(records, overlap, y_col="delta_of")
                )
                for group in GROUP_NAMES:
                    pooled[family][group].append(records[group])
    stats = {family: _cross_seed_stats(per_seed[family]) for family in VARIABLES}
    merged = {
        family: {
            group: pd.concat(frames, ignore_index=True) for group, frames in groups.items()
        }
        for family, groups in pooled.items()
    }
    return stats, {}, {}, merged


def _draw_supple3_panel(
    axis: plt.Axes,
    values: pd.DataFrame,
    title: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Render one binned delta-g curve panel from frozen records."""

    edges = quantile_bin_edges(values["absW"].to_numpy(dtype=np.float64))
    plotted = []
    for sign, color in ((1, POS_COLOR), (0, NEG_COLOR)):
        frame = values.loc[values["signpos"] == sign]
        center, mean, sem, count = binned_mean_curve(
            frame["absW"].to_numpy(), frame["delta_of"].to_numpy(), edges
        )
        valid = count > 0
        axis.errorbar(
            center[valid],
            mean[valid],
            yerr=sem[valid],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            capsize=2.5,
        )
        plotted.append((center[valid], mean[valid]))
    intercepts = []
    for center, mean in plotted:
        slope = (mean[1] - mean[0]) / (center[1] - center[0])
        intercepts.append(float(mean[0] - slope * center[0]))
    baseline = float(np.mean(intercepts))
    axis.set_xlim(0.0, 1.08 * max(float(center.max()) for center, _mean in plotted))
    axis.set_ylim(baseline - 0.30, baseline + 0.30)
    tick_start = np.ceil((baseline - 0.30) * 10.0 - 1e-12) / 10.0
    axis.set_yticks(np.arange(tick_start, baseline + 0.30 + 1e-12, 0.1))
    axis.set_xlabel(r"$|w^{\mathrm{rec}}|$" if show_xlabel else "")
    axis.set_ylabel(r"$\Delta g^{\mathrm{rec}}$" if show_ylabel else "")
    axis.set_title(title)
    axis.tick_params(axis="x", labelbottom=show_xlabel)
    axis.spines[["top", "right"]].set_visible(False)


def parse_args() -> argparse.Namespace:
    """Parse the retained Figure 7/Figure 8 inputs and combined-figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_only", action="store_true", help="Write only the requested PDF.")
    parser.add_argument("--top_row_only", action="store_true", help="Omit current panels.")
    parser.add_argument("--fig7_data_root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--digit_long",
        type=Path,
        default=FIG8_ROOT / "digit/net_recurrent_current_connection_10seed_long.csv",
    )
    parser.add_argument(
        "--sector_long",
        type=Path,
        default=FIG8_ROOT / "sector/net_recurrent_current_connection_10seed_long.csv",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=(
            PROJECT_ROOT / "results/figs/E_relevance_alignment" / f"{OUTPUT_STEM}.png"
        ),
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=PROJECT_ROOT / "results/save/iclr_figs" / f"{OUTPUT_STEM}.pdf",
    )
    return parser.parse_args()


def _compact_axis(axis: plt.Axes, *, hide_y_tick_labels: bool = False) -> None:
    """Restyle an imported standalone panel for the 5.5-inch two-row layout."""

    axis.title.set_fontsize(8.2)
    axis.xaxis.label.set_fontsize(7.5)
    axis.yaxis.label.set_fontsize(7.5)
    axis.tick_params(labelsize=6.7, length=2.5, width=0.7)
    if hide_y_tick_labels:
        axis.tick_params(axis="y", labelleft=False)
    for label in axis.get_xticklabels():
        label.set_rotation(0)
    for text in list(axis.texts):
        if text.get_text() == "*":
            text.remove()


def _load_balanced_gate(data_root: Path) -> dict:
    """Average condition means equally within each group and training seed."""

    paths = _compact_paths(data_root)
    if len(paths) != 10:
        raise ValueError(f"Expected ten compact seed files, found {len(paths)}")
    values = {family: [] for family in VARIABLES}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for family in VARIABLES:
                records = _pooled_records(arrays, family)
                values[family].append(
                    [
                        _condition_means(records[group], "delta_of").mean()
                        for group in GROUP_NAMES
                    ]
                )
    result = {family: np.asarray(rows) for family, rows in values.items()}
    if not all(np.isfinite(rows).all() for rows in result.values()):
        raise ValueError("Non-finite balanced delta gate")
    return result


def _draw_balanced_gate(
    axis: plt.Axes, values: np.ndarray, title: str, sign_stats: dict
) -> None:
    """Draw retained sign-split bars alongside balanced means, SEM, and zero tests."""

    positions = np.arange(len(GROUP_NAMES))
    width = 0.24
    series = [
        np.asarray([sign_stats[group][sign]["values"] for group in GROUP_NAMES]).T
        for sign in ("+", "-")
    ] + [values]
    for offset, seed_values, color in zip(
        (-width, 0.0, width), series, (POS_COLOR, NEG_COLOR, CURRENT_COLORS["total"])
    ):
        means = seed_values.mean(axis=0)
        sem = seed_values.std(axis=0, ddof=1) / np.sqrt(seed_values.shape[0])
        axis.bar(
            positions + offset,
            means,
            width,
            yerr=sem,
            capsize=2,
            color=color,
            edgecolor="none",
            zorder=3,
        )
        add_seed_points(axis, positions + offset, seed_values, bar_width=width)
    axis.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    axis.set_xticks(positions, ("T→T", "T→R", "R→T", "R→R"))
    axis.set_ylim(-0.46, 0.1)
    axis.set_yticks((0.1, 0.0, -0.2, -0.4))
    axis.set_title(title)
    axis.set_xlabel("Group")
    axis.spines[["top", "right"]].set_visible(False)


def render(
    fig7_pooled: dict,
    fig8_reports: dict,
    output_png: Path | None,
    output_pdf: Path,
    balanced_gate: dict,
    sign_stats: dict,
    top_row_only: bool = False,
) -> None:
    """Render delta-g, connection-current, and weight-magnitude panels in a 2-by-3 grid."""

    if bottom_axes_requested := (not top_row_only):
        from utils.analysis.clutter.fig6_net_recurrent_current import _plot_fig8_bars

    style = dict(POSTER_RC)
    style.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    with plt.rc_context(style):
        figure = plt.figure(figsize=(5.5, 2.35 if top_row_only else 2.8))
        if top_row_only:
            grid = figure.add_gridspec(
                1,
                3,
                left=0.105,
                right=0.99,
                bottom=0.18,
                top=0.82,
                wspace=0.60,
            )
            top_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
            curve_grid = grid[0, 2].subgridspec(2, 1, hspace=1.05)
            top_curve_axis = figure.add_subplot(curve_grid[0, 0])
            curve_axes = (
                top_curve_axis,
                figure.add_subplot(curve_grid[1, 0], sharex=top_curve_axis),
            )
            bottom_axes = ()
        else:
            grid = figure.add_gridspec(
                2,
                5,
                left=0.105,
                right=0.995,
                bottom=0.13,
                top=0.86,
                hspace=0.42,
                wspace=0.0,
                width_ratios=(1.33, 0.12, 1.33, 0.50, 1.0),
            )
            top_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 2]))
            bottom_axes = (
                figure.add_subplot(grid[1, 0]),
                figure.add_subplot(grid[1, 2]),
            )
            top_curve_axis = figure.add_subplot(grid[0, 4])
            curve_axes = (
                top_curve_axis,
                figure.add_subplot(grid[1, 4], sharex=top_curve_axis),
            )

        for index, (axis, family) in enumerate(zip(top_axes, VARIABLES)):
            _draw_balanced_gate(
                axis, balanced_gate[family], family.capitalize(), sign_stats[family]
            )
            axis.set_title(family.capitalize(), pad=1)
            if index == 0:
                axis.set_ylabel(r"$\Delta g^{\mathrm{rec}}$")
            if not top_row_only:
                axis.set_xlabel("")
                axis.tick_params(axis="x", labelbottom=False)
            _compact_axis(axis, hide_y_tick_labels=index > 0)

        for row, (axis, family) in enumerate(zip(curve_axes, VARIABLES)):
            _draw_supple3_panel(
                axis,
                fig7_pooled[family][CURVE_GROUP],
                f"{family.capitalize()} T→T",
                show_ylabel=True,
                show_xlabel=row == 1,
            )
            axis.set_title(
                f"{family.capitalize()} T→T",
                y=1.0 if row == 0 else 0.94,
                pad=1,
            )
            for line in axis.lines:
                if line.get_marker() == "o":
                    line.set_markersize(2.8)
                if line.get_linestyle() not in ("", "None", "none"):
                    line.set_linewidth(1.2)
            _compact_axis(axis)
        curve_axes[0].set_ylim(-0.55, 0.05)
        curve_axes[0].set_yticks((-0.5, -0.25, 0.0))
        curve_axes[1].set_ylim(-0.3, 0.3)
        curve_axes[1].set_yticks((-0.3, 0.0, 0.3))
        bar_limits = (-0.02, 0.08)
        bar_ticks = (-0.02, 0.0, 0.04, 0.08)
        for index, (axis, family) in enumerate(zip(bottom_axes, VARIABLES)):
            _plot_fig8_bars(
                axis,
                fig8_reports[family],
                family,
                "connection",
                bar_limits,
                bar_ticks,
                "%.2f",
                show_legend=False,
            )
            axis.set_title("")
            axis.set_xlabel("Group")
            if index > 0:
                axis.set_ylabel("")
            _compact_axis(axis, hide_y_tick_labels=index > 0)

        figure.legend(
            handles=(
                Patch(
                    facecolor=POS_COLOR,
                    edgecolor="none",
                    label=r"$w^{\mathrm{rec}}_{+}$",
                ),
                Patch(
                    facecolor=NEG_COLOR,
                    edgecolor="none",
                    label=r"$w^{\mathrm{rec}}_{-}$",
                ),
                Patch(
                    facecolor=CURRENT_COLORS["total"],
                    edgecolor="none",
                    label="All",
                ),
            ),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=1.0,
            columnspacing=1.2,
        )
        label_axes = (top_axes[0], curve_axes[0]) if top_row_only else (
            top_axes[0],
            curve_axes[0],
            bottom_axes[0],
        )
        for label, axis in zip("ABC", label_axes):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.025,
                position.y1 + 0.02,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        if output_png is not None:
            output_png.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_png, dpi=300)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf)
        plt.close(figure)


def main() -> None:
    """Load retained structured results and render the combined ICLR figure."""

    args = parse_args()
    stats, _gaps, _tests, pooled = _load_inputs(args.fig7_data_root)
    reports = {}
    if not args.top_row_only:
        from utils.analysis.clutter.fig6_net_recurrent_current import _load_fig8_long

        reports = {
            "digit": _load_fig8_long(args.digit_long, "digit"),
            "sector": _load_fig8_long(args.sector_long, "sector"),
        }
    render(
        pooled,
        reports,
        None if args.pdf_only else args.output_png,
        args.output_pdf,
        _load_balanced_gate(args.fig7_data_root),
        stats,
        args.top_row_only,
    )
    print(f"Saved {args.output_pdf}")
    if args.pdf_only:
        return
    data_dir = output_dir("E_relevance_alignment", SCRIPT_NAME, "data")
    output_dir("E_relevance_alignment", SCRIPT_NAME, "figs")
    (data_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(
            {
                "training_seeds": 10,
                "layout": {
                    "top": ["Digit delta-g", "Sector delta-g", "Digit TT curve"],
                    "bottom": [
                        "Digit gate-dependent current",
                        "Sector gate-dependent current",
                        "Sector TT curve",
                    ],
                },
                "current_normalization": "per nonzero recurrent connection",
                "fig7_data_root": str(args.fig7_data_root),
                "digit_current_long": str(args.digit_long),
                "sector_current_long": str(args.sector_long),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
