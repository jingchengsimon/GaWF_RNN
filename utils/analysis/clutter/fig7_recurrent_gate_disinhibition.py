"""Poster-style distillation of the recurrent-gate sign-vs-magnitude disinhibition result.

Collapses the connection-level TT/TR/RT/RR x sign(W) analysis (see
gawf_recurrent_gate_sign_vs_magnitude_disinhibition.py / ..._sector.py in utils.analysis) into one
grouped-bar figure: for each group, restricted to that group's own |W| overlap band (so + and -
are |W|-matched), mean open fraction +/- SEM for W>0 (red) vs W<0 (blue), digit and sector shown
as two side-by-side subplots. TT is highlighted (bold border + shaded background, other groups
dimmed) because it is the group where the sign gap is largest and most robust to the |W| +
context control; putting digit and sector side by side is meant to make "digit's TT gap is
large, sector's is much smaller" legible at a glance.

Re-derives the pooled data in memory from the already-collected NPZ caches. The poster uses only
the overlap-band gate means; it intentionally skips the optional OLS regression and never reruns
the expensive torch forward pass.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.multiseed_plotting import add_seed_points
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    DIGITS,
    GROUP_NAMES,
    NEG_COLOR,
    POS_COLOR,
    analyze,
)
from utils.analysis.clutter.fig7_recurrent_gate_sector_sign_magnitude import SECTORS

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DPI = 150
VARIABLES = ("digit", "sector")
CONTEXTS_BY_VARIABLE = {"digit": DIGITS, "sector": SECTORS}
HIGHLIGHT_GROUP = "TT"
DIM_ALPHA = 0.45  # non-highlighted groups' bar alpha
HIGHLIGHT_BG = "#fef9c3"


# --------------------------------------------------------------------------------------------
# |W|-matched (overlap-band-only) sign stats
# --------------------------------------------------------------------------------------------
def _condition_means(frame: pd.DataFrame, y_col: str) -> np.ndarray:
    """Return one equally weighted mean per condition from connection-level rows."""

    values = frame.groupby("context", sort=True)[y_col].mean().to_numpy(dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"No finite condition means for {y_col}.")
    return values


def print_overlap_bands(kind: str, overlap_stats: dict[str, dict]) -> None:
    print(f"\n=== {kind}: |W| overlap band per group ===")
    for g in GROUP_NAMES:
        low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
        if np.isfinite(low) and np.isfinite(high) and high > low:
            print(f"  {g}: [{low:.4f}, {high:.4f}]")
        else:
            print(f"  {g}: empty overlap band")


def overlap_band_sign_stats(
    pooled: dict[str, pd.DataFrame], overlap_stats: dict[str, dict], y_col: str = "of"
) -> dict[str, dict]:
    """Return sign statistics with equal condition weight inside the shared overlap band.

    ``y_col`` is "of" (default, raw open fraction) or "delta_of" (per-connection, cross-context
    demeaned; see collect_pooled_records in the digit disinhibition module). Rows are first
    averaged within each condition, then those condition means are averaged equally.
    """

    result: dict[str, dict] = {}
    for g in GROUP_NAMES:
        df = pooled[g]
        low, high = overlap_stats[g]["overlap_low"], overlap_stats[g]["overlap_high"]
        if not (np.isfinite(low) and np.isfinite(high) and high > low):
            result[g] = {s: {"mean": float("nan"), "sem": float("nan"), "n": 0} for s in ("+", "-")}
            continue
        in_band = df[(df["absW"] >= low) & (df["absW"] <= high)]
        stats: dict = {}
        for sign_label, signpos in (("+", 1), ("-", 0)):
            selected = in_band.loc[in_band["signpos"] == signpos]
            condition_means = _condition_means(selected, y_col)
            n = int(selected.shape[0])
            n_conditions = int(condition_means.size)
            mean = float(condition_means.mean())
            sem = (
                float(condition_means.std(ddof=1) / np.sqrt(n_conditions))
                if n_conditions > 1
                else 0.0
            )
            stats[sign_label] = {
                "mean": mean,
                "sem": sem,
                "n": n,
                "n_conditions": n_conditions,
            }
        result[g] = stats
    return result


def sign_p_value(reg: dict | None) -> float:
    """Prefer the conn-clustered robust p-value; fall back to the plain OLS p."""

    if reg is None:
        return float("nan")
    return reg.get("p_cluster", reg.get("p", float("nan")))


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------
# Tick labels spell out src->dst directionality directly (T->T, T->R, R->T, R->R) instead of
# the terse TT/TR/RT/RR group codes, so the axis no longer needs a "(src -> dst)" gloss.
GROUP_TICK_LABELS = {g: f"{g[0]}->{g[1]}" for g in GROUP_NAMES}


# Enlarged-font rc overrides, matching rnn_unit_gate_context_specificity.py's
# plot_marginalization_summary (produces 03_unit_gate_marginalization_1x3).
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

# Vertical layout only: every font bumped to 16 (separate from POSTER_RC so the horizontal
# layout's sizing is untouched).
POSTER_RC_VERTICAL = {
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _draw_poster_panel(
    ax: plt.Axes,
    stats: dict[str, dict],
    title: str,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_seed_points: bool = True,
) -> None:
    """Draw one Digit/Sector bar panel (shared by the horizontal and vertical layouts)."""

    x = np.arange(len(GROUP_NAMES))
    width = 0.32
    for gi, g in enumerate(GROUP_NAMES):
        is_highlight = g == HIGHLIGHT_GROUP
        alpha = 1.0 if is_highlight else DIM_ALPHA
        edge_color = "black" if is_highlight else "0.5"
        edge_lw = 2.2 if is_highlight else 0.6
        if is_highlight:
            ax.axvspan(x[gi] - 0.5, x[gi] + 0.5, color=HIGHLIGHT_BG, zorder=0)

        pos, neg = stats[g]["+"], stats[g]["-"]
        ax.bar(x[gi] - width / 2, pos["mean"], width, yerr=pos["sem"], capsize=3,
               color=POS_COLOR, alpha=alpha, edgecolor=edge_color, linewidth=edge_lw, zorder=3)
        ax.bar(x[gi] + width / 2, neg["mean"], width, yerr=neg["sem"], capsize=3,
               color=NEG_COLOR, alpha=alpha, edgecolor=edge_color, linewidth=edge_lw, zorder=3)
        for x_pos, values in (
            (x[gi] - width / 2, pos.get("values")),
            (x[gi] + width / 2, neg.get("values")),
        ):
            if values is not None:
                add_seed_points(
                    ax,
                    np.asarray([x_pos]),
                    np.asarray(values, dtype=np.float64)[:, None],
                    bar_width=width,
                    show=show_seed_points,
                    rng=np.random.default_rng(gi),
                )

    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_TICK_LABELS[g] for g in GROUP_NAMES])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Group")
    if show_ylabel:
        ax.set_ylabel("Gate open fraction")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=POS_COLOR,
               markeredgecolor="none", markersize=10, label="W > 0 (+)"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=NEG_COLOR,
               markeredgecolor="none", markersize=10, label="W < 0 (-)"),
    ]


def _save_png_and_pdf(fig: plt.Figure, fig_path: Path, dpi: int) -> None:
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {fig_path}")
    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    plt.close(fig)


def render_poster(
    stats_by_variable: dict[str, dict[str, dict]],
    fig_path: Path, dpi: int = DPI,
    *,
    show_seed_points: bool = True,
) -> None:
    """Horizontal layout: Digit and Sector side by side, sharing the y-axis."""

    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
        for ax, variable in zip(axes, VARIABLES):
            _draw_poster_panel(
                ax, stats_by_variable[variable], variable.capitalize(),
                show_ylabel=(ax is axes[0]),
                show_seed_points=show_seed_points,
            )
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.98))
        fig.subplots_adjust(top=0.87, bottom=0.11, wspace=0.08)
        _save_png_and_pdf(fig, fig_path, dpi)


def render_poster_vertical(
    stats_by_variable: dict[str, dict[str, dict]],
    fig_path: Path, dpi: int = DPI,
) -> None:
    """Vertical layout: Digit on top, Sector below, sharing the x-axis (each keeps its own
    y-axis -- ticks/gridlines/label are not shared, per the horizontal version's request).
    Uses POSTER_RC_VERTICAL (all fonts at 16) instead of the horizontal layout's POSTER_RC."""

    with plt.rc_context(POSTER_RC_VERTICAL):
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.6), sharex=True)
        for row, (ax, variable) in enumerate(zip(axes, VARIABLES)):
            _draw_poster_panel(ax, stats_by_variable[variable], variable.capitalize())
            if row < len(axes) - 1:
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), visible=False)

        # Legend row nudged down by 0.5 "character height" (taken as half the legend font size
        # in points, converted to a figure-fraction offset via this figure's actual height).
        fig_height_in = fig.get_size_inches()[1]
        char_height_in = POSTER_RC_VERTICAL["legend.fontsize"] / 72.0
        legend_y = 0.98 - 0.5 * char_height_in / fig_height_in
        fig.legend(handles=_legend_handles(), loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, legend_y))
        # All text at 16pt needs more headroom than POSTER_RC's mixed 13/15pt did, or the
        # legend (now taller) collides with the top ("Digit") panel's title.
        fig.subplots_adjust(top=0.85, bottom=0.07, hspace=0.22)
        _save_png_and_pdf(fig, fig_path, dpi)


def parse_args() -> argparse.Namespace:
    """Parse isolated cache locations and an optional 10-seed output destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dirs", nargs="+", type=Path, default=None)
    parser.add_argument("--fig_dir", type=Path, default=None)
    parser.add_argument("--output_stem", default="recurrent_gate_disinhibition_poster")
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each 10-seed bar.",
    )
    return parser.parse_args()


def _cross_seed_stats(records: list[dict[str, dict]]) -> dict[str, dict]:
    """Collapse per-seed overlap-band means to cross-seed mean ± SEM."""

    result: dict[str, dict] = {}
    for group in GROUP_NAMES:
        result[group] = {}
        for sign in ("+", "-"):
            values = np.asarray(
                [record[group][sign]["mean"] for record in records], dtype=np.float64
            )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Non-finite per-seed Fig7 statistic for {group}/{sign}")
            result[group][sign] = {
                "mean": float(values.mean()),
                "sem": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                "n": int(values.size),
                "values": values.tolist(),
            }
    return result


def exact_sign_flip_p(values: np.ndarray) -> float:
    """Return the exact two-sided sign-flip p-value for paired seed-level values."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Sign-flip values must be a finite one-dimensional array of size >= 2.")
    observed = abs(float(values.mean()))
    null_abs_means = np.asarray(
        [abs(float(np.mean(values * signs))) for signs in product((-1.0, 1.0), repeat=len(values))]
    )
    return float(np.mean(null_abs_means >= observed - np.finfo(np.float64).eps))


def seed_level_gap_p(records: list[dict[str, dict]], group: str) -> float:
    """Return the exact two-sided sign-flip p-value for paired per-seed sign gaps."""

    gaps = np.asarray(
        [record[group]["+"]["mean"] - record[group]["-"]["mean"] for record in records],
        dtype=np.float64,
    )
    return exact_sign_flip_p(gaps)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    """Render a single-seed or cross-seed poster from structured cache data."""

    args = parse_args()
    if args.cache_dirs is not None and len(args.cache_dirs) < 2:
        raise ValueError("--cache_dirs requires at least two per-seed cache directories")
    stats_by_variable: dict[str, dict[str, dict]] = {}
    gap_by_variable: dict[str, dict[str, float]] = {}
    p_by_variable: dict[str, dict[str, float]] = {}

    for kind in VARIABLES:
        if args.cache_dirs is None:
            pooled, _meta, overlap_stats, _regs_full, regs_clean = analyze(
                CONTEXTS_BY_VARIABLE[kind], kind, run_regressions=False
            )
            print_overlap_bands(kind, overlap_stats)
            stats = overlap_band_sign_stats(pooled, overlap_stats)
            stats_by_variable[kind] = stats
            p_by_variable[kind] = {g: float("nan") for g in GROUP_NAMES}
        else:
            per_seed_stats: list[dict[str, dict]] = []
            for cache_dir in args.cache_dirs:
                pooled, _meta, overlap_stats, _regs_full, _regs_clean = analyze(
                    CONTEXTS_BY_VARIABLE[kind], kind, cache_dir=cache_dir, run_regressions=False
                )
                per_seed_stats.append(overlap_band_sign_stats(pooled, overlap_stats))
            stats = _cross_seed_stats(per_seed_stats)
            stats_by_variable[kind] = stats
            p_by_variable[kind] = {
                g: seed_level_gap_p(per_seed_stats, g) for g in GROUP_NAMES
            }
        gap_by_variable[kind] = {
            g: stats[g]["+"]["mean"] - stats[g]["-"]["mean"] for g in GROUP_NAMES
        }

    print("\n=== summary (|W|-matched overlap band, controlling for |W| + context) ===")
    header = (
        f"{'kind':>7} {'group':>6} {'n_+':>8} {'n_-':>8} {'of+':>8} {'of-':>8} "
        f"{'gap':>8} {'p (paired seeds)':>22}"
    )
    print(header)
    for kind in VARIABLES:
        stats = stats_by_variable[kind]
        for g in GROUP_NAMES:
            pos, neg = stats[g]["+"], stats[g]["-"]
            print(f"{kind:>7} {g:>6} {pos['n']:>8} {neg['n']:>8} {pos['mean']:>8.4f} "
                  f"{neg['mean']:>8.4f} {gap_by_variable[kind][g]:>8.4f} "
                  f"{p_by_variable[kind][g]:>22.3e}")

    fig_dir = (
        args.fig_dir
        if args.fig_dir is not None
        else output_dir(CATEGORY, SCRIPT_NAME, "figs")
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    render_poster(
        stats_by_variable,
        fig_dir / f"{args.output_stem}.png",
        show_seed_points=args.show_seed_points,
    )
    if args.cache_dirs is None:
        render_poster_vertical(stats_by_variable, fig_dir / f"{args.output_stem}_vertical.png")
    if args.cache_dirs is not None:
        summary_path = fig_dir / f"{args.output_stem}_metadata.json"
        summary_path.write_text(
            json.dumps(
                {
                    "n_seeds": len(args.cache_dirs),
                    "cache_dirs": [str(path.resolve()) for path in args.cache_dirs],
                    "aggregation": "per-seed overlap-band means; bars show cross-seed mean +/- SEM",
                    "test": "exact two-sided sign-flip test of paired per-seed (+ minus -) gaps",
                    "seed_level_gap_p": p_by_variable,
                    "seed42_included": False,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
