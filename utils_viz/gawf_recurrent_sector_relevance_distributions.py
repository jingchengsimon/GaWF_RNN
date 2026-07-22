"""Plot top-10% versus remaining recurrent SOURCE-gate distributions by sector.

Input is ``recurrent_sector_top10_gate_distributions.npz`` from the matching analysis module.
Output is one independent PNG density figure for each of the nine sectors.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#3b82f6", "#f97316")
GROUP_LABELS = ("Top 10% units", "Remaining 90% units")


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_sector_relevance_distributions",
                "data",
            )
            / "recurrent_sector_top10_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_sector_relevance_distributions",
                "figs",
            )
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {path}")


def plot_context_distribution(
    gate_name: str,
    factor: str,
    context: int,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    context_d: float,
    relevant_units: int,
    remaining_units: int,
    fig_dir: Path,
    dpi: int,
    *,
    density_limit: float,
) -> Path:
    """Write one context's normalized SOURCE-gate density comparison."""

    if gate_name not in ("input", "recurrent"):
        raise ValueError("gate_name must be 'input' or 'recurrent'")
    if factor not in ("sector", "digit"):
        raise ValueError("factor must be 'sector' or 'digit'")

    widths = np.diff(bin_edges)
    density = hist_counts / (group_count[:, None] * widths[None, :])
    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    for group_index, (label, color, unit_count) in enumerate(
        zip(GROUP_LABELS, COLORS, (relevant_units, remaining_units))
    ):
        axis.stairs(
            density[group_index],
            bin_edges,
            color=color,
            linewidth=1.7,
            fill=True,
            alpha=0.22,
            label=f"{label} ({unit_count} units; mean={group_mean[group_index]:.3f})",
        )
        axis.axvline(
            group_mean[group_index],
            color=color,
            linewidth=1.2,
            linestyle="--",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, density_limit)
    axis.set_xlabel(f"Raw {gate_name} gate (mean over 256 destinations)")
    axis.set_ylabel("Density")
    context_label = factor.capitalize()
    axis.set_title(
        f"{context_label} {context} — {gate_name} SOURCE-gate distribution\n"
        f"Top 10% versus remaining 90% eligible units; Cohen's d = {context_d:.3f}"
    )
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    destination = fig_dir / f"{gate_name}_{factor}_{context}_top10_vs_remaining_distribution.png"
    _save(fig, destination, dpi)
    return destination


def plot_sector_distribution(
    sector: int,
    bin_edges: np.ndarray,
    hist_counts: np.ndarray,
    group_mean: np.ndarray,
    group_count: np.ndarray,
    sector_d: float,
    relevant_units: int,
    remaining_units: int,
    fig_dir: Path,
    dpi: int,
    *,
    density_limit: float,
) -> Path:
    """Write one sector's normalized recurrent-gate density comparison."""

    return plot_context_distribution(
        "recurrent",
        "sector",
        sector,
        bin_edges,
        hist_counts,
        group_mean,
        group_count,
        sector_d,
        relevant_units,
        remaining_units,
        fig_dir,
        dpi,
        density_limit=density_limit,
    )


def main() -> None:
    """Load saved histograms and render all nine sector figures."""

    args = parse_args()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.data).expanduser().resolve(), allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        hist_counts = np.asarray(data["hist_counts"], dtype=np.int64)
        group_mean = np.asarray(data["group_mean"], dtype=np.float64)
        group_count = np.asarray(data["group_count"], dtype=np.int64)
        sector_d = np.asarray(data["sector_cohens_d"], dtype=np.float64)
        relevant_mask = np.asarray(data["relevant_mask"], dtype=bool)
        eligible = np.asarray(data["eligible_mask"], dtype=bool)
    widths = np.diff(bin_edges)
    all_density = hist_counts / (group_count[..., None] * widths[None, None, :])
    density_limit = 1.08 * float(all_density.max())
    for sector in range(9):
        relevant_units = int(relevant_mask[sector].sum())
        plot_sector_distribution(
            sector,
            bin_edges,
            hist_counts[sector],
            group_mean[sector],
            group_count[sector],
            float(sector_d[sector]),
            relevant_units,
            int(eligible.sum() - relevant_units),
            fig_dir,
            args.dpi,
            density_limit=density_limit,
        )


if __name__ == "__main__":
    main()
