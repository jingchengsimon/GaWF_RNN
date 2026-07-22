"""Plot the three remaining Part-2 top-10% SOURCE-gate distribution families.

Input is ``remaining_top10_gate_distributions.npz`` from the matching analysis module. Outputs
are nine input/sector, ten input/digit, and ten recurrent/digit independent PNG density figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_viz.gawf_recurrent_sector_relevance_distributions import (
    plot_context_distribution,
)


CELL_SPECS = {
    "input_sector": ("input", "sector", 9),
    "input_digit": ("input", "digit", 10),
    "recurrent_digit": ("recurrent", "digit", 10),
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir("E_relevance_alignment", "gawf_remaining_relevance_distributions", "data")
            / "remaining_top10_gate_distributions.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(
            output_dir("E_relevance_alignment", "gawf_remaining_relevance_distributions", "figs")
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    """Load saved histograms and render all 29 context figures."""

    args = parse_args()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.data).expanduser().resolve(), allow_pickle=False) as data:
        bin_edges = np.asarray(data["bin_edges"], dtype=np.float64)
        widths = np.diff(bin_edges)
        for cell, (gate_name, factor, levels) in CELL_SPECS.items():
            hist_counts = np.asarray(data[f"{cell}_hist_counts"], dtype=np.int64)
            group_mean = np.asarray(data[f"{cell}_group_mean"], dtype=np.float64)
            group_count = np.asarray(data[f"{cell}_group_count"], dtype=np.int64)
            context_d = np.asarray(data[f"{cell}_context_cohens_d"], dtype=np.float64)
            relevant_mask = np.asarray(data[f"{cell}_relevant_mask"], dtype=bool)
            eligible = np.asarray(data[f"{cell}_eligible_mask"], dtype=bool)
            all_density = hist_counts / (group_count[..., None] * widths[None, None, :])
            density_limit = 1.08 * float(all_density.max())
            for context in range(levels):
                relevant_units = int(relevant_mask[context].sum())
                plot_context_distribution(
                    gate_name,
                    factor,
                    context,
                    bin_edges,
                    hist_counts[context],
                    group_mean[context],
                    group_count[context],
                    float(context_d[context]),
                    relevant_units,
                    int(eligible.sum() - relevant_units),
                    fig_dir,
                    args.dpi,
                    density_limit=density_limit,
                )


if __name__ == "__main__":
    main()
