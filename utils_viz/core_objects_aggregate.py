"""Regenerate the poster-style four-core-object aggregate figure from saved NPZ summaries.

Inputs are the four unified variance-decomposition NPZ files.  The development PNG remains in
``results/anal_figs/D_variance_decomposition/`` and the official PDF is written to the configured
publication-figure directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.publication_paths import publication_figures_dir
from utils_anal.anal_paths import output_dir
from utils_anal.run_unified_variance_decomposition import _plot_compact_aggregate
from utils_anal.variance_decomposition import CM_FACTORS, RepeatedDecomposition


OBJECTS = ("input_gate", "recurrent_gate", "encoder_activation", "hidden_state")


def parse_args() -> argparse.Namespace:
    """Parse saved-data and figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=output_dir("D_variance_decomposition", "unified", "data"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=output_dir("D_variance_decomposition", "unified", "figs"),
    )
    parser.add_argument("--publication_fig_dir", type=Path, default=None)
    return parser.parse_args()


def load_saved_results(data_dir: Path) -> dict[str, RepeatedDecomposition]:
    """Load the repeated condition-mean aggregate fractions needed by the compact figure."""

    results: dict[str, RepeatedDecomposition] = {}
    for object_name in OBJECTS:
        path = data_dir / f"{object_name}_per_unit_distributions.npz"
        with np.load(path, allow_pickle=False) as arrays:
            aggregate_cm = {
                factor: np.asarray(arrays[f"aggregate_cm_{factor}"], dtype=np.float64)
                for factor in CM_FACTORS
            }
        results[object_name] = RepeatedDecomposition(
            aggregate_cm=aggregate_cm,
            aggregate_trial={},
            per_unit_cm={},
            per_unit_trial={},
            unweighted_per_unit_mean_cm={},
            unweighted_per_unit_mean_trial={},
            consistency={},
        )
    return results


def main() -> None:
    """Load the saved summaries and regenerate the official compact aggregate figure."""

    args = parse_args()
    publication_dir = publication_figures_dir(args.publication_fig_dir, create=True)
    destination = _plot_compact_aggregate(
        args.figure_dir,
        load_saved_results(args.data_dir),
        publication_dir,
    )
    print(f"Saved {destination}")
    if publication_dir is not None:
        print(f"Saved {publication_dir / 'core_objects_aggregate_2x2.pdf'}")
    else:
        print("Skipped publication PDF: no publication figure directory is configured")


if __name__ == "__main__":
    main()
