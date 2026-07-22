"""Plot the seven requested GaWF gate-distribution figures from saved analysis arrays."""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", default=str(output_dir("A_raw_gate", "gawf_gate_distribution", "data"))
    )
    parser.add_argument(
        "--digit_data_dir",
        default=str(output_dir("B_gate_by_context", "gawf_gate_digit_distribution", "data")),
    )
    parser.add_argument(
        "--raw_dir", default=str(output_dir("A_raw_gate", "gawf_gate_distribution", "figs"))
    )
    parser.add_argument(
        "--context_dir",
        default=str(output_dir("B_gate_by_context", "gawf_gate_distribution", "figs")),
    )
    parser.add_argument(
        "--delta_dir", default=str(output_dir("C_delta_gate", "gawf_gate_distribution", "figs"))
    )
    parser.add_argument(
        "--relevance_dir",
        default=str(output_dir("E_relevance_alignment", "gawf_gate_distribution", "figs")),
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="Deprecated compatibility override; sends every figure to one directory.",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    return parser.parse_args()


def _density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    total = float(np.asarray(counts).sum())
    return np.asarray(counts, dtype=np.float64) / (total * widths)


def _finish(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {path}")


def _gate_axes(
    axes: np.ndarray,
    edges: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_all"]
        stats = metadata["distribution"][kind]
        axis.plot(centers, _density(counts, edges), color="#2b6cb0", linewidth=1.5)
        axis.axvline(0.5, color="black", linestyle="--", label="0.5")
        axis.axvline(stats["mean"], color="#d53f8c", label=f"mean={stats['mean']:.4f}")
        axis.axvline(
            stats["median"], color="#38a169", linestyle=":", label=f"median={stats['median']:.4f}"
        )
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
        axis.legend(fontsize=8)


def main() -> None:
    """Read analysis outputs and save seven independent figures."""

    args = parse_args()
    if args.save_dir:
        args.raw_dir = args.context_dir = args.delta_dir = args.relevance_dir = args.save_dir
    for directory in (args.raw_dir, args.context_dir, args.delta_dir, args.relevance_dir):
        os.makedirs(directory, exist_ok=True)
    stats_path = os.path.join(args.data_dir, "gawf_gate_distribution_stats.npz")
    metadata_path = os.path.join(args.data_dir, "gawf_gate_distribution_meta.json")
    with np.load(stats_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with open(metadata_path, encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    suffix = args.format

    edges = arrays["gate_edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    _gate_axes(axes, edges, arrays, metadata)
    fig.suptitle("GaWF gate values pooled across all test frames")
    _finish(fig, os.path.join(args.raw_dir, f"01_pooled_histogram.{suffix}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_sign"]
        axis.plot(centers, _density(counts[0], edges), label="W > 0", color="#c53030")
        axis.plot(centers, _density(counts[1], edges), label="W < 0", color="#2b6cb0")
        axis.axvline(0.5, color="black", linestyle="--", linewidth=0.9)
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
        axis.legend()
    fig.suptitle("Gate distributions split by corresponding weight sign")
    _finish(fig, os.path.join(args.raw_dir, f"02_weight_sign_histogram.{suffix}"))

    delta_edges = arrays["delta_edges"]
    delta_centers = (delta_edges[:-1] + delta_edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        counts = arrays[f"hist_{kind}_delta"]
        axis.plot(delta_centers, _density(counts, delta_edges), color="#805ad5")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set(
            title=title,
            xlabel=r"Group-mean gate $\Delta g$",
            ylabel="Density",
            xlim=(-0.75, 0.75),
        )
    fig.suptitle("Sector-centered gate distributions")
    sector_centered_name = f"03_sector_centered_gate_histogram.{suffix}"
    _finish(fig, os.path.join(args.delta_dir, sector_centered_name))

    digit_stats_path = os.path.join(args.digit_data_dir, "gawf_gate_digit_stats.npz")
    if os.path.isfile(digit_stats_path):
        with np.load(digit_stats_path) as loaded:
            digit_arrays = {key: loaded[key] for key in loaded.files}
        digit_delta_edges = digit_arrays["delta_edges"]
        if not np.array_equal(delta_edges, digit_delta_edges):
            raise RuntimeError("Sector and digit delta histograms must use identical bin edges")
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 7.2),
            sharex=True,
            sharey="row",
        )
        for row, kind in enumerate(("input", "recurrent")):
            for col, (conditioning, source) in enumerate(
                (("Sector", arrays), ("Digit", digit_arrays))
            ):
                axis = axes[row, col]
                counts = source[f"hist_{kind}_delta"]
                axis.plot(delta_centers, _density(counts, delta_edges), color="#805ad5")
                axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
                axis.set_xlim(-0.75, 0.75)
                axis.set_title(f"{kind.title()} gate — {conditioning}")
                axis.set_xlabel(r"Group-mean gate $\Delta g$")
                axis.set_ylabel("Density")
        fig.suptitle("Corrected group-mean gate deviations on shared axes")
        fig.tight_layout()
        combined_name = f"03_sector_digit_group_mean_delta_histogram.{suffix}"
        _finish(fig, os.path.join(args.delta_dir, combined_name))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 9))
    for axis, kind, title in zip(axes, ("input", "recurrent"), ("Input gate", "Recurrent gate")):
        context_counts = arrays[f"hist_{kind}_context"]
        for sector in range(9):
            axis.plot(
                centers,
                _density(context_counts[sector], edges),
                color=colors[sector],
                linewidth=1.0,
                label=str(sector),
            )
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
    axes[1].legend(title="Sector", ncol=3, fontsize=8)
    fig.suptitle("Gate distributions by foreground sector")
    _finish(fig, os.path.join(args.context_dir, f"04_per_context_histogram.{suffix}"))

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    relevance = arrays["hist_input_relevance"]
    axis.plot(centers, _density(relevance[0], edges), label="Relevant spatial sector")
    axis.plot(centers, _density(relevance[1], edges), label="Other spatial sectors")
    effect = metadata["task_relevance"]["cohens_d_relevant_minus_irrelevant"]
    axis.set(
        xlabel="Input-gate value",
        ylabel="Density",
        title=f"Task-relevance proxy (Cohen's d={effect:.4f})",
        xlim=(0.0, 1.0),
    )
    axis.legend()
    _finish(fig, os.path.join(args.relevance_dir, f"05_task_relevance_histogram.{suffix}"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    sectors = np.arange(9)
    for column, kind in enumerate(("input", "recurrent")):
        records = metadata["sparsity"][kind]
        axes[0, column].plot(
            sectors, [row["top_5pct_mass_fraction"] for row in records], marker="o", label="top 5%"
        )
        axes[0, column].plot(
            sectors,
            [row["top_10pct_mass_fraction"] for row in records],
            marker="s",
            label="top 10%",
        )
        axes[0, column].set(title=f"{kind.capitalize()} gate mass", ylabel="Mass fraction")
        axes[0, column].legend()
        axes[1, column].plot(sectors, [row["gini"] for row in records], marker="o", label="Gini")
        axes[1, column].plot(
            sectors,
            [row["normalized_participation_ratio"] for row in records],
            marker="s",
            label="Normalized PR",
        )
        axes[1, column].set(xlabel="Sector", ylabel="Index")
        axes[1, column].legend()
    fig.suptitle("Gate sparsity and concentration by context")
    _finish(fig, os.path.join(args.context_dir, f"06_sparsity_by_context.{suffix}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    kinds = ("input", "recurrent")
    titles = ("Input weights", "Recurrent weights")
    for axis, kind, title in zip(axes, kinds, titles):
        effective_edges = arrays[f"effective_edges_{kind}"]
        effective_centers = (effective_edges[:-1] + effective_edges[1:]) / 2.0
        axis.plot(
            effective_centers,
            _density(arrays[f"hist_weight_{kind}"], effective_edges),
            label="W",
            color="black",
        )
        axis.plot(
            effective_centers,
            _density(arrays[f"hist_effective_{kind}"], effective_edges),
            label=r"$G\odot W$",
            color="#dd6b20",
        )
        ratio = metadata["distribution"][kind]["frame_effective_norm_ratio"]["mean"]
        axis.set(title=f"{title} (mean norm ratio={ratio:.4f})", xlabel="Weight", ylabel="Density")
        axis.legend()
    fig.suptitle("Base and effective-weight distributions")
    _finish(fig, os.path.join(args.raw_dir, f"07_effective_weight_histogram.{suffix}"))


if __name__ == "__main__":
    main()
