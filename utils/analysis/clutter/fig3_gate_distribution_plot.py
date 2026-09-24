"""Plot the seven requested GaWF gate-distribution figures from saved analysis arrays."""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(
    _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
)
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils.analysis.anal_paths import output_dir

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SAVE_DATA_ROOT = os.path.join(_ANAL_PROJECT_ROOT, "results", "save_data", "fig3")


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments.

    Directory defaults are left ``None`` here and resolved lazily in ``main()`` so that
    ``output_dir(...)`` (which creates its directory as a side effect) only runs for a category
    actually needed by this invocation. Building the parser must never, by itself, recreate
    figure directories for categories a ``--save_dir`` override makes irrelevant.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--digit_data_dir", default=None)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--context_dir", default=None)
    parser.add_argument("--delta_dir", default=None)
    parser.add_argument("--relevance_dir", default=None)
    parser.add_argument(
        "--save_dir",
        default="",
        help="Deprecated compatibility override; sends every figure to one directory.",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    parser.add_argument(
        "--seed_dirs",
        nargs="+",
        default=None,
        help="Per-seed Fig3 directories. Histograms are pooled before plotting.",
    )
    parser.add_argument(
        "--only_gate_weight_2x2",
        action="store_true",
        help="Render only the final gate/weight 2x2 summary.",
    )
    parser.add_argument(
        "--gate_weight_stem",
        default="07_gate_and_weight_distributions_2x2",
        help="Filename stem used with --only_gate_weight_2x2.",
    )
    parser.add_argument(
        "--gate_weight_layout",
        choices=("2x2", "1x4"),
        default="2x2",
        help="Panel layout used with --only_gate_weight_2x2.",
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional JSON destination for multi-seed aggregation provenance.",
    )
    parser.add_argument(
        "--exclude_zero_feedback_reset",
        action="store_true",
        help="Remove the known t=0 (all-zero-feedback, gate=0.5) contribution from each seed.",
    )
    parser.add_argument(
        "--effective_weight_stats",
        default=None,
        help="Optional NPZ overriding only the input/recurrent W and G⊙W histograms.",
    )
    return parser.parse_args()


def _resolve_output_dirs(args: argparse.Namespace) -> None:
    """Fill in any directory left unset by the CLI, calling ``output_dir`` only when needed."""

    if args.save_dir:
        args.raw_dir = args.context_dir = args.delta_dir = args.relevance_dir = args.save_dir
    if args.data_dir is None:
        args.data_dir = os.path.join(SAVE_DATA_ROOT, "raw_statistics")
    if args.digit_data_dir is None:
        args.digit_data_dir = os.path.join(SAVE_DATA_ROOT, "digit_statistics")
    if args.raw_dir is None:
        args.raw_dir = str(output_dir("A_raw_gate", "gawf_gate_distribution", "figs"))
    if args.context_dir is None:
        args.context_dir = str(output_dir("B_gate_by_context", "gawf_gate_distribution", "figs"))
    if args.delta_dir is None:
        args.delta_dir = str(output_dir("C_delta_gate", "gawf_gate_distribution", "figs"))
    if args.relevance_dir is None:
        args.relevance_dir = str(
            output_dir("E_relevance_alignment", "gawf_gate_distribution", "figs")
        )


def _density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    total = float(np.asarray(counts).sum())
    return np.asarray(counts, dtype=np.float64) / (total * widths)


def _probability_percent(counts: np.ndarray) -> np.ndarray:
    """Return each bin's share of the total count, in percent, matching 01's pooled histogram."""

    counts_float = np.asarray(counts, dtype=np.float64)
    return 100.0 * counts_float / counts_float.sum()


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


def _style_probability_axis(axis: plt.Axes) -> None:
    """Apply the shared spine style used by the pooled gate and weight panels."""

    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


GATE_YTICKS = {"input": (0, 30, 60)}
GATE_XTICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_XTICKS = (-2, -1, 0, 1, 2)
GATE_REBIN_FACTOR = 4  # 400 saved bins at width 0.0025 -> 100 bins at width 0.01.


def _rebin_counts(
    counts: np.ndarray, edges: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sum consecutive equal-width bins by ``factor``, widening the bin width accordingly."""

    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape[-1] % factor != 0:
        raise ValueError(f"bin count {counts.shape[-1]} not divisible by rebin factor {factor}")
    rebinned_counts = counts.reshape(-1, factor).sum(axis=-1)
    rebinned_edges = edges[::factor]
    return rebinned_counts, rebinned_edges


def _pooled_gate_probability_axes(
    axes: np.ndarray,
    edges: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    """Draw the input/recurrent pooled-gate probability panels (row 1 of the 2-by-2 summary).

    The saved histogram uses 0.0025-wide bins; those are re-binned here to 0.01-wide bins
    (``GATE_REBIN_FACTOR``) so each curve carries enough probability mass to read clearly.
    """

    for axis, kind, title in zip(
        axes,
        ("input", "recurrent"),
        (r"$g^{\mathrm{in}}$", r"$g^{\mathrm{rec}}$"),
    ):
        counts, coarse_edges = _rebin_counts(arrays[f"hist_{kind}_all"], edges, GATE_REBIN_FACTOR)
        centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2.0
        probability = _probability_percent(counts)
        axis.plot(
            centers,
            probability,
            color="#2b6cb0",
            linewidth=1.5,
            label=r"$g$",
        )
        axis.set(title=title, xlabel="Gate value", ylabel="Probability (%)", xlim=(-0.01, 1.01))
        axis.set_title(title, pad=0)
        axis.set_xticks(GATE_XTICKS, ("0", "0.25", "0.5", "0.75", "1"))
        if kind == "recurrent":
            maximum_tick = int(np.rint(float(probability.max())))
            midpoint_tick = int(np.rint(maximum_tick / 2.0))
            axis.set_yticks((0, midpoint_tick, maximum_tick))
        else:
            axis.set_yticks(GATE_YTICKS[kind])
        _style_probability_axis(axis)


def _weight_probability_axes(axes: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    """Draw the base/effective input & recurrent weight probability panels (row 2)."""

    titles = (r"$\widetilde{w}^{\mathrm{in}}$", r"$\widetilde{w}^{\mathrm{rec}}$")
    for axis, kind, title in zip(axes, ("input", "recurrent"), titles):
        effective_edges = arrays[f"effective_edges_{kind}"]
        effective_centers = (effective_edges[:-1] + effective_edges[1:]) / 2.0
        axis.plot(
            effective_centers,
            _probability_percent(arrays[f"hist_weight_{kind}"]),
            label=r"$w$",
            color="black",
        )
        axis.plot(
            effective_centers,
            _probability_percent(arrays[f"hist_effective_{kind}"]),
            label=r"$\widetilde{w}$",
            color="#dd6b20",
        )
        axis.set(
            title=title,
            xlabel="Weight value",
            ylabel="Probability (%)",
            xlim=(-2.0, 2.0),
        )
        axis.set_title(title, pad=0)
        axis.set_xticks(WEIGHT_XTICKS)
        if effective_edges.size <= 1001:
            upper = int(np.ceil(axis.get_ylim()[1]))
            axis.set_ylim(0, upper)
            axis.set_yticks((0, int(np.floor(upper / 2 + 0.5)), upper))
        _style_probability_axis(axis)


def _gate_weight_summary(
    edges: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    layout: str,
) -> plt.Figure:
    """Return the gate/weight probability summary in the requested panel layout."""

    if layout == "1x4":
        figure, axes = plt.subplots(1, 4, figsize=(5.5, 1.3))
        gate_axes, weight_axes = axes[:2], axes[2:]
    else:
        figure, axes = plt.subplots(2, 2, figsize=(0.7 * 2 * 5.05, 2 * 4.1))
        gate_axes, weight_axes = axes[0], axes[1]
    _pooled_gate_probability_axes(gate_axes, edges, arrays, metadata)
    _weight_probability_axes(weight_axes, arrays)
    handles = gate_axes[0].get_legend_handles_labels()[0]
    handles += weight_axes[0].get_legend_handles_labels()[0]
    labels = gate_axes[0].get_legend_handles_labels()[1]
    labels += weight_axes[0].get_legend_handles_labels()[1]
    if layout == "1x4":
        for axis in axes:
            axis.set_ylabel("")
        figure.supylabel("Probability (%)", x=0.012, fontsize=8)
        figure.subplots_adjust(
            left=0.085, right=0.995, bottom=0.29, top=0.72, wspace=0.42
        )
        for label, axis in zip("ABCD", axes):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.012,
                position.y1 + 0.035,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        legend_y = 0.99
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        char_height_frac = (16.0 / 72.0) / (2 * 4.1)
        legend_y = 0.965 - 0.4 * char_height_frac
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(labels),
        frameon=False,
    )
    return figure


def _histogram_median(counts: np.ndarray, edges: np.ndarray) -> float:
    """Return the fixed-bin median represented by a nonempty histogram."""

    cumulative = np.cumsum(np.asarray(counts, dtype=np.int64))
    if cumulative.size == 0 or cumulative[-1] <= 0:
        raise ValueError("Cannot compute a median from an empty histogram")
    index = int(np.searchsorted(cumulative, (int(cumulative[-1]) - 1) // 2 + 1))
    return float((edges[index] + edges[index + 1]) / 2.0)


def _reproject_histogram_counts(
    counts: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Project a histogram to different bins, uniformly within each source interval.

    Weight ranges differ across independently trained checkpoints.  The saved counts contain no
    sample-level values, so overlap-weighted projection is the exact aggregation permitted by
    the histogram representation under its standard uniform-within-bin assumption.
    """

    source_counts = np.asarray(counts, dtype=np.float64)
    source_edges = np.asarray(source_edges, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    if source_counts.ndim != 1 or source_edges.size != source_counts.size + 1:
        raise ValueError("Histogram counts and source edges have incompatible shapes")
    if target_edges.ndim != 1 or target_edges.size < 2 or np.any(np.diff(target_edges) <= 0):
        raise ValueError("Target histogram edges must be strictly increasing")
    projected = np.zeros(target_edges.size - 1, dtype=np.float64)
    for index, count in enumerate(source_counts):
        if count == 0:
            continue
        left = source_edges[index]
        right = source_edges[index + 1]
        width = right - left
        if width <= 0:
            raise ValueError("Source histogram edges must be strictly increasing")
        start = max(0, int(np.searchsorted(target_edges, left, side="right") - 1))
        stop = min(projected.size, int(np.searchsorted(target_edges, right, side="left") + 1))
        for target_index in range(start, stop):
            overlap = min(right, target_edges[target_index + 1]) - max(
                left, target_edges[target_index]
            )
            if overlap > 0:
                projected[target_index] += count * overlap / width
    if not np.isclose(projected.sum(), source_counts.sum(), rtol=1e-10, atol=1e-6):
        raise ValueError("Target histogram range does not cover all source bins")
    return projected


def _fixed_histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count values in the fixed-edge convention used by the saved Fig3 histograms."""

    bins = edges.size - 1
    indices = np.floor((np.asarray(values).reshape(-1) - edges[0]) * bins / (edges[-1] - edges[0]))
    indices = np.clip(indices.astype(np.int64), 0, bins - 1)
    return np.bincount(indices, minlength=bins).astype(np.int64)


def _mean_sem(values: np.ndarray) -> dict[str, object]:
    """Return the mean, seed-level SEM, and values for independent training seeds."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"Expected at least two one-dimensional seed values, got {array.shape}")
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / np.sqrt(array.size)),
        "seed_values": array.tolist(),
    }


def _endpoint_fraction_summary(
    arrays_by_seed: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    """Summarize reset-excluded gate fractions below 0.1 and at or above 0.9."""

    edges = np.asarray(arrays_by_seed[0]["gate_edges"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    below = centers < 0.1
    above = centers >= 0.9
    summary: dict[str, object] = {
        "definition": {"closed": "[0, 0.1)", "open": "[0.9, 1]"},
        "aggregation": "fraction within seed, then mean and SEM across training seeds",
    }
    for kind in ("input", "recurrent"):
        fractions = []
        for arrays in arrays_by_seed:
            counts = np.asarray(arrays[f"hist_{kind}_all"], dtype=np.int64)
            total = int(counts.sum())
            if total <= 0:
                raise ValueError(f"Empty {kind} gate histogram")
            fractions.append(
                (float(counts[below].sum() / total), float(counts[above].sum() / total))
            )
        values = np.asarray(fractions, dtype=np.float64)
        summary[kind] = {
            "below_0_1": _mean_sem(values[:, 0]),
            "at_or_above_0_9": _mean_sem(values[:, 1]),
        }
    return summary


def _exclude_reset_histograms(
    arrays: dict[str, np.ndarray],
    trajectory_path: str,
) -> int:
    """Exactly remove all-zero-feedback t=0 gate and effective-weight histogram mass."""

    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        feedback = trajectory["feedback"]
        reset_frames = int(np.all(feedback == 0.0, axis=-1).sum())
        if reset_frames == 0:
            raise RuntimeError(f"Expected reset frames in {trajectory_path}, found none")
        for kind, weight_key in (("input", "weight_ih"), ("recurrent", "weight_hh")):
            weight = trajectory[weight_key]
            gate_hist = arrays[f"hist_{kind}_all"].copy()
            midpoint_bin = int(np.searchsorted(arrays["gate_edges"], 0.5, side="right") - 1)
            gate_hist[midpoint_bin] -= reset_frames * weight.size
            if gate_hist[midpoint_bin] < 0:
                raise RuntimeError("Reset correction exceeded the saved gate histogram mass")
            arrays[f"hist_{kind}_all"] = gate_hist
            arrays[f"hist_effective_{kind}"] = arrays[f"hist_effective_{kind}"] - (
                reset_frames * _fixed_histogram(0.5 * weight, arrays[f"effective_edges_{kind}"])
            )
    return reset_frames


def _load_multiseed_gate_statistics(
    seed_dirs: list[str], *, exclude_zero_feedback_reset: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Pool same-bin Fig3 histograms and preserve an explicit 10-seed provenance record."""

    if len(seed_dirs) < 2:
        raise ValueError("--seed_dirs requires at least two independent seed directories")
    arrays_by_seed: list[dict[str, np.ndarray]] = []
    metadata_by_seed: list[dict[str, object]] = []
    for directory in seed_dirs:
        stats_path = os.path.join(directory, "gawf_gate_distribution_stats.npz")
        metadata_path = os.path.join(directory, "gawf_gate_distribution_meta.json")
        with np.load(stats_path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        if exclude_zero_feedback_reset:
            _exclude_reset_histograms(arrays, os.path.join(directory, "gawf_gate_trajectory.npz"))
        arrays_by_seed.append(arrays)
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata_by_seed.append(json.load(file_obj))
    reference = arrays_by_seed[0]
    pooled: dict[str, np.ndarray] = {}
    weight_hist_keys = {
        "hist_weight_input",
        "hist_weight_recurrent",
        "hist_effective_input",
        "hist_effective_recurrent",
    }
    for key, value in reference.items():
        values = [item[key] for item in arrays_by_seed]
        if any(value.shape != item.shape for item in values):
            raise ValueError(f"Per-seed Fig3 array shape mismatch for {key}")
        if key in weight_hist_keys or key.startswith("effective_edges_"):
            continue
        if key.startswith("hist_") or key == "context_counts":
            pooled[key] = np.sum(np.stack(values, axis=0), axis=0, dtype=np.int64)
        elif key.startswith("context_mean_"):
            pooled[key] = np.mean(
                np.stack(values, axis=0), axis=0, dtype=np.float64
            ).astype(np.float32)
        else:
            if any(not np.array_equal(value, item) for item in values[1:]):
                raise ValueError(f"Per-seed Fig3 bin/weight array mismatch for {key}")
            pooled[key] = value
    for kind in ("input", "recurrent"):
        edge_key = f"effective_edges_{kind}"
        histogram_size = reference[edge_key].size - 1
        max_abs = max(
            float(np.max(np.abs(item[edge_key]))) for item in arrays_by_seed
        )
        target_edges = np.linspace(-max_abs, max_abs, histogram_size + 1, dtype=np.float32)
        pooled[edge_key] = target_edges
        for prefix in ("hist_weight", "hist_effective"):
            pooled[f"{prefix}_{kind}"] = np.sum(
                np.stack(
                    [
                        _reproject_histogram_counts(
                            item[f"{prefix}_{kind}"], item[edge_key], target_edges
                        )
                        for item in arrays_by_seed
                    ],
                    axis=0,
                ),
                axis=0,
                dtype=np.float64,
            )
    metadata = dict(metadata_by_seed[0])
    distribution = dict(metadata.get("distribution", {}))
    for kind in ("input", "recurrent"):
        record = dict(distribution.get(kind, {}))
        record["median"] = _histogram_median(pooled[f"hist_{kind}_all"], pooled["gate_edges"])
        distribution[kind] = record
    metadata["distribution"] = distribution
    metadata["seed_level_endpoint_fractions"] = _endpoint_fraction_summary(arrays_by_seed)
    sources_exclude_reset = all(
        int(item.get("reset_frames_excluded", 0)) > 0 for item in metadata_by_seed
    )
    metadata["multiseed_aggregation"] = {
        "n_seeds": len(seed_dirs),
        "seed_dirs": [os.path.abspath(directory) for directory in seed_dirs],
        "definition": (
            "Same-bin gate histograms are summed across seeds and gate medians are recomputed "
            "from the pooled histogram. Weight and effective-weight histograms are first "
            "reprojected by bin overlap to a common symmetric range before summation."
        ),
        "seed42_included": False,
        "reset_frames_excluded": bool(sources_exclude_reset or exclude_zero_feedback_reset),
        "additional_reset_subtraction_at_aggregation": bool(exclude_zero_feedback_reset),
    }
    return pooled, metadata


def main() -> None:
    """Read analysis outputs and save seven independent figures."""

    args = parse_args()
    _resolve_output_dirs(args)
    for directory in (args.raw_dir, args.context_dir, args.delta_dir, args.relevance_dir):
        os.makedirs(directory, exist_ok=True)
    if args.seed_dirs is None:
        stats_path = os.path.join(args.data_dir, "gawf_gate_distribution_stats.npz")
        metadata_path = os.path.join(args.data_dir, "gawf_gate_distribution_meta.json")
        with np.load(stats_path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        with open(metadata_path, encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
    else:
        arrays, metadata = _load_multiseed_gate_statistics(
            args.seed_dirs,
            exclude_zero_feedback_reset=args.exclude_zero_feedback_reset,
        )
    if args.metadata_path is not None:
        with open(args.metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)
    if args.effective_weight_stats is not None:
        with np.load(args.effective_weight_stats, allow_pickle=False) as loaded:
            required = {
                f"{prefix}_{kind}"
                for prefix in ("effective_edges", "hist_weight", "hist_effective")
                for kind in ("input", "recurrent")
            }
            missing = sorted(required.difference(loaded.files))
            if missing:
                raise ValueError(f"Effective-weight override is missing arrays: {missing}")
            for key in required:
                arrays[key] = np.asarray(loaded[key])
    suffix = args.format

    edges = arrays["gate_edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    if args.only_gate_weight_2x2:
        with plt.rc_context(
            {
                "font.size": 7 if args.gate_weight_layout == "1x4" else 16,
                "axes.labelsize": 8 if args.gate_weight_layout == "1x4" else 16,
                "axes.titlesize": 8 if args.gate_weight_layout == "1x4" else 16,
                "xtick.labelsize": 7 if args.gate_weight_layout == "1x4" else 16,
                "ytick.labelsize": 7 if args.gate_weight_layout == "1x4" else 16,
                "legend.fontsize": 6.5 if args.gate_weight_layout == "1x4" else 16,
                "axes.linewidth": 0.6 if args.gate_weight_layout == "1x4" else 0.8,
            }
        ):
            fig = _gate_weight_summary(
                edges,
                arrays,
                metadata,
                args.gate_weight_layout,
            )
            stem = args.gate_weight_stem
            extensions = ("png", "pdf") if args.format == "png" else (args.format,)
            for extension in extensions:
                path = os.path.join(args.raw_dir, f"{stem}.{extension}")
                if args.gate_weight_layout == "1x4":
                    fig.savefig(path, dpi=300)
                else:
                    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
                print(f"Saved figure: {path}")
            plt.close(fig)
        return

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
    digit_meta_path = os.path.join(args.digit_data_dir, "gawf_gate_digit_meta.json")
    digit_arrays: dict[str, np.ndarray] | None = None
    digit_metadata: dict[str, object] | None = None
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
        if os.path.isfile(digit_meta_path):
            with open(digit_meta_path, encoding="utf-8") as file_obj:
                digit_metadata = json.load(file_obj)

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

    sectors = np.arange(9)
    digits = np.arange(10)
    factor_columns = [("Sector", sectors, "Sector", metadata["sparsity"])]
    if digit_metadata is not None:
        factor_columns.append(("Digit", digits, "Foreground digit", digit_metadata["sparsity"]))
    # Row 1 (mass fraction) and row 2 (index) use distinct color pairs so the two series pairs
    # stay visually separable once their legends are merged into one shared legend.
    mass_colors = ("#2b6cb0", "#dd6b20")
    index_colors = ("#38a169", "#805ad5")
    fig, axes = plt.subplots(
        2, 2 * len(factor_columns), figsize=(5 * len(factor_columns), 7.4), sharex="col"
    )
    axes = np.atleast_2d(axes)
    for factor_index, (factor_label, x_values, x_label, sparsity_by_kind) in enumerate(
        factor_columns
    ):
        for kind_index, kind in enumerate(("input", "recurrent")):
            column = 2 * factor_index + kind_index
            records = sparsity_by_kind[kind]
            axes[0, column].plot(
                x_values,
                [row["top_5pct_mass_fraction"] for row in records],
                marker="o",
                color=mass_colors[0],
                label="top 5%",
            )
            axes[0, column].plot(
                x_values,
                [row["top_10pct_mass_fraction"] for row in records],
                marker="s",
                color=mass_colors[1],
                label="top 10%",
            )
            axes[0, column].set(
                title=f"{kind.capitalize()} gate mass ({factor_label})", ylabel="Mass fraction"
            )
            axes[1, column].plot(
                x_values,
                [row["gini"] for row in records],
                marker="o",
                color=index_colors[0],
                label="Gini",
            )
            axes[1, column].plot(
                x_values,
                [row["normalized_participation_ratio"] for row in records],
                marker="s",
                color=index_colors[1],
                label="Normalized PR",
            )
            axes[1, column].set(xlabel=x_label, ylabel="Index", xticks=x_values)
    # One text-line height (10pt, the default legend/tick font) expressed as a fraction of this
    # figure's fixed 7.4-inch height. Both the title and the shared legend drop by two of these
    # units; the title drops by one extra unit so the title-to-legend gap tightens as well.
    char_height_frac = (10.0 / 72.0) / 7.4
    title_y = 0.985 - 3 * char_height_frac
    legend_y = 0.93 - 2 * char_height_frac
    fig.suptitle("Gate sparsity and concentration by sector and digit", y=title_y)
    handles = (
        axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
    )
    labels = (
        axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
    )
    _finish(fig, os.path.join(args.context_dir, f"06_sparsity_by_sector_digit.{suffix}"))

    # Per-panel width:height ratio matches core_objects_aggregate_2x2.png (5.05in x 4.1in per
    # panel), just laid out as two columns instead of one, then narrowed by another 30% in
    # width only. No composite title: the four panel titles plus the shared
    # G/median/W/G-odot-W legend already carry everything it would add.
    with plt.rc_context(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    ):
        fig = _gate_weight_summary(edges, arrays, metadata, "2x2")
        gate_weight_png = os.path.join(args.raw_dir, "07_gate_and_weight_distributions_2x2.png")
        gate_weight_pdf = os.path.join(args.raw_dir, "07_gate_and_weight_distributions_2x2.pdf")
        fig.savefig(gate_weight_png, dpi=150, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(gate_weight_pdf, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    print(f"Saved figure: {gate_weight_png}")
    print(f"Saved figure: {gate_weight_pdf}")


if __name__ == "__main__":
    main()
