"""Compare input-gate means against input-weight magnitude, split by sign.

For each of ten saved GaWF trajectories, ``collect`` reconstructs sequential input gates under the
original equal-n sector protocol. The legacy output retains Sector 0 connection arrays;
``--all_sectors`` stores compact nine-sector gate means and ``plot --all_sectors`` pools ten seeds
and nine sectors, writes the two-panel figures, and saves seed-level overlap-gap and slope stats.
``--connection_stats_only`` writes explicitly diagnostic per-seed and pooled connection-row tests.
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
from scipy.stats import linregress, ttest_ind  # noqa: E402

from utils.analysis.clutter.fig3_gate_distribution import (
    _spatial_sector_indices,
    exclude_zero_feedback_reset_frames,
    iter_gate_chunks,
)
from utils.analysis.clutter.fig6_sector_gate_sequential import NUM_SECTORS, equal_n_sector_mask
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    MAX_SCATTER_PER_SIGN,
    NEG_COLOR,
    POS_COLOR,
    _draw_panel,
    compute_overlap_band,
)


RESULT_NAME = "input_gate_sign_magnitude_sector0.npz"
ALL_SECTORS_RESULT_NAME = "input_gate_sign_magnitude_9sector.npz"
GROUPS = ("sector0_sources", "other_sources")
GROUP_LABELS = {"sector0_sources": "matching", "other_sources": "other"}
SECTOR = 0


def parse_args() -> argparse.Namespace:
    """Parse one-seed collection and ten-seed pooling commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--trajectory", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--gate_tau", type=float, default=0.5)
    collect_parser.add_argument("--gate_chunk_size", type=int, default=16)
    collect_parser.add_argument("--device", default="cpu")
    collect_parser.add_argument("--all_sectors", action="store_true")

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--data_root", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--all_sectors", action="store_true")
    plot_parser.add_argument("--connection_stats_only", action="store_true")
    return parser.parse_args()


def _source_group_masks(input_size: int, sector: int = SECTOR) -> dict[str, np.ndarray]:
    """Return one sector's 128 matching source units and their 1024-unit complement."""

    matching = np.zeros(input_size, dtype=bool)
    matching[_spatial_sector_indices(input_size)[sector]] = True
    other = ~matching
    if int(matching.sum()) != 128 or int(other.sum()) != 1024:
        raise RuntimeError("Sector-0 source partition must be 128 matching and 1024 other units.")
    return {"sector0_sources": matching, "other_sources": other}


def _collect_sector_means(
    feedback: np.ndarray,
    labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weight_ih: np.ndarray,
    *,
    selection_seed: int,
    gate_tau: float,
    gate_chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return equal-n sequential input-gate means for all nine sectors."""

    if weight_ih.ndim != 2:
        raise ValueError(f"Expected two-dimensional input weights, got {weight_ih.shape}.")
    feedback, labels, _reset_frames = exclude_zero_feedback_reset_frames(feedback, labels)
    sectors = np.asarray(labels, dtype=np.int64).reshape(-1, 2)[:, 1]
    selected, target, original_counts = equal_n_sector_mask(sectors, selection_seed)
    sums = np.zeros((NUM_SECTORS, *weight_ih.shape), dtype=np.float64)
    counts = np.zeros(NUM_SECTORS, dtype=np.int64)
    for start, end, gate_input, _gate_recurrent in iter_gate_chunks(
        feedback, u, v, weight_ih.shape[1], gate_tau, gate_chunk_size, device=device
    ):
        chunk_selected = selected[start:end]
        chunk_sectors = sectors[start:end]
        for sector in np.unique(chunk_sectors[chunk_selected]):
            use = chunk_selected & (chunk_sectors == sector)
            values = gate_input[use]
            sums[sector] += values.sum(axis=0, dtype=np.float64)
            counts[sector] += values.shape[0]
    if not np.all(counts == target):
        raise RuntimeError(
            f"Equal-n accumulation mismatch: expected {target}, got {counts.tolist()}."
        )
    return (sums / counts[:, None, None]).astype(np.float32), original_counts, target


def _collect_connection_means(
    feedback: np.ndarray,
    labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weight_ih: np.ndarray,
    *,
    selection_seed: int,
    gate_tau: float,
    gate_chunk_size: int,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Return Sector-0 raw and delta gates for each source partition and static input weight."""

    sector_means, original_counts, target = _collect_sector_means(
        feedback,
        labels,
        u,
        v,
        weight_ih,
        selection_seed=selection_seed,
        gate_tau=gate_tau,
        gate_chunk_size=gate_chunk_size,
        device=device,
    )
    source_masks = _source_group_masks(weight_ih.shape[1])
    results: dict[str, np.ndarray] = {}
    sector0_gate = sector_means[SECTOR]
    delta = sector0_gate - sector_means.mean(axis=0)
    for name, source_mask in source_masks.items():
        weights = weight_ih[:, source_mask]
        keep = weights != 0.0
        results[f"{name}_weight"] = weights[keep].astype(np.float32)
        results[f"{name}_gate"] = sector0_gate[:, source_mask][keep].astype(np.float32)
        results[f"{name}_delta"] = delta[:, source_mask][keep].astype(np.float32)
    return results, original_counts, target


def collect(args: argparse.Namespace) -> Path:
    """Save compact input-connection gate data for one trajectory."""

    if args.gate_tau <= 0 or args.gate_chunk_size <= 0:
        raise ValueError("gate_tau and gate_chunk_size must be positive.")
    destination = args.output_dir / (
        ALL_SECTORS_RESULT_NAME if args.all_sectors else RESULT_NAME
    )
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    with np.load(args.trajectory, allow_pickle=False) as arrays:
        common = {
            "feedback": arrays["feedback"].astype(np.float32, copy=False),
            "labels": arrays["labels"].astype(np.int64, copy=False),
            "u": arrays["U"].astype(np.float32, copy=False),
            "v": arrays["V"].astype(np.float32, copy=False),
            "weight_ih": arrays["weight_ih"].astype(np.float32, copy=False),
            "selection_seed": args.seed,
            "gate_tau": args.gate_tau,
            "gate_chunk_size": args.gate_chunk_size,
            "device": args.device,
        }
        if args.all_sectors:
            sector_means, original_counts, target = _collect_sector_means(**common)
            values = {
                "weight": common["weight_ih"],
                "sector_gate_mean": sector_means,
            }
        else:
            values, original_counts, target = _collect_connection_means(**common)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(destination, **values)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "selection": "original equal-n sector selection",
                "selection_seed": args.seed,
                "sectors": list(range(NUM_SECTORS)) if args.all_sectors else [SECTOR],
                "selected_frames_per_sector": target,
                "original_frames_by_sector": original_counts.astype(int).tolist(),
                "matching_sources_per_sector": 128,
                "other_sources_per_sector": 1024,
                "destination_units": int(common["weight_ih"].shape[0]),
                "delta_definition": "Sector mean minus unweighted mean of nine sector means",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_pooled(data_root: Path) -> dict[str, pd.DataFrame]:
    """Pool compact connection means from exactly ten seeds into the two source groups."""

    paths = sorted(data_root.glob(f"seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed files in {data_root}, found {len(paths)}.")
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in GROUPS}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for name in GROUPS:
                weight = np.asarray(arrays[f"{name}_weight"], dtype=np.float64)
                gate = np.asarray(arrays[f"{name}_gate"], dtype=np.float64)
                delta = np.asarray(arrays[f"{name}_delta"], dtype=np.float64)
                if not (weight.shape == gate.shape == delta.shape) or weight.size == 0:
                    raise RuntimeError(f"Invalid {name} data in {path}.")
                pooled[name].append(
                    pd.DataFrame(
                        {
                            "absW": np.abs(weight),
                            "signpos": (weight > 0.0).astype(np.int64),
                            "gate": gate,
                            "delta_gate": delta,
                        }
                    )
                )
    return {name: pd.concat(rows, ignore_index=True) for name, rows in pooled.items()}


def _load_all_sector_by_seed(data_root: Path) -> dict[str, list[pd.DataFrame]]:
    """Load exactly ten compact seeds and partition every sector into matching/other sources."""

    paths = sorted(data_root.glob(f"seed*/{ALL_SECTORS_RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed files in {data_root}, found {len(paths)}.")
    by_seed: dict[str, list[pd.DataFrame]] = {name: [] for name in GROUPS}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            weight = np.asarray(arrays["weight"], dtype=np.float64)
            means = np.asarray(arrays["sector_gate_mean"], dtype=np.float64)
        if means.shape != (NUM_SECTORS, *weight.shape):
            raise RuntimeError(f"Invalid nine-sector data in {path}: {means.shape}.")
        grand = means.mean(axis=0)
        frames: dict[str, list[pd.DataFrame]] = {name: [] for name in GROUPS}
        for sector in range(NUM_SECTORS):
            sector_mean = means[sector]
            for name, source_mask in _source_group_masks(weight.shape[1], sector).items():
                selected_weight = weight[:, source_mask]
                selected_gate = sector_mean[:, source_mask]
                selected_delta = selected_gate - grand[:, source_mask]
                keep = selected_weight != 0.0
                frames[name].append(
                    pd.DataFrame(
                        {
                            "absW": np.abs(selected_weight[keep]),
                            "signpos": (selected_weight[keep] > 0.0).astype(np.int64),
                            "gate": selected_gate[keep],
                            "delta_gate": selected_delta[keep],
                            "context": sector,
                        }
                    )
                )
        for name in GROUPS:
            by_seed[name].append(pd.concat(frames[name], ignore_index=True))
    return by_seed


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the ordinary-least-squares slope with an intercept."""

    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise RuntimeError("Cannot fit a slope to constant |W| values.")
    return float(np.dot(centered, y - y.mean()) / denominator)


def _mean_sem(values: list[float]) -> dict[str, object]:
    """Summarize ten independent seed values with cross-seed SEM."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,):
        raise RuntimeError(f"Expected ten seed values, got {array.shape}.")
    mean = float(array.mean())
    return {
        "mean": mean,
        "sem": float(array.std(ddof=1) / np.sqrt(array.size)),
        "seed_values": array.tolist(),
    }


def _seed_level_stats(by_seed: dict[str, list[pd.DataFrame]]) -> dict[str, object]:
    """Compute existing Figure-5.9 overlap gaps, sign slopes, and group delta levels."""

    result: dict[str, object] = {}
    for name in GROUPS:
        collected = {
            "positive_overlap_mean": [],
            "negative_overlap_mean": [],
            "overlap_gap": [],
            "positive_overlap_slope": [],
            "negative_overlap_slope": [],
            "overall_delta_level": [],
        }
        for frame in by_seed[name]:
            positive_w = frame.loc[frame["signpos"] == 1, "absW"].to_numpy()
            negative_w = frame.loc[frame["signpos"] == 0, "absW"].to_numpy()
            overlap = compute_overlap_band(positive_w, negative_w)
            in_band = frame["absW"].between(
                overlap["overlap_low"], overlap["overlap_high"], inclusive="both"
            )
            band = frame.loc[in_band]
            sign_values = {}
            for sign, label in ((1, "positive"), (0, "negative")):
                selected = band.loc[band["signpos"] == sign]
                sign_values[label] = float(
                    selected.groupby("context", sort=True)["delta_gate"].mean().mean()
                )
                collected[f"{label}_overlap_mean"].append(sign_values[label])
                collected[f"{label}_overlap_slope"].append(
                    _slope(
                        selected["absW"].to_numpy(),
                        selected["delta_gate"].to_numpy(),
                    )
                )
            collected["overlap_gap"].append(
                sign_values["positive"] - sign_values["negative"]
            )
            collected["overall_delta_level"].append(
                float(frame.groupby("context", sort=True)["delta_gate"].mean().mean())
            )
        result[name] = {key: _mean_sem(values) for key, values in collected.items()}
    return result


def _connection_level_stats(frame: pd.DataFrame) -> dict[str, object]:
    """Return diagnostic overlap-gap and pooled-sign OLS statistics for connection rows."""

    positive_w = frame.loc[frame["signpos"] == 1, "absW"].to_numpy()
    negative_w = frame.loc[frame["signpos"] == 0, "absW"].to_numpy()
    overlap = compute_overlap_band(positive_w, negative_w)
    in_band = frame["absW"].between(
        overlap["overlap_low"], overlap["overlap_high"], inclusive="both"
    )
    band = frame.loc[in_band]
    positive = band.loc[band["signpos"] == 1, "delta_gate"].to_numpy()
    negative = band.loc[band["signpos"] == 0, "delta_gate"].to_numpy()
    if positive.size < 2 or negative.size < 2:
        raise RuntimeError("Connection-level sign test requires at least two rows per sign.")
    gap_test = ttest_ind(positive, negative, equal_var=False)
    slope_fit = linregress(
        band["absW"].to_numpy(),
        band["delta_gate"].to_numpy(),
    )
    return {
        "overlap_low": float(overlap["overlap_low"]),
        "overlap_high": float(overlap["overlap_high"]),
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "positive_delta_mean": float(positive.mean()),
        "negative_delta_mean": float(negative.mean()),
        "gap": float(positive.mean() - negative.mean()),
        "gap_p_value": float(gap_test.pvalue),
        "gap_test_statistic": float(gap_test.statistic),
        "ols_slope": float(slope_fit.slope),
        "ols_slope_p_value": float(slope_fit.pvalue),
        "ols_intercept": float(slope_fit.intercept),
        "ols_r_squared": float(slope_fit.rvalue**2),
        "ols_n": int(len(band)),
    }


def _connection_level_stats_by_seed(
    by_seed: dict[str, list[pd.DataFrame]],
) -> dict[str, object]:
    """Return ten single-seed and one all-seed-pooled diagnostic result per group."""

    result: dict[str, object] = {}
    for name in GROUPS:
        frames = by_seed[name]
        if len(frames) != 10:
            raise RuntimeError(f"Expected ten {name} seed frames, got {len(frames)}.")
        group = {
            f"seed{seed:02d}": _connection_level_stats(frame)
            for seed, frame in enumerate(frames, start=1)
        }
        group["all_seeds_pooled"] = _connection_level_stats(
            pd.concat(frames, ignore_index=True)
        )
        result[GROUP_LABELS[name]] = group
    return result


def write_connection_level_stats(data_root: Path, output_dir: Path) -> Path:
    """Write connection-row diagnostics from existing ten-seed compact arrays."""

    by_seed = _load_all_sector_by_seed(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (
        "Supple2_input_gate_sign_vs_mag_9sector_connection_level_diagnostics.json"
    )
    destination.write_text(
        json.dumps(
            {
                "scope": "ten single-seed results plus all ten seeds pooled",
                "unit": "seed x sector x connection row within each source group",
                "gap": "mean(delta_g | W>0) - mean(delta_g | W<0) in shared |W| band",
                "gap_test": "two-sided Welch independent-samples t-test over rows",
                "ols": (
                    "Ordinary Least Squares: delta_g ~ intercept + |W| over both signs "
                    "within the shared |W| band"
                ),
                "ols_test": "two-sided t-test of slope = 0",
                "interpretation": (
                    "Diagnostic only: rows share seeds, sectors, destinations, and physical "
                    "connections, so p-values do not replace seed-level inference. Statistical "
                    "significance with a negligible gap or slope is not treated as a finding."
                ),
                "groups": _connection_level_stats_by_seed(by_seed),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _render_zoomed(
    pooled: dict[str, pd.DataFrame],
    figure_path: Path,
    *,
    value_column: str,
    y_label: str,
    title: str,
    all_sectors: bool = False,
) -> None:
    """Render matching versus other sources with Fig7's sign/magnitude zoom protocol."""

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    labels = {
        "sector0_sources": (
            "Matching sources across 9 sectors\n(128 inputs/sector x 256 destinations)"
            if all_sectors
            else "Sector 0 matching sources\n(128 inputs x 256 destinations)"
        ),
        "other_sources": (
            "Other sources across 9 sectors\n(1024 inputs/sector x 256 destinations)"
            if all_sectors
            else "Other sources\n(1024 inputs x 256 destinations)"
        ),
    }
    for axis, name in zip(axes, GROUPS):
        values = pooled[name]
        positive = values.loc[values["signpos"] == 1, "absW"].to_numpy()
        negative = values.loc[values["signpos"] == 0, "absW"].to_numpy()
        overlap = compute_overlap_band(positive, negative)
        maxima = _draw_panel(
            axis,
            values,
            overlap["overlap_low"],
            overlap["overlap_high"],
            rng,
            y_col=value_column,
            y_label=y_label,
        )
        candidates = [value for value in maxima if np.isfinite(value)]
        curve_max = max(candidates) if candidates else float(values["absW"].max())
        axis.set_xlim(0.0, max(curve_max * 1.08, 1e-6))
        axis.set_title(labels[name], fontsize=11)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=POS_COLOR, label="W > 0 (+)"),
        plt.Line2D([], [], marker="o", linestyle="none", color=NEG_COLOR, label="W < 0 (-)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        f"{title}\n{'All 9 sectors' if all_sectors else 'Sector 0'}; "
        "binned mean +/- SEM; shaded = shared |W| range; zoomed to curves",
        fontsize=11,
        y=1.00,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.76, wspace=0.25)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Write the gate and delta-gate Supplementary 2 PNGs plus a compact count summary."""

    if args.all_sectors:
        by_seed = _load_all_sector_by_seed(args.data_root)
        pooled = {name: pd.concat(frames, ignore_index=True) for name, frames in by_seed.items()}
    else:
        by_seed = None
        pooled = _load_pooled(args.data_root)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_10seed_9sector" if args.all_sectors else ""
    gate = args.figure_dir / f"Supple2_input_gate_sign_vs_mag_sector_zoom{suffix}.png"
    delta = args.figure_dir / f"Supple2_input_gate_sign_vs_mag_sector_delta_zoom{suffix}.png"
    _render_zoomed(
        pooled,
        gate,
        value_column="gate",
        y_label=("Input gate mean" if args.all_sectors else "Sector-0 input gate mean"),
        title="Input gate vs. |W|, split by sign(W)",
        all_sectors=args.all_sectors,
    )
    _render_zoomed(
        pooled,
        delta,
        value_column="delta_gate",
        y_label="Delta input gate\n(vs. this connection's mean across sectors)",
        title="Input gate delta-g vs. |W|, split by sign(W)",
        all_sectors=args.all_sectors,
    )
    summary = args.figure_dir / "Supple_2_input_gate_sign_vs_mag_sector_summary.npz"
    np.savez_compressed(
        summary,
        **{f"{name}_count": np.int64(len(values)) for name, values in pooled.items()},
    )
    if by_seed is not None:
        stats_path = args.figure_dir / "Supple2_input_gate_sign_vs_mag_9sector_10seed_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "scope": "ten training seeds pooled equally over all nine sectors",
                    "gap": "mean(delta_g | W>0) - mean(delta_g | W<0) in shared |W| band",
                    "slopes": "per-seed OLS delta_g ~ |W| within the same shared |W| band",
                    "levels": "per-seed mean delta_g over all group connections and sectors",
                    "groups": _seed_level_stats(by_seed),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return gate, delta, summary


def main() -> None:
    """Dispatch per-seed collection or pooled Supplementary 2 rendering."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    elif args.connection_stats_only:
        if not args.all_sectors:
            raise ValueError("--connection_stats_only requires --all_sectors.")
        print(f"Saved {write_connection_level_stats(args.data_root, args.figure_dir)}")
    else:
        for path in plot(args):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
