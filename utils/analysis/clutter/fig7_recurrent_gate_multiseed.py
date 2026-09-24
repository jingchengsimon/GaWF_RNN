"""Build compact ten-seed recurrent-gate inputs and render Figure 7/Supplementary 3.

``collect`` reconstructs recurrent gates from one saved trajectory, accumulates equal-n Digit
and Sector condition means, attaches seed-specific hidden tuned masks, and saves one compact NPZ.
``plot`` reads exactly ten NPZ files, performs seed-level Figure 7 inference, writes horizontal
and vertical PDFs, and renders pooled ten-seed Supplementary 3 gate/delta-gate PNG diagnostics.
``--supple3_stats_only`` writes seed-level sign-specific slopes and overall delta levels without
rerendering figures. Its ``--afferent`` view transposes the recurrent connection axes before
grouping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.stats import f as f_distribution

from utils.analysis.clutter.fig6_encoder_sector_patterns import _equal_n_condition_mask
from utils.analysis.clutter.fig3_gate_distribution import exclude_zero_feedback_reset_frames
from utils.analysis.clutter.fig7_recurrent_gate_cache import group_masks
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import (
    _cross_seed_stats,
    exact_sign_flip_p,
    overlap_band_sign_stats,
)
from utils.analysis.clutter.fig7_recurrent_gate_disinhibition_delta import (
    _render_poster_delta,
)
from utils.analysis.clutter.fig7_recurrent_gate_sign_magnitude import (
    DELTA_TITLE,
    DELTA_Y_LABEL,
    GROUP_NAMES,
    compute_overlap_band,
    render_figure_zoomed,
)
from utils.analysis.clutter.fig7_relevance_stats import relevance_masks

RESULT_NAME = "recurrent_gate_condition_means.npz"
VARIABLES = ("digit", "sector")
CONFIG = {"digit": (10, 0), "sector": (9, 1)}
TOP_FRACTION = 0.10


def parse_args() -> argparse.Namespace:
    """Parse compact per-seed collection or ten-seed plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--trajectory", required=True, type=Path)
    collect_parser.add_argument("--selectivity", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--selection_seed", type=int, default=260718)
    collect_parser.add_argument("--gate_tau", type=float, default=0.5)
    collect_parser.add_argument("--gate_chunk_size", type=int, default=64)
    collect_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--data_root", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--summary_dir", required=True, type=Path)
    plot_parser.add_argument("--fig7_only", action="store_true")
    plot_parser.add_argument("--supple3_delta_only", action="store_true")
    plot_parser.add_argument("--supple3_stats_only", action="store_true")
    plot_parser.add_argument("--afferent", action="store_true")
    return parser.parse_args()


def _recurrent_gate_chunks(
    feedback: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    input_size: int,
    tau: float,
    chunk_size: int,
    device: str,
):
    """Yield exact recurrent-gate chunks without materializing the larger input gate."""

    flat_feedback = feedback.reshape(-1, feedback.shape[-1])
    target = torch.device(device)
    u_tensor = torch.from_numpy(u).to(target)
    recurrent_v = torch.from_numpy(v[:, input_size:]).to(target)
    for start in range(0, flat_feedback.shape[0], chunk_size):
        end = min(start + chunk_size, flat_feedback.shape[0])
        feedback_tensor = torch.from_numpy(flat_feedback[start:end]).to(target)
        with torch.no_grad():
            bounded = feedback_tensor.to(torch.float32).clamp(-10, 10).unsqueeze(2)
            scaled_u = u_tensor.unsqueeze(0) * bounded.transpose(1, 2)
            gate = torch.sigmoid(torch.matmul(scaled_u, recurrent_v) / tau)
        yield start, end, gate.cpu().numpy()


def _hidden_tuned_masks(selectivity_path: Path, kind: str) -> np.ndarray:
    """Return context-by-hidden tuned masks using the established FDR eligibility rule."""

    with np.load(selectivity_path, allow_pickle=False) as arrays:
        tuning = np.asarray(arrays[f"primary_hidden_tuning_{kind}"], dtype=np.float64)
        passed = np.asarray(arrays[f"primary_hidden_passed_{kind}"], dtype=bool)
        dominant = np.asarray(arrays["primary_hidden_interaction_dominant"], dtype=bool)
    return relevance_masks(tuning, passed & ~dominant, TOP_FRACTION)


def collect(args: argparse.Namespace) -> Path:
    """Save one seed's compact recurrent-gate condition means and tuned masks."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    with np.load(args.trajectory, allow_pickle=False) as arrays:
        raw_feedback = np.asarray(arrays["feedback"], dtype=np.float32)
        raw_labels = np.asarray(arrays["labels"], dtype=np.int64)
        u = np.asarray(arrays["U"], dtype=np.float32)
        v = np.asarray(arrays["V"], dtype=np.float32)
        weight = np.asarray(arrays["weight_hh"], dtype=np.float32)
        input_size = int(np.asarray(arrays["weight_ih"]).shape[1])
    feedback, labels, _reset_frames = exclude_zero_feedback_reset_frames(raw_feedback, raw_labels)
    selections = {}
    sums = {}
    counts = {}
    original_counts = {}
    targets = {}
    for kind, (levels, label_column) in CONFIG.items():
        selected, target, original = _equal_n_condition_mask(
            labels[:, label_column], levels, args.selection_seed
        )
        selections[kind] = selected
        sums[kind] = np.zeros((levels, *weight.shape), dtype=np.float64)
        counts[kind] = np.zeros(levels, dtype=np.int64)
        original_counts[kind] = original
        targets[kind] = target
    for start, end, gate in _recurrent_gate_chunks(
        feedback, u, v, input_size, args.gate_tau, args.gate_chunk_size, args.device
    ):
        chunk_labels = labels[start:end]
        for kind, (levels, label_column) in CONFIG.items():
            chosen = selections[kind][start:end]
            for level in range(levels):
                use = chosen & (chunk_labels[:, label_column] == level)
                if np.any(use):
                    sums[kind][level] += gate[use].sum(axis=0, dtype=np.float64)
                    counts[kind][level] += int(use.sum())
    for kind in VARIABLES:
        if not np.all(counts[kind] == targets[kind]):
            raise RuntimeError(
                f"{kind} equal-n mismatch: expected {targets[kind]}, got {counts[kind].tolist()}"
            )
    payload: dict[str, np.ndarray] = {
        "seed": np.asarray(args.seed, dtype=np.int64),
        "selection_seed": np.asarray(args.selection_seed, dtype=np.int64),
        "weight": weight,
    }
    for kind in VARIABLES:
        payload[f"{kind}_gate_mean"] = (sums[kind] / counts[kind][:, None, None]).astype(
            np.float32
        )
        payload[f"{kind}_tuned"] = _hidden_tuned_masks(args.selectivity, kind).astype(np.uint8)
        payload[f"{kind}_selected_count"] = counts[kind]
        payload[f"{kind}_original_count"] = original_counts[kind].astype(np.int64)
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / RESULT_NAME
    np.savez_compressed(destination, **payload)
    return destination


def _pooled_records(
    arrays: np.lib.npyio.NpzFile, kind: str, *, afferent: bool = False
) -> dict[str, pd.DataFrame]:
    """Build group records; the afferent view transposes source/destination axes."""

    weight = np.asarray(arrays["weight"], dtype=np.float64)
    means = np.asarray(arrays[f"{kind}_gate_mean"], dtype=np.float64)
    if afferent:
        weight = weight.T
        means = means.transpose(0, 2, 1)
    tuned = np.asarray(arrays[f"{kind}_tuned"], dtype=bool)
    grand = means.mean(axis=0)
    hidden_size = weight.shape[0]
    rows: dict[str, list[pd.DataFrame]] = {group: [] for group in GROUP_NAMES}
    for context in range(means.shape[0]):
        masks = group_masks(tuned[context], ~tuned[context])
        for group, (source, destination) in masks.items():
            mask = destination[:, None] & source[None, :] & (weight != 0.0)
            dst, src = np.where(mask)
            values = weight[dst, src]
            rows[group].append(
                pd.DataFrame(
                    {
                        "absW": np.abs(values),
                        "of": means[context, dst, src],
                        "delta_of": means[context, dst, src] - grand[dst, src],
                        "signpos": (values > 0.0).astype(np.int64),
                        "context": context,
                        "conn": dst.astype(np.int64) * hidden_size + src,
                    }
                )
            )
    return {group: pd.concat(group_rows, ignore_index=True) for group, group_rows in rows.items()}


def _overlap_stats(pooled: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Return the shared positive/negative weight-magnitude band for each group."""

    return {
        group: compute_overlap_band(
            frame.loc[frame["signpos"] == 1, "absW"].to_numpy(),
            frame.loc[frame["signpos"] == 0, "absW"].to_numpy(),
        )
        for group, frame in pooled.items()
    }


def _compact_paths(data_root: Path) -> list[Path]:
    """Return per-seed compact files from their canonical compact subdirectories."""

    return sorted(data_root.glob(f"seed*/compact/{RESULT_NAME}"))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Ordinary Least Squares slope with an intercept."""

    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise RuntimeError("Cannot fit a slope to constant |W| values.")
    return float(np.dot(centered, y - y.mean()) / denominator)


def _sign_magnitude_seed_metrics(pooled: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """Return one seed's sign-specific overlap slopes and overall delta levels."""

    overlap = _overlap_stats(pooled)
    result: dict[str, dict[str, float]] = {}
    for group in GROUP_NAMES:
        frame = pooled[group]
        band = frame.loc[
            frame["absW"].between(
                overlap[group]["overlap_low"],
                overlap[group]["overlap_high"],
                inclusive="both",
            )
        ]
        result[group] = {"overall_delta_level": float(frame["delta_of"].mean())}
        for sign, label in ((1, "positive_overlap_slope"), (0, "negative_overlap_slope")):
            selected = band.loc[band["signpos"] == sign]
            result[group][label] = _ols_slope(
                selected["absW"].to_numpy(), selected["delta_of"].to_numpy()
            )
    return result


def _mean_sem(values: list[float]) -> dict[str, object]:
    """Summarize exactly ten independent training-seed values with SEM."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,):
        raise RuntimeError(f"Expected ten seed values, got {array.shape}.")
    mean = float(array.mean())
    return {
        "mean": mean,
        "sem": float(array.std(ddof=1) / np.sqrt(array.size)),
        "seed_values": array.tolist(),
    }


def _group_variable_interaction(summary_dir: Path) -> dict[str, float | int]:
    """Fit the seed-level group-by-variable interaction on saved sign-gap cells."""

    summary_path = summary_dir / "fig7_seed_level_summary.npz"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing Figure 7 seed-level summary: {summary_path}")
    values = np.empty((10, len(GROUP_NAMES), len(VARIABLES)), dtype=np.float64)
    with np.load(summary_path, allow_pickle=False) as arrays:
        for group_index, group in enumerate(GROUP_NAMES):
            for variable_index, variable in enumerate(VARIABLES):
                cell = np.asarray(arrays[f"{variable}_{group}_gap_seed_values"], dtype=np.float64)
                if cell.shape != (10,):
                    raise RuntimeError(
                        f"Expected ten {variable}/{group} sign-gap values, got {cell.shape}."
                    )
                values[:, group_index, variable_index] = cell
    grand = values.mean()
    subject_mean = values.mean(axis=(1, 2), keepdims=True)
    group_mean = values.mean(axis=(0, 2), keepdims=True)
    variable_mean = values.mean(axis=(0, 1), keepdims=True)
    cell_mean = values.mean(axis=0, keepdims=True)
    interaction = cell_mean - group_mean - variable_mean + grand
    residual = (
        values
        - values.mean(axis=2, keepdims=True)
        - values.mean(axis=1, keepdims=True)
        - cell_mean
        + subject_mean
        + group_mean
        + variable_mean
        - grand
    )
    df_num = (len(GROUP_NAMES) - 1) * (len(VARIABLES) - 1)
    df_den = (values.shape[0] - 1) * df_num
    ss_effect = float(values.shape[0] * np.square(interaction).sum())
    ss_error = float(np.square(residual).sum())
    statistic = (ss_effect / df_num) / (ss_error / df_den)
    return {
        "f_statistic": float(statistic),
        "df_num": df_num,
        "df_den": df_den,
        "p_value": float(f_distribution.sf(statistic, df_num, df_den)),
        "ss_effect": ss_effect,
        "ss_error": ss_error,
    }


def write_supple3_seed_stats(args: argparse.Namespace) -> Path:
    """Write Session-6 seed-level sign/magnitude statistics from existing compact arrays."""

    paths = _compact_paths(args.data_root)
    if len(paths) != 10:
        raise RuntimeError(
            f"Expected ten compact seed files in {args.data_root}, found {len(paths)}"
        )
    collected = {
        kind: {
            group: {
                "positive_overlap_slope": [],
                "negative_overlap_slope": [],
                "overall_delta_level": [],
            }
            for group in GROUP_NAMES
        }
        for kind in VARIABLES
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for kind in VARIABLES:
                metrics = _sign_magnitude_seed_metrics(
                    _pooled_records(arrays, kind, afferent=args.afferent)
                )
                for group in GROUP_NAMES:
                    for metric, value in metrics[group].items():
                        collected[kind][group][metric].append(value)
    summary = {
        kind: {
            group: {metric: _mean_sem(values) for metric, values in metrics.items()}
            for group, metrics in groups.items()
        }
        for kind, groups in collected.items()
    }
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_afferent" if args.afferent else ""
    destination = args.summary_dir / f"supple3_seed_level_sign_magnitude_stats{suffix}.json"
    destination.write_text(
        json.dumps(
            {
                "scope": "ten training seeds; Digit/Sector x TT/TR/RT/RR",
                "slope": (
                    "Per seed and sign, OLS delta_g ~ |W| within that seed/group shared "
                    "positive/negative |W| overlap band"
                ),
                "level": "Per-seed mean delta_g over all group connections and contexts",
                "inference_unit": "training seed; mean +/- SEM",
                "group_variable_interaction": _group_variable_interaction(args.summary_dir),
                "groups": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def plot(args: argparse.Namespace) -> list[Path]:
    """Render formal ten-seed Figure 7 and pooled ten-seed Supplementary 3 outputs."""

    paths = _compact_paths(args.data_root)
    if len(paths) != 10:
        raise RuntimeError(
            f"Expected ten compact seed files in {args.data_root}, found {len(paths)}"
        )
    per_seed: dict[str, list[dict[str, dict]]] = {kind: [] for kind in VARIABLES}
    supplementary: dict[str, dict[str, list[pd.DataFrame]]] = {
        kind: {group: [] for group in GROUP_NAMES} for kind in VARIABLES
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for kind in VARIABLES:
                pooled = _pooled_records(arrays, kind, afferent=args.afferent)
                overlap = _overlap_stats(pooled)
                per_seed[kind].append(overlap_band_sign_stats(pooled, overlap, y_col="delta_of"))
                for group in GROUP_NAMES:
                    supplementary[kind][group].append(pooled[group])
    stats_by_variable = {kind: _cross_seed_stats(per_seed[kind]) for kind in VARIABLES}
    gap_by_variable = {
        kind: {
            group: stats_by_variable[kind][group]["+"]["mean"]
            - stats_by_variable[kind][group]["-"]["mean"]
            for group in GROUP_NAMES
        }
        for kind in VARIABLES
    }
    raw_p: dict[tuple[str, str, str], float] = {}
    seed_values: dict[tuple[str, str, str], np.ndarray] = {}
    for kind in VARIABLES:
        for group in GROUP_NAMES:
            positive = np.asarray([record[group]["+"]["mean"] for record in per_seed[kind]])
            negative = np.asarray([record[group]["-"]["mean"] for record in per_seed[kind]])
            seed_values[(kind, group, "+")] = positive
            seed_values[(kind, group, "-")] = negative
            seed_values[(kind, group, "gap")] = positive - negative
            for test in ("+", "-", "gap"):
                raw_p[(kind, group, test)] = exact_sign_flip_p(seed_values[(kind, group, test)])
    gap_p = {
        kind: {group: raw_p[(kind, group, "gap")] for group in GROUP_NAMES}
        for kind in VARIABLES
    }
    bar_tests = {
        kind: {
            group: {test: raw_p[(kind, group, test)] for test in ("+", "-", "gap")}
            for group in GROUP_NAMES
        }
        for kind in VARIABLES
    }
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_afferent" if args.afferent else ""
    horizontal = args.figure_dir / f"Fig7_recurrent_gate_disinhibition_poster_delta_10seed{suffix}"
    vertical = args.figure_dir / (
        f"Fig7_recurrent_gate_disinhibition_poster_delta_vertical_10seed{suffix}"
    )
    _render_poster_delta(
        stats_by_variable,
        gap_by_variable,
        gap_p,
        horizontal,
        significance_tests=bar_tests,
    )
    _render_poster_delta(
        stats_by_variable,
        gap_by_variable,
        gap_p,
        vertical,
        vertical=True,
        significance_tests=bar_tests,
    )
    outputs = [horizontal.with_suffix(".pdf"), vertical.with_suffix(".pdf")]
    if not args.fig7_only:
        for kind in VARIABLES:
            pooled = {
                group: pd.concat(supplementary[kind][group], ignore_index=True)
                for group in GROUP_NAMES
            }
            overlap = _overlap_stats(pooled)
            gate = args.figure_dir / (
                f"Supple3_rec_gate_sign_vs_mag_disinh_{kind}_zoom_10seed{suffix}.png"
            )
            delta = args.figure_dir / (
                f"Supple3_rec_gate_sign_vs_mag_disinh_{kind}_delta_zoom_10seed{suffix}.png"
            )
            if not args.supple3_delta_only:
                render_figure_zoomed(
                    pooled,
                    overlap,
                    gate,
                    y_col="of",
                    y_label="gate open fraction",
                    title=f"Recurrent gate vs. |W|, split by sign(W) - {kind}, 10 seeds",
                )
                outputs.append(gate)
            render_figure_zoomed(
                pooled,
                overlap,
                delta,
                y_col="delta_of",
                y_label=DELTA_Y_LABEL.format(kind=kind),
                title=f"{DELTA_TITLE.format(kind=kind)} - 10 seeds",
            )
            outputs.append(delta)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        f"{kind}_{group}_{test}_seed_values": values.astype(np.float32)
        for (kind, group, test), values in seed_values.items()
    }
    summary_payload.update(
        {
            f"{kind}_{group}_{test}_p_raw": np.asarray(raw_p[(kind, group, test)])
            for kind, group, test in raw_p
        }
    )
    np.savez_compressed(
        args.summary_dir / f"fig7_seed_level_summary{suffix}.npz", **summary_payload
    )
    return outputs


def main() -> None:
    """Dispatch compact collection or formal ten-seed rendering."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    elif args.supple3_stats_only:
        print(f"Saved {write_supple3_seed_stats(args)}")
    else:
        for output in plot(args):
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
