"""Analyze GaWF gate context specificity after the mandatory Part 0 audit.

The script reconstructs gates in chunks and immediately reduces them.  Full-gate point
estimates are exact; trial bootstrap and label-shuffle inference use a fixed reproducible
synapse sample because saving the full trial-by-synapse gate tensor is intentionally avoided.

Inputs: the compact GaWF trajectory, trained checkpoint, and Clutter test stimulus.
Outputs: compact NPZ/JSON summaries for Parts 1--3; no dense per-frame gate array is saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset
from utils_anal.gawf_gate_distribution import _gini, iter_gate_chunks


@dataclass
class GateAggregate:
    """Streaming sufficient statistics for one gate tensor."""

    shape: tuple[int, int]
    sample_index: np.ndarray
    joint_sum: np.ndarray
    joint_half: np.ndarray
    equal_joint_sum: np.ndarray
    equal_joint_half: np.ndarray
    equal_marginal_sum: dict[str, np.ndarray]
    equal_marginal_half: dict[str, np.ndarray]
    split_sum: dict[str, np.ndarray]
    split_half: dict[str, np.ndarray]
    joint_sumsq: np.ndarray
    equal_joint_sumsq: np.ndarray
    sample_values: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="./results/anal_data/gawf_gate_audit/gawf_gate_trajectory.npz",
    )
    parser.add_argument(
        "--part0",
        default="./results/anal_data/gawf_gate_context_specificity/part0_prerequisites.json",
    )
    parser.add_argument(
        "--save_dir",
        default="./results/anal_data/gawf_gate_context_specificity",
    )
    parser.add_argument(
        "--ckpt",
        default=(
            "./results/train_data/clutter/best_6model_param_matched_40h/"
            "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
        ),
    )
    parser.add_argument("--data_dir", default="./stimuli")
    parser.add_argument(
        "--data_suffix",
        default="40h-float32-nonjoint-10digit-unique-bg-causal-continuous",
    )
    parser.add_argument("--gate_chunk_size", type=int, default=128)
    parser.add_argument("--gate_tau", type=float, default=0.5)
    parser.add_argument("--point_tolerance", type=float, default=1e-6)
    parser.add_argument("--sample_synapses", type=int, default=128)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _balanced_masks(
    digits: np.ndarray, sectors: np.ndarray, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return reproducible equal-n marginal and equal-cell masks."""

    rng = np.random.default_rng(seed)
    masks: dict[str, np.ndarray] = {}
    for name, labels, levels in (("sector", sectors, 9), ("digit", digits, 10)):
        target = min(int(np.count_nonzero(labels == level)) for level in range(levels))
        mask = np.zeros(labels.size, dtype=bool)
        for level in range(levels):
            indices = np.flatnonzero(labels == level)
            mask[rng.choice(indices, size=target, replace=False)] = True
        masks[name] = mask
    joint = sectors * 10 + digits
    target_joint = min(int(np.count_nonzero(joint == cell)) for cell in range(90))
    joint_mask = np.zeros(joint.size, dtype=bool)
    for cell in range(90):
        indices = np.flatnonzero(joint == cell)
        joint_mask[rng.choice(indices, size=target_joint, replace=False)] = True
    return masks, joint_mask


def _split_codes(labels: np.ndarray, levels: int, seed: int) -> np.ndarray:
    """Assign each observation to a reproducible within-level split half."""

    rng = np.random.default_rng(seed)
    halves = np.empty(labels.size, dtype=np.int8)
    for level in range(levels):
        indices = np.flatnonzero(labels == level)
        indices = rng.permutation(indices)
        halves[indices[: indices.size // 2]] = 0
        halves[indices[indices.size // 2 :]] = 1
    return labels * 2 + halves


def _new_aggregate(
    shape: tuple[int, int], n_frames: int, sample_synapses: int, rng: np.random.Generator
) -> GateAggregate:
    """Allocate one gate's sufficient statistics."""

    features = int(np.prod(shape))
    sample_index = np.sort(rng.choice(features, size=min(sample_synapses, features), replace=False))
    zeros90 = lambda dtype: np.zeros((90, features), dtype=dtype)
    return GateAggregate(
        shape=shape,
        sample_index=sample_index,
        joint_sum=zeros90(np.float64),
        joint_half=zeros90(np.int32),
        equal_joint_sum=zeros90(np.float64),
        equal_joint_half=zeros90(np.int32),
        equal_marginal_sum={
            "sector": np.zeros((9, features), dtype=np.float64),
            "digit": np.zeros((10, features), dtype=np.float64),
        },
        equal_marginal_half={
            "sector": np.zeros((9, features), dtype=np.int32),
            "digit": np.zeros((10, features), dtype=np.int32),
        },
        split_sum={
            "sector": np.zeros((18, features), dtype=np.float64),
            "digit": np.zeros((20, features), dtype=np.float64),
        },
        split_half={
            "sector": np.zeros((18, features), dtype=np.int32),
            "digit": np.zeros((20, features), dtype=np.int32),
        },
        joint_sumsq=np.zeros(features, dtype=np.float64),
        equal_joint_sumsq=np.zeros(features, dtype=np.float64),
        sample_values=np.empty((n_frames, sample_index.size), dtype=np.float32),
    )


def _add_by_code(target: np.ndarray, values: np.ndarray, codes: np.ndarray) -> None:
    """Add rows of a chunk into feature sums indexed by a small integer code."""

    flat = values.reshape(values.shape[0], -1)
    for code in np.unique(codes):
        mask = codes == code
        target[int(code)] += flat[mask].sum(axis=0, dtype=np.float64)


def _add_half_by_code(
    target: np.ndarray, values: np.ndarray, codes: np.ndarray, tolerance: float
) -> None:
    """Count entries near 0.5 by code."""

    flat = values.reshape(values.shape[0], -1)
    near = np.abs(flat - 0.5) < tolerance
    for code in np.unique(codes):
        target[int(code)] += near[codes == code].sum(axis=0, dtype=np.int32)


def stream_gate_aggregates(
    trajectory: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, GateAggregate], dict[str, Any]]:
    """Reconstruct all gates once and collect exact sufficient statistics."""

    feedback = trajectory["feedback"].astype(np.float32, copy=False)
    labels = trajectory["labels"].reshape(-1, 2).astype(np.int64, copy=False)
    digits, sectors = labels[:, 0], labels[:, 1]
    joint = sectors * 10 + digits
    equal_masks, equal_joint_mask = _balanced_masks(digits, sectors, args.seed)
    split_codes = {
        "sector": _split_codes(sectors, 9, args.seed + 11),
        "digit": _split_codes(digits, 10, args.seed + 17),
    }
    rng = np.random.default_rng(args.seed + 23)
    aggregates = {
        "input": _new_aggregate(
            tuple(trajectory["weight_ih"].shape), labels.shape[0], args.sample_synapses, rng
        ),
        "recurrent": _new_aggregate(
            tuple(trajectory["weight_hh"].shape), labels.shape[0], args.sample_synapses, rng
        ),
    }
    input_size = int(trajectory["weight_ih"].shape[1])
    started = time.perf_counter()
    total_chunks = int(np.ceil(labels.shape[0] / args.gate_chunk_size))
    for chunk_idx, (start, end, gate_input, gate_recurrent) in enumerate(
        iter_gate_chunks(
            feedback,
            trajectory["U"].astype(np.float32, copy=False),
            trajectory["V"].astype(np.float32, copy=False),
            input_size,
            args.gate_tau,
            args.gate_chunk_size,
        )
    ):
        for name, gate in (("input", gate_input), ("recurrent", gate_recurrent)):
            agg = aggregates[name]
            flat = gate.reshape(gate.shape[0], -1)
            codes = joint[start:end]
            _add_by_code(agg.joint_sum, gate, codes)
            _add_half_by_code(agg.joint_half, gate, codes, args.point_tolerance)
            agg.joint_sumsq += np.square(flat, dtype=np.float64).sum(axis=0)
            selected = equal_joint_mask[start:end]
            if np.any(selected):
                _add_by_code(agg.equal_joint_sum, gate[selected], codes[selected])
                _add_half_by_code(
                    agg.equal_joint_half, gate[selected], codes[selected], args.point_tolerance
                )
                agg.equal_joint_sumsq += np.square(flat[selected], dtype=np.float64).sum(axis=0)
            for factor, factor_labels in (("sector", sectors), ("digit", digits)):
                marginal_selected = equal_masks[factor][start:end]
                chunk_factor = factor_labels[start:end]
                if np.any(marginal_selected):
                    _add_by_code(
                        agg.equal_marginal_sum[factor],
                        gate[marginal_selected],
                        chunk_factor[marginal_selected],
                    )
                    _add_half_by_code(
                        agg.equal_marginal_half[factor],
                        gate[marginal_selected],
                        chunk_factor[marginal_selected],
                        args.point_tolerance,
                    )
                _add_by_code(agg.split_sum[factor], gate, split_codes[factor][start:end])
                _add_half_by_code(
                    agg.split_half[factor],
                    gate,
                    split_codes[factor][start:end],
                    args.point_tolerance,
                )
            agg.sample_values[start:end] = flat[:, agg.sample_index]
        if (chunk_idx + 1) % 25 == 0 or chunk_idx + 1 == total_chunks:
            print(
                f"context aggregate chunks {chunk_idx + 1}/{total_chunks} | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    design = {
        "digits": digits,
        "sectors": sectors,
        "joint": joint,
        "equal_masks": equal_masks,
        "equal_joint_mask": equal_joint_mask,
        "split_codes": split_codes,
    }
    return aggregates, design


def _means(
    sums: np.ndarray,
    counts: np.ndarray,
    half_counts: np.ndarray | None,
    exclude_half: bool,
) -> np.ndarray:
    """Convert sums to means, optionally treating entries near 0.5 as missing."""

    denom = np.asarray(counts, dtype=np.int64)
    while denom.ndim < sums.ndim:
        denom = denom[..., None]
    if not exclude_half:
        return sums / denom
    if half_counts is None:
        raise ValueError("half_counts required for point-mass exclusion")
    valid = denom - half_counts
    return np.divide(
        sums - 0.5 * half_counts,
        valid,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=valid > 0,
    )


def _delta_summary(delta: np.ndarray) -> dict[str, float | int]:
    """Summarize a condition-by-synapse delta array."""

    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "iqr": float(q75 - q25),
        "fraction_abs_gt_0_1": float(np.mean(np.abs(values) > 0.1)),
        "fraction_abs_gt_0_25": float(np.mean(np.abs(values) > 0.25)),
    }


def _corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid].reshape(-1), b[valid].reshape(-1))[0, 1])


def part1_point_estimates(
    aggregates: dict[str, GateAggregate], design: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute full/equal-n and point-included/excluded delta summaries."""

    sectors, digits, joint = design["sectors"], design["digits"], design["joint"]
    joint_counts = np.bincount(joint, minlength=90).reshape(9, 10)
    output: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for gate_name, agg in aggregates.items():
        output[gate_name] = {}
        for exclude in (False, True):
            point_key = "point_excluded" if exclude else "point_included"
            output[gate_name][point_key] = {}
            for factor, axis, labels, levels in (
                ("sector", 1, sectors, 9),
                ("digit", 0, digits, 10),
            ):
                sums = agg.joint_sum.reshape(9, 10, -1)
                halves = agg.joint_half.reshape(9, 10, -1)
                if factor == "sector":
                    full_sum = sums.sum(axis=1)
                    full_half = halves.sum(axis=1)
                else:
                    full_sum = sums.sum(axis=0)
                    full_half = halves.sum(axis=0)
                full_mean = _means(
                    full_sum,
                    np.bincount(labels, minlength=levels),
                    full_half,
                    exclude,
                )
                grand = _means(
                    full_sum.sum(axis=0, keepdims=True),
                    np.asarray([labels.size]),
                    full_half.sum(axis=0, keepdims=True),
                    exclude,
                )[0]
                delta = full_mean - grand
                arrays[f"delta_{gate_name}_{factor}_full_{point_key}"] = delta.astype(np.float32)

                equal_count = int(np.count_nonzero(design["equal_masks"][factor])) // levels
                equal_mean = _means(
                    agg.equal_marginal_sum[factor],
                    np.full(levels, equal_count),
                    agg.equal_marginal_half[factor],
                    exclude,
                )
                if exclude:
                    equal_grand = _means(
                        agg.equal_marginal_sum[factor].sum(axis=0, keepdims=True),
                        np.asarray([equal_count * levels]),
                        agg.equal_marginal_half[factor].sum(axis=0, keepdims=True),
                        True,
                    )[0]
                else:
                    equal_grand = equal_mean.mean(axis=0)
                equal_delta = equal_mean - equal_grand
                arrays[f"delta_{gate_name}_{factor}_equal_n_{point_key}"] = equal_delta.astype(
                    np.float32
                )

                split_count = np.bincount(design["split_codes"][factor], minlength=2 * levels)
                split_mean = _means(
                    agg.split_sum[factor], split_count, agg.split_half[factor], exclude
                ).reshape(levels, 2, -1)
                split_sum = agg.split_sum[factor].reshape(levels, 2, -1).sum(axis=0)
                split_near = agg.split_half[factor].reshape(levels, 2, -1).sum(axis=0)
                split_grand = _means(
                    split_sum,
                    split_count.reshape(levels, 2).sum(axis=0),
                    split_near,
                    exclude,
                )
                split_delta = split_mean - split_grand[None, :, :]
                split_a, split_b = split_delta[:, 0], split_delta[:, 1]
                output[gate_name][point_key][factor] = {
                    "full": _delta_summary(delta),
                    "equal_n": _delta_summary(equal_delta),
                    "split_half": {
                        "correlation": _corr_flat(split_a, split_b),
                        "std_difference": float(np.nanstd(split_a - split_b)),
                    },
                }

            spatial_source = arrays[
                f"delta_{gate_name}_sector_equal_n_{point_key}"
            ]
            if gate_name == "input":
                sector_maps = spatial_source.reshape(9, 256, 32, 6, 6).mean(axis=(1, 2))
                digit_source = arrays[f"delta_input_digit_equal_n_{point_key}"]
                digit_maps = digit_source.reshape(10, 256, 32, 6, 6).mean(axis=(1, 2))
                arrays[f"spatial_sector_{point_key}"] = sector_maps.astype(np.float32)
                arrays[f"spatial_digit_{point_key}"] = digit_maps.astype(np.float32)
    return output, arrays


def _sample_delta_std(
    values: np.ndarray,
    order: np.ndarray,
    counts: np.ndarray,
    exclude_half: bool,
    equal_n: bool,
    tolerance: float,
) -> float:
    """Compute shuffled marginal delta spread on sampled synapses."""

    if equal_n:
        count = int(counts.min())
        use_n = count * counts.size
        ordered = values[order[:use_n]].reshape(counts.size, count, -1)
        valid = np.abs(ordered - 0.5) >= tolerance
        if exclude_half:
            denominator = valid.sum(axis=1)
            means = np.divide(
                np.where(valid, ordered, 0.0).sum(axis=1),
                denominator,
                out=np.full((counts.size, values.shape[1]), np.nan, dtype=np.float64),
                where=denominator > 0,
            )
        else:
            means = ordered.mean(axis=1)
    else:
        ordered = values[order]
        boundaries = np.r_[0, np.cumsum(counts)[:-1]]
        if exclude_half:
            valid = np.abs(ordered - 0.5) >= tolerance
            sums = np.add.reduceat(np.where(valid, ordered, 0.0), boundaries, axis=0)
            denom = np.add.reduceat(valid, boundaries, axis=0)
            means = np.divide(
                sums,
                denom,
                out=np.full_like(sums, np.nan, dtype=np.float64),
                where=denom > 0,
            )
        else:
            means = np.add.reduceat(ordered, boundaries, axis=0) / counts[:, None]
    return float(np.nanstd(means - np.nanmean(means, axis=0)))


def label_shuffle_inference(
    aggregates: dict[str, GateAggregate],
    design: dict[str, Any],
    part1: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run 1000 sampled-synapse label shuffles for both marginal factors."""

    rng = np.random.default_rng(args.seed + 101)
    result: dict[str, Any] = {"method": {
        "resamples": args.resamples,
        "synapses_per_gate": args.sample_synapses,
        "note": "Full-gate point estimates; shuffled null uses fixed sampled synapses.",
    }}
    for factor, labels, levels in (
        ("sector", design["sectors"], 9),
        ("digit", design["digits"], 10),
    ):
        counts = np.bincount(labels, minlength=levels)
        result[factor] = {}
        orders = [rng.permutation(labels.size) for _ in range(args.resamples)]
        observed_orders = {
            "full": np.argsort(labels, kind="stable"),
            "equal_n": np.concatenate(
                [
                    rng.permutation(np.flatnonzero(labels == level))[: counts.min()]
                    for level in range(levels)
                ]
            ),
        }
        for gate_name, agg in aggregates.items():
            result[factor][gate_name] = {}
            for exclude in (False, True):
                point_key = "point_excluded" if exclude else "point_included"
                result[factor][gate_name][point_key] = {}
                for equal_n in (False, True):
                    balance_key = "equal_n" if equal_n else "full"
                    null = np.asarray([
                        _sample_delta_std(
                            agg.sample_values,
                            order,
                            counts,
                            exclude,
                            equal_n,
                            args.point_tolerance,
                        )
                        for order in orders
                    ])
                    observed_full = float(
                        part1[gate_name][point_key][factor][balance_key]["std"]
                    )
                    observed_sample = _sample_delta_std(
                        agg.sample_values,
                        observed_orders[balance_key],
                        counts,
                        exclude,
                        equal_n,
                        args.point_tolerance,
                    )
                    null_mean = float(null.mean())
                    null_std = float(null.std(ddof=1))
                    result[factor][gate_name][point_key][balance_key] = {
                        "observed_full_gate_std": observed_full,
                        "observed_sampled_synapse_std": observed_sample,
                        "null_sampled_synapse_mean": null_mean,
                        "null_sampled_synapse_std": null_std,
                        "z_score": (observed_sample - null_mean) / null_std,
                        "empirical_p_greater": float(
                            (1 + np.count_nonzero(null >= observed_sample))
                            / (args.resamples + 1)
                        ),
                        "null_quantiles_2_5_50_97_5": np.quantile(
                            null, [0.025, 0.5, 0.975]
                        ).tolist(),
                    }
    return result


def _marginal_variance(cell_mean: np.ndarray) -> dict[str, Any]:
    """Match utils_anal/5_dpca_marginalized_variance.py on a balanced 9x10 tensor."""

    grand = np.nanmean(cell_mean, axis=(0, 1))
    sector = np.nanmean(cell_mean, axis=1) - grand
    digit = np.nanmean(cell_mean, axis=0) - grand
    interaction = cell_mean - grand - sector[:, None] - digit[None, :]
    components = {
        "sector": float(np.nansum(np.square(sector[:, None])) * cell_mean.shape[1]),
        "digit": float(np.nansum(np.square(digit[None, :])) * cell_mean.shape[0]),
        "interaction": float(np.nansum(np.square(interaction))),
    }
    total = float(np.nansum(np.square(cell_mean - grand)))
    return {
        "fractions": {key: value / total for key, value in components.items()},
        "sum_check": sum(components.values()) / total,
        "components": components,
        "total_condition_mean_variance": total,
        "sector_effect": sector,
        "digit_effect": digit,
    }


def part2_point_estimates(
    aggregates: dict[str, GateAggregate], design: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute exact full-gate condition-mean and balanced trial-level decompositions."""

    joint_counts = np.bincount(design["joint"], minlength=90).reshape(9, 10)
    equal_n = int(np.count_nonzero(design["equal_joint_mask"])) // 90
    report: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for gate_name, agg in aggregates.items():
        report[gate_name] = {}
        for exclude in (False, True):
            point_key = "point_excluded" if exclude else "point_included"
            full_cell = _means(
                agg.joint_sum.reshape(9, 10, -1),
                joint_counts,
                agg.joint_half.reshape(9, 10, -1),
                exclude,
            )
            equal_cell = _means(
                agg.equal_joint_sum.reshape(9, 10, -1),
                np.full((9, 10), equal_n),
                agg.equal_joint_half.reshape(9, 10, -1),
                exclude,
            )
            full_result = _marginal_variance(full_cell)
            equal_result = _marginal_variance(equal_cell)
            report[gate_name][point_key] = {
                "full_condition_mean": {k: v for k, v in full_result.items() if "effect" not in k},
                "equal_cell_condition_mean": {
                    k: v for k, v in equal_result.items() if "effect" not in k
                },
            }
            arrays[f"digit_effect_{gate_name}_{point_key}"] = equal_result[
                "digit_effect"
            ].astype(np.float32)
            if not exclude:
                grand = agg.equal_joint_sum.sum(axis=0) / (equal_n * 90)
                total_trial_ss = float(
                    np.sum(agg.equal_joint_sumsq - equal_n * 90 * np.square(grand))
                )
                factor_ss = {
                    key: value * equal_n for key, value in equal_result["components"].items()
                }
                residual = max(0.0, total_trial_ss - sum(factor_ss.values()))
                report[gate_name][point_key]["equal_cell_trial_total"] = {
                    "percent": {
                        **{key: 100.0 * value / total_trial_ss for key, value in factor_ss.items()},
                        "residual": 100.0 * residual / total_trial_ss,
                    },
                    "total_trial_sum_squares": total_trial_ss,
                    "residual_sum_squares": residual,
                }
            else:
                report[gate_name][point_key]["equal_cell_trial_total"] = {
                    "unsupported": (
                        "Entrywise point exclusion gives feature- and cell-specific sample sizes; "
                        "the orthogonal balanced ANOVA residual is therefore not defined."
                    )
                }
    return report, arrays


def _bootstrap_cell_means(
    values: np.ndarray,
    joint: np.ndarray,
    selected: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
    exclude_half: bool,
    tolerance: float,
) -> np.ndarray:
    """Bootstrap equal-cell trial means for sampled features."""

    feature_count = values.shape[1]
    output = np.empty((resamples, 9, 10, feature_count), dtype=np.float32)
    for cell in range(90):
        cell_values = values[selected & (joint == cell)]
        n_cell = cell_values.shape[0]
        weights = rng.multinomial(n_cell, np.full(n_cell, 1.0 / n_cell), size=resamples)
        if exclude_half:
            valid = np.abs(cell_values - 0.5) >= tolerance
            numerator = weights @ np.where(valid, cell_values, 0.0)
            denominator = weights @ valid.astype(np.float32)
            means = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=np.float64),
                where=denominator > 0,
            )
        else:
            means = (weights @ cell_values) / n_cell
        sector, digit = divmod(cell, 10)
        output[:, sector, digit] = means.astype(np.float32)
    return output


def bootstrap_part2(
    aggregates: dict[str, GateAggregate], design: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Compute sampled-synapse 1000-draw trial bootstrap CIs."""

    rng = np.random.default_rng(args.seed + 303)
    result: dict[str, Any] = {"method": {
        "resamples": args.resamples,
        "synapses_per_gate": args.sample_synapses,
        "sampling": "trial bootstrap within each of 90 equal-n cells",
    }}
    for gate_name, agg in aggregates.items():
        result[gate_name] = {}
        for exclude in (False, True):
            point_key = "point_excluded" if exclude else "point_included"
            cells = _bootstrap_cell_means(
                agg.sample_values,
                design["joint"],
                design["equal_joint_mask"],
                args.resamples,
                rng,
                exclude,
                args.point_tolerance,
            )
            grand = np.nanmean(cells, axis=(1, 2), keepdims=True)
            sector = np.nanmean(cells, axis=2, keepdims=True) - grand
            digit = np.nanmean(cells, axis=1, keepdims=True) - grand
            interaction = cells - grand - sector - digit
            ss = np.stack(
                [
                    np.nansum(np.square(sector), axis=(1, 2, 3)) * 10,
                    np.nansum(np.square(digit), axis=(1, 2, 3)) * 9,
                    np.nansum(np.square(interaction), axis=(1, 2, 3)),
                ],
                axis=1,
            )
            fractions = ss / ss.sum(axis=1, keepdims=True)
            result[gate_name][point_key] = {
                factor: {
                    "bootstrap_mean": float(fractions[:, idx].mean()),
                    "ci95": np.quantile(fractions[:, idx], [0.025, 0.975]).tolist(),
                }
                for idx, factor in enumerate(("sector", "digit", "interaction"))
            }
    return result


def _load_positions(tsv_path: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with open(tsv_path, "r", newline="") as file_obj:
        for row in csv.DictReader(file_obj, delimiter="\t"):
            xs.append(float(row["fg_char_x"]))
            ys.append(float(row["fg_char_y"]))
    return np.asarray(xs[start:stop]), np.asarray(ys[start:stop])


def _crop_proxy(frame: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Return nonzero area and intensity in a clipped 28x28 composite crop."""

    x0, y0 = int(round(x - 14)), int(round(y - 14))
    x1, y1 = x0 + 28, y0 + 28
    crop = frame[max(0, y0) : min(frame.shape[0], y1), max(0, x0) : min(frame.shape[1], x1)]
    return float(np.count_nonzero(crop)), float(np.sum(crop, dtype=np.float64))


def collect_encoder_and_proxy(
    args: argparse.Namespace, design: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the exact model encoder and compute the composite-crop ink proxy."""

    dataset, num_pos = build_test_dataset(args)
    model = build_model_from_ckpt(
        os.path.abspath(args.ckpt), num_pos, torch.device("cpu"), args.chan_num
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n_frames = design["digits"].size
    encoded = np.empty((n_frames, 1152), dtype=np.float32)
    raw_path = os.path.join(args.data_dir, f"stimulus_reg-test-{args.data_suffix}.npy")
    tsv_path = os.path.join(args.data_dir, f"stimulus_reg-test-{args.data_suffix}.tsv")
    raw = np.load(raw_path, mmap_mode="r")
    start_frame = args.chan_num
    xs, ys = _load_positions(tsv_path, start_frame, start_frame + n_frames)
    proxy_area = np.empty(n_frames, dtype=np.float64)
    proxy_intensity = np.empty(n_frames, dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames = batch[0].to(dtype=torch.float32)
            batch_encoded = model.encode_frames(frames).cpu().numpy().reshape(-1, 1152)
            end = cursor + batch_encoded.shape[0]
            encoded[cursor:end] = batch_encoded
            raw_indices = np.arange(start_frame + cursor, start_frame + end)
            for local, raw_index in enumerate(raw_indices):
                proxy_area[cursor + local], proxy_intensity[cursor + local] = _crop_proxy(
                    raw[raw_index], xs[cursor + local], ys[cursor + local]
                )
            cursor = end
            if (batch_idx + 1) % 25 == 0 or batch_idx + 1 == len(loader):
                print(f"encoder batches {batch_idx + 1}/{len(loader)}", flush=True)
    if cursor != n_frames:
        raise RuntimeError(f"Encoder frame count mismatch: {cursor} != {n_frames}")
    metrics = {
        "ink_area_proxy": proxy_area,
        "ink_intensity_proxy": proxy_intensity,
        "encoder_l1": np.abs(encoded).sum(axis=1, dtype=np.float64),
        "encoder_l2": np.sqrt(np.square(encoded, dtype=np.float64).sum(axis=1)),
        "encoder_active_gt_0": np.count_nonzero(encoded > 0.0, axis=1).astype(np.float64),
    }
    return encoded, metrics


def _group_mean(values: np.ndarray, labels: np.ndarray, levels: int) -> np.ndarray:
    return np.asarray([values[labels == level].mean(axis=0) for level in range(levels)])


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _gini_by_factor(
    agg: GateAggregate,
    labels: np.ndarray,
    levels: int,
    exclude_half: bool = False,
) -> np.ndarray:
    """Compute exact Gini of each marginal mean gate, with optional 0.5 exclusion."""

    joint_sum = agg.joint_sum.reshape(9, 10, -1)
    joint_half = agg.joint_half.reshape(9, 10, -1)
    sums = joint_sum.sum(axis=1) if levels == 9 else joint_sum.sum(axis=0)
    halves = joint_half.sum(axis=1) if levels == 9 else joint_half.sum(axis=0)
    counts = np.bincount(labels, minlength=levels)
    means = _means(sums, counts, halves, exclude_half)
    return np.asarray([_gini(row) for row in means])


def part3_analysis(
    aggregates: dict[str, GateAggregate],
    design: dict[str, Any],
    encoder: np.ndarray,
    metrics: dict[str, np.ndarray],
    part2_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the proxy-ink, encoder-norm, regression, and symmetric sector controls."""

    report: dict[str, Any] = {
        "limitation": (
            "No MNIST exemplar index/mask is logged. Ink variables are 28x28 composite-crop "
            "proxies and can include overlapping background digits."
        ),
        "encoder_activation_threshold": 0.0,
    }
    arrays: dict[str, np.ndarray] = {}
    for factor, labels, levels in (
        ("digit", design["digits"], 10),
        ("sector", design["sectors"], 9),
    ):
        report[factor] = {
            "group_metrics": {},
            "gate_gini": {},
            "correlations": {},
            "gate_gini_by_point_mass": {},
            "correlations_by_point_mass": {},
        }
        for metric_name, values in metrics.items():
            grouped = _group_mean(values[:, None], labels, levels)[:, 0]
            report[factor]["group_metrics"][metric_name] = grouped.tolist()
            arrays[f"{factor}_{metric_name}"] = grouped.astype(np.float32)
        for gate_name, agg in aggregates.items():
            for exclude_half in (False, True):
                point_key = "point_excluded" if exclude_half else "point_included"
                gini = _gini_by_factor(agg, labels, levels, exclude_half)
                report[factor]["gate_gini_by_point_mass"].setdefault(point_key, {})[
                    gate_name
                ] = gini.tolist()
                arrays[f"{factor}_gini_{gate_name}_{point_key}"] = gini.astype(np.float32)
                correlations = {
                    metric_name: _correlations(
                        np.asarray(report[factor]["group_metrics"][metric_name]), gini
                    )
                    for metric_name in metrics
                }
                ink = np.asarray(report[factor]["group_metrics"]["ink_area_proxy"])
                slope, intercept = np.polyfit(ink, gini, 1)
                residual = gini - (intercept + slope * ink)
                regression = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "raw_range": float(np.ptp(gini)),
                    "residual_range": float(np.ptp(residual)),
                }
                if factor == "digit":
                    keep = np.arange(levels) != 1
                    regression["ink_correlation_excluding_digit_1"] = _correlations(
                        ink[keep], gini[keep]
                    )
                correlations["ink_regression"] = regression
                report[factor]["correlations_by_point_mass"].setdefault(point_key, {})[
                    gate_name
                ] = correlations
                if not exclude_half:
                    report[factor]["gate_gini"][gate_name] = gini.tolist()
                    report[factor]["correlations"][gate_name] = correlations
                    arrays[f"{factor}_gini_{gate_name}"] = gini.astype(np.float32)

    report["encoder_control"] = {}
    cell_means = _group_mean(encoder, design["joint"], 90).reshape(9, 10, -1)
    equal_cell_means = np.asarray(
        [
            encoder[design["equal_joint_mask"] & (design["joint"] == cell)].mean(axis=0)
            for cell in range(90)
        ]
    ).reshape(9, 10, -1)
    encoder_full = _marginal_variance(cell_means)
    encoder_equal = _marginal_variance(equal_cell_means)
    report["encoder_control"]["full_condition_mean"] = {
        key: value for key, value in encoder_full.items() if "effect" not in key
    }
    report["encoder_control"]["equal_cell_condition_mean"] = {
        key: value for key, value in encoder_equal.items() if "effect" not in key
    }
    equal_n = int(np.count_nonzero(design["equal_joint_mask"])) // 90
    selected_encoder = encoder[design["equal_joint_mask"]]
    encoder_grand = selected_encoder.mean(axis=0, dtype=np.float64)
    total_trial_ss = float(
        np.square(selected_encoder.astype(np.float64) - encoder_grand).sum()
    )
    factor_ss = {
        key: equal_n * value for key, value in encoder_equal["components"].items()
    }
    residual_ss = max(0.0, total_trial_ss - sum(factor_ss.values()))
    report["encoder_control"]["equal_cell_trial_total"] = {
        "percent": {
            **{key: 100.0 * value / total_trial_ss for key, value in factor_ss.items()},
            "residual": 100.0 * residual_ss / total_trial_ss,
        }
    }
    report["digit"]["variance_contribution_by_point_mass"] = {}
    for point_key in ("point_included", "point_excluded"):
        report["digit"]["variance_contribution_by_point_mass"][point_key] = {}
        for gate_name in aggregates:
            digit_effect = part2_arrays[f"digit_effect_{gate_name}_{point_key}"]
            per_digit_contribution = np.nansum(np.square(digit_effect), axis=1) * 9
            contribution = {
                "definition": (
                    "9 * sum_synapse b_d^2; level contribution, not a fraction by itself"
                ),
                "values": per_digit_contribution.tolist(),
                "correlation_with_metrics": {
                    metric_name: _correlations(
                        np.asarray(report["digit"]["group_metrics"][metric_name]),
                        per_digit_contribution,
                    )
                    for metric_name in metrics
                },
            }
            report["digit"]["variance_contribution_by_point_mass"][point_key][
                gate_name
            ] = contribution
            arrays[f"digit_variance_contribution_{gate_name}_{point_key}"] = (
                per_digit_contribution.astype(np.float32)
            )
            if point_key == "point_included":
                report["digit"].setdefault("variance_contribution", {})[
                    gate_name
                ] = contribution
                arrays[f"digit_variance_contribution_{gate_name}"] = (
                    per_digit_contribution.astype(np.float32)
                )
    return report, arrays


def bootstrap_encoder_control(
    encoder: np.ndarray, design: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Bootstrap the raw encoder decomposition on a fixed feature sample."""

    rng = np.random.default_rng(args.seed + 707)
    sample_size = min(args.sample_synapses, encoder.shape[1])
    sample_index = np.sort(rng.choice(encoder.shape[1], size=sample_size, replace=False))
    cells = _bootstrap_cell_means(
        encoder[:, sample_index],
        design["joint"],
        design["equal_joint_mask"],
        args.resamples,
        rng,
        False,
        args.point_tolerance,
    )
    grand = cells.mean(axis=(1, 2), keepdims=True)
    sector = cells.mean(axis=2, keepdims=True) - grand
    digit = cells.mean(axis=1, keepdims=True) - grand
    interaction = cells - grand - sector - digit
    sums = np.stack(
        [
            np.square(sector).sum(axis=(1, 2, 3)) * 10,
            np.square(digit).sum(axis=(1, 2, 3)) * 9,
            np.square(interaction).sum(axis=(1, 2, 3)),
        ],
        axis=1,
    )
    fractions = sums / sums.sum(axis=1, keepdims=True)
    return {
        "method": {
            "resamples": args.resamples,
            "sampled_encoder_features": sample_size,
            "sampling": "trial bootstrap within each equal-n joint cell",
        },
        **{
            factor: {
                "bootstrap_mean": float(fractions[:, index].mean()),
                "ci95": np.quantile(fractions[:, index], [0.025, 0.975]).tolist(),
            }
            for index, factor in enumerate(("sector", "digit", "interaction"))
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def compact_output_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace dense condition-by-synapse deltas with compact shared-bin histograms."""

    delta_edges = np.linspace(-0.75, 0.75, 1201, dtype=np.float32)
    compact: dict[str, np.ndarray] = {"delta_edges": delta_edges}
    for key, values in arrays.items():
        if key.startswith("delta_"):
            counts, _ = np.histogram(values, bins=delta_edges)
            compact[f"hist_{key}"] = counts.astype(np.int64)
        elif key.startswith("digit_effect_"):
            continue
        else:
            compact[key] = values
    return compact


def main() -> None:
    """Run Parts 1--3 and save compact results."""

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    with np.load(args.trajectory) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    with open(args.part0, "r", encoding="utf-8") as file_obj:
        part0 = json.load(file_obj)
    if not part0["point_mass_at_0_5"]["repeat_downstream_with_point_mass_excluded"]:
        raise RuntimeError("Part 0 does not request point-mass-excluded repeats")
    aggregates, design = stream_gate_aggregates(trajectory, args)
    part1, part1_arrays = part1_point_estimates(aggregates, design)
    shuffle = label_shuffle_inference(aggregates, design, part1, args)
    part2, part2_arrays = part2_point_estimates(aggregates, design)
    bootstrap = bootstrap_part2(aggregates, design, args)
    encoder, proxy_metrics = collect_encoder_and_proxy(args, design)
    part3, part3_arrays = part3_analysis(
        aggregates, design, encoder, proxy_metrics, part2_arrays
    )
    part3["encoder_control"]["bootstrap"] = bootstrap_encoder_control(
        encoder, design, args
    )
    report = {
        "method": {
            "trajectory": os.path.abspath(args.trajectory),
            "gate_shapes": {
                name: list(agg.shape) for name, agg in aggregates.items()
            },
            "n_frames": int(design["digits"].size),
            "random_seed": args.seed,
            "resamples": args.resamples,
            "sample_synapses_per_gate_for_inference": args.sample_synapses,
            "dense_gate_storage": False,
            "point_estimates": "all synapses, exact streamed reductions",
            "inference": "fixed synapse sample, trial resampling",
        },
        "part1": part1,
        "part1_shuffle": shuffle,
        "part2": part2,
        "part2_bootstrap": bootstrap,
        "part3": part3,
    }
    compact_arrays = compact_output_arrays({**part1_arrays, **part2_arrays, **part3_arrays})
    np.savez_compressed(os.path.join(args.save_dir, "parts123_compact.npz"), **compact_arrays)
    result_path = os.path.join(args.save_dir, "parts123_results.json")
    with open(result_path, "w", encoding="utf-8") as file_obj:
        json.dump(_json_ready(report), file_obj, indent=2)
    print(f"Saved Parts 1-3: {args.save_dir}")


if __name__ == "__main__":
    main()
