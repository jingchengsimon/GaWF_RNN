"""Audit GaWF gate provenance, matrix axes, activation confounds, and robustness.

Inputs are the compact continuous-test gate trajectory, the saved validation selectivity from
the symmetric relevance analysis, the matching GaWF checkpoint, and the uint8 test split.
Outputs are compact JSON/CSV/NPZ summaries; no trial-by-full-gate tensor is saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt
from utils_anal.gawf_gate_context_parts123 import _balanced_masks
from utils_anal.gawf_gate_distribution import iter_gate_chunks
from utils_anal.gawf_symmetric_relevance_timing import (
    _gate_tensors,
    _trajectory_with_measurements,
)
from utils_anal.gawf_symmetric_stats import (
    cohens_d_from_moments,
    relevance_masks,
    trial_relevance_moments,
)

FACTORS = ("sector", "digit", "interaction")
GATES = ("input", "recurrent")
VARIABLES = ("sector", "digit")
VIEWS = ("source", "destination")
POLICIES = ("interaction_excluded", "interaction_included")


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default=str(
            output_dir(
                "A_raw_gate",
                "gawf_gate_distribution",
                "data",
            ) / "gawf_gate_trajectory.npz"
        ),
    )
    parser.add_argument(
        "--selectivity",
        default=str(
            output_dir(
                "D_variance_decomposition",
                "gawf_symmetric_relevance_timing",
                "data",
            )
            / "part1_selectivity.npz"
        ),
    )
    parser.add_argument(
        "--old_relevance",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_symmetric_relevance_timing",
                "data",
            )
            / "part2_results.json"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=(
            "./results/train_data/clutter/best_6model_param_matched_40h/"
            "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
        ),
    )
    parser.add_argument("--data_dir", default="./stimuli")
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument(
        "--save_dir", default=str(output_dir("H_controls", "gawf_gate_robustness", "data"))
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--gate_chunk_size", type=int, default=128)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--ci_resamples", type=int, default=200)
    parser.add_argument("--correlation_resamples", type=int, default=5000)
    parser.add_argument("--bootstrap_batch_size", type=int, default=20)
    parser.add_argument("--max_sample_synapses", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess_ci_only", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decomposition(cell_mean: np.ndarray) -> dict[str, Any]:
    """Return balanced two-way condition-mean sums of squares and fractions."""

    cell = np.asarray(cell_mean, dtype=np.float64)
    grand = cell.mean(axis=(0, 1))
    sector = cell.mean(axis=1) - grand
    digit = cell.mean(axis=0) - grand
    interaction = cell - grand - sector[:, None] - digit[None, :]
    components = np.asarray(
        [
            cell.shape[1] * np.square(sector).sum(),
            cell.shape[0] * np.square(digit).sum(),
            np.square(interaction).sum(),
        ],
        dtype=np.float64,
    )
    fractions = components / components.sum()
    return {
        "components": dict(zip(FACTORS, components.astype(float))),
        "fractions": dict(zip(FACTORS, fractions.astype(float))),
    }


def _group_deltas(
    joint_sum: np.ndarray, joint_counts: np.ndarray
) -> dict[str, np.ndarray]:
    """Return corrected sector and digit group-mean deviations from a weighted grand."""

    sums = joint_sum.reshape(9, 10, -1)
    grand = sums.sum(axis=(0, 1)) / joint_counts.sum()
    sector_sum = sums.sum(axis=1)
    digit_sum = sums.sum(axis=0)
    sector_mean = sector_sum / joint_counts.sum(axis=1)[:, None]
    digit_mean = digit_sum / joint_counts.sum(axis=0)[:, None]
    return {"sector": sector_mean - grand, "digit": digit_mean - grand}


def _crossings(x: np.ndarray, first: np.ndarray, second: np.ndarray) -> list[float]:
    difference = np.asarray(first) - np.asarray(second)
    output: list[float] = []
    for index in np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:])):
        x0, x1 = float(x[index]), float(x[index + 1])
        y0, y1 = float(difference[index]), float(difference[index + 1])
        output.append(x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0)
    return output


def _survival_report(
    deltas: dict[str, dict[str, np.ndarray]], thresholds: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report: dict[str, Any] = {"cells": {}}
    arrays: dict[str, np.ndarray] = {"survival_thresholds": thresholds.astype(np.float32)}
    for gate in GATES:
        for variable in VARIABLES:
            values = np.abs(deltas[gate][variable].reshape(-1))
            survival = np.asarray([(values > threshold).mean() for threshold in thresholds])
            quantiles = np.quantile(values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
            cell = f"{gate}_{variable}"
            arrays[f"survival_{cell}"] = survival.astype(np.float32)
            report["cells"][cell] = {
                "count": int(values.size),
                "abs_delta_quantiles": dict(
                    zip(("50", "75", "90", "95", "99", "99.9"), quantiles.astype(float))
                ),
                "survival_at_0_5": float((values > 0.5).mean()),
            }
    recurrent_crossings = _crossings(
        thresholds,
        arrays["survival_recurrent_sector"],
        arrays["survival_recurrent_digit"],
    )
    report["recurrent_sector_minus_digit_crossings"] = recurrent_crossings
    report["reconciliation"] = (
        "crossing" if recurrent_crossings else "no_crossing_genuine_inconsistency"
    )
    return report, arrays


def _stream_continuous(
    trajectory: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Stream full gates into joint sums and nested sampled-synapse trial matrices."""

    feedback = trajectory["feedback"].astype(np.float32, copy=False)
    labels = trajectory["labels"].reshape(-1, 2).astype(np.int64, copy=False)
    digits, sectors = labels[:, 0], labels[:, 1]
    joint = sectors * 10 + digits
    _, equal_mask = _balanced_masks(digits, sectors, args.seed)
    full_counts = np.bincount(joint, minlength=90).reshape(9, 10)
    equal_counts = np.bincount(joint[equal_mask], minlength=90).reshape(9, 10)
    shapes = {
        "input": tuple(trajectory["weight_ih"].shape),
        "recurrent": tuple(trajectory["weight_hh"].shape),
    }
    full_sums = {
        name: np.zeros((90, int(np.prod(shape))), dtype=np.float64)
        for name, shape in shapes.items()
    }
    equal_sums = {name: np.zeros_like(values) for name, values in full_sums.items()}
    rng = np.random.default_rng(args.seed + 501)
    sample_indices = {
        name: rng.choice(
            int(np.prod(shape)),
            size=min(args.max_sample_synapses, int(np.prod(shape))),
            replace=False,
        )
        for name, shape in shapes.items()
    }
    equal_size = int(equal_mask.sum())
    sample_values = {
        name: np.empty((equal_size, indices.size), dtype=np.float32)
        for name, indices in sample_indices.items()
    }
    selected_position = 0
    started = time.perf_counter()
    for chunk_idx, (start, end, gate_input, gate_recurrent) in enumerate(
        iter_gate_chunks(
            feedback,
            trajectory["U"].astype(np.float32, copy=False),
            trajectory["V"].astype(np.float32, copy=False),
            shapes["input"][1],
            0.5,
            args.gate_chunk_size,
        ),
        start=1,
    ):
        chunk_joint = joint[start:end]
        selected = equal_mask[start:end]
        selected_count = int(selected.sum())
        for name, gate in (("input", gate_input), ("recurrent", gate_recurrent)):
            flat = gate.reshape(gate.shape[0], -1)
            for code in np.unique(chunk_joint):
                code_mask = chunk_joint == code
                full_sums[name][int(code)] += flat[code_mask].sum(axis=0, dtype=np.float64)
                equal_code = code_mask & selected
                if np.any(equal_code):
                    equal_sums[name][int(code)] += flat[equal_code].sum(
                        axis=0, dtype=np.float64
                    )
            if selected_count:
                sample_values[name][selected_position : selected_position + selected_count] = (
                    flat[selected][:, sample_indices[name]]
                )
        selected_position += selected_count
        if chunk_idx % 50 == 0:
            print(
                f"continuous gate chunks {chunk_idx} | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if selected_position != equal_size:
        raise RuntimeError(f"equal-cell sample fill mismatch: {selected_position} vs {equal_size}")
    metadata = {
        "n_trials": int(labels.shape[0]),
        "full_joint_count_range": [int(full_counts.min()), int(full_counts.max())],
        "equal_joint_count_range": [int(equal_counts.min()), int(equal_counts.max())],
        "sample_synapses": {key: int(value.size) for key, value in sample_indices.items()},
    }
    sums = {
        **{f"full_{key}": value for key, value in full_sums.items()},
        **{f"equal_{key}": value for key, value in equal_sums.items()},
        "full_counts": full_counts,
        "equal_counts": equal_counts,
    }
    return metadata, sums, sample_values, joint[equal_mask]


def _leave_one_out(equal_sums: dict[str, np.ndarray], counts: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {"baseline": {}, "drop_digit": [], "drop_sector": []}
    cells: dict[str, np.ndarray] = {}
    for gate in GATES:
        cell = equal_sums[gate].reshape(9, 10, -1) / counts[..., None]
        cells[gate] = cell
        report["baseline"][gate] = _decomposition(cell)
    for variable, levels in (("digit", 10), ("sector", 9)):
        destination = report[f"drop_{variable}"]
        for dropped in range(levels):
            row: dict[str, Any] = {"dropped_level": dropped, "gates": {}}
            keep = np.arange(levels) != dropped
            for gate in GATES:
                reduced = cells[gate][:, keep] if variable == "digit" else cells[gate][keep]
                row["gates"][gate] = _decomposition(reduced)
            destination.append(row)
    report["ranges"] = {}
    report["moves_over_3_percentage_points"] = []
    for drop_key in ("drop_digit", "drop_sector"):
        report["ranges"][drop_key] = {}
        for gate in GATES:
            report["ranges"][drop_key][gate] = {}
            for factor in FACTORS:
                values = np.asarray(
                    [row["gates"][gate]["fractions"][factor] for row in report[drop_key]]
                )
                report["ranges"][drop_key][gate][factor] = [
                    float(values.min()),
                    float(values.max()),
                ]
                baseline = report["baseline"][gate]["fractions"][factor]
                for row, value in zip(report[drop_key], values):
                    move = 100.0 * float(value - baseline)
                    if abs(move) > 3.0:
                        report["moves_over_3_percentage_points"].append(
                            {
                                "drop": drop_key,
                                "level": row["dropped_level"],
                                "gate": gate,
                                "factor": factor,
                                "move_percentage_points": move,
                            }
                        )
    return report


def _bootstrap_fraction_from_cells(cell_means: np.ndarray) -> np.ndarray:
    grand = cell_means.mean(axis=(1, 2), keepdims=True)
    sector = cell_means.mean(axis=2, keepdims=True) - grand
    digit = cell_means.mean(axis=1, keepdims=True) - grand
    interaction = cell_means - grand - sector - digit
    sums = np.stack(
        [
            cell_means.shape[2] * np.square(sector).sum(axis=(1, 2, 3)),
            cell_means.shape[1] * np.square(digit).sum(axis=(1, 2, 3)),
            np.square(interaction).sum(axis=(1, 2, 3)),
        ],
        axis=1,
    )
    return sums / sums.sum(axis=1, keepdims=True)


def _ci_convergence_gate(
    values: np.ndarray,
    joint: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Bootstrap balanced-cell trials for nested synapse samples without dense gate draws."""

    sizes = [size for size in (128, 512, 2048, 8192) if size <= values.shape[1]]
    counts = np.bincount(joint, minlength=90)
    if np.unique(counts).size != 1:
        raise RuntimeError(f"CI convergence requires equal cell counts, got {counts.tolist()}")
    n_cell = int(counts[0])
    rows = [np.flatnonzero(joint == cell) for cell in range(90)]
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_cell, 1.0 / n_cell)
    weights = np.stack(
        [rng.multinomial(n_cell, probabilities, size=resamples) for _ in range(90)], axis=1
    ).astype(np.float32)
    weights /= n_cell
    sums = np.zeros((resamples, 3), dtype=np.float64)
    output: dict[int, np.ndarray] = {}
    start = 0
    for stop in sizes:
        for block_start in range(start, stop, 128):
            block_stop = min(stop, block_start + 128)
            width = block_stop - block_start
            cell_means = np.empty((resamples, 9, 10, width), dtype=np.float32)
            for cell in range(90):
                sector, digit = divmod(cell, 10)
                cell_values = values[rows[cell], block_start:block_stop]
                cell_means[:, sector, digit] = weights[:, cell] @ cell_values
            fractions = _bootstrap_fraction_from_cells(cell_means)
            # Accumulate component sums rather than averaging blockwise fractions.
            grand = cell_means.mean(axis=(1, 2), keepdims=True)
            sector_effect = cell_means.mean(axis=2, keepdims=True) - grand
            digit_effect = cell_means.mean(axis=1, keepdims=True) - grand
            interaction = cell_means - grand - sector_effect - digit_effect
            sums[:, 0] += 10 * np.square(sector_effect).sum(axis=(1, 2, 3))
            sums[:, 1] += 9 * np.square(digit_effect).sum(axis=(1, 2, 3))
            sums[:, 2] += np.square(interaction).sum(axis=(1, 2, 3))
            del fractions
        output[stop] = sums / sums.sum(axis=1, keepdims=True)
        start = stop
    return output


def _ci_convergence(
    sample_values: dict[str, np.ndarray],
    joint: np.ndarray,
    point: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report: dict[str, Any] = {
        "method": {
            "resamples": args.ci_resamples,
            "trial_bootstrap": "within each of 90 equal-n cells",
            "synapse_samples": [128, 512, 2048, 8192],
            "plateau_rule": "CI width at 8192 is at least 90% of width at 2048",
        },
        "gates": {},
    }
    arrays: dict[str, np.ndarray] = {}
    all_plateau = True
    for gate_idx, gate in enumerate(GATES):
        draws = _ci_convergence_gate(
            sample_values[gate], joint, args.ci_resamples, args.seed + 700 + gate_idx
        )
        report["gates"][gate] = {}
        for size, fractions in draws.items():
            arrays[f"ci_draws_{gate}_{size}"] = fractions.astype(np.float32)
            report["gates"][gate][str(size)] = {}
            for factor_idx, factor in enumerate(FACTORS):
                low, high = np.quantile(fractions[:, factor_idx], [0.025, 0.975])
                report["gates"][gate][str(size)][factor] = {
                    "ci95": [float(low), float(high)],
                    "width": float(high - low),
                }
        for factor in ("sector", "digit"):
            width_2048 = report["gates"][gate]["2048"][factor]["width"]
            width_8192 = report["gates"][gate]["8192"][factor]["width"]
            plateau = bool(width_8192 >= 0.9 * width_2048)
            report["gates"][gate]["plateau_by_2048_" + factor] = plateau
            all_plateau &= plateau
        final = report["gates"][gate]["8192"]
        final_draws = draws[8192]
        report["gates"][gate]["final"] = {
            factor: {
                "point": point[gate]["fractions"][factor],
                "ci95": [
                    float(
                        point[gate]["fractions"][factor]
                        - np.quantile(
                            final_draws[:, factor_idx] - final_draws[:, factor_idx].mean(),
                            0.975,
                        )
                    ),
                    float(
                        point[gate]["fractions"][factor]
                        - np.quantile(
                            final_draws[:, factor_idx] - final_draws[:, factor_idx].mean(),
                            0.025,
                        )
                    ),
                ],
                "raw_sampled_percentile_ci95": final[factor]["ci95"],
                "status": (
                    "8192-synapse sampled trial bootstrap, basic interval recentered on "
                    "the exact full-gate point"
                ),
            }
            for factor_idx, factor in enumerate(FACTORS)
        }
    report["all_four_headlines_plateau_by_2048"] = all_plateau
    report["full_gate_bootstrap_required"] = not all_plateau
    return report, arrays


def _load_selectivity(path: str) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path) as loaded:
        return {
            population: {
                name: loaded[f"primary_{population}_{name}"]
                for name in (
                    "eta_sector",
                    "eta_digit",
                    "eta_interaction",
                    "tuning_sector",
                    "tuning_digit",
                    "interaction_dominant",
                    "passed_sector",
                    "passed_digit",
                )
            }
            for population in ("encoder", "hidden")
        }


def _variant_masks(
    selectivity: dict[str, dict[str, np.ndarray]], top_percent: Iterable[int]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for gate in GATES:
        for view in VIEWS:
            population = "encoder" if gate == "input" and view == "source" else "hidden"
            result = selectivity[population]
            for variable in VARIABLES:
                for policy in POLICIES:
                    passed = result[f"passed_{variable}"].astype(bool)
                    dominant = result["interaction_dominant"].astype(bool)
                    eligible = passed & (~dominant if policy == "interaction_excluded" else True)
                    tuning = result[f"tuning_{variable}"]
                    for percent in top_percent:
                        key = f"{gate}|{view}|{variable}|{policy}|{percent}"
                        output[key] = {
                            "gate": gate,
                            "view": view,
                            "variable": variable,
                            "policy": policy,
                            "percent": percent,
                            "population": population,
                            "eligible": eligible,
                            "relevance": relevance_masks(tuning, eligible, percent / 100.0),
                            "eligible_count": int(eligible.sum()),
                        }
    return output


def _view_moments(
    views: dict[tuple[str, str], np.ndarray],
    labels: np.ndarray,
    masks: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, record in masks.items():
        context = labels[:, 1] if record["variable"] == "sector" else labels[:, 0]
        output[key] = trial_relevance_moments(
            views[(record["gate"], record["view"])],
            context,
            record["relevance"],
            record["eligible"],
        ).astype(np.float32)
    return output


def _collect_relevance_test(
    args: argparse.Namespace,
    masks: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Collect trial moments plus recurrent source/destination gates and activation means."""

    args.use_sector_mode = True
    args.predict_all_chars = False
    dataset, num_pos = build_eval_dataset(args, "test")
    device = torch.device("cpu")
    model = build_model_from_ckpt(args.ckpt, num_pos, device, chan_num=args.chan_num)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    moment_blocks: dict[str, list[np.ndarray]] = {key: [] for key in masks}
    recurrent_blocks = {"source": [], "destination": []}
    labels_blocks: list[np.ndarray] = []
    activation_sum = np.zeros(256, dtype=np.float64)
    activation_abs_sum = np.zeros(256, dtype=np.float64)
    activation_count = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels_t = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            trajectory = _trajectory_with_measurements(encoded, model, record_gates=True)
            source_input = trajectory["input_gate"].reshape(-1, encoded.shape[-1]).numpy()
            source_recurrent = trajectory["recurrent_gate"].reshape(-1, 256).numpy()
            flat_feedback = trajectory["feedback"].reshape(-1, model.feedback_dim)
            destination_input_blocks: list[np.ndarray] = []
            destination_recurrent_blocks: list[np.ndarray] = []
            for start in range(0, flat_feedback.shape[0], args.gate_chunk_size):
                gate_input, gate_recurrent = _gate_tensors(
                    flat_feedback[start : start + args.gate_chunk_size], model, encoded.shape[-1]
                )
                destination_input_blocks.append(gate_input.mean(dim=2).numpy())
                destination_recurrent_blocks.append(gate_recurrent.mean(dim=2).numpy())
            destination_input = np.concatenate(destination_input_blocks)
            destination_recurrent = np.concatenate(destination_recurrent_blocks)
            labels = labels_t.numpy().reshape(-1, 2).astype(np.int64)
            views = {
                ("input", "source"): source_input,
                ("input", "destination"): destination_input,
                ("recurrent", "source"): source_recurrent,
                ("recurrent", "destination"): destination_recurrent,
            }
            block_moments = _view_moments(views, labels, masks)
            for key, values in block_moments.items():
                moment_blocks[key].append(values)
            hidden = trajectory["hidden"].reshape(-1, 256).numpy().astype(np.float64)
            activation_sum += hidden.sum(axis=0)
            activation_abs_sum += np.abs(hidden).sum(axis=0)
            activation_count += hidden.shape[0]
            recurrent_blocks["source"].append(source_recurrent.astype(np.float32))
            recurrent_blocks["destination"].append(destination_recurrent.astype(np.float32))
            labels_blocks.append(labels)
            samples_done = min((batch_idx + 1) * args.batch_size, len(dataset))
            if samples_done % 200 < args.batch_size or batch_idx + 1 == len(loader):
                print(
                    f"relevance test sequences {samples_done}/{len(dataset)} | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    moments = {key: np.concatenate(value) for key, value in moment_blocks.items()}
    recurrent = {key: np.concatenate(value) for key, value in recurrent_blocks.items()}
    labels = np.concatenate(labels_blocks)
    metadata = {
        "n_trials": int(labels.shape[0]),
        "mean_activation": activation_sum / activation_count,
        "mean_absolute_activation": activation_abs_sum / activation_count,
        "labels": labels,
    }
    return moments, recurrent, metadata


def _d_from_sums(sums: np.ndarray) -> np.ndarray:
    sum_rel, ss_rel, n_rel, sum_other, ss_other, n_other = np.moveaxis(sums, -1, 0)
    mean_rel = sum_rel / n_rel
    mean_other = sum_other / n_other
    var_rel = np.maximum(0.0, (ss_rel - sum_rel * sum_rel / n_rel) / (n_rel - 1.0))
    var_other = np.maximum(
        0.0, (ss_other - sum_other * sum_other / n_other) / (n_other - 1.0)
    )
    pooled = np.sqrt(
        ((n_rel - 1) * var_rel + (n_other - 1) * var_other) / (n_rel + n_other - 2)
    )
    return (mean_rel - mean_other) / pooled


def _bootstrap_moment_variants(
    moments: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    keys = list(moments)
    n_trials = moments[keys[0]].shape[0]
    matrix = np.concatenate([moments[key] for key in keys], axis=1).astype(np.float32)
    point = {
        key: cohens_d_from_moments(moments[key].sum(axis=0, dtype=np.float64)) for key in keys
    }
    draws = {key: np.empty(args.resamples, dtype=np.float32) for key in keys}
    rng = np.random.default_rng(args.seed + 900)
    probabilities = np.full(n_trials, 1.0 / n_trials)
    for start in range(0, args.resamples, args.bootstrap_batch_size):
        stop = min(args.resamples, start + args.bootstrap_batch_size)
        weights = rng.multinomial(
            n_trials, probabilities, size=stop - start
        ).astype(np.float32)
        boot_sums = (weights @ matrix).reshape(stop - start, len(keys), 6)
        boot_d = _d_from_sums(boot_sums)
        for index, key in enumerate(keys):
            draws[key][start:stop] = boot_d[:, index]
        print(f"relevance bootstrap {stop}/{args.resamples}", flush=True)
    return point, draws


def _linear_prediction(gate_mean: np.ndarray, activation: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(activation.size), activation])
    return design @ np.linalg.lstsq(design, gate_mean, rcond=None)[0]


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(stats.pearsonr(first, second).statistic)


def _partial_correlation(first: np.ndarray, second: np.ndarray, control: np.ndarray) -> float:
    design = np.column_stack([np.ones(control.size), control])
    first_residual = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    second_residual = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return _pearson(first_residual, second_residual)


def _correlation_ci(
    function: Any,
    arrays: tuple[np.ndarray, ...],
    resamples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    size = arrays[0].size
    for index in range(resamples):
        sampled = rng.integers(0, size, size=size)
        draws[index] = function(*(array[sampled] for array in arrays))
    return np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def _stratified_d(
    gates: np.ndarray,
    labels: np.ndarray,
    record: dict[str, Any],
    activation: np.ndarray,
) -> dict[str, Any]:
    boundaries = np.quantile(activation, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(activation, boundaries, right=True)
    context = labels[:, 1] if record["variable"] == "sector" else labels[:, 0]
    d_values: list[float] = []
    weights: list[float] = []
    for stratum in range(5):
        eligible = record["eligible"] & (strata == stratum)
        relevance = record["relevance"] & eligible[None, :]
        moment = trial_relevance_moments(gates, context, relevance, eligible).sum(axis=0)
        d_value = cohens_d_from_moments(moment)
        d_values.append(float(d_value))
        n_rel, n_other = float(moment[2]), float(moment[5])
        weights.append(n_rel * n_other / (n_rel + n_other) if n_rel + n_other else 0.0)
    valid = np.isfinite(d_values) & (np.asarray(weights) > 0)
    pooled = float(np.average(np.asarray(d_values)[valid], weights=np.asarray(weights)[valid]))
    return {
        "activation_quintile_d": d_values,
        "pooled_within_quintile_d": pooled,
        "quintile_boundaries": boundaries.astype(float).tolist(),
    }


def _relevance_and_activation(
    selectivity: dict[str, dict[str, np.ndarray]],
    masks: dict[str, dict[str, Any]],
    moments: dict[str, np.ndarray],
    recurrent: dict[str, np.ndarray],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    labels = metadata["labels"]
    activation = metadata["mean_activation"]
    activation_abs = metadata["mean_absolute_activation"]
    controlled_keys: list[str] = []
    for key, record in masks.items():
        if record["gate"] != "recurrent":
            continue
        prediction = _linear_prediction(recurrent[record["view"]].mean(axis=0), activation)
        residual = recurrent[record["view"]] - prediction[None, :]
        context = labels[:, 1] if record["variable"] == "sector" else labels[:, 0]
        controlled_key = "controlled|" + key
        moments[controlled_key] = trial_relevance_moments(
            residual,
            context,
            record["relevance"],
            record["eligible"],
        ).astype(np.float32)
        controlled_keys.append(controlled_key)
    points, draws = _bootstrap_moment_variants(moments, args)
    d_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for key, record in masks.items():
        low, high = np.quantile(draws[key], [0.025, 0.975])
        d_rows.append(
            {
                **{name: record[name] for name in ("gate", "view", "variable", "policy")},
                "top_percent": record["percent"],
                "population": record["population"],
                "eligible_units": record["eligible_count"],
                "cohens_d": points[key],
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
        )
        if record["gate"] == "recurrent":
            controlled_key = "controlled|" + key
            controlled_low, controlled_high = np.quantile(
                draws[controlled_key], [0.025, 0.975]
            )
            stratified = _stratified_d(
                recurrent[record["view"]], labels, record, activation
            )
            control_rows.append(
                {
                    "gate": "recurrent",
                    "view": record["view"],
                    "variable": record["variable"],
                    "policy": record["policy"],
                    "top_percent": record["percent"],
                    "d_before": points[key],
                    "d_before_ci95_low": float(low),
                    "d_before_ci95_high": float(high),
                    "d_linear_control": points[controlled_key],
                    "d_linear_ci95_low": float(controlled_low),
                    "d_linear_ci95_high": float(controlled_high),
                    "d_stratified_pooled": stratified["pooled_within_quintile_d"],
                    "d_quintile_1": stratified["activation_quintile_d"][0],
                    "d_quintile_2": stratified["activation_quintile_d"][1],
                    "d_quintile_3": stratified["activation_quintile_d"][2],
                    "d_quintile_4": stratified["activation_quintile_d"][3],
                    "d_quintile_5": stratified["activation_quintile_d"][4],
                }
            )
    correlations: list[dict[str, Any]] = []
    for view_idx, view in enumerate(VIEWS):
        gate_mean = recurrent[view].mean(axis=0, dtype=np.float64)
        for factor_idx, factor in enumerate(VARIABLES):
            eta = selectivity["hidden"][f"eta_{factor}"].astype(np.float64)
            partial = _partial_correlation(gate_mean, eta, activation)
            correlations.append(
                {
                    "view": view,
                    "variable": factor,
                    "r_gate_mean_activation": _pearson(gate_mean, activation),
                    "r_gate_mean_abs_activation": _pearson(gate_mean, activation_abs),
                    "r_selectivity_activation": _pearson(eta, activation),
                    "partial_r_gate_selectivity_controlling_activation": partial,
                    "partial_r_ci95": _correlation_ci(
                        _partial_correlation,
                        (gate_mean, eta, activation),
                        args.correlation_resamples,
                        args.seed + 1200 + view_idx * 10 + factor_idx,
                    ),
                }
            )
    report = {
        "matrix_convention": {
            "gate_shape": "(batch, destination, source)",
            "einsum": 'torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)',
            "interpretation": "G[i,j] modulates the connection FROM source j TO destination i",
        },
        "activation": {
            "hidden_is_post_relu": True,
            "max_abs_mean_minus_mean_abs": float(np.max(np.abs(activation - activation_abs))),
            "correlations": correlations,
        },
        "negative_d_survival": {},
    }
    for row in control_rows:
        key = f"{row['view']}_{row['variable']}_{row['policy']}_top{row['top_percent']}"
        report["negative_d_survival"][key] = {
            "linear_control_same_negative_sign": bool(row["d_linear_control"] < 0),
            "stratified_same_negative_sign": bool(row["d_stratified_pooled"] < 0),
        }
    arrays = {
        "mean_hidden_activation": activation.astype(np.float32),
        "mean_absolute_hidden_activation": activation_abs.astype(np.float32),
        **{
            f"bootstrap_d_{index}": draws[key].astype(np.float32)
            for index, key in enumerate(masks)
        },
    }
    return report, d_rows, control_rows, arrays


def _old_source_check(rows: list[dict[str, Any]], old_path: str) -> dict[str, Any]:
    with open(old_path, encoding="utf-8") as file_obj:
        old = json.load(file_obj)["primary_validation_estimate_test_effect"]
    differences: list[float] = []
    for row in rows:
        if row["view"] != "source":
            continue
        cell = f"{row['gate']}_{row['variable']}"
        old_value = old[row["policy"]]["cells"][cell]["top_percent"][
            str(row["top_percent"])
        ]["cohens_d"]
        differences.append(abs(row["cohens_d"] - old_value))
    return {
        "max_abs_source_d_difference_from_prior_analysis": float(max(differences)),
        "source_values_match_at_5e_5": bool(max(differences) < 5e-5),
    }


def _loo_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for drop in ("drop_digit", "drop_sector"):
        for record in report[drop]:
            for gate in GATES:
                fractions = record["gates"][gate]["fractions"]
                rows.append(
                    {
                        "drop_variable": drop.removeprefix("drop_"),
                        "dropped_level": record["dropped_level"],
                        "gate": gate,
                        **{f"{factor}_percent": 100.0 * fractions[factor] for factor in FACTORS},
                    }
                )
    return rows


def _ci_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in GATES:
        for factor in FACTORS:
            final = report["gates"][gate]["final"][factor]
            rows.append(
                {
                    "gate": gate,
                    "factor": factor,
                    "point_percent": 100.0 * final["point"],
                    "ci95_low_percent": 100.0 * final["ci95"][0],
                    "ci95_high_percent": 100.0 * final["ci95"][1],
                    "ci_status": final["status"],
                }
            )
    return rows


def _postprocess_ci_outputs(save_dir: str) -> None:
    """Recenter saved sampled-trial intervals on the exact full-gate point."""

    result_path = os.path.join(save_dir, "robustness_results.json")
    compact_path = os.path.join(save_dir, "robustness_compact.npz")
    with open(result_path, encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    ci_report = report["part5_ci_convergence"]
    with np.load(compact_path) as compact:
        for gate in GATES:
            for factor_idx, factor in enumerate(FACTORS):
                draws = compact[f"ci_draws_{gate}_8192"][:, factor_idx].astype(np.float64)
                centered = draws - draws.mean()
                point_value = ci_report["gates"][gate]["final"][factor]["point"]
                raw_ci = ci_report["gates"][gate]["8192"][factor]["ci95"]
                ci_report["gates"][gate]["final"][factor].update(
                    {
                        "ci95": [
                            float(point_value - np.quantile(centered, 0.975)),
                            float(point_value - np.quantile(centered, 0.025)),
                        ],
                        "raw_sampled_percentile_ci95": raw_ci,
                        "status": (
                            "8192-synapse sampled trial bootstrap, basic interval recentered "
                            "on the exact full-gate point"
                        ),
                    }
                )
    source_check = report["part2_and_part3"]["prior_source_reproduction"]
    maximum = source_check["max_abs_source_d_difference_from_prior_analysis"]
    source_check.pop("source_values_match_at_1e_6", None)
    source_check["source_values_match_at_5e_5"] = bool(maximum < 5e-5)
    with open(result_path, "w", encoding="utf-8") as file_obj:
        json.dump(_json_ready(report), file_obj, indent=2, allow_nan=False)
    _write_csv(os.path.join(save_dir, "part5_final_ci.csv"), _ci_rows(ci_report))
    print(f"Postprocessed CI outputs: {os.path.abspath(save_dir)}")


def main() -> None:
    """Run all five requested audit parts and save compact deliverables."""

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.postprocess_ci_only:
        _postprocess_ci_outputs(args.save_dir)
        return
    with np.load(args.trajectory) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    continuous_meta, sums, sample_values, equal_joint = _stream_continuous(trajectory, args)
    deltas = {
        gate: _group_deltas(sums[f"full_{gate}"], sums["full_counts"]) for gate in GATES
    }
    survival, compact = _survival_report(deltas, np.linspace(0.0, 0.8, 801))
    equal_sums = {gate: sums[f"equal_{gate}"] for gate in GATES}
    leave_one_out = _leave_one_out(equal_sums, sums["equal_counts"])
    point = {
        gate: _decomposition(
            equal_sums[gate].reshape(9, 10, -1) / sums["equal_counts"][..., None]
        )
        for gate in GATES
    }
    del sums, equal_sums, deltas
    ci_report, ci_arrays = _ci_convergence(
        sample_values, equal_joint, point, args
    )
    compact.update(ci_arrays)
    del sample_values

    selectivity = _load_selectivity(args.selectivity)
    masks = _variant_masks(selectivity, (10, 20, 30))
    moments, recurrent, relevance_meta = _collect_relevance_test(args, masks)
    relevance_report, d_rows, control_rows, relevance_arrays = _relevance_and_activation(
        selectivity, masks, moments, recurrent, relevance_meta, args
    )
    relevance_report["prior_source_reproduction"] = _old_source_check(d_rows, args.old_relevance)
    compact.update(relevance_arrays)

    report = {
        "provenance": {
            "continuous_parts": ["part1_reconciliation", "part4_leave_one_out", "part5_ci"],
            "continuous_trajectory": os.path.abspath(args.trajectory),
            "relevance_parts": ["part2_gate_axes", "part3_activation_control"],
            "relevance_test_suffix": args.data_suffix,
            "selectivity": os.path.abspath(args.selectivity),
            "checkpoint": os.path.abspath(args.ckpt),
            "protocol_warning": (
                "Parts 1/4/5 use the continuous gate-audit test trajectory; Parts 2/3 use the "
                "40h-uint8 relevance test split paired with validation-estimated selectivity."
            ),
        },
        "continuous_metadata": continuous_meta,
        "part1_reconciliation": survival,
        "part2_and_part3": relevance_report,
        "part4_leave_one_out": leave_one_out,
        "part5_ci_convergence": ci_report,
    }
    with open(os.path.join(args.save_dir, "robustness_results.json"), "w", encoding="utf-8") as f:
        json.dump(_json_ready(report), f, indent=2, allow_nan=False)
    np.savez_compressed(os.path.join(args.save_dir, "robustness_compact.npz"), **compact)
    _write_csv(os.path.join(args.save_dir, "part2_source_destination_d.csv"), d_rows)
    _write_csv(os.path.join(args.save_dir, "part3_activation_controlled_d.csv"), control_rows)
    _write_csv(os.path.join(args.save_dir, "part4_leave_one_out.csv"), _loo_rows(leave_one_out))
    _write_csv(os.path.join(args.save_dir, "part5_final_ci.csv"), _ci_rows(ci_report))
    print(json.dumps(_json_ready(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
