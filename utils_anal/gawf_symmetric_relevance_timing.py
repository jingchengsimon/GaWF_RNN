"""Run symmetric 2x2 GaWF relevance and target-switch timing analyses.

Inputs are one single-layer GaWF sector checkpoint plus distinct Clutter validation/test splits.
The script records pre-gate encoder activations, hidden activations, feedback-derived gate-column
means, and readout logits. It saves compact Part 0--3 NPZ/JSON outputs in ``--save_dir``. The
validation split estimates selectivity; test trials test gate effects. A sequence-level test
split-half estimate/test swap is reported as a robustness analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
from utils_anal.gawf_symmetric_stats import (
    NUM_DIGITS,
    NUM_SECTORS,
    SelectivityResult,
    architecture_axis_variance,
    bootstrap_d,
    bootstrap_onset,
    cosine_alignment,
    first_crossing,
    first_negative_to_nonnegative,
    interaction_dominant,
    joint_design,
    paired_lead_test,
    permutation_selectivity,
    relevance_label_null,
    relevance_masks,
    trial_relevance_moments,
    two_way_decomposition,
)


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
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
        "--decomposition_dir",
        default=str(
            output_dir("D_variance_decomposition", "gawf_symmetric_relevance_timing", "data")
        ),
    )
    parser.add_argument(
        "--relevance_dir",
        default=str(output_dir("E_relevance_alignment", "gawf_symmetric_relevance_timing", "data")),
    )
    parser.add_argument(
        "--timing_dir",
        default=str(output_dir("F_timing", "gawf_symmetric_relevance_timing", "data")),
    )
    parser.add_argument(
        "--control_dir",
        default=str(output_dir("H_controls", "gawf_symmetric_relevance_timing", "data")),
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="Deprecated compatibility override; sends every artifact to one directory.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("AIM3_NUM_WORKERS", "0"))
    )
    parser.add_argument("--gate_chunk_size", type=int, default=16)
    parser.add_argument("--permutation_batch_size", type=int, default=10)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--top_percent", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--post_frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_split_half", action="store_true")
    parser.add_argument("--postprocess_part3_only", action="store_true")
    return parser.parse_args()


def _gate_tensors(
    feedback: torch.Tensor,
    model: torch.nn.Module,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the single-layer GaWF gate calculation exactly."""

    feedback = feedback.to(dtype=torch.float32).clamp(-10, 10).unsqueeze(2)
    scaled_u = model.U.unsqueeze(0) * feedback.transpose(1, 2)
    gate = torch.sigmoid(torch.matmul(scaled_u, model.V) / model.gate_tau)
    return gate[..., :input_size], gate[..., input_size:]


def _trajectory_with_measurements(
    encoded: torch.Tensor,
    model: torch.nn.Module,
    record_gates: bool,
    *,
    record_input_gate: bool = True,
) -> dict[str, torch.Tensor]:
    """Run a reset trajectory and retain activation/readout/gate summaries."""

    batch_size, frame_num, input_size = encoded.shape
    hidden = torch.zeros(
        batch_size, model.rnn.hidden_size, dtype=encoded.dtype, device=encoded.device
    )
    feedback = torch.zeros(
        batch_size, model.feedback_dim, dtype=torch.float32, device=encoded.device
    )
    hidden_steps: list[torch.Tensor] = []
    feedback_steps: list[torch.Tensor] = []
    char_steps: list[torch.Tensor] = []
    sector_steps: list[torch.Tensor] = []
    input_gate_steps: list[torch.Tensor] = []
    recurrent_gate_steps: list[torch.Tensor] = []
    for time_idx in range(frame_num):
        feedback_steps.append(feedback.detach().clone())
        gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
        if record_gates and record_input_gate:
            input_gate_steps.append(gate_input.mean(dim=1))
        if record_gates:
            recurrent_gate_steps.append(gate_recurrent.mean(dim=1))
        input_term = torch.einsum(
            "bi,bhi,hi->bh",
            encoded[:, time_idx],
            gate_input,
            model.rnn.weight_ih_l0,
        )
        recurrent_term = torch.einsum(
            "bi,bhi,hi->bh", hidden, gate_recurrent, model.rnn.weight_hh_l0
        )
        preactivation = input_term + recurrent_term
        if model.rnn.bias_ih_l0 is not None:
            preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
        if model.rnn.bias_hh_l0 is not None:
            preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
        hidden = torch.relu(model.LNormRNN(torch.tanh(preactivation)))
        char_logits, sector_logits = model.classifier(hidden)
        feedback = torch.cat([char_logits, sector_logits], dim=-1).to(torch.float32)
        hidden_steps.append(hidden)
        char_steps.append(char_logits)
        sector_steps.append(sector_logits)
    output = {
        "hidden": torch.stack(hidden_steps, dim=1),
        "feedback": torch.stack(feedback_steps, dim=1),
        "char_logits": torch.stack(char_steps, dim=1),
        "sector_logits": torch.stack(sector_steps, dim=1),
    }
    if record_gates:
        if record_input_gate:
            output["input_gate"] = torch.stack(input_gate_steps, dim=1)
        output["recurrent_gate"] = torch.stack(recurrent_gate_steps, dim=1)
    return output


def collect_split(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    record_gates: bool,
) -> dict[str, np.ndarray]:
    """Collect pre-gate encoder and recurrent measurements for one held-out split."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    collected: dict[str, list[np.ndarray]] = {
        "encoder": [],
        "hidden": [],
        "feedback": [],
        "char_logits": [],
        "sector_logits": [],
        "labels": [],
    }
    if record_gates:
        collected["input_gate"] = []
        collected["recurrent_gate"] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            labels = labels.to(dtype=torch.int64)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoder = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            trajectory = _trajectory_with_measurements(encoder, model, record_gates)
            collected["encoder"].append(encoder.cpu().numpy().astype(np.float32))
            collected["labels"].append(labels.numpy().astype(np.int64))
            for key, value in trajectory.items():
                collected[key].append(value.cpu().numpy().astype(np.float32))
            samples_done = min((batch_idx + 1) * batch_size, len(dataset))
            if samples_done % 200 < batch_size or batch_idx + 1 == len(loader):
                print(
                    f"  collected {samples_done}/{len(dataset)} sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    output = {
        key: np.concatenate(values, axis=0).reshape(-1, values[0].shape[-1])
        for key, values in collected.items()
    }
    frame_num = int(dataset.frame_num)
    output["sequence_id"] = np.repeat(np.arange(len(dataset), dtype=np.int64), frame_num)
    output["raw_frame"] = np.arange(
        int(dataset.chan_num), int(dataset.chan_num) + len(dataset) * frame_num, dtype=np.int64
    )
    return output


def _selectivity_payload(result: SelectivityResult, inference: dict[str, np.ndarray]) -> dict:
    return {
        "eta_sector": result.eta_sector.astype(np.float32),
        "eta_digit": result.eta_digit.astype(np.float32),
        "eta_interaction": result.eta_interaction.astype(np.float32),
        "eta_residual": result.eta_residual.astype(np.float32),
        "tuning_sector": result.tuning_sector.astype(np.float32),
        "tuning_digit": result.tuning_digit.astype(np.float32),
        "interaction_dominant": interaction_dominant(result).astype(np.uint8),
        **{
            key: value.astype(np.uint8) if key.startswith("passed_") else value
            for key, value in inference.items()
        },
    }


def analyze_selectivity(
    encoder: np.ndarray,
    hidden: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed_offset: int,
) -> tuple[dict[str, SelectivityResult], dict[str, dict[str, np.ndarray]]]:
    """Compute activation selectivity and per-unit permutation inference for both spaces."""

    results: dict[str, SelectivityResult] = {}
    inference: dict[str, dict[str, np.ndarray]] = {}
    for pop_idx, (population, values) in enumerate((("encoder", encoder), ("hidden", hidden))):
        print(f"Part 1: {population} two-way decomposition ({values.shape})", flush=True)
        result = two_way_decomposition(values, labels)
        inferred = permutation_selectivity(
            values,
            labels,
            result,
            resamples=args.resamples,
            seed=args.seed + seed_offset + pop_idx * 10000,
            device=device,
            permutation_batch_size=args.permutation_batch_size,
            fdr_alpha=args.fdr_alpha,
        )
        results[population] = result
        inference[population] = inferred
        print(
            f"  {population}: sector FDR={int(inferred['passed_sector'].sum())}, "
            f"digit FDR={int(inferred['passed_digit'].sum())}, "
            f"interaction-dominant={int(interaction_dominant(result).sum())}",
            flush=True,
        )
    return results, inference


def _cell_inputs(
    gate_name: str,
    factor: str,
    selectivity: dict[str, SelectivityResult],
    inference: dict[str, dict[str, np.ndarray]],
    gate_columns: dict[str, np.ndarray],
) -> tuple[np.ndarray, SelectivityResult, np.ndarray, np.ndarray]:
    population = "encoder" if gate_name == "input" else "hidden"
    result = selectivity[population]
    passed = inference[population][f"passed_{factor}"].astype(bool)
    tuning = result.tuning_sector if factor == "sector" else result.tuning_digit
    return gate_columns[gate_name], result, passed, tuning


def run_part2(
    selectivity: dict[str, SelectivityResult],
    inference: dict[str, dict[str, np.ndarray]],
    gates: dict[str, np.ndarray],
    labels: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
    *,
    tag: str,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run both interaction policies for all four gate x variable cells."""

    selected_labels = labels[test_mask]
    selected_gates = {key: values[test_mask] for key, values in gates.items()}
    report: dict[str, Any] = {"tag": tag, "n_test_trials": int(test_mask.sum())}
    arrays: dict[str, np.ndarray] = {}
    for policy_idx, exclude_interaction in enumerate((True, False)):
        policy = "interaction_excluded" if exclude_interaction else "interaction_included"
        policy_report: dict[str, Any] = {"cells": {}}
        for gate_idx, gate_name in enumerate(("input", "recurrent")):
            for factor_idx, factor in enumerate(("sector", "digit")):
                cell = f"{gate_name}_{factor}"
                gate_values, result, passed, tuning = _cell_inputs(
                    gate_name, factor, selectivity, inference, selected_gates
                )
                dominant = interaction_dominant(result)
                eligible = passed & (~dominant if exclude_interaction else True)
                removed = int(np.count_nonzero(passed & dominant)) if exclude_interaction else 0
                context = selected_labels[:, 1] if factor == "sector" else selected_labels[:, 0]
                cell_report: dict[str, Any] = {
                    "fdr_selective_units": int(passed.sum()),
                    "interaction_dominant_removed": removed,
                    "eligible_units": int(eligible.sum()),
                    "top_percent": {},
                }
                for percent_idx, percent in enumerate(args.top_percent):
                    fraction = percent / 100.0
                    masks = relevance_masks(tuning, eligible, fraction)
                    moments = trial_relevance_moments(gate_values, context, masks, eligible)
                    common_seed = args.seed + seed_offset + policy_idx * 100000 + percent_idx
                    point, bootstrap = bootstrap_d(
                        moments, resamples=args.resamples, seed=common_seed
                    )
                    null = relevance_label_null(
                        gate_values,
                        selected_labels,
                        factor,
                        tuning,
                        eligible,
                        fraction,
                        resamples=args.resamples,
                        seed=(
                            args.seed
                            + seed_offset
                            + policy_idx * 100000
                            + gate_idx * 10000
                            + factor_idx * 1000
                            + percent_idx
                        ),
                    )
                    p_value = float((1 + np.count_nonzero(null >= point)) / (args.resamples + 1))
                    percent_key = str(percent)
                    cell_report["top_percent"][percent_key] = {
                        "cohens_d": float(point),
                        "bootstrap_ci95": [
                            float(x) for x in np.quantile(bootstrap, [0.025, 0.975])
                        ],
                        "relevant_units_per_level": masks.sum(axis=1).astype(int).tolist(),
                        "relevance_shuffle_p_value": p_value,
                    }
                    prefix = f"{tag}_{policy}_{cell}_top{percent}"
                    arrays[f"{prefix}_bootstrap_d"] = bootstrap.astype(np.float32)
                    arrays[f"{prefix}_relevance_null_d"] = null.astype(np.float32)

                alignment = cosine_alignment(
                    tuning,
                    gate_values,
                    selected_labels,
                    factor,
                    eligible,
                    resamples=args.resamples,
                    seed=args.seed + seed_offset + 500000 + gate_idx * 1000 + factor_idx,
                )
                cell_report["continuous_alignment"] = {
                    "diagonal_minus_off_diagonal": alignment["diagonal_minus_off_diagonal"],
                    "permutation_p_value": alignment["permutation_p_value"],
                    "permutation_alternative": alignment["permutation_alternative"],
                }
                arrays[f"{tag}_{policy}_{cell}_alignment_matrix"] = alignment["matrix"]
                arrays[f"{tag}_{policy}_{cell}_alignment_null"] = alignment["permutation_null"]
                policy_report["cells"][cell] = cell_report

        policy_report["direct_differences"] = _part2_direct_differences(
            arrays, tag, policy, args.top_percent
        )
        report[policy] = policy_report
    report["old_sector_input_proxy"] = {
        "reported_cohens_d": 0.660599811129108,
        "definition": "fixed 3x3 spatial proxy over all encoder columns",
        "new_definition": "FDR-selective activation tuning relevance; not numerically identical",
    }
    return report, arrays


def _part2_direct_differences(
    arrays: dict[str, np.ndarray],
    tag: str,
    policy: str,
    percentages: list[int],
) -> dict[str, Any]:
    """Test the predicted gate x variable dissociation with paired bootstrap draws."""

    output: dict[str, Any] = {}
    for percent in percentages:

        def boot(cell: str) -> np.ndarray:
            return arrays[f"{tag}_{policy}_{cell}_top{percent}_bootstrap_d"].astype(np.float64)

        input_sector = boot("input_sector")
        input_digit = boot("input_digit")
        recurrent_sector = boot("recurrent_sector")
        recurrent_digit = boot("recurrent_digit")
        contrasts = {
            "input_sector_minus_digit": input_sector - input_digit,
            "recurrent_digit_minus_sector": recurrent_digit - recurrent_sector,
            "dissociation_interaction": (
                input_sector - input_digit + recurrent_digit - recurrent_sector
            ),
            "sector_input_minus_recurrent": input_sector - recurrent_sector,
            "digit_recurrent_minus_input": recurrent_digit - input_digit,
        }
        output[str(percent)] = {}
        for name, values in contrasts.items():
            output[str(percent)][name] = {
                "mean_bootstrap_difference": float(values.mean()),
                "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
                "one_sided_p_for_predicted_positive": float(
                    (1 + np.count_nonzero(values <= 0)) / (values.size + 1)
                ),
            }
    return output


def _event_indices(
    dataset: torch.utils.data.Dataset,
    trajectory: dict[str, np.ndarray],
    post_frames: int,
    sequence_mask: np.ndarray,
) -> np.ndarray:
    """Map eligible raw fg switches to flat post1 indices without crossing sequences."""

    raw_start = int(trajectory["raw_frame"][0])
    raw_stop = int(trajectory["raw_frame"][-1])
    indices: list[int] = []
    for raw_switch in np.flatnonzero(np.asarray(dataset.fg_switch) != 0):
        flat = int(raw_switch - raw_start)
        if flat < 0 or raw_switch + post_frames - 1 > raw_stop:
            continue
        sequence_id = int(trajectory["sequence_id"][flat])
        if not sequence_mask[sequence_id]:
            continue
        if np.any(trajectory["sequence_id"][flat : flat + post_frames] != sequence_id):
            continue
        if np.any(dataset.fg_switch[raw_switch + 1 : raw_switch + post_frames] != 0):
            continue
        indices.append(flat)
    if not indices:
        raise RuntimeError("No eligible foreground-switch events remain")
    return np.asarray(indices, dtype=np.int64)


def _cosine_to_profile(gates: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    numerator = np.einsum("etu,eu->et", gates, profiles)
    denominator = np.linalg.norm(gates, axis=2) * np.linalg.norm(profiles, axis=1)[:, None]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def run_part3(
    selectivity: dict[str, SelectivityResult],
    inference: dict[str, dict[str, np.ndarray]],
    trajectory: dict[str, np.ndarray],
    dataset: torch.utils.data.Dataset,
    sequence_mask: np.ndarray,
    args: argparse.Namespace,
    *,
    tag: str,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Analyze gate, argmax, and graded-evidence timing around foreground switches."""

    event_start = _event_indices(dataset, trajectory, args.post_frames, sequence_mask)
    offsets = np.arange(args.post_frames, dtype=np.int64)
    event_indices = event_start[:, None] + offsets[None, :]
    raw_switch = trajectory["raw_frame"][event_start]
    old_labels = dataset.labels_sector[raw_switch - 1].astype(np.int64)
    new_labels = dataset.labels_sector[raw_switch].astype(np.int64)
    report: dict[str, Any] = {
        "tag": tag,
        "post_frame_definition": (
            "post1 is the raw fg_switch frame; with chan_num=2 its input straddles old/new"
        ),
        "onset_baseline": "post1 (the switch/straddling frame)",
        "information_theoretic_floor": "post3",
        "eligible_fg_switch_events_before_factor_filter": int(event_start.size),
        "cells": {},
        "readout": {},
    }
    arrays: dict[str, np.ndarray] = {
        f"{tag}_event_start_flat": event_start.astype(np.int64),
        f"{tag}_event_raw_switch": raw_switch.astype(np.int64),
    }
    gates = {"input": trajectory["input_gate"], "recurrent": trajectory["recurrent_gate"]}

    for factor_idx, factor in enumerate(("sector", "digit")):
        label_column = 1 if factor == "sector" else 0
        changed = old_labels[:, label_column] != new_labels[:, label_column]
        selected_indices = event_indices[changed]
        old_context = old_labels[changed, label_column]
        new_context = new_labels[changed, label_column]
        logits_key = "sector_logits" if factor == "sector" else "char_logits"
        logits = trajectory[logits_key][selected_indices]
        prediction = logits.argmax(axis=2)
        correct = (prediction == new_context[:, None]).astype(np.float64)
        shifted = logits - logits.max(axis=2, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=2, keepdims=True)
        graded = np.take_along_axis(probability, new_context[:, None, None], axis=2)[..., 0]
        correct_onset = bootstrap_onset(
            correct, resamples=args.resamples, seed=args.seed + seed_offset + factor_idx * 100
        )
        graded_onset = bootstrap_onset(
            graded,
            resamples=args.resamples,
            seed=args.seed + seed_offset + factor_idx * 100 + 1,
        )
        first_correct = first_crossing(correct, threshold=1.0)
        graded_delta = graded - graded[:, [0]]
        graded_delta[:, 0] = -np.inf
        first_graded_rise = first_crossing(graded_delta, threshold=np.finfo(np.float64).eps)
        report["readout"][factor] = {
            "n_changed_events": int(changed.sum()),
            "argmax_accuracy_onset": _compact_onset(correct_onset),
            "graded_probability_onset": _compact_onset(graded_onset),
            "first_correct_rate_within_window": float(np.isfinite(first_correct).mean()),
            "first_graded_rise_rate_within_window": float(np.isfinite(first_graded_rise).mean()),
        }
        arrays[f"{tag}_{factor}_readout_correct"] = correct.astype(np.float32)
        arrays[f"{tag}_{factor}_readout_graded"] = graded.astype(np.float32)
        arrays[f"{tag}_{factor}_first_correct"] = first_correct.astype(np.float32)
        arrays[f"{tag}_{factor}_first_graded_rise"] = first_graded_rise.astype(np.float32)

        for gate_idx, gate_name in enumerate(("input", "recurrent")):
            population = "encoder" if gate_name == "input" else "hidden"
            result = selectivity[population]
            passed = inference[population][f"passed_{factor}"].astype(bool)
            eligible = passed & ~interaction_dominant(result)
            tuning = result.tuning_sector if factor == "sector" else result.tuning_digit
            gate_events = gates[gate_name][selected_indices][:, :, eligible].astype(np.float64)
            new_profile = tuning[new_context][:, eligible]
            old_profile = tuning[old_context][:, eligible]
            alignment = _cosine_to_profile(gate_events, new_profile) - _cosine_to_profile(
                gate_events, old_profile
            )
            onset = bootstrap_onset(
                alignment,
                resamples=args.resamples,
                seed=args.seed + seed_offset + 1000 + gate_idx * 100 + factor_idx,
            )
            gate_crossing = first_negative_to_nonnegative(alignment)
            paired_argmax = paired_lead_test(gate_crossing, first_correct)
            paired_graded = paired_lead_test(gate_crossing, first_graded_rise)
            cell = f"{gate_name}_{factor}"
            report["cells"][cell] = {
                "eligible_units": int(eligible.sum()),
                "gate_alignment_onset": _compact_onset(onset),
                "mean_curve_direction": _mean_curve_direction(alignment),
                "zero_crossing_rate_within_window": float(np.isfinite(gate_crossing).mean()),
                "paired_gate_vs_first_correct": _compact_paired(paired_argmax),
                "paired_gate_vs_first_graded_rise": _compact_paired(paired_graded),
                "architectural_interpretation": _timing_interpretation(
                    paired_argmax, paired_graded
                ),
            }
            arrays[f"{tag}_{cell}_alignment_difference"] = alignment.astype(np.float32)
            arrays[f"{tag}_{cell}_gate_zero_crossing"] = gate_crossing.astype(np.float32)
            arrays[f"{tag}_{cell}_argmax_minus_gate"] = paired_argmax["differences"]
            arrays[f"{tag}_{cell}_graded_rise_minus_gate"] = paired_graded["differences"]
    return report, arrays


def _compact_onset(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "onset_post_frame": int(result["onset_post_frame"]),
        "onset_ci95": result["onset_ci95"],
        "mean_by_post_frame": result["mean"].astype(float).tolist(),
        "delta_ci95_lower": result["delta_ci95_lower"].astype(float).tolist(),
        "delta_ci95_upper": result["delta_ci95_upper"].astype(float).tolist(),
    }


def _compact_paired(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in ("differences", "valid_mask")}


def _mean_curve_direction(alignment: np.ndarray) -> dict[str, Any]:
    mean = np.asarray(alignment, dtype=np.float64).mean(axis=0)
    crossing = first_negative_to_nonnegative(mean[None, :])[0]
    return {
        "post1_mean": float(mean[0]),
        "starts_negative_as_assumed": bool(mean[0] < 0),
        "negative_to_nonnegative_crossing_post_frame": (
            int(crossing) if np.isfinite(crossing) else None
        ),
    }


def _timing_interpretation(paired_argmax: dict[str, Any], paired_graded: dict[str, Any]) -> str:
    if paired_argmax["n_paired_events"] == 0:
        return "no directional gate crossing/readout pairs; causal timing is unsupported"
    argmax_lead = (
        paired_argmax["mean_difference"] > 0 and paired_argmax["wilcoxon_greater_p_value"] < 0.05
    )
    graded_lead = (
        paired_graded["n_paired_events"] > 0
        and paired_graded["mean_difference"] > 0
        and paired_graded["wilcoxon_greater_p_value"] < 0.05
    )
    if argmax_lead and not graded_lead:
        return "gate leads argmax but not graded rise: amplification-consistent pattern"
    if argmax_lead and graded_lead:
        return (
            "gate apparently leads both argmax and graded rise; this conflicts with one-step "
            "readout feedback, so a causal gate lead is not supported"
        )
    if paired_argmax["mean_difference"] < 0:
        return "gate does not lead argmax within events; reflection/mixed pattern"
    return "no significant per-event gate lead over argmax; causal amplification is unsupported"


def postprocess_part3_directional_crossings(save_dir: str) -> None:
    """Recompute saved paired timing with strict negative-to-nonnegative crossings."""

    event_path = os.path.join(save_dir, "part3_events.npz")
    result_path = os.path.join(save_dir, "part3_results.json")
    with np.load(event_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    report = _load_json(result_path)
    report_groups = [report["primary_validation_profiles_test_switches"]]
    report_groups.extend(report.get("test_split_half_folds", []))
    for group in report_groups:
        tag = group["tag"]
        group["zero_crossing_definition"] = (
            "first negative-to-nonnegative crossing after at least one negative post frame; "
            "initial nonnegative post1 is not a reconfiguration"
        )
        for cell, cell_report in group["cells"].items():
            factor = "sector" if cell.endswith("sector") else "digit"
            alignment_key = f"{tag}_{cell}_alignment_difference"
            alignment = arrays[alignment_key].astype(np.float64)
            gate_crossing = first_negative_to_nonnegative(alignment)
            first_correct = arrays[f"{tag}_{factor}_first_correct"]
            first_graded = arrays[f"{tag}_{factor}_first_graded_rise"]
            paired_argmax = paired_lead_test(gate_crossing, first_correct)
            paired_graded = paired_lead_test(gate_crossing, first_graded)
            cell_report["mean_curve_direction"] = _mean_curve_direction(alignment)
            cell_report["zero_crossing_rate_within_window"] = float(
                np.isfinite(gate_crossing).mean()
            )
            cell_report["paired_gate_vs_first_correct"] = _compact_paired(paired_argmax)
            cell_report["paired_gate_vs_first_graded_rise"] = _compact_paired(paired_graded)
            cell_report["architectural_interpretation"] = _timing_interpretation(
                paired_argmax, paired_graded
            )
            arrays[f"{tag}_{cell}_gate_zero_crossing"] = gate_crossing.astype(np.float32)
            arrays[f"{tag}_{cell}_argmax_minus_gate"] = paired_argmax["differences"]
            arrays[f"{tag}_{cell}_graded_rise_minus_gate"] = paired_graded["differences"]
    report["split_half_average"] = _average_nested_numeric(report.get("test_split_half_folds", []))
    temporary_npz = f"{event_path}.tmp.npz"
    np.savez_compressed(temporary_npz, **arrays)
    os.replace(temporary_npz, event_path)
    _save_json(result_path, report)
    print(f"Recomputed directional Part 3 crossings: {os.path.abspath(save_dir)}")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _split_half_sequence_masks(
    sequence_ids: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n_sequences = int(sequence_ids.max()) + 1
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_sequences)
    half = n_sequences // 2
    sequence_a = np.zeros(n_sequences, dtype=bool)
    sequence_a[order[:half]] = True
    sequence_b = ~sequence_a
    return sequence_a, sequence_b


def _frame_mask(sequence_ids: np.ndarray, sequence_mask: np.ndarray) -> np.ndarray:
    return sequence_mask[np.asarray(sequence_ids, dtype=np.int64)]


def _save_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(_json_ready(value), file_obj, indent=2, allow_nan=False)


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


def _print_part0(part0: dict[str, Any]) -> None:
    print("\nPART 0 — SPLITS (before plotting)")
    for split, report in part0.items():
        if not isinstance(report, dict) or "n_trials" not in report:
            continue
        print(
            f"{split}: n={report['n_trials']}, chi2={report['chi_square']:.3f}, "
            f"df={report['chi_square_degrees_freedom']}, "
            f"p={report['chi_square_p_value']:.6g}, V={report['cramers_v']:.4f}"
        )
        print(np.asarray(report["joint_frequency_sector_rows_digit_columns"], dtype=np.int64))
        if not report["independent_at_alpha_0_05"]:
            print("  sector and digit are not independent; marginal results are confounded.")


def _print_primary_part2(report: dict[str, Any], percentages: list[int]) -> None:
    print("\nPRIMARY PART 2 — validation estimate / test effect")
    for policy in ("interaction_excluded", "interaction_included"):
        print(f"\n{policy}")
        for percent in percentages:
            print(f"top {percent}%: gate | sector d [CI] | digit d [CI] | sector-digit")
            for gate in ("input", "recurrent"):
                sector = report[policy]["cells"][f"{gate}_sector"]["top_percent"][str(percent)]
                digit = report[policy]["cells"][f"{gate}_digit"]["top_percent"][str(percent)]
                difference = sector["cohens_d"] - digit["cohens_d"]
                print(
                    f"  {gate:9s} | {sector['cohens_d']:+.3f} {sector['bootstrap_ci95']} | "
                    f"{digit['cohens_d']:+.3f} {digit['bootstrap_ci95']} | {difference:+.3f}"
                )
        interaction = report[policy]["direct_differences"]["20"]["dissociation_interaction"]
        print(f"direct dissociation contrast (top20): {interaction}")


def _print_primary_part3(report: dict[str, Any]) -> None:
    print("\nPRIMARY PART 3 — per-event paired timing")
    for cell, cell_report in report["cells"].items():
        paired = cell_report["paired_gate_vs_first_correct"]
        graded = cell_report["paired_gate_vs_first_graded_rise"]
        print(
            f"{cell}: argmax difference mean={paired['mean_difference']:+.3f}, "
            f"median={paired['median_difference']:+.3f}, "
            f"p={paired['wilcoxon_greater_p_value']:.6g}; "
            f"graded-rise mean={graded['mean_difference']:+.3f}, "
            f"p={graded['wilcoxon_greater_p_value']:.6g}"
        )
        print(f"  {cell_report['architectural_interpretation']}")


def main() -> None:
    """Run the full validation/test and split-half analysis."""

    args = parse_args()
    if args.resamples <= 0 or args.post_frames < 2:
        raise ValueError("resamples must be positive and post_frames must be at least 2")
    if any(percent <= 0 or percent >= 100 for percent in args.top_percent):
        raise ValueError("top_percent values must lie strictly between 0 and 100")
    if args.save_dir:
        args.decomposition_dir = args.relevance_dir = args.timing_dir = args.control_dir = (
            args.save_dir
        )
    for directory in (
        args.decomposition_dir,
        args.relevance_dir,
        args.timing_dir,
        args.control_dir,
    ):
        os.makedirs(directory, exist_ok=True)
    if args.postprocess_part3_only:
        postprocess_part3_directional_crossings(args.timing_dir)
        return
    device = resolve_device(args.device, require_cuda_if_requested=True)
    args.use_sector_mode = True
    args.predict_all_chars = False

    validation_ds, validation_num_pos = build_eval_dataset(args, "validation")
    test_ds, test_num_pos = build_eval_dataset(args, "test")
    if validation_num_pos != test_num_pos:
        raise RuntimeError("validation/test num_pos mismatch")
    model = build_model_from_ckpt(args.ckpt, test_num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis currently requires a single-layer GaWF checkpoint")
    architecture = {
        "encoder_map": [32, 6, 6],
        "encoder_flat": int(model.encoder_flatten_size),
        "hidden": int(model.rnn.hidden_size),
        "input_gate": [int(model.rnn.hidden_size), int(model.encoder_flatten_size)],
        "recurrent_gate": [int(model.rnn.hidden_size), int(model.rnn.hidden_size)],
        "encoder_activation_is_pre_gate": True,
        "evidence": "encoder_module output is recorded before _trajectory_with_measurements",
    }
    print(f"ARCHITECTURE: {architecture}")

    print("Collecting VALIDATION activations (no gates)")
    validation = collect_split(
        validation_ds,
        model,
        device,
        args.batch_size,
        args.num_workers,
        record_gates=False,
    )
    print("Collecting TEST activations, gates, and readouts")
    test = collect_split(
        test_ds,
        model,
        device,
        args.batch_size,
        args.num_workers,
        record_gates=True,
    )
    validation_labels = validation["labels"].astype(np.int64)
    test_labels = test["labels"].astype(np.int64)
    sequence_a, sequence_b = _split_half_sequence_masks(test["sequence_id"], args.seed)
    frame_a = _frame_mask(test["sequence_id"], sequence_a)
    frame_b = _frame_mask(test["sequence_id"], sequence_b)
    part0 = {
        "validation": joint_design(validation_labels),
        "test": joint_design(test_labels),
        "test_half_a": joint_design(test_labels[frame_a]),
        "test_half_b": joint_design(test_labels[frame_b]),
        "split_half_unit": "model sequence (32 outputs), preventing temporal leakage",
    }
    _print_part0(part0)
    _save_json(os.path.join(args.control_dir, "part0_splits.json"), part0)

    primary_selectivity, primary_inference = analyze_selectivity(
        validation["encoder"],
        validation["hidden"],
        validation_labels,
        args,
        device,
        seed_offset=0,
    )
    part1_summary = {
        "architecture": architecture,
        "fdr_alpha": args.fdr_alpha,
        "permutations": args.resamples,
        "tuning_warning": (
            "within-unit z-scoring forces a preferred level even for non-selective units; "
            "downstream analyses therefore require the FDR mask"
        ),
        "encoder_architectural_assumption_test": architecture_axis_variance(
            primary_selectivity["encoder"]
        ),
        "interaction_dominant": {},
    }
    for population in ("encoder", "hidden"):
        dominant = interaction_dominant(primary_selectivity[population])
        part1_summary["interaction_dominant"][population] = {
            "count": int(dominant.sum()),
            "total": int(dominant.size),
            "fraction": float(dominant.mean()),
        }
    part1_arrays: dict[str, np.ndarray] = {}
    for population in ("encoder", "hidden"):
        for key, value in _selectivity_payload(
            primary_selectivity[population], primary_inference[population]
        ).items():
            part1_arrays[f"primary_{population}_{key}"] = value

    gates = {"input": test["input_gate"], "recurrent": test["recurrent_gate"]}
    primary_part2, primary_part2_arrays = run_part2(
        primary_selectivity,
        primary_inference,
        gates,
        test_labels,
        np.ones(test_labels.shape[0], dtype=bool),
        args,
        tag="primary",
        seed_offset=0,
    )
    primary_part3, primary_part3_arrays = run_part3(
        primary_selectivity,
        primary_inference,
        test,
        test_ds,
        np.ones(sequence_a.size, dtype=bool),
        args,
        tag="primary",
        seed_offset=0,
    )

    split_part2: list[dict[str, Any]] = []
    split_part3: list[dict[str, Any]] = []
    split_arrays: dict[str, np.ndarray] = {}
    if not args.skip_split_half:
        for fold_idx, (estimate_frame, test_frame, estimate_sequence, test_sequence) in enumerate(
            ((frame_a, frame_b, sequence_a, sequence_b), (frame_b, frame_a, sequence_b, sequence_a))
        ):
            fold_tag = "split_a_est_b_test" if fold_idx == 0 else "split_b_est_a_test"
            fold_selectivity, fold_inference = analyze_selectivity(
                test["encoder"][estimate_frame],
                test["hidden"][estimate_frame],
                test_labels[estimate_frame],
                args,
                device,
                seed_offset=(fold_idx + 1) * 1000000,
            )
            for population in ("encoder", "hidden"):
                for key, value in _selectivity_payload(
                    fold_selectivity[population], fold_inference[population]
                ).items():
                    part1_arrays[f"{fold_tag}_{population}_{key}"] = value
            fold_part2, fold_part2_arrays = run_part2(
                fold_selectivity,
                fold_inference,
                gates,
                test_labels,
                test_frame,
                args,
                tag=fold_tag,
                seed_offset=(fold_idx + 1) * 1000000,
            )
            fold_part3, fold_part3_arrays = run_part3(
                fold_selectivity,
                fold_inference,
                test,
                test_ds,
                test_sequence,
                args,
                tag=fold_tag,
                seed_offset=(fold_idx + 1) * 1000000,
            )
            split_part2.append(fold_part2)
            split_part3.append(fold_part3)
            split_arrays.update(fold_part2_arrays)
            split_arrays.update(fold_part3_arrays)

    part2_report = {
        "primary_validation_estimate_test_effect": primary_part2,
        "test_split_half_folds": split_part2,
        "split_half_average": _average_nested_numeric(split_part2),
    }
    part3_report = {
        "primary_validation_profiles_test_switches": primary_part3,
        "test_split_half_folds": split_part3,
        "split_half_average": _average_nested_numeric(split_part3),
    }
    _print_primary_part2(primary_part2, args.top_percent)
    _print_primary_part3(primary_part3)
    np.savez_compressed(
        os.path.join(args.decomposition_dir, "part1_selectivity.npz"), **part1_arrays
    )
    np.savez_compressed(
        os.path.join(args.relevance_dir, "part2_inference.npz"),
        **primary_part2_arrays,
        **{key: value for key, value in split_arrays.items() if "top" in key or "alignment" in key},
    )
    np.savez_compressed(
        os.path.join(args.timing_dir, "part3_events.npz"),
        **primary_part3_arrays,
        **{
            key: value
            for key, value in split_arrays.items()
            if "top" not in key and "alignment_matrix" not in key and "alignment_null" not in key
        },
    )
    _save_json(os.path.join(args.decomposition_dir, "part1_summary.json"), part1_summary)
    _save_json(os.path.join(args.relevance_dir, "part2_results.json"), part2_report)
    _save_json(os.path.join(args.timing_dir, "part3_results.json"), part3_report)
    metadata = {
        "checkpoint": os.path.abspath(args.ckpt),
        "data_dir": os.path.abspath(args.data_dir),
        "data_suffix": args.data_suffix,
        "output_dirs": {
            "decomposition": os.path.abspath(args.decomposition_dir),
            "relevance": os.path.abspath(args.relevance_dir),
            "timing": os.path.abspath(args.timing_dir),
            "controls": os.path.abspath(args.control_dir),
        },
        "architecture": architecture,
        "validation_frames": int(validation_labels.shape[0]),
        "test_frames": int(test_labels.shape[0]),
        "frame_num": int(test_ds.frame_num),
        "chan_num": int(test_ds.chan_num),
        "aggregation": "gate matrix averaged over rows; columns retain encoder/hidden units",
        "random_seed": args.seed,
        "resamples": args.resamples,
        "num_workers": args.num_workers,
        "top_percent": args.top_percent,
        "split_half_enabled": not args.skip_split_half,
    }
    _save_json(os.path.join(args.control_dir, "run_metadata.json"), metadata)
    print(f"Saved decomposition outputs to: {os.path.abspath(args.decomposition_dir)}")
    print(f"Saved relevance outputs to: {os.path.abspath(args.relevance_dir)}")
    print(f"Saved timing outputs to: {os.path.abspath(args.timing_dir)}")
    print(f"Saved control outputs to: {os.path.abspath(args.control_dir)}")


def _average_nested_numeric(records: list[dict[str, Any]]) -> Any:
    """Average like-shaped numeric leaves across the two split-half fold reports."""

    if not records:
        return {}
    first = records[0]
    if isinstance(first, dict):
        shared = set(first)
        for record in records[1:]:
            shared &= set(record)
        return {
            key: _average_nested_numeric([record[key] for record in records])
            for key in sorted(shared)
        }
    if isinstance(first, list):
        if all(len(record) == len(first) for record in records):
            return [
                _average_nested_numeric([record[idx] for record in records])
                for idx in range(len(first))
            ]
        return first
    if isinstance(first, (int, float, np.integer, np.floating)) and not isinstance(first, bool):
        values = np.asarray(records, dtype=np.float64)
        return float(np.nanmean(values))
    return first


if __name__ == "__main__":
    main()
