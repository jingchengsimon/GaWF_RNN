"""Analyze context variance of GaWF, LSTM, and GRU unit-level gates on Clutter.

The analysis uses the same balanced 9-sector by 10-digit design as the GaWF gate
context-specificity Figure 03. GaWF connection gates are projected to destination units by
averaging incoming input or recurrent synapses before decomposition. LSTM contributes input,
forget, and output gates; GRU contributes reset and update gates. Candidate activations are
intentionally excluded.

Inputs: a compact GaWF trajectory (and optional saved gate mmap arrays), trained LSTM/GRU
checkpoints, the continuous Clutter test set, and reference labels.
Outputs: compact JSON/NPZ/CSV summaries; no trial-by-unit gate tensor is saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset
from utils_anal.gawf_gate_context_parts123 import _balanced_masks, _marginal_variance
from utils_anal.anal_paths import output_dir


GATE_NAMES = {
    "gawf": ("input_mean", "recurrent_mean"),
    "lstm": ("input", "forget", "output"),
    "gru": ("reset", "update"),
}


@dataclass
class UnitGateAggregate:
    """Balanced-cell sufficient statistics for one unit-level gate."""

    joint_sum: np.ndarray
    joint_sumsq: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    base = "./results/train_data/clutter/best_6model_param_matched_40h"
    parser.add_argument(
        "--lstm_ckpt",
        default=(
            f"{base}/lstm_sector_acc_h80_lr0.001_wd0.001_cdo0.0_"
            "rdo0.5_model.pth"
        ),
    )
    parser.add_argument(
        "--gru_ckpt",
        default=(
            f"{base}/gru_sector_acc_h105_lr0.005_wd0.001_cdo0.0_"
            "rdo0.5_model.pth"
        ),
    )
    parser.add_argument(
        "--trajectory",
        default=str(
            output_dir("A_raw_gate", "gawf_gate_distribution", "data")
            / "gawf_gate_trajectory.npz"
        ),
    )
    parser.add_argument(
        "--gawf_input_manifest",
        type=Path,
        default=output_dir(
            "D_variance_decomposition", "export_unified_variance_sources", "data"
        )
        / "input_manifest.json",
        help="Manifest containing saved trial-level GaWF gate arrays.",
    )
    parser.add_argument(
        "--include_gawf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include destination-unit means of GaWF incoming synapse gates.",
    )
    parser.add_argument("--gawf_gate_tau", type=float, default=0.5)
    parser.add_argument(
        "--reuse_existing_rnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse aligned LSTM/GRU results already present in save_dir.",
    )
    parser.add_argument(
        "--save_dir",
        default=str(
            output_dir(
                "D_variance_decomposition",
                "rnn_unit_gate_context_specificity",
                "data",
            )
        ),
    )
    parser.add_argument("--data_dir", default="./stimuli")
    parser.add_argument(
        "--data_suffix",
        default="40h-float32-nonjoint-10digit-unique-bg-causal-continuous",
    )
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gawf_frame_batch_size", type=int, default=64)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def lstm_unit_gates(
    encoded: torch.Tensor, rnn: torch.nn.LSTM
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reproduce a one-layer PyTorch LSTM and return its three sigmoid gates."""

    if rnn.num_layers != 1 or rnn.bidirectional:
        raise ValueError("Unit-gate extraction currently requires one unidirectional LSTM layer.")
    batch_size, steps, _features = encoded.shape
    hidden_size = rnn.hidden_size
    hidden = encoded.new_zeros((batch_size, hidden_size))
    cell = encoded.new_zeros((batch_size, hidden_size))
    outputs: list[torch.Tensor] = []
    gates: dict[str, list[torch.Tensor]] = {name: [] for name in GATE_NAMES["lstm"]}
    for step in range(steps):
        affine = F.linear(encoded[:, step], rnn.weight_ih_l0, rnn.bias_ih_l0)
        affine = affine + F.linear(hidden, rnn.weight_hh_l0, rnn.bias_hh_l0)
        input_raw, forget_raw, candidate_raw, output_raw = affine.chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_raw)
        forget_gate = torch.sigmoid(forget_raw)
        candidate = torch.tanh(candidate_raw)
        output_gate = torch.sigmoid(output_raw)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        gates["input"].append(input_gate)
        gates["forget"].append(forget_gate)
        gates["output"].append(output_gate)
        outputs.append(hidden)
    stacked = {name: torch.stack(values, dim=1) for name, values in gates.items()}
    return stacked, torch.stack(outputs, dim=1)


def gru_unit_gates(
    encoded: torch.Tensor, rnn: torch.nn.GRU
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Reproduce a one-layer PyTorch GRU and return its reset and update gates."""

    if rnn.num_layers != 1 or rnn.bidirectional:
        raise ValueError("Unit-gate extraction currently requires one unidirectional GRU layer.")
    batch_size, steps, _features = encoded.shape
    hidden_size = rnn.hidden_size
    hidden = encoded.new_zeros((batch_size, hidden_size))
    outputs: list[torch.Tensor] = []
    gates: dict[str, list[torch.Tensor]] = {name: [] for name in GATE_NAMES["gru"]}
    for step in range(steps):
        input_affine = F.linear(encoded[:, step], rnn.weight_ih_l0, rnn.bias_ih_l0)
        hidden_affine = F.linear(hidden, rnn.weight_hh_l0, rnn.bias_hh_l0)
        input_reset, input_update, input_candidate = input_affine.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = hidden_affine.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset_gate * hidden_candidate)
        hidden = candidate + update_gate * (hidden - candidate)
        gates["reset"].append(reset_gate)
        gates["update"].append(update_gate)
        outputs.append(hidden)
    stacked = {name: torch.stack(values, dim=1) for name, values in gates.items()}
    return stacked, torch.stack(outputs, dim=1)


def _new_aggregates(model_type: str, hidden_size: int) -> dict[str, UnitGateAggregate]:
    """Allocate balanced 90-cell sufficient statistics."""

    return {
        name: UnitGateAggregate(
            joint_sum=np.zeros((90, hidden_size), dtype=np.float64),
            joint_sumsq=np.zeros(hidden_size, dtype=np.float64),
        )
        for name in GATE_NAMES[model_type]
    }


def _add_by_code(target: np.ndarray, values: np.ndarray, codes: np.ndarray) -> None:
    """Accumulate rows into joint-cell sums."""

    for code in np.unique(codes):
        target[int(code)] += values[codes == code].sum(axis=0, dtype=np.float64)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    """Resolve one path stored relative to a GaWF source manifest."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _accumulate_destination_unit_gate(
    gate_path: Path,
    *,
    hidden_size: int,
    source_size: int,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    batch_size: int,
) -> UnitGateAggregate:
    """Average incoming GaWF synapses per destination unit and stream balanced moments."""

    if hidden_size <= 0 or source_size <= 0 or batch_size <= 0:
        raise ValueError("hidden_size, source_size, and batch_size must be positive")
    raw = np.load(gate_path, mmap_mode="r", allow_pickle=False)
    n_frames = reference_labels.shape[0]
    expected_size = n_frames * hidden_size * source_size
    if raw.size != expected_size:
        raise ValueError(
            f"{gate_path} contains {raw.size} values; expected {expected_size} "
            f"for ({n_frames}, {hidden_size}, {source_size})"
        )
    gates = raw.reshape(n_frames, hidden_size, source_size)
    joint = reference_labels[:, 1] * 10 + reference_labels[:, 0]
    aggregate = UnitGateAggregate(
        joint_sum=np.zeros((90, hidden_size), dtype=np.float64),
        joint_sumsq=np.zeros(hidden_size, dtype=np.float64),
    )
    qualified = 0
    next_report = 200
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        selected = equal_joint_mask[start:end]
        if not np.any(selected):
            continue
        destination_means = np.mean(
            np.asarray(gates[start:end][selected]), axis=2, dtype=np.float64
        )
        selected_codes = joint[start:end][selected]
        _add_by_code(aggregate.joint_sum, destination_means, selected_codes)
        aggregate.joint_sumsq += np.square(destination_means).sum(axis=0, dtype=np.float64)
        qualified += int(destination_means.shape[0])
        if qualified >= next_report:
            print(
                f"GaWF destination-unit gates: qualified frames={qualified}",
                flush=True,
            )
            next_report = ((qualified // 200) + 1) * 200
    expected_qualified = int(np.count_nonzero(equal_joint_mask))
    if qualified != expected_qualified:
        raise RuntimeError(
            f"Accumulated {qualified} balanced GaWF frames; expected {expected_qualified}"
        )
    return aggregate


def _accumulate_reconstructed_gawf_unit_gates(
    trajectory_path: Path,
    *,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    batch_size: int,
    gate_tau: float,
    device: torch.device,
) -> tuple[dict[str, UnitGateAggregate], tuple[int, int], tuple[int, int]]:
    """Reconstruct exact float32 GaWF gates and immediately reduce incoming synapses."""

    if batch_size <= 0 or gate_tau <= 0.0:
        raise ValueError("batch_size and gate_tau must be positive")
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        required = {"feedback", "labels", "U", "V", "weight_ih", "weight_hh"}
        missing = sorted(required - set(trajectory.files))
        if missing:
            raise ValueError(f"GaWF trajectory is missing arrays: {missing}")
        raw_feedback = np.asarray(trajectory["feedback"], dtype=np.float32)
        if raw_feedback.ndim < 2:
            raise ValueError("GaWF feedback must have a trailing feature dimension")
        feedback = raw_feedback.reshape(-1, raw_feedback.shape[-1])
        gawf_labels = np.asarray(trajectory["labels"], dtype=np.int64).reshape(-1, 2)
        u = np.asarray(trajectory["U"], dtype=np.float32)
        v = np.asarray(trajectory["V"], dtype=np.float32)
        weight_ih_shape = tuple(int(value) for value in trajectory["weight_ih"].shape)
        weight_hh_shape = tuple(int(value) for value in trajectory["weight_hh"].shape)
    if not np.array_equal(gawf_labels, reference_labels):
        raise RuntimeError("GaWF trajectory labels do not align with the LSTM/GRU labels")
    hidden_size, input_size = weight_ih_shape
    if u.shape != (hidden_size, feedback.shape[1]):
        raise ValueError(f"Unexpected GaWF U shape {u.shape}")
    if v.shape != (feedback.shape[1], input_size + hidden_size):
        raise ValueError(f"Unexpected GaWF V shape {v.shape}")
    if weight_hh_shape != (hidden_size, hidden_size):
        raise ValueError(f"Unexpected GaWF recurrent weight shape {weight_hh_shape}")
    aggregates = {
        name: UnitGateAggregate(
            joint_sum=np.zeros((90, hidden_size), dtype=np.float64),
            joint_sumsq=np.zeros(hidden_size, dtype=np.float64),
        )
        for name in GATE_NAMES["gawf"]
    }
    u_tensor = torch.from_numpy(u).to(device=device)
    v_tensor = torch.from_numpy(v).to(device=device)
    joint = reference_labels[:, 1] * 10 + reference_labels[:, 0]
    qualified = 0
    next_report = 200
    with torch.no_grad():
        for start in range(0, feedback.shape[0], batch_size):
            end = min(start + batch_size, feedback.shape[0])
            selected = equal_joint_mask[start:end]
            if not np.any(selected):
                continue
            feedback_tensor = torch.from_numpy(feedback[start:end][selected]).to(device=device)
            scaled_u = u_tensor.unsqueeze(0) * feedback_tensor.clamp(-10, 10).unsqueeze(1)
            gates = torch.sigmoid(torch.matmul(scaled_u, v_tensor) / gate_tau)
            gate_values = gates.cpu().numpy()
            destination_means = {
                "input_mean": np.mean(
                    gate_values[..., :input_size], axis=2, dtype=np.float64
                ),
                "recurrent_mean": np.mean(
                    gate_values[..., input_size:], axis=2, dtype=np.float64
                ),
            }
            selected_codes = joint[start:end][selected]
            for gate_name, values in destination_means.items():
                aggregate = aggregates[gate_name]
                _add_by_code(aggregate.joint_sum, values, selected_codes)
                aggregate.joint_sumsq += np.square(values).sum(
                    axis=0, dtype=np.float64
                )
            qualified += int(selected_codes.size)
            if qualified >= next_report:
                print(
                    f"GaWF destination-unit gates: qualified frames={qualified}",
                    flush=True,
                )
                next_report = ((qualified // 200) + 1) * 200
    expected_qualified = int(np.count_nonzero(equal_joint_mask))
    if qualified != expected_qualified:
        raise RuntimeError(
            f"Accumulated {qualified} balanced GaWF frames; expected {expected_qualified}"
        )
    return aggregates, weight_ih_shape, weight_hh_shape


def analyze_gawf(
    manifest_path: Path,
    trajectory_path: Path,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    batch_size: int,
    gate_tau: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Analyze destination-unit means from saved GaWF input/recurrent gate matrices."""

    manifest_path = manifest_path.expanduser().resolve()
    trajectory_path = trajectory_path.expanduser().resolve()
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        gawf_labels = np.asarray(trajectory["labels"], dtype=np.int64).reshape(-1, 2)
        weight_ih_shape = tuple(int(value) for value in trajectory["weight_ih"].shape)
        weight_hh_shape = tuple(int(value) for value in trajectory["weight_hh"].shape)
    if not np.array_equal(gawf_labels, reference_labels):
        raise RuntimeError("Saved GaWF labels do not align with the LSTM/GRU reference labels")
    if len(weight_ih_shape) != 2 or len(weight_hh_shape) != 2:
        raise ValueError("Saved GaWF weights must both be matrices")
    hidden_size, input_size = weight_ih_shape
    if weight_hh_shape != (hidden_size, hidden_size):
        raise ValueError(
            f"Incompatible GaWF weight shapes: input={weight_ih_shape}, "
            f"recurrent={weight_hh_shape}"
        )
    equal_n = int(np.count_nonzero(equal_joint_mask)) // 90
    report: dict[str, Any] = {
        "hidden_size": hidden_size,
        "gate_level": "destination_unit_mean_of_incoming_synapses",
        "source_axis_reduction": "arithmetic_mean_of_raw_sigmoid_gates",
        "gate_tau": gate_tau,
        "gates": {},
    }
    arrays: dict[str, np.ndarray] = {}
    gate_specs = (
        ("input_mean", "input_gate", input_size),
        ("recurrent_mean", "recurrent_gate", hidden_size),
    )
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_trajectory_value = payload.get("trajectory_npz")
        if not isinstance(manifest_trajectory_value, str):
            raise ValueError("GaWF input manifest must contain trajectory_npz")
        manifest_trajectory_path = _resolve_manifest_path(
            manifest_path, manifest_trajectory_value
        )
        with np.load(manifest_trajectory_path, allow_pickle=False) as manifest_trajectory:
            manifest_labels = np.asarray(
                manifest_trajectory["labels"], dtype=np.int64
            ).reshape(-1, 2)
            manifest_ih_shape = tuple(
                int(value) for value in manifest_trajectory["weight_ih"].shape
            )
            manifest_hh_shape = tuple(
                int(value) for value in manifest_trajectory["weight_hh"].shape
            )
        if not np.array_equal(manifest_labels, reference_labels):
            raise RuntimeError("Saved GaWF mmap labels do not align with reference labels")
        if manifest_ih_shape != weight_ih_shape or manifest_hh_shape != weight_hh_shape:
            raise RuntimeError("Saved GaWF mmap weight shapes do not match compact trajectory")
        objects = payload.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("GaWF input manifest must contain an objects dictionary")
        report["source_manifest"] = str(manifest_path)
        report["gate_source"] = "saved_float32_mmap"
        aggregates = {}
        for gate_name, object_name, source_size in gate_specs:
            object_spec = objects.get(object_name)
            if not isinstance(object_spec, dict) or not isinstance(
                object_spec.get("path"), str
            ):
                raise ValueError(f"GaWF input manifest is missing objects.{object_name}.path")
            gate_path = _resolve_manifest_path(manifest_path, object_spec["path"])
            aggregates[gate_name] = _accumulate_destination_unit_gate(
                gate_path,
                hidden_size=hidden_size,
                source_size=source_size,
                reference_labels=reference_labels,
                equal_joint_mask=equal_joint_mask,
                batch_size=batch_size,
            )
    else:
        print(
            f"GaWF source manifest not found at {manifest_path}; reconstructing gates "
            "from compact trajectory",
            flush=True,
        )
        report["source_trajectory"] = str(trajectory_path)
        report["gate_source"] = "reconstructed_float32_from_feedback_U_V"
        aggregates, reconstructed_ih_shape, reconstructed_hh_shape = (
            _accumulate_reconstructed_gawf_unit_gates(
                trajectory_path,
                reference_labels=reference_labels,
                equal_joint_mask=equal_joint_mask,
                batch_size=batch_size,
                gate_tau=gate_tau,
                device=device,
            )
        )
        if reconstructed_ih_shape != weight_ih_shape or reconstructed_hh_shape != weight_hh_shape:
            raise RuntimeError("Reconstructed GaWF gate shapes changed during analysis")
    for gate_name, _object_name, _source_size in gate_specs:
        gate_report, cell_mean = _summarize_gate(aggregates[gate_name], equal_n)
        report["gates"][gate_name] = gate_report
        arrays[f"gawf_{gate_name}_equal_cell_mean"] = cell_mean
    return report, arrays


def _summarize_gate(
    aggregate: UnitGateAggregate, equal_n: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute condition-mean and trial-total variance decompositions."""

    cell_mean = aggregate.joint_sum.reshape(9, 10, -1) / equal_n
    condition = _marginal_variance(cell_mean)
    grand = aggregate.joint_sum.sum(axis=0) / (90 * equal_n)
    total_trial_ss = float(
        np.sum(aggregate.joint_sumsq - 90 * equal_n * np.square(grand))
    )
    factor_ss = {
        name: float(value * equal_n) for name, value in condition["components"].items()
    }
    residual = total_trial_ss - sum(factor_ss.values())
    if residual < 0.0 and abs(residual) < 1e-8 * total_trial_ss:
        residual = 0.0
    report = {
        "shape_per_trial": [int(cell_mean.shape[-1])],
        "equal_cell_condition_mean": {
            "fractions": condition["fractions"],
            "components": condition["components"],
            "total_condition_mean_variance": condition[
                "total_condition_mean_variance"
            ],
            "sum_check": condition["sum_check"],
        },
        "equal_cell_trial_total": {
            "percent": {
                **{
                    name: 100.0 * value / total_trial_ss
                    for name, value in factor_ss.items()
                },
                "residual": 100.0 * residual / total_trial_ss,
            },
            "total_trial_sum_squares": total_trial_ss,
            "residual_sum_squares": residual,
            "sum_check_percent": 100.0 * (sum(factor_ss.values()) + residual)
            / total_trial_ss,
        },
    }
    return report, cell_mean.astype(np.float32)


def analyze_model(
    model_type: str,
    checkpoint: str,
    dataset: torch.utils.data.Dataset,
    num_pos: int,
    reference_labels: np.ndarray,
    equal_joint_mask: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Extract one model's unit gates and stream exact balanced statistics."""

    model = build_model_from_ckpt(checkpoint, num_pos, device, chan_num=args.chan_num)
    if model.core.num_layers != 1:
        raise ValueError(f"{model_type} checkpoint must have one recurrent layer.")
    rnn = model.core.rnn
    expected_class = torch.nn.LSTM if model_type == "lstm" else torch.nn.GRU
    if not isinstance(rnn, expected_class):
        raise TypeError(f"Expected {expected_class.__name__}, got {type(rnn).__name__}.")
    aggregates = _new_aggregates(model_type, rnn.hidden_size)
    joint = reference_labels[:, 1] * 10 + reference_labels[:, 0]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    frame_offset = 0
    parity_max = 0.0
    extraction = lstm_unit_gates if model_type == "lstm" else gru_unit_gates
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, labels = batch[:2]
            frames = frames.to(device=device, dtype=torch.float32)
            labels_flat = labels.numpy().reshape(-1, 2).astype(np.int64, copy=False)
            end = frame_offset + labels_flat.shape[0]
            if not np.array_equal(labels_flat, reference_labels[frame_offset:end]):
                raise RuntimeError(
                    f"{model_type} dataset labels diverge from reference at batch {batch_index}."
                )
            encoded = model.encode_frames(frames)
            gates, manual_output = extraction(encoded, rnn)
            native_output, _state = rnn(encoded)
            parity_max = max(
                parity_max,
                float(torch.max(torch.abs(manual_output - native_output)).item()),
            )
            selected = equal_joint_mask[frame_offset:end]
            selected_codes = joint[frame_offset:end][selected]
            for gate_name, tensor in gates.items():
                values = tensor.detach().cpu().numpy().reshape(-1, rnn.hidden_size)
                selected_values = values[selected]
                aggregate = aggregates[gate_name]
                _add_by_code(aggregate.joint_sum, selected_values, selected_codes)
                aggregate.joint_sumsq += np.square(
                    selected_values, dtype=np.float64
                ).sum(axis=0)
            frame_offset = end
            if (batch_index + 1) % 25 == 0 or frame_offset == reference_labels.shape[0]:
                print(
                    f"{model_type.upper()} batches {batch_index + 1}/{len(loader)} | "
                    f"frames={frame_offset}/{reference_labels.shape[0]}",
                    flush=True,
                )
    if frame_offset != reference_labels.shape[0]:
        raise RuntimeError(
            f"Processed {frame_offset} frames, expected {reference_labels.shape[0]}."
        )
    if parity_max > 1e-4:
        raise RuntimeError(f"{model_type} recurrence parity failed: max_abs={parity_max:.3e}")

    equal_n = int(np.count_nonzero(equal_joint_mask)) // 90
    report: dict[str, Any] = {
        "checkpoint": os.path.abspath(checkpoint),
        "hidden_size": int(rnn.hidden_size),
        "gate_level": "unit",
        "candidate_activation_excluded": True,
        "native_recurrence_parity_max_abs": parity_max,
        "gates": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for gate_name, aggregate in aggregates.items():
        gate_report, cell_mean = _summarize_gate(aggregate, equal_n)
        report["gates"][gate_name] = gate_report
        arrays[f"{model_type}_{gate_name}_equal_cell_mean"] = cell_mean
    return report, arrays


def _write_csv(report: dict[str, Any], path: str) -> None:
    """Write plot-ready percentages to a compact CSV."""

    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            ["model", "gate", "factor", "condition_mean_percent", "trial_total_percent"]
        )
        for model_type, model_report in report["models"].items():
            for gate_name, gate_report in model_report["gates"].items():
                condition = gate_report["equal_cell_condition_mean"]["fractions"]
                trial = gate_report["equal_cell_trial_total"]["percent"]
                for factor in ("sector", "digit", "interaction", "residual"):
                    writer.writerow(
                        [
                            model_type,
                            gate_name,
                            factor,
                            "" if factor == "residual" else 100.0 * condition[factor],
                            trial[factor],
                        ]
                    )


def _load_existing_rnn_results(
    report_path: Path,
    arrays_path: Path,
    *,
    args: argparse.Namespace,
    n_frames: int,
    equal_n: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """Reuse aligned LSTM/GRU results, rejecting any provenance mismatch."""

    if not args.reuse_existing_rnn or not report_path.is_file() or not arrays_path.is_file():
        return None
    existing = json.loads(report_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "dataset": args.data_suffix,
        "n_frames": n_frames,
        "balance_seed": int(args.seed),
        "equal_joint_cell_n": equal_n,
    }
    mismatches = {
        key: (existing.get(key), expected)
        for key, expected in expected_metadata.items()
        if existing.get(key) != expected
    }
    for model_type, checkpoint in (("lstm", args.lstm_ckpt), ("gru", args.gru_ckpt)):
        observed = existing.get("models", {}).get(model_type, {}).get("checkpoint")
        expected = os.path.abspath(checkpoint)
        if observed != expected:
            mismatches[f"{model_type}_checkpoint"] = (observed, expected)
    if mismatches:
        formatted = ", ".join(
            f"{key}: observed={observed!r}, expected={expected!r}"
            for key, (observed, expected) in mismatches.items()
        )
        raise RuntimeError(f"Existing LSTM/GRU report is not reusable: {formatted}")
    models = {name: existing["models"][name] for name in ("lstm", "gru")}
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key.startswith(("lstm_", "gru_"))
        }
    print(f"Reusing aligned LSTM/GRU results from {report_path}", flush=True)
    return models, arrays


def main() -> None:
    """Run GaWF/LSTM/GRU analyses and save compact outputs."""

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable.")
    dataset, num_pos = build_test_dataset(args)
    with np.load(args.trajectory) as loaded:
        label_tensor = loaded["labels"].astype(np.int64, copy=False)
    if label_tensor.shape[:1] != (len(dataset),) or label_tensor.shape[-1] != 2:
        raise RuntimeError(
            f"Reference labels have shape {label_tensor.shape}; expected "
            f"({len(dataset)}, sequence_length, 2)."
        )
    reference_labels = label_tensor.reshape(-1, 2)
    digits, sectors = reference_labels[:, 0], reference_labels[:, 1]
    _marginal_masks, equal_joint_mask = _balanced_masks(digits, sectors, args.seed)
    equal_n = int(np.count_nonzero(equal_joint_mask)) // 90

    json_path = Path(args.save_dir) / "unit_gate_context_variance.json"
    npz_path = Path(args.save_dir) / "unit_gate_context_variance.npz"
    csv_path = Path(args.save_dir) / "unit_gate_context_variance.csv"
    reused = _load_existing_rnn_results(
        json_path,
        npz_path,
        args=args,
        n_frames=reference_labels.shape[0],
        equal_n=equal_n,
    )
    if reused is None:
        models: dict[str, Any] = {}
        arrays: dict[str, np.ndarray] = {}
    else:
        models, arrays = reused
    if args.include_gawf:
        gawf_report, gawf_arrays = analyze_gawf(
            args.gawf_input_manifest,
            Path(args.trajectory),
            reference_labels,
            equal_joint_mask,
            args.gawf_frame_batch_size,
            args.gawf_gate_tau,
            device,
        )
        models["gawf"] = gawf_report
        arrays.update(gawf_arrays)
    if reused is None:
        for model_type, checkpoint in (("lstm", args.lstm_ckpt), ("gru", args.gru_ckpt)):
            model_report, model_arrays = analyze_model(
                model_type,
                checkpoint,
                dataset,
                num_pos,
                reference_labels,
                equal_joint_mask,
                device,
                args,
            )
            models[model_type] = model_report
            arrays.update(model_arrays)

    report = {
        "analysis": "GaWF/LSTM/GRU context variance decomposition of unit-level gates",
        "dataset": args.data_suffix,
        "n_sequences": int(len(dataset)),
        "sequence_length": int(reference_labels.shape[0] // len(dataset)),
        "n_frames": int(reference_labels.shape[0]),
        "balance_seed": int(args.seed),
        "equal_joint_cell_n": equal_n,
        "labels": {"digit_levels": 10, "sector_levels": 9},
        "gate_convention": {
            "gawf": (
                "destination-unit arithmetic mean across raw sigmoid input or recurrent "
                "synapse gates"
            ),
            "lstm": "PyTorch i/f/g/o order; sigmoid i/f/o reported, candidate g excluded",
            "gru": "PyTorch r/z/n order; sigmoid reset/update reported, candidate n excluded",
        },
        "models": models,
    }
    json_partial = json_path.with_name(f"{json_path.stem}.partial{json_path.suffix}")
    npz_partial = npz_path.with_name(f"{npz_path.stem}.partial{npz_path.suffix}")
    csv_partial = csv_path.with_name(f"{csv_path.stem}.partial{csv_path.suffix}")
    with json_partial.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    np.savez_compressed(npz_partial, **arrays)
    _write_csv(report, str(csv_partial))
    json_partial.replace(json_path)
    npz_partial.replace(npz_path)
    csv_partial.replace(csv_path)
    print(f"Saved {json_path}")
    print(f"Saved {npz_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
