"""Decompose LSTM and GRU unit-level gate variance into sector, digit, and interaction.

GaWF modulates a gate per *synapse*, so its gate array is (hidden x input). LSTM and GRU
instead carry one gate value per *unit*, which makes the two architectures directly
comparable only after the unit-level gates are put through the same balanced
sector-by-digit decomposition used for GaWF.

The gates are recomputed from the exact PyTorch recurrence rather than read from a hook,
so ``lstm_unit_gates`` and ``gru_unit_gates`` return both the per-gate tensors and the
hidden sequence they imply; the hidden sequence must reproduce ``rnn(encoded)`` to
floating-point tolerance, which is what pins the gate definitions to PyTorch's own
gate ordering and bias placement.

Output: ``unit_gate_context_variance.json``, read by
``utils_viz/rnn_unit_gate_context_specificity.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_anal.anal_helpers import (
    build_eval_dataset,
    build_model_from_ckpt,
    resolve_device,
)

N_SECTORS = 9
N_DIGITS = 10
LSTM_GATE_NAMES = ("input", "forget", "output")
GRU_GATE_NAMES = ("reset", "update")


@dataclass
class UnitGateAggregate:
    """Balanced per-cell sums for one gate.

    ``joint_sum`` is (n_cells, n_units), summed over the equal-n trials inside each
    sector-by-digit cell. ``joint_sumsq`` is (n_units,), the total sum of squares over
    every retained trial. Keeping only these two moments means the decomposition never
    needs the per-trial gate array in memory.
    """

    joint_sum: np.ndarray
    joint_sumsq: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse unit-gate context-specificity arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm_ckpt", required=True)
    parser.add_argument("--gru_ckpt", required=True)
    parser.add_argument("--data_dir", default="./stimuli")
    parser.add_argument("--data_suffix", default="40h-uint8")
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
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=2000)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260719)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _split_bias(rnn: torch.nn.RNNBase, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input and hidden bias vectors, or zeros when the module is bias-free."""

    if rnn.bias:
        return rnn.bias_ih_l0, rnn.bias_hh_l0
    zero = torch.zeros(
        rnn.weight_ih_l0.shape[0], dtype=encoded.dtype, device=encoded.device
    )
    return zero, zero


def lstm_unit_gates(
    encoded: torch.Tensor, rnn: torch.nn.LSTM
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Recompute LSTM input/forget/output gates and the hidden sequence they produce.

    PyTorch packs the LSTM gates as ``[i, f, g, o]`` in a single (4H, *) weight, and adds
    ``bias_ih`` and ``bias_hh`` independently. ``g`` is the candidate cell, not a gate, so
    it is used for the recurrence but not returned as a gate.
    """

    if rnn.num_layers != 1 or rnn.bidirectional:
        raise ValueError("lstm_unit_gates supports a single-layer unidirectional LSTM")
    if not rnn.batch_first:
        raise ValueError("lstm_unit_gates expects batch_first=True")
    batch_size, frame_num, _input_size = encoded.shape
    hidden_size = rnn.hidden_size
    bias_ih, bias_hh = _split_bias(rnn, encoded)
    hidden = torch.zeros(
        batch_size, hidden_size, dtype=encoded.dtype, device=encoded.device
    )
    cell = torch.zeros_like(hidden)
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in LSTM_GATE_NAMES}
    hidden_steps = []
    for time_idx in range(frame_num):
        pre = (
            encoded[:, time_idx] @ rnn.weight_ih_l0.T
            + bias_ih
            + hidden @ rnn.weight_hh_l0.T
            + bias_hh
        )
        input_gate = torch.sigmoid(pre[:, 0:hidden_size])
        forget_gate = torch.sigmoid(pre[:, hidden_size : 2 * hidden_size])
        candidate = torch.tanh(pre[:, 2 * hidden_size : 3 * hidden_size])
        output_gate = torch.sigmoid(pre[:, 3 * hidden_size : 4 * hidden_size])
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        collected["input"].append(input_gate)
        collected["forget"].append(forget_gate)
        collected["output"].append(output_gate)
        hidden_steps.append(hidden)
    gates = {name: torch.stack(values, dim=1) for name, values in collected.items()}
    return gates, torch.stack(hidden_steps, dim=1)


def gru_unit_gates(
    encoded: torch.Tensor, rnn: torch.nn.GRU
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Recompute GRU reset/update gates and the hidden sequence they produce.

    PyTorch packs the GRU gates as ``[r, z, n]``. The reset gate multiplies the *whole*
    hidden contribution to the candidate, including ``bias_hn`` -- applying it before the
    bias is the usual off-by-one-term error and would break the parity check.
    """

    if rnn.num_layers != 1 or rnn.bidirectional:
        raise ValueError("gru_unit_gates supports a single-layer unidirectional GRU")
    if not rnn.batch_first:
        raise ValueError("gru_unit_gates expects batch_first=True")
    batch_size, frame_num, _input_size = encoded.shape
    hidden_size = rnn.hidden_size
    bias_ih, bias_hh = _split_bias(rnn, encoded)
    hidden = torch.zeros(
        batch_size, hidden_size, dtype=encoded.dtype, device=encoded.device
    )
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in GRU_GATE_NAMES}
    hidden_steps = []
    for time_idx in range(frame_num):
        gate_input = encoded[:, time_idx] @ rnn.weight_ih_l0.T + bias_ih
        gate_hidden = hidden @ rnn.weight_hh_l0.T + bias_hh
        reset_gate = torch.sigmoid(
            gate_input[:, 0:hidden_size] + gate_hidden[:, 0:hidden_size]
        )
        update_gate = torch.sigmoid(
            gate_input[:, hidden_size : 2 * hidden_size]
            + gate_hidden[:, hidden_size : 2 * hidden_size]
        )
        candidate = torch.tanh(
            gate_input[:, 2 * hidden_size : 3 * hidden_size]
            + reset_gate * gate_hidden[:, 2 * hidden_size : 3 * hidden_size]
        )
        hidden = (1.0 - update_gate) * candidate + update_gate * hidden
        collected["reset"].append(reset_gate)
        collected["update"].append(update_gate)
        hidden_steps.append(hidden)
    gates = {name: torch.stack(values, dim=1) for name, values in collected.items()}
    return gates, torch.stack(hidden_steps, dim=1)


def _summarize_gate(
    aggregate: UnitGateAggregate, equal_n: int
) -> tuple[dict[str, object], np.ndarray]:
    """Decompose one gate into sector, digit, and interaction variance.

    Two views are reported because they answer different questions:

    * ``equal_cell_condition_mean`` decomposes the variance *of the cell means*, which is
      the quantity comparable to a hidden-state marginalization. Its fractions sum to 1.
    * ``equal_cell_trial_total`` decomposes total trial-level variance and therefore also
      carries a residual (within-cell) term. Its percentages sum to 100.

    The design is balanced by construction (``equal_n`` trials in every one of the 90
    cells), so the three effects are orthogonal and the sums of squares add exactly.
    Sums of squares are pooled across units before forming ratios, so a unit with a
    larger dynamic range contributes proportionally rather than being weighted equally.
    """

    if equal_n <= 0:
        raise ValueError("equal_n must be positive")
    n_cells = aggregate.joint_sum.shape[0]
    if n_cells != N_SECTORS * N_DIGITS:
        raise ValueError(
            f"Expected {N_SECTORS * N_DIGITS} joint cells, got {n_cells}"
        )
    n_units = aggregate.joint_sum.shape[1]
    cell_mean = (aggregate.joint_sum / equal_n).reshape(N_SECTORS, N_DIGITS, n_units)

    grand = cell_mean.mean(axis=(0, 1))
    sector_mean = cell_mean.mean(axis=1)
    digit_mean = cell_mean.mean(axis=0)
    sector_effect = sector_mean - grand
    digit_effect = digit_mean - grand
    interaction_effect = (
        cell_mean - sector_mean[:, None, :] - digit_mean[None, :, :] + grand
    )

    condition_ss = {
        "sector": float(N_DIGITS * np.square(sector_effect).sum()),
        "digit": float(N_SECTORS * np.square(digit_effect).sum()),
        "interaction": float(np.square(interaction_effect).sum()),
    }
    condition_total = sum(condition_ss.values())
    if condition_total <= 0.0:
        raise RuntimeError("Condition-mean variance is zero; gate is constant")
    fractions = {key: value / condition_total for key, value in condition_ss.items()}

    n_trials = n_cells * equal_n
    trial_total_ss = float(
        aggregate.joint_sumsq.sum() - n_trials * float(np.square(grand).sum())
    )
    if trial_total_ss <= 0.0:
        raise RuntimeError("Trial-total variance is non-positive; check the aggregate")
    trial_ss = {key: equal_n * value for key, value in condition_ss.items()}
    trial_ss["residual"] = trial_total_ss - sum(trial_ss.values())
    percent = {key: 100.0 * value / trial_total_ss for key, value in trial_ss.items()}

    report = {
        "n_units": int(n_units),
        "equal_n_per_cell": int(equal_n),
        "n_trials": int(n_trials),
        "equal_cell_condition_mean": {
            "sum_of_squares": condition_ss,
            "total_sum_of_squares": condition_total,
            "fractions": fractions,
        },
        "equal_cell_trial_total": {
            "sum_of_squares": trial_ss,
            "total_sum_of_squares": trial_total_ss,
            "percent": percent,
        },
    }
    return report, cell_mean


def _balanced_cell_indices(
    sectors: np.ndarray, digits: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Draw the same number of frames from every sector-by-digit cell.

    Returns the selected flat indices ordered cell-major, so a later reshape to
    (n_cells, equal_n) lines up with ``UnitGateAggregate``.
    """

    per_cell: list[np.ndarray] = []
    for sector in range(N_SECTORS):
        for digit in range(N_DIGITS):
            members = np.flatnonzero((sectors == sector) & (digits == digit))
            if members.size == 0:
                raise RuntimeError(
                    f"Sector {sector} x digit {digit} cell is empty; cannot balance"
                )
            per_cell.append(members)
    equal_n = min(int(members.size) for members in per_cell)
    selected = np.stack(
        [rng.choice(members, size=equal_n, replace=False) for members in per_cell]
    )
    return selected, equal_n


def collect_unit_gates(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    model_type: str,
    batch_size: int,
    num_workers: int,
    max_sequences: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Run the encoder and recurrence, returning per-frame gates and their labels."""

    gate_fn = lstm_unit_gates if model_type == "lstm" else gru_unit_gates
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    collected: dict[str, list[np.ndarray]] = {}
    label_chunks: list[np.ndarray] = []
    seen = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            gates, _hidden = gate_fn(encoded, model.rnn)
            for name, value in gates.items():
                collected.setdefault(name, []).append(
                    value.reshape(-1, value.shape[-1]).cpu().numpy().astype(np.float32)
                )
            label_chunks.append(
                labels.reshape(-1, labels.shape[-1]).numpy().astype(np.int64)
            )
            seen += frames.shape[0]
            print(
                f"  {model_type}: {min(seen, max_sequences)}/{max_sequences} sequences | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if seen >= max_sequences:
                break
    gate_arrays = {
        name: np.concatenate(chunks, axis=0) for name, chunks in collected.items()
    }
    labels_array = np.concatenate(label_chunks, axis=0)
    return gate_arrays, labels_array[:, 1], labels_array[:, 0]


def analyze_model(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    model_type: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Collect gates for one recurrent model and decompose each of its gates."""

    gate_arrays, sectors, digits = collect_unit_gates(
        model,
        dataset,
        device,
        model_type,
        args.batch_size,
        args.num_workers,
        args.max_sequences,
    )
    rng = np.random.default_rng(args.seed)
    selected, equal_n = _balanced_cell_indices(sectors, digits, rng)
    gates_report: dict[str, object] = {}
    for name, values in gate_arrays.items():
        cells = values[selected]
        aggregate = UnitGateAggregate(
            joint_sum=cells.sum(axis=1).astype(np.float64),
            joint_sumsq=np.square(cells.astype(np.float64)).sum(axis=(0, 1)),
        )
        gate_report, _cell_mean = _summarize_gate(aggregate, equal_n)
        gates_report[name] = gate_report
    return {
        "n_frames_collected": int(sectors.size),
        "equal_n_per_cell": int(equal_n),
        "gates": gates_report,
    }


def main() -> None:
    """Decompose LSTM and GRU unit gates and save one compact JSON report."""

    args = parse_args()
    device = resolve_device(args.device)
    dataset, num_pos = build_eval_dataset(args, "test")
    report: dict[str, object] = {"models": {}}
    for model_type, ckpt in (("lstm", args.lstm_ckpt), ("gru", args.gru_ckpt)):
        print(f"Analyzing {model_type} from {ckpt}", flush=True)
        model = build_model_from_ckpt(
            ckpt, num_pos, device, chan_num=int(args.chan_num)
        )
        report["models"][model_type] = analyze_model(
            model, dataset, device, model_type, args
        )
        report["models"][model_type]["checkpoint"] = os.path.abspath(ckpt)
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, "unit_gate_context_variance.json")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    print(f"Saved unit-gate decomposition: {output_path}")


if __name__ == "__main__":
    main()
