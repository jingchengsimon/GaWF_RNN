"""Aggregate exact GaWF gates by foreground digit without saving per-frame gate tensors.

The input is the compact trajectory produced by ``gawf_gate_distribution.py``. Gates are
reconstructed in chunks from feedback and U/V, grouped by ``fg_char_id``, and immediately
reduced to histograms and per-digit sparsity statistics. Outputs contain no dense gate matrices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.gawf_gate_distribution import (
    _group_mean_delta,
    _hist,
    _sparsity,
    iter_gate_chunks,
)


def parse_args() -> argparse.Namespace:
    """Parse digit-conditioned aggregation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="./results/anal_data/gawf_gate_audit/gawf_gate_trajectory.npz",
    )
    parser.add_argument(
        "--save_dir",
        default="./results/anal_data/gawf_gate_audit_digit",
    )
    parser.add_argument("--gate_chunk_size", type=int, default=128)
    parser.add_argument("--hist_bins", type=int, default=400)
    parser.add_argument("--gate_tau", type=float, default=0.5)
    return parser.parse_args()


def aggregate_by_digit(
    trajectory: dict[str, np.ndarray],
    gate_tau: float,
    chunk_size: int,
    hist_bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Stream all gates into ten digit-conditioned histograms and average-gate summaries."""

    feedback = trajectory["feedback"].astype(np.float32, copy=False)
    labels = trajectory["labels"].astype(np.int64, copy=False)
    u = trajectory["U"].astype(np.float32, copy=False)
    v = trajectory["V"].astype(np.float32, copy=False)
    weight_ih = trajectory["weight_ih"].astype(np.float32, copy=False)
    weight_hh = trajectory["weight_hh"].astype(np.float32, copy=False)
    hidden_size, input_size = weight_ih.shape
    if weight_hh.shape != (hidden_size, hidden_size):
        raise RuntimeError(f"Unexpected recurrent weight shape: {weight_hh.shape}")

    digits = labels[..., 0].reshape(-1)
    if np.any((digits < 0) | (digits > 9)):
        raise RuntimeError("Foreground digit labels must be in [0, 9]")
    edges = np.linspace(0.0, 1.0, hist_bins + 1, dtype=np.float64)
    hist_input = np.zeros((10, hist_bins), dtype=np.int64)
    hist_recurrent = np.zeros((10, hist_bins), dtype=np.int64)
    digit_counts = np.zeros(10, dtype=np.int64)
    sum_input = np.zeros((10, hidden_size, input_size), dtype=np.float64)
    sum_recurrent = np.zeros((10, hidden_size, hidden_size), dtype=np.float64)

    total_chunks = int(np.ceil(digits.size / chunk_size))
    start_time = time.perf_counter()
    for chunk_idx, (start, end, gate_ih, gate_hh) in enumerate(
        iter_gate_chunks(feedback, u, v, input_size, gate_tau, chunk_size)
    ):
        chunk_digits = digits[start:end]
        for digit in np.unique(chunk_digits):
            digit_int = int(digit)
            mask = chunk_digits == digit_int
            selected_input = gate_ih[mask]
            selected_recurrent = gate_hh[mask]
            count = int(np.count_nonzero(mask))
            digit_counts[digit_int] += count
            sum_input[digit_int] += selected_input.sum(axis=0, dtype=np.float64)
            sum_recurrent[digit_int] += selected_recurrent.sum(axis=0, dtype=np.float64)
            hist_input[digit_int] += _hist(selected_input, edges)
            hist_recurrent[digit_int] += _hist(selected_recurrent, edges)

        if (chunk_idx + 1) % 25 == 0 or chunk_idx + 1 == total_chunks:
            elapsed = time.perf_counter() - start_time
            print(
                f"digit aggregation chunks {chunk_idx + 1}/{total_chunks} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if np.any(digit_counts == 0):
        raise RuntimeError(f"At least one digit has no frames: {digit_counts.tolist()}")
    mean_input, _digit_center_input, delta_input = _group_mean_delta(
        sum_input, digit_counts
    )
    mean_recurrent, _digit_center_recurrent, delta_recurrent = _group_mean_delta(
        sum_recurrent, digit_counts
    )
    sparsity = {
        "input": [_sparsity(mean_input[digit]) for digit in range(10)],
        "recurrent": [_sparsity(mean_recurrent[digit]) for digit in range(10)],
    }

    delta_edges = np.linspace(-1.0, 1.0, hist_bins * 2 + 1, dtype=np.float64)
    hist_delta_input = _hist(delta_input, delta_edges)
    hist_delta_recurrent = _hist(delta_recurrent, delta_edges)
    delta_moments = {
        "input": {"count": 0, "sum": 0.0, "sum2": 0.0},
        "recurrent": {"count": 0, "sum": 0.0, "sum2": 0.0},
    }
    for kind, delta in (("input", delta_input), ("recurrent", delta_recurrent)):
        delta_moments[kind]["count"] = int(delta.size)
        delta_moments[kind]["sum"] = float(delta.sum(dtype=np.float64))
        delta_moments[kind]["sum2"] = float(np.square(delta).sum(dtype=np.float64))

    centered_summary: dict[str, dict[str, float | int]] = {}
    for kind, values in delta_moments.items():
        count = int(values["count"])
        mean = float(values["sum"]) / count
        variance = max(0.0, float(values["sum2"]) / count - mean * mean)
        centered_summary[kind] = {"count": count, "mean": mean, "std": variance**0.5}

    arrays = {
        "gate_edges": edges.astype(np.float32),
        "delta_edges": delta_edges.astype(np.float32),
        "hist_input_digit": hist_input,
        "hist_recurrent_digit": hist_recurrent,
        "hist_input_delta": hist_delta_input,
        "hist_recurrent_delta": hist_delta_recurrent,
        "digit_counts": digit_counts,
    }
    metadata = {
        "conditioning": "foreground digit identity (fg_char_id)",
        "digits": list(range(10)),
        "digit_counts": [int(value) for value in digit_counts],
        "digit_gate_means": {
            "input": [float(mean_input[digit].mean()) for digit in range(10)],
            "recurrent": [float(mean_recurrent[digit].mean()) for digit in range(10)],
        },
        "digit_centered": centered_summary,
        "sparsity": sparsity,
        "gate_tau": gate_tau,
        "n_frames": int(digits.size),
        "input_gate_shape_per_frame": list(weight_ih.shape),
        "recurrent_gate_shape_per_frame": list(weight_hh.shape),
        "histogram_bins": hist_bins,
        "storage_note": (
            "Only digit-conditioned and digit-centered histogram counts plus scalar summaries "
            "are saved; "
            "no per-frame or per-digit dense gate matrices are written."
        ),
    }
    return arrays, metadata


def main() -> None:
    """Load the compact trajectory, aggregate by digit, and save compact statistics."""

    args = parse_args()
    if args.gate_chunk_size <= 0 or args.hist_bins <= 0:
        raise ValueError("gate_chunk_size and hist_bins must be positive")
    trajectory_path = os.path.abspath(args.trajectory)
    if not os.path.isfile(trajectory_path):
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")
    with np.load(trajectory_path) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}

    arrays, metadata = aggregate_by_digit(
        trajectory,
        gate_tau=args.gate_tau,
        chunk_size=args.gate_chunk_size,
        hist_bins=args.hist_bins,
    )
    metadata["trajectory"] = trajectory_path
    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = os.path.join(args.save_dir, "gawf_gate_digit_stats.npz")
    metadata_path = os.path.join(args.save_dir, "gawf_gate_digit_meta.json")
    np.savez_compressed(stats_path, **arrays)
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved digit statistics: {stats_path}")
    print(f"Saved digit metadata: {metadata_path}")


if __name__ == "__main__":
    main()
