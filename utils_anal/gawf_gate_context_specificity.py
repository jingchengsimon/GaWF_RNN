"""Analyze GaWF context specificity, beginning with mandatory design prerequisites.

The Part 0 mode reads the compact evaluation trajectory, reports the joint sector-by-digit
design and balance, and reconstructs all gates in chunks to measure the exact 0.5 point mass.
It writes only compact JSON statistics and never saves per-frame dense gate tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import chi2_contingency

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.gawf_gate_distribution import iter_gate_chunks


def parse_args() -> argparse.Namespace:
    """Parse context-specificity analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="./results/anal_data/gawf_gate_audit/gawf_gate_trajectory.npz",
    )
    parser.add_argument(
        "--save_dir",
        default="./results/anal_data/gawf_gate_context_specificity",
    )
    parser.add_argument("--gate_chunk_size", type=int, default=128)
    parser.add_argument("--gate_tau", type=float, default=0.5)
    parser.add_argument("--point_tolerance", type=float, default=1e-6)
    parser.add_argument("--part0-only", action="store_true")
    return parser.parse_args()


def joint_design(labels: np.ndarray) -> dict[str, object]:
    """Compute the 9x10 joint table, marginal counts, chi-square test, and Cramer's V."""

    flat_labels = labels.reshape(-1, labels.shape[-1]).astype(np.int64, copy=False)
    digits = flat_labels[:, 0]
    sectors = flat_labels[:, 1]
    if np.any((digits < 0) | (digits > 9)) or np.any((sectors < 0) | (sectors > 8)):
        raise RuntimeError("Expected digit labels in [0,9] and sector labels in [0,8]")
    table = np.zeros((9, 10), dtype=np.int64)
    np.add.at(table, (sectors, digits), 1)
    chi2, p_value, degrees_freedom, expected = chi2_contingency(table, correction=False)
    n_trials = int(table.sum())
    cramers_v = float(np.sqrt(chi2 / (n_trials * min(table.shape[0] - 1, 9))))
    sector_counts = table.sum(axis=1)
    digit_counts = table.sum(axis=0)
    return {
        "n_trials": n_trials,
        "joint_frequency_sector_rows_digit_columns": table.tolist(),
        "sector_counts": sector_counts.tolist(),
        "digit_counts": digit_counts.tolist(),
        "chi_square": float(chi2),
        "chi_square_p_value": float(p_value),
        "chi_square_degrees_freedom": int(degrees_freedom),
        "cramers_v": cramers_v,
        "minimum_expected_count": float(expected.min()),
        "independent_at_alpha_0_05": bool(p_value >= 0.05),
        "balanced": {
            "sector": bool(np.all(sector_counts == sector_counts[0])),
            "digit": bool(np.all(digit_counts == digit_counts[0])),
            "joint_cells": bool(np.all(table == table.flat[0])),
        },
        "equal_n_subsampling": {
            "per_sector": int(sector_counts.min()),
            "per_digit": int(digit_counts.min()),
            "per_sector_digit_cell": int(table.min()),
        },
    }


def point_mass_at_half(
    trajectory: dict[str, np.ndarray],
    gate_tau: float,
    chunk_size: int,
    tolerance: float,
) -> dict[str, object]:
    """Reconstruct every gate and count entries satisfying ``abs(g - 0.5) < tolerance``."""

    feedback = trajectory["feedback"].astype(np.float32, copy=False)
    u = trajectory["U"].astype(np.float32, copy=False)
    v = trajectory["V"].astype(np.float32, copy=False)
    input_size = int(trajectory["weight_ih"].shape[1])
    counts = {"input": 0, "recurrent": 0}
    totals = {"input": 0, "recurrent": 0}
    n_frames = int(np.prod(feedback.shape[:-1]))
    total_chunks = int(np.ceil(n_frames / chunk_size))
    start_time = time.perf_counter()

    for chunk_idx, (_start, _end, gate_ih, gate_hh) in enumerate(
        iter_gate_chunks(feedback, u, v, input_size, gate_tau, chunk_size)
    ):
        for kind, gate in (("input", gate_ih), ("recurrent", gate_hh)):
            counts[kind] += int(np.count_nonzero(np.abs(gate - 0.5) < tolerance))
            totals[kind] += int(gate.size)
        if (chunk_idx + 1) % 25 == 0 or chunk_idx + 1 == total_chunks:
            elapsed = time.perf_counter() - start_time
            print(
                f"Part 0 point-mass chunks {chunk_idx + 1}/{total_chunks} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    n_sequences, frames_per_sequence = feedback.shape[:2]
    initialization_fraction = 1.0 / frames_per_sequence
    return {
        "tolerance": tolerance,
        "input": {
            "count": counts["input"],
            "total": totals["input"],
            "fraction": counts["input"] / totals["input"],
        },
        "recurrent": {
            "count": counts["recurrent"],
            "total": totals["recurrent"],
            "fraction": counts["recurrent"] / totals["recurrent"],
        },
        "sequence_initialization": {
            "n_sequences": int(n_sequences),
            "frames_per_sequence": int(frames_per_sequence),
            "guaranteed_zero_feedback_frame_fraction": initialization_fraction,
            "interpretation": (
                "Every reset sequence begins with zero feedback, so its first-frame gates are "
                "exactly 0.5; this contribution is initialization, not evidence of dead units."
            ),
        },
        "repeat_downstream_with_point_mass_excluded": bool(
            counts["input"] / totals["input"] > 0.01
            or counts["recurrent"] / totals["recurrent"] > 0.01
        ),
    }


def main() -> None:
    """Run mandatory Part 0 and save its compact report."""

    args = parse_args()
    if args.gate_chunk_size <= 0 or args.point_tolerance <= 0:
        raise ValueError("gate_chunk_size and point_tolerance must be positive")
    trajectory_path = os.path.abspath(args.trajectory)
    with np.load(trajectory_path) as loaded:
        trajectory = {key: loaded[key] for key in loaded.files}
    labels = trajectory["labels"]
    shapes = {
        "input_gate_per_trial": list(trajectory["weight_ih"].shape),
        "recurrent_gate_per_trial": list(trajectory["weight_hh"].shape),
        "feedback": list(trajectory["feedback"].shape),
        "labels": list(labels.shape),
    }
    report = {
        "trajectory": trajectory_path,
        "shapes": shapes,
        "joint_design": joint_design(labels),
        "point_mass_at_0_5": point_mass_at_half(
            trajectory,
            gate_tau=args.gate_tau,
            chunk_size=args.gate_chunk_size,
            tolerance=args.point_tolerance,
        ),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, "part0_prerequisites.json")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    print(f"Saved Part 0 report: {output_path}")
    if not args.part0_only:
        raise RuntimeError("Only --part0-only is implemented; complete Part 0 before later parts")


if __name__ == "__main__":
    main()
