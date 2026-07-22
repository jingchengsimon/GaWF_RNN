"""Export sequential-feedback, equal-n sector input-gate spatial means.

Input is the compact trajectory from ``gawf_gate_distribution.py`` containing aligned pre-step
feedback, labels, U/V, and static recurrent weights. Exact input gates are reconstructed in
chunks, with an equal number of frames sampled per sector. Output is one compressed NPZ with
``point_included`` and ``point_excluded`` arrays of shape ``(9, 6, 6)`` and dtype float32, plus
JSON metadata describing the selection, gate protocol, and point-mass filtering.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir


NUM_SECTORS = 9
ENCODER_SHAPE = (32, 6, 6)


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default=str(
            output_dir("A_raw_gate", "gawf_gate_distribution", "data") / "gawf_gate_trajectory.npz"
        ),
    )
    parser.add_argument(
        "--save_dir",
        default=str(output_dir("B_gate_by_context", "sector_sigmoid_gate_sequential", "data")),
    )
    parser.add_argument("--gate_tau", type=float, default=0.5)
    parser.add_argument("--gate_chunk_size", type=int, default=16)
    parser.add_argument("--point_tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def equal_n_sector_mask(sectors: np.ndarray, seed: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Select the same minimum observed frame count independently from every sector."""

    sectors = np.asarray(sectors, dtype=np.int64).reshape(-1)
    if np.any((sectors < 0) | (sectors >= NUM_SECTORS)):
        raise ValueError("sector labels must lie in [0, 8]")
    original_counts = np.bincount(sectors, minlength=NUM_SECTORS).astype(np.int64)
    if np.any(original_counts == 0):
        missing = np.flatnonzero(original_counts == 0).tolist()
        raise RuntimeError(f"No frames found for sector(s): {missing}")
    target = int(original_counts.min())
    selected = np.zeros(sectors.size, dtype=bool)
    rng = np.random.default_rng(seed)
    for sector in range(NUM_SECTORS):
        indices = np.flatnonzero(sectors == sector)
        selected[rng.choice(indices, size=target, replace=False)] = True
    return selected, target, original_counts


def accumulate_equal_n_input_gates(
    chunks: Iterable[tuple[int, int, np.ndarray]],
    sectors: np.ndarray,
    selected: np.ndarray,
    gate_shape: tuple[int, int],
    point_tolerance: float,
    *,
    progress_every: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream included/excluded per-synapse sums and counts by sector."""

    if point_tolerance < 0:
        raise ValueError("point_tolerance must be nonnegative")
    sectors = np.asarray(sectors, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    if sectors.shape != selected.shape:
        raise ValueError("sectors and selected must have matching shapes")
    included_sum = np.zeros((NUM_SECTORS, *gate_shape), dtype=np.float64)
    excluded_sum = np.zeros_like(included_sum)
    included_count = np.zeros(NUM_SECTORS, dtype=np.int64)
    excluded_count = np.zeros((NUM_SECTORS, *gate_shape), dtype=np.int64)
    started = time.perf_counter()
    for chunk_index, (start, end, gate_input) in enumerate(chunks, start=1):
        if gate_input.shape != (end - start, *gate_shape):
            raise ValueError(
                f"gate chunk {start}:{end} has shape {gate_input.shape}; "
                f"expected {(end - start, *gate_shape)}"
            )
        chunk_selected = selected[start:end]
        chunk_sectors = sectors[start:end]
        for sector in np.unique(chunk_sectors[chunk_selected]):
            use = chunk_selected & (chunk_sectors == sector)
            values = np.asarray(gate_input[use], dtype=np.float32)
            included_sum[sector] += values.sum(axis=0, dtype=np.float64)
            included_count[sector] += values.shape[0]
            valid = np.abs(values - 0.5) >= point_tolerance
            excluded_sum[sector] += np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
            excluded_count[sector] += valid.sum(axis=0, dtype=np.int64)
        if progress_every > 0 and chunk_index % progress_every == 0:
            print(
                f"  processed {end}/{sectors.size} frames | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return included_sum, included_count, excluded_sum, excluded_count


def spatial_gate_means(
    included_sum: np.ndarray,
    included_count: np.ndarray,
    excluded_sum: np.ndarray,
    excluded_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-synapse accumulators to included/excluded 9x6x6 spatial maps."""

    hidden_size, input_size = included_sum.shape[1:]
    if input_size != int(np.prod(ENCODER_SHAPE)):
        raise ValueError(f"input gate width must be 1152, got {input_size}")
    if np.any(included_count <= 0):
        raise RuntimeError("Every sector requires at least one selected frame")
    if np.any(excluded_count <= 0):
        raise RuntimeError("At least one synapse has no non-0.5 observations")
    included = included_sum / included_count[:, None, None]
    excluded = excluded_sum / excluded_count
    included_maps = included.reshape(NUM_SECTORS, hidden_size, *ENCODER_SHAPE).mean(axis=(1, 2))
    excluded_maps = excluded.reshape(NUM_SECTORS, hidden_size, *ENCODER_SHAPE).mean(axis=(1, 2))
    return (
        included_maps.astype(np.float32, copy=False),
        excluded_maps.astype(np.float32, copy=False),
    )


def main() -> None:
    """Reconstruct sequential gates, balance sectors, and save both mean-map definitions."""

    args = parse_args()
    if args.gate_tau <= 0 or args.gate_chunk_size <= 0:
        raise ValueError("gate_tau and gate_chunk_size must be positive")
    os.makedirs(args.save_dir, exist_ok=True)
    trajectory_path = os.path.abspath(args.trajectory)
    with np.load(trajectory_path, allow_pickle=False) as loaded:
        feedback = loaded["feedback"].astype(np.float32, copy=False)
        labels = loaded["labels"].reshape(-1, 2).astype(np.int64, copy=False)
        u = loaded["U"].astype(np.float32, copy=False)
        v = loaded["V"].astype(np.float32, copy=False)
        gate_shape = tuple(int(value) for value in loaded["weight_ih"].shape)
    sectors = labels[:, 1]
    selected, target, original_counts = equal_n_sector_mask(sectors, args.seed)
    print(
        f"Sequential input gates: equal-n={target} per sector, "
        f"selected={int(selected.sum())}/{sectors.size}",
        flush=True,
    )

    from utils_anal.gawf_gate_distribution import iter_gate_chunks

    reconstructed = iter_gate_chunks(
        feedback,
        u,
        v,
        gate_shape[1],
        args.gate_tau,
        args.gate_chunk_size,
        device=args.device,
    )
    input_chunks = ((start, end, gate_input) for start, end, gate_input, _ in reconstructed)
    accumulators = accumulate_equal_n_input_gates(
        input_chunks,
        sectors,
        selected,
        gate_shape,
        args.point_tolerance,
    )
    included_sum, included_count, excluded_sum, excluded_count = accumulators
    if not np.all(included_count == target):
        raise RuntimeError(
            f"Equal-n accumulation mismatch: expected {target}, got {included_count.tolist()}"
        )
    included_maps, excluded_maps = spatial_gate_means(*accumulators)

    arrays_path = os.path.join(args.save_dir, "sector_gate_mean_sequential_equal_n.npz")
    metadata_path = os.path.join(args.save_dir, "sector_gate_mean_sequential_equal_n_meta.json")
    np.savez_compressed(
        arrays_path,
        point_included=included_maps.astype(np.float32, copy=False),
        point_excluded=excluded_maps.astype(np.float32, copy=False),
    )
    metadata = {
        "trajectory": trajectory_path,
        "protocol": "sequential pre-step feedback gates",
        "label_alignment": "gate applied at timestep t grouped by the sector label at timestep t",
        "selection": "equal-n random subsample independently within each sector",
        "seed": args.seed,
        "original_frames_by_sector": original_counts.astype(int).tolist(),
        "selected_frames_per_sector": target,
        "selected_frames_total": int(selected.sum()),
        "gate_tau": args.gate_tau,
        "point_tolerance": args.point_tolerance,
        "input_gate_shape_per_frame": list(gate_shape),
        "encoder_shape": list(ENCODER_SHAPE),
        "aggregation": "per-synapse frame mean, then mean over hidden units and channels",
        "point_included_shape": list(included_maps.shape),
        "point_excluded_shape": list(excluded_maps.shape),
        "point_excluded_valid_count_min": int(excluded_count.min()),
        "point_excluded_valid_count_max": int(excluded_count.max()),
        "point_excluded_valid_fraction": float(
            excluded_count.sum() / (target * excluded_count.size)
        ),
        "device": args.device,
    }
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved arrays: {arrays_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
