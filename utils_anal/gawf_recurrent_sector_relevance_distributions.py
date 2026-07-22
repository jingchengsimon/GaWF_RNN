"""Export recurrent SOURCE-gate distributions for sector-relevant hidden units.

Inputs are the exact GaWF checkpoint/test split used by the symmetric Part-2 analysis plus its
saved Part-1 selectivity arrays and Part-2 report. For every sector, this analysis reproduces the
interaction-excluded top-10% relevance mask, runs the same reset sequence trajectories, averages
each raw post-sigmoid recurrent gate matrix over its 256 destination rows, and accumulates
distributions for top-10% versus remaining eligible SOURCE units. Outputs are one compressed NPZ
plus JSON metadata.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir


NUM_SECTORS = 9
GROUP_NAMES = ("top10", "remaining")


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
        "--selectivity",
        default=str(
            output_dir("D_variance_decomposition", "gawf_symmetric_relevance_timing", "data")
            / "part1_selectivity.npz"
        ),
    )
    parser.add_argument(
        "--part2_report",
        default=str(
            output_dir("E_relevance_alignment", "gawf_symmetric_relevance_timing", "data")
            / "part2_results.json"
        ),
    )
    parser.add_argument(
        "--split_report",
        default=str(
            output_dir("H_controls", "gawf_symmetric_relevance_timing", "data")
            / "part0_splits.json"
        ),
    )
    parser.add_argument(
        "--save_dir",
        default=str(
            output_dir(
                "E_relevance_alignment",
                "gawf_recurrent_sector_relevance_distributions",
                "data",
            )
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("AIM3_NUM_WORKERS", "0"))
    )
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hist_bins", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def collect_test_recurrent_source_gates(
    dataset: object,
    model: object,
    device: object,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact Part-2 test trajectories and return recurrent SOURCE gates and labels."""

    import torch
    from torch.utils.data import DataLoader

    from utils_anal.gawf_symmetric_relevance_timing import _trajectory_with_measurements

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    gate_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            trajectory = _trajectory_with_measurements(
                encoded, model, record_gates=True, record_input_gate=False
            )
            gate_batches.append(
                trajectory["recurrent_gate"].cpu().numpy().astype(np.float32, copy=False)
            )
            label_batches.append(labels.numpy().astype(np.int64, copy=False))
            samples_done = min((batch_index + 1) * batch_size, len(dataset))
            if samples_done % 200 < batch_size or batch_index + 1 == len(loader):
                print(
                    f"  collected {samples_done}/{len(dataset)} test sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    gates = np.concatenate(gate_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    return gates.reshape(-1, gates.shape[-1]), labels.reshape(-1, labels.shape[-1])


def accumulate_context_group_distributions(
    chunks: Iterable[tuple[int, int, np.ndarray]],
    contexts: np.ndarray,
    relevant_masks: np.ndarray,
    eligible: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate histogram counts and first two moments for both groups by context."""

    contexts = np.asarray(contexts, dtype=np.int64).reshape(-1)
    relevant_masks = np.asarray(relevant_masks, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if relevant_masks.ndim != 2 or relevant_masks.shape[1] != eligible.size:
        raise ValueError(
            f"relevant_masks must have shape (contexts, {eligible.size}), "
            f"got {relevant_masks.shape}"
        )
    num_contexts = relevant_masks.shape[0]
    if contexts.size == 0 or contexts.min() < 0 or contexts.max() >= num_contexts:
        raise ValueError("contexts must be non-empty indices covered by relevant_masks")
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional sequence")
    hist = np.zeros((num_contexts, len(GROUP_NAMES), bin_edges.size - 1), dtype=np.int64)
    sums = np.zeros((num_contexts, len(GROUP_NAMES)), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    counts = np.zeros((num_contexts, len(GROUP_NAMES)), dtype=np.int64)
    for start, end, source_mean in chunks:
        expected_shape = (end - start, eligible.size)
        if source_mean.shape != expected_shape:
            raise ValueError(
                f"Chunk {start}:{end} has shape {source_mean.shape}, {expected_shape=}"
            )
        chunk_contexts = contexts[start:end]
        for context in np.unique(chunk_contexts):
            frame_values = np.asarray(source_mean[chunk_contexts == context], dtype=np.float32)
            masks = (relevant_masks[context], eligible & ~relevant_masks[context])
            for group_index, group_mask in enumerate(masks):
                values = frame_values[:, group_mask].reshape(-1)
                hist[context, group_index] += np.histogram(values, bins=bin_edges)[0]
                sums[context, group_index] += values.sum(dtype=np.float64)
                sums_sq[context, group_index] += np.square(values, dtype=np.float64).sum(
                    dtype=np.float64
                )
                counts[context, group_index] += values.size
    if np.any(counts == 0):
        raise RuntimeError("Every context and relevance group must contain gate observations")
    return hist, sums, sums_sq, counts


def accumulate_sector_group_distributions(
    chunks: Iterable[tuple[int, int, np.ndarray]],
    sectors: np.ndarray,
    relevant_masks: np.ndarray,
    eligible: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible nine-sector wrapper for the generic accumulator."""

    if np.asarray(relevant_masks).shape[0] != NUM_SECTORS:
        raise ValueError(f"Sector relevance masks must contain {NUM_SECTORS} rows")
    return accumulate_context_group_distributions(
        chunks, sectors, relevant_masks, eligible, bin_edges
    )


def summarize_group_moments(
    sums: np.ndarray,
    sums_sq: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return group means, sample SDs, per-context d, and pooled global d."""

    sums = np.asarray(sums, dtype=np.float64)
    sums_sq = np.asarray(sums_sq, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts <= 1):
        raise RuntimeError("Every context and group requires at least two gate observations")
    means = sums / counts
    variances = (sums_sq - np.square(sums) / counts) / (counts - 1)
    np.maximum(variances, 0.0, out=variances)
    stds = np.sqrt(variances)

    def cohens_d(local_sums: np.ndarray, local_sums_sq: np.ndarray, local_n: np.ndarray) -> float:
        local_means = local_sums / local_n
        local_vars = (local_sums_sq - np.square(local_sums) / local_n) / (local_n - 1)
        pooled = np.sqrt(
            ((local_n[0] - 1) * local_vars[0] + (local_n[1] - 1) * local_vars[1])
            / (local_n.sum() - 2)
        )
        return float((local_means[0] - local_means[1]) / pooled)

    context_d = np.asarray(
        [cohens_d(sums[index], sums_sq[index], counts[index]) for index in range(sums.shape[0])],
        dtype=np.float64,
    )
    global_d = cohens_d(sums.sum(axis=0), sums_sq.sum(axis=0), counts.sum(axis=0))
    return means, stds, context_d, global_d


def main() -> None:
    """Collect exact test gates and export sector-specific relevance distributions."""

    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.hist_bins <= 1:
        raise ValueError(
            "batch_size and hist_bins must be positive; num_workers cannot be negative"
        )
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    selectivity_path = Path(args.selectivity).expanduser().resolve()
    report_path = Path(args.part2_report).expanduser().resolve()
    split_report_path = Path(args.split_report).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    from utils_anal.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device

    args.use_sector_mode = True
    args.predict_all_chars = False
    device = resolve_device(args.device, require_cuda_if_requested=True)
    test_dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(checkpoint_path), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis requires a single-layer GaWF checkpoint")
    hidden_size = int(model.rnn.hidden_size)
    with np.load(selectivity_path, allow_pickle=False) as selectivity:
        tuning = np.asarray(selectivity["primary_hidden_tuning_sector"], dtype=np.float64)
        passed = np.asarray(selectivity["primary_hidden_passed_sector"], dtype=bool)
        dominant = np.asarray(selectivity["primary_hidden_interaction_dominant"], dtype=bool)
    eligible = passed & ~dominant
    from utils_anal.gawf_symmetric_stats import relevance_masks

    relevant_masks = relevance_masks(tuning, eligible, 0.10)
    if relevant_masks.shape != (NUM_SECTORS, hidden_size):
        raise RuntimeError("Relevance masks do not align with recurrent SOURCE units")

    gates, labels = collect_test_recurrent_source_gates(
        test_dataset,
        model,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    frame_sector = labels[:, 1]
    frame_counts = np.bincount(frame_sector, minlength=NUM_SECTORS).astype(np.int64)
    with split_report_path.open(encoding="utf-8") as file_obj:
        expected_frame_counts = np.asarray(
            json.load(file_obj)["test"]["sector_counts"], dtype=np.int64
        )
    if not np.array_equal(frame_counts, expected_frame_counts):
        raise RuntimeError(
            "Local test split does not match Part-2: "
            f"observed sector counts {frame_counts.tolist()}, "
            f"expected {expected_frame_counts.tolist()}"
        )

    bin_edges = np.linspace(0.0, 1.0, args.hist_bins + 1, dtype=np.float64)
    hist, sums, sums_sq, counts = accumulate_sector_group_distributions(
        ((0, gates.shape[0], gates),), frame_sector, relevant_masks, eligible, bin_edges
    )
    means, stds, sector_d, global_d = summarize_group_moments(sums, sums_sq, counts)
    with report_path.open(encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    reference_cell = report["primary_validation_estimate_test_effect"]["interaction_excluded"][
        "cells"
    ]["recurrent_sector"]
    reference_d = float(reference_cell["top_percent"]["10"]["cohens_d"])
    reproduction_tolerance = 5e-5
    if not np.isclose(global_d, reference_d, atol=reproduction_tolerance, rtol=0.0):
        raise RuntimeError(
            f"Reconstructed global d={global_d:.9f} does not match Part-2 d={reference_d:.9f}"
        )

    arrays_path = save_dir / "recurrent_sector_top10_gate_distributions.npz"
    metadata_path = save_dir / "recurrent_sector_top10_gate_distributions_meta.json"
    np.savez_compressed(
        arrays_path,
        bin_edges=bin_edges.astype(np.float32),
        hist_counts=hist.astype(np.int64),
        group_mean=means.astype(np.float32),
        group_std=stds.astype(np.float32),
        group_count=counts.astype(np.int64),
        sector_cohens_d=sector_d.astype(np.float32),
        relevant_mask=relevant_masks.astype(np.uint8),
        eligible_mask=eligible.astype(np.uint8),
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "data_suffix": args.data_suffix,
        "selectivity": str(selectivity_path),
        "part2_report": str(report_path),
        "split_report": str(split_report_path),
        "trajectory": "reset sequential trajectory, identical to Part-2 test collection",
        "gate": "recurrent SOURCE gate averaged over 256 destination rows",
        "gate_values": "raw post-sigmoid; 0.5 point mass included",
        "selection": "sector FDR-selective, interaction-dominant excluded, top 10% per sector",
        "groups": list(GROUP_NAMES),
        "eligible_units": int(eligible.sum()),
        "top10_units_per_sector": relevant_masks.sum(axis=1).astype(int).tolist(),
        "remaining_units_per_sector": (eligible.sum() - relevant_masks.sum(axis=1))
        .astype(int)
        .tolist(),
        "frames_by_sector": frame_counts.astype(int).tolist(),
        "hist_bins": args.hist_bins,
        "gate_tau": float(model.gate_tau),
        "sector_cohens_d": sector_d.tolist(),
        "global_cohens_d": global_d,
        "part2_reference_cohens_d": reference_d,
        "global_d_absolute_difference": abs(global_d - reference_d),
        "global_d_reproduction_tolerance": reproduction_tolerance,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved {arrays_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
