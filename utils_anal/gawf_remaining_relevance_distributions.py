"""Export the three remaining Part-2 top-10% SOURCE-gate distributions.

The exact GaWF checkpoint and test split used by the symmetric Part-2 analysis are replayed with
reset sequential trajectories. Input and recurrent gate matrices are averaged over their 256
destination rows. For input/sector, input/digit, and recurrent/digit, the saved validation-defined
FDR masks are combined with interaction-dominant exclusion and context-specific top-10% tuning
masks. One NPZ stores float32 bin edges, means, SDs, Cohen's d values and uint8 masks plus int64
histogram/group counts; one JSON stores provenance and numerical checks against Part 2.
"""

from __future__ import annotations

import argparse
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
from utils_anal.gawf_recurrent_sector_relevance_distributions import (
    accumulate_context_group_distributions,
    summarize_group_moments,
)


CELL_SPECS = {
    "input_sector": ("input", "encoder", "sector", 9, 1),
    "input_digit": ("input", "encoder", "digit", 10, 0),
    "recurrent_digit": ("recurrent", "hidden", "digit", 10, 0),
}
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
            output_dir("E_relevance_alignment", "gawf_remaining_relevance_distributions", "data")
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


def collect_test_source_gates(
    dataset: object,
    model: object,
    device: object,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run exact Part-2 test trajectories and return both SOURCE-gate views and labels."""

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
    gate_batches: dict[str, list[np.ndarray]] = {"input": [], "recurrent": []}
    label_batches: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            trajectory = _trajectory_with_measurements(encoded, model, record_gates=True)
            for gate_name in gate_batches:
                gate_batches[gate_name].append(
                    trajectory[f"{gate_name}_gate"].cpu().numpy().astype(np.float32, copy=False)
                )
            label_batches.append(labels.numpy().astype(np.int64, copy=False))
            samples_done = min((batch_index + 1) * batch_size, len(dataset))
            if samples_done % 200 < batch_size or batch_index + 1 == len(loader):
                print(
                    f"  collected {samples_done}/{len(dataset)} test sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    gates = {}
    for gate_name, batches in gate_batches.items():
        concatenated = np.concatenate(batches, axis=0)
        gates[gate_name] = concatenated.reshape(-1, concatenated.shape[-1])
    labels = np.concatenate(label_batches, axis=0)
    return gates, labels.reshape(-1, labels.shape[-1])


def main() -> None:
    """Collect exact test gates and export the three remaining distribution families."""

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
    from utils_anal.gawf_symmetric_stats import relevance_masks

    args.use_sector_mode = True
    args.predict_all_chars = False
    device = resolve_device(args.device, require_cuda_if_requested=True)
    test_dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(checkpoint_path), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis requires a single-layer GaWF checkpoint")

    selections = {}
    with np.load(selectivity_path, allow_pickle=False) as selectivity:
        for cell, (_gate, population, factor, levels, _label_column) in CELL_SPECS.items():
            tuning = np.asarray(
                selectivity[f"primary_{population}_tuning_{factor}"], dtype=np.float64
            )
            passed = np.asarray(selectivity[f"primary_{population}_passed_{factor}"], dtype=bool)
            dominant = np.asarray(
                selectivity[f"primary_{population}_interaction_dominant"], dtype=bool
            )
            eligible = passed & ~dominant
            relevant = relevance_masks(tuning, eligible, 0.10)
            if relevant.shape != (levels, eligible.size):
                raise RuntimeError(f"{cell} masks have unexpected shape {relevant.shape}")
            selections[cell] = (eligible, relevant)

    gates, labels = collect_test_source_gates(
        test_dataset,
        model,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    with split_report_path.open(encoding="utf-8") as file_obj:
        split_report = json.load(file_obj)["test"]
    observed_counts = {
        "sector": np.bincount(labels[:, 1], minlength=9).astype(np.int64),
        "digit": np.bincount(labels[:, 0], minlength=10).astype(np.int64),
    }
    for factor, counts in observed_counts.items():
        expected = np.asarray(split_report[f"{factor}_counts"], dtype=np.int64)
        if not np.array_equal(counts, expected):
            raise RuntimeError(
                f"Local test {factor} counts {counts.tolist()} do not match "
                f"Part-2 counts {expected.tolist()}"
            )

    with report_path.open(encoding="utf-8") as file_obj:
        report_cells = json.load(file_obj)["primary_validation_estimate_test_effect"][
            "interaction_excluded"
        ]["cells"]
    bin_edges = np.linspace(0.0, 1.0, args.hist_bins + 1, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {"bin_edges": bin_edges.astype(np.float32)}
    cell_metadata = {}
    reproduction_tolerance = 5e-5
    for cell, (gate_name, population, factor, levels, label_column) in CELL_SPECS.items():
        eligible, relevant = selections[cell]
        values = gates[gate_name]
        contexts = labels[:, label_column]
        hist, sums, sums_sq, counts = accumulate_context_group_distributions(
            ((0, values.shape[0], values),), contexts, relevant, eligible, bin_edges
        )
        means, stds, context_d, global_d = summarize_group_moments(sums, sums_sq, counts)
        reference_d = float(report_cells[cell]["top_percent"]["10"]["cohens_d"])
        if not np.isclose(global_d, reference_d, atol=reproduction_tolerance, rtol=0.0):
            raise RuntimeError(
                f"{cell} reconstructed d={global_d:.9f} does not match "
                f"Part-2 d={reference_d:.9f}"
            )
        arrays.update(
            {
                f"{cell}_hist_counts": hist.astype(np.int64),
                f"{cell}_group_mean": means.astype(np.float32),
                f"{cell}_group_std": stds.astype(np.float32),
                f"{cell}_group_count": counts.astype(np.int64),
                f"{cell}_context_cohens_d": context_d.astype(np.float32),
                f"{cell}_relevant_mask": relevant.astype(np.uint8),
                f"{cell}_eligible_mask": eligible.astype(np.uint8),
            }
        )
        top_count = relevant.sum(axis=1).astype(int)
        cell_metadata[cell] = {
            "gate": f"{gate_name} SOURCE gate averaged over 256 destination rows",
            "activation_population": population,
            "factor": factor,
            "context_levels": levels,
            "selection": (
                f"{factor} FDR-selective, interaction-dominant excluded, "
                "top 10% independently per context"
            ),
            "eligible_units": int(eligible.sum()),
            "top10_units_per_context": top_count.tolist(),
            "remaining_units_per_context": (eligible.sum() - top_count).astype(int).tolist(),
            "frames_per_context": observed_counts[factor].astype(int).tolist(),
            "context_cohens_d": context_d.tolist(),
            "global_cohens_d": global_d,
            "part2_reference_cohens_d": reference_d,
            "global_d_absolute_difference": abs(global_d - reference_d),
        }

    arrays_path = save_dir / "remaining_top10_gate_distributions.npz"
    metadata_path = save_dir / "remaining_top10_gate_distributions_meta.json"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "data_suffix": args.data_suffix,
        "selectivity": str(selectivity_path),
        "part2_report": str(report_path),
        "split_report": str(split_report_path),
        "trajectory": "reset sequential trajectory, identical to Part-2 test collection",
        "gate_values": "raw post-sigmoid; 0.5 point mass included",
        "groups": list(GROUP_NAMES),
        "hist_bins": args.hist_bins,
        "gate_tau": float(model.gate_tau),
        "global_d_reproduction_tolerance": reproduction_tolerance,
        "cells": cell_metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved {arrays_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
