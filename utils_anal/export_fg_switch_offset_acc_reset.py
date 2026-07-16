"""Export fg-switch offset accuracies with model state reset at every fg switch.

This ablation reads sector checkpoints and the balanced joint-switch test split. It preserves
the original input tensors and labels, but splits each sequence at every ``fg_switch`` frame.
Each segment is forwarded independently, so every model uses its standard zero initial state
without receiving an explicit switch/context feature. Segments of equal length are batched.

Outputs (in ``--save_dir``):
- ``fg_switch_offset_acc_<ckpt_tag>.npz``  — char/sector accuracy for preN..postN.
- ``fg_switch_offset_meta_<ckpt_tag>.json`` — checkpoint, counts, and reset protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.train_acceleration import run_forward_with_feedback
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils_anal.export_fg_switch_offset_acc import (
    _build_offset_targets_from_switch,
    _collect_ckpts,
    _parse_model_key,
    build_offset_labels,
    build_offset_order,
)


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, balanced-data, reset-evaluation, and output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", default="./results/train_data")
    parser.add_argument("--ckpts", nargs="*", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--data_dir", default="")
    parser.add_argument(
        "--data_suffix",
        default="40h-float32-jointswitch-balanced",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_radius", type=int, default=10)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _segment_groups(
    reset_mask: np.ndarray,
) -> DefaultDict[int, list[tuple[int, int, int]]]:
    """Group ``(sample, start, end)`` segments by length for zero-state batched forwards."""

    groups: DefaultDict[int, list[tuple[int, int, int]]] = defaultdict(list)
    batch_size, seq_len = reset_mask.shape
    for sample_idx in range(batch_size):
        starts = [0]
        starts.extend(
            int(idx)
            for idx in np.flatnonzero(reset_mask[sample_idx]).tolist()
            if int(idx) > 0
        )
        starts = sorted(set(starts))
        ends = starts[1:] + [seq_len]
        for start, end in zip(starts, ends):
            if end > start:
                groups[end - start].append((sample_idx, start, end))
    return groups


def evaluate_with_switch_resets(
    ckpt_path: str,
    model: torch.nn.Module,
    test_ds,
    device: torch.device,
    batch_size: int,
    window_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate one checkpoint after resetting its standard state at each fg switch."""

    switch_arr = getattr(test_ds, "fg_switch", None)
    if switch_arr is None:
        raise RuntimeError("Test dataset does not expose fg_switch.")

    offset_order = build_offset_order(window_radius)
    offset_targets = _build_offset_targets_from_switch(switch_arr, window_radius)
    stats = {
        offset: {"char_correct": 0, "sector_correct": 0, "total": 0}
        for offset in offset_order
    }
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    model_key = _parse_model_key(ckpt_path)
    use_feedback = True if model_key == "gawf" else None
    seq_len = int(getattr(test_ds, "frame_num", 32))
    chan_num = int(getattr(test_ds, "chan_num", 2))
    reset_count = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, labels = batch[0], batch[1]
            if inputs.dtype == torch.float64:
                inputs = inputs.float()
            labels_np = labels.detach().cpu().numpy()
            current_batch = int(inputs.shape[0])
            global_start = batch_idx * batch_size * seq_len + chan_num
            global_end = global_start + current_batch * seq_len
            batch_switch = np.asarray(switch_arr[global_start:global_end]).reshape(
                current_batch, seq_len
            )
            reset_mask = batch_switch != 0
            reset_count += int(reset_mask.sum())

            pred_char = np.empty((current_batch, seq_len), dtype=np.int64)
            pred_sector = np.empty((current_batch, seq_len), dtype=np.int64)
            for _length, segments in sorted(_segment_groups(reset_mask).items()):
                for chunk_start in range(0, len(segments), batch_size):
                    chunk = segments[chunk_start : chunk_start + batch_size]
                    segment_inputs = torch.stack(
                        [inputs[sample, start:end] for sample, start, end in chunk]
                    ).to(device)
                    out_char, out_sector = run_forward_with_feedback(
                        model,
                        segment_inputs,
                        use_feedback=use_feedback,
                    )
                    char_np = torch.argmax(out_char, dim=2).detach().cpu().numpy()
                    sector_np = torch.argmax(out_sector, dim=2).detach().cpu().numpy()
                    for local_idx, (sample, start, end) in enumerate(chunk):
                        pred_char[sample, start:end] = char_np[local_idx]
                        pred_sector[sample, start:end] = sector_np[local_idx]

            batch_offsets = offset_targets[global_start:global_end].reshape(
                current_batch, seq_len
            )
            for offset in offset_order:
                mask = batch_offsets == offset
                count = int(mask.sum())
                if count == 0:
                    continue
                stats[offset]["char_correct"] += int(
                    ((pred_char == labels_np[:, :, 0]) & mask).sum()
                )
                stats[offset]["sector_correct"] += int(
                    ((pred_sector == labels_np[:, :, 1]) & mask).sum()
                )
                stats[offset]["total"] += count

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                print(
                    f"  [reset eval] batches={batch_idx + 1}/{len(loader)} "
                    f"switch_resets={reset_count}"
                )

    char_acc = np.zeros(len(offset_order), dtype=np.float32)
    sector_acc = np.zeros(len(offset_order), dtype=np.float32)
    frame_counts = np.zeros(len(offset_order), dtype=np.int64)
    for idx, offset in enumerate(offset_order):
        total = stats[offset]["total"]
        if total <= 0:
            continue
        frame_counts[idx] = total
        char_acc[idx] = 100.0 * stats[offset]["char_correct"] / total
        sector_acc[idx] = 100.0 * stats[offset]["sector_correct"] / total
    return char_acc, sector_acc, frame_counts, reset_count


def save_outputs(
    save_dir: str,
    ckpt_path: str,
    char_acc: np.ndarray,
    sector_acc: np.ndarray,
    frame_counts: np.ndarray,
    reset_count: int,
    offset_order: list[int],
    offset_labels: list[str],
) -> None:
    """Save reset-ablation arrays and protocol metadata without changing base filenames."""

    os.makedirs(save_dir, exist_ok=True)
    ckpt_tag = os.path.basename(ckpt_path).replace("_model.pth", "")
    npz_path = os.path.join(save_dir, f"fg_switch_offset_acc_{ckpt_tag}.npz")
    meta_path = os.path.join(save_dir, f"fg_switch_offset_meta_{ckpt_tag}.json")
    np.savez(
        npz_path,
        offset_order=np.asarray(offset_order, dtype=np.int64),
        offset_labels=np.asarray(offset_labels, dtype="<U8"),
        char_acc=char_acc.astype(np.float32),
        sector_acc=sector_acc.astype(np.float32),
        frame_counts=frame_counts.astype(np.int64),
    )
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "switch_target": "fg",
                "state_reset_at_fg_switch": True,
                "reset_state": "model_standard_zero_initial_state",
                "reset_implementation": "split_at_switch_and_forward_each_segment_independently",
                "explicit_switch_feature_added": False,
                "input_frames_modified": False,
                "reset_count": int(reset_count),
                "window_radius": len(offset_order) // 2,
                "offset_order": offset_order,
                "offset_labels": offset_labels,
                "frame_counts": frame_counts.astype(np.int64).tolist(),
            },
            stream,
            indent=2,
        )
    print(f"Saved npz:  {npz_path}")
    print(f"Saved meta: {meta_path}")


def main() -> None:
    """Run the six-model (or explicit checkpoint) balanced reset ablation."""

    args = parse_args()
    setattr(args, "switch_target", "fg")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device, require_cuda_if_requested=False)
    test_ds, num_pos = build_test_dataset(args)
    offset_order = build_offset_order(args.window_radius)
    offset_labels = build_offset_labels(offset_order)
    ckpt_paths = _collect_ckpts(args)
    print(f"Test dataset size: {len(test_ds)}; checkpoints: {len(ckpt_paths)}")

    for ckpt_path in ckpt_paths:
        print(f"\n[reset eval] {ckpt_path}")
        model = build_model_from_ckpt(ckpt_path, num_pos=num_pos, device=device)
        char_acc, sector_acc, frame_counts, reset_count = evaluate_with_switch_resets(
            ckpt_path,
            model,
            test_ds,
            device,
            args.batch_size,
            args.window_radius,
        )
        save_outputs(
            args.save_dir,
            ckpt_path,
            char_acc,
            sector_acc,
            frame_counts,
            reset_count,
            offset_order,
            offset_labels,
        )


if __name__ == "__main__":
    main()
