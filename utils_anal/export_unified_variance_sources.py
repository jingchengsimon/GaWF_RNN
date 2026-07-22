"""Export exact saved sources for the unified GaWF variance decomposition.

Inputs are a single-layer GaWF sector checkpoint and the Clutter test split.  The exporter
loads both through the canonical analysis helpers, evaluates reset-feedback trajectories, and
writes frame-major float32 ``.npy`` arrays for encoder activations, input/recurrent gate
synapses, and hidden states.  Gate arrays are written directly to memory maps, so the complete
trial-by-synapse tensor is never resident in CPU or GPU memory.  A compact trajectory NPZ stores
the aligned labels, pre-step feedback, and static weights; ``input_manifest.json`` is consumed by
``run_unified_variance_decomposition.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.recurrent_cores.gawf import _compute_gawf_transforms
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils_anal.anal_paths import output_dir


CATEGORY = "D_variance_decomposition"
SCRIPT_NAME = "export_unified_variance_sources"
ARRAY_NAMES = (
    "encoder_activation",
    "input_gate",
    "hidden_state",
    "recurrent_gate",
)
COMPACT_NAMES = (
    "gawf_gate_trajectory.npz",
    "input_manifest.json",
    "source_provenance.json",
)


class WritableArray(Protocol):
    """Minimal indexed-write interface shared by ndarrays and memory maps."""

    def __setitem__(self, key: object, value: object) -> None:
        """Write one indexed block."""


def parse_args() -> argparse.Namespace:
    """Parse source-export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/train_data/clutter/best_6model_param_matched_40h"
            / "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
        ),
    )
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "stimuli")
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--num_workers", type=int, default=int(os.environ.get("AIM3_NUM_WORKERS", "2"))
    )
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush_every_batches", type=int, default=25)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_values(
    model: torch.nn.Module,
    feedback: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact eager GaWF gates using the shared recurrent-core transform."""

    fb_t = feedback.to(dtype=torch.float32).clamp(-10, 10).unsqueeze(2)
    transform_ih, transform_hh = _compute_gawf_transforms(
        model.U, fb_t, model.V, input_size
    )
    return (
        torch.sigmoid(transform_ih / model.gate_tau),
        torch.sigmoid(transform_hh / model.gate_tau),
    )


def export_encoded_batch(
    encoded: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    arrays: dict[str, WritableArray],
    feedback_array: WritableArray,
    label_array: WritableArray,
    sequence_offset: int,
) -> None:
    """Run one reset-feedback batch and write all frame-major source values."""

    batch_size, frame_num, input_size = encoded.shape
    start = sequence_offset * frame_num
    stop = (sequence_offset + batch_size) * frame_num
    arrays["encoder_activation"][start:stop] = (
        encoded.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, input_size)
    )
    label_array[start:stop] = (
        labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1, 2)
    )

    hidden = model.core.initial_state(batch_size, encoded.device, encoded.dtype)
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("Unified source export requires a single-layer GaWF core")
    feedback = torch.zeros(
        batch_size,
        model.feedback_dim,
        dtype=torch.float32,
        device=encoded.device,
    )
    row_base = start + np.arange(batch_size, dtype=np.int64) * frame_num
    for time_idx in range(frame_num):
        rows = row_base + time_idx
        gate_ih, gate_hh = _gate_values(model, feedback, input_size)
        arrays["input_gate"][rows] = (
            gate_ih.detach().cpu().numpy().astype(np.float32, copy=False).reshape(batch_size, -1)
        )
        arrays["recurrent_gate"][rows] = (
            gate_hh.detach().cpu().numpy().astype(np.float32, copy=False).reshape(batch_size, -1)
        )
        feedback_array[rows] = feedback.detach().cpu().numpy().astype(np.float32, copy=False)

        hidden = model.core.step(encoded[:, time_idx], hidden, feedback)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Unexpected multi-layer state during single-layer export")
        arrays["hidden_state"][rows] = (
            hidden.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        char_logits, sector_logits = model.classifier(hidden)
        feedback = model._compute_feedback(char_logits, sector_logits).to(dtype=torch.float32)


def _array_shapes(
    n_frames: int,
    input_size: int,
    hidden_size: int,
) -> dict[str, tuple[int, int]]:
    return {
        "encoder_activation": (n_frames, input_size),
        "input_gate": (n_frames, hidden_size * input_size),
        "hidden_state": (n_frames, hidden_size),
        "recurrent_gate": (n_frames, hidden_size * hidden_size),
    }


def _required_bytes(shapes: dict[str, tuple[int, int]]) -> int:
    return sum(int(np.prod(shape)) for shape in shapes.values()) * np.dtype(np.float32).itemsize


def _open_partial_arrays(
    data_dir: Path,
    shapes: dict[str, tuple[int, int]],
) -> tuple[dict[str, np.memmap], dict[str, Path]]:
    partial_paths = {name: data_dir / f"{name}.partial.npy" for name in ARRAY_NAMES}
    final_paths = {name: data_dir / f"{name}.npy" for name in ARRAY_NAMES}
    compact_paths = [data_dir / name for name in COMPACT_NAMES]
    compact_partial_paths = [
        path.with_name(f"{path.stem}.partial{path.suffix}") for path in compact_paths
    ]
    occupied = [
        path
        for path in (
            *partial_paths.values(),
            *final_paths.values(),
            *compact_paths,
            *compact_partial_paths,
            data_dir / "manifest.json",
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing unified source arrays: "
            + ", ".join(str(path) for path in occupied)
        )
    arrays = {
        name: np.lib.format.open_memmap(
            partial_paths[name], mode="w+", dtype=np.float32, shape=shapes[name]
        )
        for name in ARRAY_NAMES
    }
    return arrays, partial_paths


def _flush(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _finalize_arrays(
    arrays: dict[str, np.memmap],
    partial_paths: dict[str, Path],
    data_dir: Path,
) -> dict[str, Path]:
    _flush(arrays)
    arrays.clear()
    gc.collect()
    final_paths = {name: data_dir / f"{name}.npy" for name in ARRAY_NAMES}
    for name in ARRAY_NAMES:
        partial_paths[name].replace(final_paths[name])
    return final_paths


def _compact_paths(data_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    final = {name: data_dir / name for name in COMPACT_NAMES}
    partial = {
        name: path.with_name(f"{path.stem}.partial{path.suffix}")
        for name, path in final.items()
    }
    return final, partial


def _input_manifest_payload(
    trajectory_path: Path,
    final_paths: dict[str, Path],
    source: dict[str, object],
) -> dict[str, object]:
    """Build the exact saved-source schema consumed by the unified runner."""

    return {
        "trajectory_npz": trajectory_path.name,
        "objects": {name: {"path": final_paths[name].name} for name in ARRAY_NAMES},
        "source": source,
    }


def _finalize_compact_artifacts(
    final_paths: dict[str, Path],
    partial_paths: dict[str, Path],
) -> None:
    """Atomically publish each compact artifact after every partial write succeeds."""

    for name in COMPACT_NAMES:
        partial_paths[name].replace(final_paths[name])


def main() -> None:
    """Export every required trial-level representation to canonical saved arrays."""

    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.flush_every_batches <= 0:
        raise ValueError(
            "batch_size/flush_every_batches must be positive and num_workers nonnegative"
        )
    device = resolve_device(args.device, require_cuda_if_requested=True)
    data_dir = output_dir(CATEGORY, SCRIPT_NAME, "data")

    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        data_suffix=args.data_suffix,
        use_mmap=args.use_mmap,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
    )
    dataset, num_pos = build_test_dataset(dataset_args)
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("Unified source export requires a single-layer GaWF checkpoint")

    frame_num = int(dataset.frame_num)
    n_sequences = len(dataset)
    n_frames = n_sequences * frame_num
    input_size = int(model.encoder_flatten_size)
    hidden_size = int(model.rnn.hidden_size)
    if (input_size, hidden_size, model.feedback_dim) != (1152, 256, 19):
        raise RuntimeError(
            "Unified source contract requires input/hidden/feedback sizes 1152/256/19, got "
            f"{input_size}/{hidden_size}/{model.feedback_dim}"
        )
    shapes = _array_shapes(n_frames, input_size, hidden_size)
    required_bytes = _required_bytes(shapes)
    free_bytes = shutil.disk_usage(data_dir).free
    if free_bytes < int(required_bytes * 1.05):
        raise OSError(
            f"Insufficient free disk: need at least {required_bytes * 1.05 / 1024**3:.1f} GiB, "
            f"have {free_bytes / 1024**3:.1f} GiB"
        )

    arrays, partial_paths = _open_partial_arrays(data_dir, shapes)
    feedback = np.empty((n_frames, model.feedback_dim), dtype=np.float32)
    labels = np.empty((n_frames, 2), dtype=np.int64)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"Exporting {n_sequences} sequences x {frame_num} frames; "
        f"saved arrays={required_bytes / 1024**3:.1f} GiB",
        flush=True,
    )
    started = time.perf_counter()
    sequence_offset = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, batch_labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            encoded = model.encode_frames(frames)
            export_encoded_batch(
                encoded,
                batch_labels,
                model,
                arrays,
                feedback,
                labels,
                sequence_offset,
            )
            sequence_offset += int(frames.shape[0])
            if (batch_idx + 1) % args.flush_every_batches == 0 or sequence_offset == n_sequences:
                _flush(arrays)
                print(
                    f"exported {sequence_offset}/{n_sequences} sequences | "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    if sequence_offset != n_sequences:
        raise RuntimeError(f"Export stopped at {sequence_offset}/{n_sequences} sequences")
    final_paths = _finalize_arrays(arrays, partial_paths, data_dir)

    compact_final, compact_partial = _compact_paths(data_dir)
    trajectory_path = compact_final["gawf_gate_trajectory.npz"]
    np.savez_compressed(
        compact_partial["gawf_gate_trajectory.npz"],
        feedback=feedback.reshape(n_sequences, frame_num, model.feedback_dim),
        labels=labels.reshape(n_sequences, frame_num, 2),
        weight_ih=model.rnn.weight_ih_l0.detach().cpu().numpy().astype(np.float32),
        weight_hh=model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float32),
    )
    source = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": _sha256(args.ckpt),
        "data_dir": str(args.data_dir.resolve()),
        "data_suffix": args.data_suffix,
        "n_sequences": n_sequences,
        "frames_per_sequence": frame_num,
        "n_frames": n_frames,
    }
    input_manifest = _input_manifest_payload(trajectory_path, final_paths, source)
    manifest_path = compact_final["input_manifest.json"]
    compact_partial["input_manifest.json"].write_text(
        json.dumps(input_manifest, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        **source,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "arrays": {
            name: {"path": str(path), "shape": list(shapes[name]), "dtype": "float32"}
            for name, path in final_paths.items()
        },
        "trajectory": str(trajectory_path),
        "input_manifest": str(manifest_path),
    }
    compact_partial["source_provenance.json"].write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    _finalize_compact_artifacts(compact_final, compact_partial)
    print(f"Saved unified source manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
