#!/usr/bin/env python3
"""Losslessly convert Clutter stimulus arrays from float32 storage to uint8.

The converter is chunked and resumable.  A target is published under its final
name only after every source value has been checked to be a finite integer in
``[0, 255]`` and the uint8 round trip has been verified exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.lib.format as npfmt


DEFAULT_SOURCE_SUFFIX = "40h-float32"
DEFAULT_TARGET_SUFFIX = "40h-uint8"
SPLIT_STEMS = {
    "train": "stimulus_reg-train",
    "validation": "stimulus_reg-validation",
    "test": "stimulus_reg-test",
}


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, array: np.ndarray) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _validate_chunk(chunk: np.ndarray, start: int, end: int) -> np.ndarray:
    if not np.isfinite(chunk).all():
        raise ValueError(f"non-finite source value in frames {start}:{end}")
    if np.any(chunk < 0.0) or np.any(chunk > 255.0):
        raise ValueError(f"source value outside [0, 255] in frames {start}:{end}")
    converted = chunk.astype(np.uint8)
    if not np.array_equal(converted.astype(np.float32), chunk):
        raise ValueError(f"non-integral or non-round-trippable source value in frames {start}:{end}")
    return converted


def convert_array(source: Path, target: Path, chunk_frames: int) -> dict:
    """Convert one array, resuming a compatible partial conversion when present."""

    source = source.resolve()
    target = target.resolve()
    partial = target.with_name(f"{target.name}.partial")
    progress_path = target.with_name(f"{target.name}.progress.json")
    source_array = np.load(source, mmap_mode="r")
    if source_array.dtype != np.float32:
        raise TypeError(f"expected float32 source, got {source_array.dtype}: {source}")
    if source_array.ndim != 3:
        raise ValueError(f"expected a (frames, height, width) array, got {source_array.shape}")

    identity = _source_identity(source, source_array)
    completed = 0
    if target.exists():
        existing = np.load(target, mmap_mode="r")
        if existing.dtype != np.uint8 or existing.shape != source_array.shape:
            raise ValueError(f"incompatible existing target: {target}")
        total = source_array.shape[0]
        for start in range(0, total, chunk_frames):
            end = min(start + chunk_frames, total)
            source_chunk = np.asarray(source_array[start:end])
            _validate_chunk(source_chunk, start, end)
            if not np.array_equal(np.asarray(existing[start:end]).astype(np.float32), source_chunk):
                raise ValueError(f"existing target differs from source at frames {start}:{end}")
        return {"status": "already_complete", "source": identity, "target": str(target)}

    if partial.exists() or progress_path.exists():
        if not partial.exists() or not progress_path.exists():
            raise RuntimeError(f"incomplete resume pair: {partial} / {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("source") != identity:
            raise RuntimeError(f"source changed since partial conversion: {source}")
        completed = int(progress.get("completed_frames", 0))
        target_array = np.load(partial, mmap_mode="r+")
        if target_array.dtype != np.uint8 or target_array.shape != source_array.shape:
            raise RuntimeError(f"partial target is incompatible: {partial}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target_array = npfmt.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=source_array.shape,
        )
        _atomic_json(
            progress_path,
            {
                "status": "converting",
                "source": identity,
                "target": str(target),
                "completed_frames": 0,
            },
        )

    total = source_array.shape[0]
    if completed < 0 or completed > total:
        raise RuntimeError(f"invalid completed_frames={completed} for {progress_path}")

    for start in range(completed, total, chunk_frames):
        end = min(start + chunk_frames, total)
        source_chunk = np.asarray(source_array[start:end])
        converted = _validate_chunk(source_chunk, start, end)
        target_array[start:end] = converted
        target_array.flush()
        if not np.array_equal(np.asarray(target_array[start:end]).astype(np.float32), source_chunk):
            raise RuntimeError(f"written target failed round-trip verification at {start}:{end}")
        _atomic_json(
            progress_path,
            {
                "status": "converting",
                "source": identity,
                "target": str(target),
                "completed_frames": end,
            },
        )
        print(f"[{target.name}] {end}/{total} frames", flush=True)

    del target_array
    os.replace(partial, target)
    final = {
        "status": "complete",
        "source": identity,
        "target": str(target),
        "target_dtype": "uint8",
        "target_shape": list(source_array.shape),
        "completed_frames": total,
    }
    _atomic_json(progress_path, final)
    return final


def copy_labels(source: Path, target: Path) -> dict:
    """Copy a label TSV atomically and verify that its content is unchanged."""

    source = source.resolve()
    target = target.resolve()
    source_hash = _sha256(source)
    if target.exists():
        target_hash = _sha256(target)
        if target_hash != source_hash:
            raise ValueError(f"existing label target differs from source: {target}")
        return {"status": "already_complete", "sha256": source_hash, "target": str(target)}
    partial = target.with_name(f"{target.name}.partial")
    shutil.copy2(source, partial)
    target_hash = _sha256(partial)
    if target_hash != source_hash:
        raise RuntimeError(f"label copy verification failed: {source} -> {target}")
    os.replace(partial, target)
    return {"status": "complete", "sha256": source_hash, "target": str(target)}


def convert_splits(
    data_dir: Path,
    source_suffix: str,
    target_suffix: str,
    splits: Iterable[str],
    chunk_frames: int,
) -> dict:
    results = {}
    for split in splits:
        stem = SPLIT_STEMS[split]
        source_npy = data_dir / f"{stem}-{source_suffix}.npy"
        target_npy = data_dir / f"{stem}-{target_suffix}.npy"
        source_tsv = data_dir / f"{stem}-{source_suffix}.tsv"
        target_tsv = data_dir / f"{stem}-{target_suffix}.tsv"
        for required in (source_npy, source_tsv):
            if not required.is_file():
                raise FileNotFoundError(required)
        results[split] = {
            "array": convert_array(source_npy, target_npy, chunk_frames),
            "labels": copy_labels(source_tsv, target_tsv),
        }
    manifest = data_dir / f"conversion-{source_suffix}-to-{target_suffix}.json"
    _atomic_json(manifest, {"status": "complete", "splits": results})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-suffix", default=DEFAULT_SOURCE_SUFFIX)
    parser.add_argument("--target-suffix", default=DEFAULT_TARGET_SUFFIX)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_STEMS),
        default=list(SPLIT_STEMS),
    )
    parser.add_argument("--chunk-frames", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be positive")
    if args.source_suffix == args.target_suffix:
        parser.error("source and target suffixes must differ")
    return args


def main() -> None:
    args = parse_args()
    convert_splits(
        args.data_dir.expanduser().resolve(),
        args.source_suffix,
        args.target_suffix,
        args.splits,
        args.chunk_frames,
    )


if __name__ == "__main__":
    main()
