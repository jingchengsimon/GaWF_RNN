#!/usr/bin/env python3
"""Benchmark Clutter dtype, casting, layout, locality, and staging choices."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_model import MC_RNN_Dataset
from utils.clutter_data_pipeline import BlockShuffleSampler, prepare_clutter_inputs
from utils.clutter_task_models import RNNConv


@dataclass(frozen=True)
class Variant:
    name: str
    suffix: str
    cast_mode: str
    frame_layout: str
    block_size: int = 0
    staged: bool = False


VARIANTS = (
    Variant("float32_sample_stacked_global", "40h-float32", "sample", "stacked"),
    Variant("uint8_sample_stacked_global", "40h-uint8", "sample", "stacked"),
    Variant("uint8_batchcpu_stacked_global", "40h-uint8", "batch_cpu", "stacked"),
    Variant("uint8_device_stacked_global", "40h-uint8", "device", "stacked"),
    Variant("uint8_device_compact_global", "40h-uint8", "device", "compact"),
    Variant("uint8_device_compact_block64", "40h-uint8", "device", "compact", 64),
    Variant("uint8_device_compact_block256", "40h-uint8", "device", "compact", 256),
    Variant("uint8_device_compact_block1024", "40h-uint8", "device", "compact", 1024),
    Variant("uint8_device_compact_block4096", "40h-uint8", "device", "compact", 4096),
    Variant("uint8_sample_stacked_block256_staged", "40h-uint8", "sample", "stacked", 256, True),
    Variant("uint8_batchcpu_stacked_block256_staged", "40h-uint8", "batch_cpu", "stacked", 256, True),
    Variant("uint8_device_stacked_block256_staged", "40h-uint8", "device", "stacked", 256, True),
    Variant("uint8_device_compact_block256_staged", "40h-uint8", "device", "compact", 256, True),
    Variant(
        "uint8_device_compact_block1024_staged",
        "40h-uint8",
        "device",
        "compact",
        1024,
        True,
    ),
)


def _proc_io() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            out[key] = int(value.strip())
    except OSError:
        pass
    return out


def _drop_cache_hint(path: Path) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def _stage_file(source: Path, stage_dir: Path) -> tuple[Path, float]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / source.name
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return target, 0.0
    partial = target.with_name(target.name + ".partial")
    t0 = time.perf_counter()
    with source.open("rb") as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(partial, target)
    return target, time.perf_counter() - t0


def _load_labels(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "stimulus_reg-train-40h-uint8.tsv", sep="\t")


def _make_loader(
    variant: Variant,
    data_dir: Path,
    labels: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    stage_dir: Path | None,
):
    source = data_dir / f"stimulus_reg-train-{variant.suffix}.npy"
    stage_seconds = 0.0
    if variant.staged:
        if stage_dir is None:
            raise ValueError("staged variant requires --stage-dir")
        source, stage_seconds = _stage_file(source, stage_dir)
    data = np.load(source, mmap_mode="r")
    dataset = MC_RNN_Dataset(
        data,
        labels,
        frame_num=32,
        chan_num=2,
        use_sector=True,
        input_cast_mode=variant.cast_mode,
        frame_layout=variant.frame_layout,
    )
    sampler = None
    shuffle = variant.block_size == 0
    if variant.block_size:
        sampler = BlockShuffleSampler(dataset, variant.block_size, seed=seed)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and num_workers > 0,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs), source, stage_seconds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_loader(
    loader,
    variant: Variant,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
    num_batches: int,
) -> dict:
    iterator = iter(loader)
    checksum = 0.0
    for _ in range(warmup_batches):
        batch = next(iterator)
        x = prepare_clutter_inputs(
            batch[0],
            device=device,
            cast_mode=variant.cast_mode,
            frame_layout=variant.frame_layout,
            chan_num=2,
            non_blocking=bool(loader.pin_memory),
        )
        checksum += float(x[0, 0, 0, 0, 0].item())
    _sync(device)
    before_io = _proc_io()
    before_ru = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    done = 0
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        x = prepare_clutter_inputs(
            batch[0],
            device=device,
            cast_mode=variant.cast_mode,
            frame_layout=variant.frame_layout,
            chan_num=2,
            non_blocking=bool(loader.pin_memory),
        )
        checksum += float(x[0, 0, 0, 0, 0].item())
        done += 1
    _sync(device)
    elapsed = time.perf_counter() - t0
    after_ru = resource.getrusage(resource.RUSAGE_SELF)
    after_io = _proc_io()
    samples = done * batch_size
    return {
        "mode": "loader",
        "batches": done,
        "samples": samples,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples / elapsed,
        "batches_per_sec": done / elapsed,
        "source_frames_per_sec": samples * 33 / elapsed,
        "minor_faults": after_ru.ru_minflt - before_ru.ru_minflt,
        "major_faults": after_ru.ru_majflt - before_ru.ru_majflt,
        "proc_read_bytes": after_io.get("read_bytes", 0) - before_io.get("read_bytes", 0),
        "max_rss_kib": after_ru.ru_maxrss,
        "checksum": checksum,
    }


def _benchmark_e2e(
    loader,
    variant: Variant,
    device: torch.device,
    warmup_batches: int,
    num_batches: int,
) -> dict:
    if device.type != "cuda":
        raise ValueError("end-to-end mode requires CUDA")
    model = RNNConv(
        num_classes=10,
        num_pos=9,
        hidden_size=64,
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
        device=str(device),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    iterator = iter(loader)

    def step(batch):
        x = prepare_clutter_inputs(
            batch[0],
            device=device,
            cast_mode=variant.cast_mode,
            frame_layout=variant.frame_layout,
            chan_num=2,
            non_blocking=bool(loader.pin_memory),
        )
        y = batch[1].to(device=device, non_blocking=bool(loader.pin_memory))
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            char_out, pos_out = model(x)
            loss = torch.nn.functional.cross_entropy(
                char_out.reshape(-1, 10), y[..., 0].reshape(-1)
            ) + torch.nn.functional.cross_entropy(
                pos_out.reshape(-1, 9), y[..., 1].reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().item())

    for _ in range(warmup_batches):
        step(next(iterator))
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    loss = 0.0
    done = 0
    for _ in range(num_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        loss += step(batch)
        done += 1
    _sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "mode": "e2e_amp_train",
        "batches": done,
        "elapsed_sec": elapsed,
        "batches_per_sec": done / elapsed,
        "samples_per_sec": done * loader.batch_size / elapsed,
        "mean_loss": loss / max(done, 1),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--mode", choices=("loader", "e2e"), default="loader")
    parser.add_argument("--drop-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [v for v in VARIANTS if not args.variants or v.name in args.variants]
    unknown = set(args.variants) - {v.name for v in VARIANTS}
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    labels = _load_labels(args.data_dir)
    device = torch.device(args.device)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in selected:
        source_hint = args.data_dir / f"stimulus_reg-train-{variant.suffix}.npy"
        if args.drop_cache and not variant.staged:
            cache_drop_requested = _drop_cache_hint(source_hint)
        else:
            cache_drop_requested = False
        loader, source, stage_seconds = _make_loader(
            variant,
            args.data_dir,
            labels,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.seed,
            args.stage_dir,
        )
        print(f"benchmarking {variant.name} from {source}", flush=True)
        if args.mode == "e2e":
            metrics = _benchmark_e2e(
                loader, variant, device, args.warmup_batches, args.num_batches
            )
        else:
            metrics = _benchmark_loader(
                loader,
                variant,
                device,
                args.batch_size,
                args.warmup_batches,
                args.num_batches,
            )
        row = {
            "variant": asdict(variant),
            "source": str(source),
            "source_size_bytes": source.stat().st_size,
            "stage_copy_sec": stage_seconds,
            "cache_drop_requested": cache_drop_requested,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory and args.num_workers > 0),
            **metrics,
        }
        results.append(row)
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
