"""Configurable Clutter input preparation and locality-aware sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import Sampler


CAST_MODES = ("sample", "batch_cpu", "device")
FRAME_LAYOUTS = ("stacked", "compact")


@dataclass(frozen=True)
class ClutterDataPipelineConfig:
    """Input-pipeline choices shared by train, eval, and benchmarks."""

    cast_mode: str = "device"
    frame_layout: str = "compact"
    shuffle_block_size: int = -1

    def __post_init__(self) -> None:
        if self.cast_mode not in CAST_MODES:
            raise ValueError(f"cast_mode must be one of {CAST_MODES}, got {self.cast_mode!r}")
        if self.frame_layout not in FRAME_LAYOUTS:
            raise ValueError(
                f"frame_layout must be one of {FRAME_LAYOUTS}, got {self.frame_layout!r}"
            )
        if self.shuffle_block_size < -1:
            raise ValueError("shuffle_block_size must be -1, 0, or a positive integer")


class BlockShuffleSampler(Sampler[int]):
    """Shuffle contiguous sample blocks and independently shuffle within each block."""

    def __init__(self, data_source, block_size: int, seed: int = 0) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.data_source = data_source
        self.block_size = int(block_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        num_blocks = (n + self.block_size - 1) // self.block_size
        for block_idx in torch.randperm(num_blocks, generator=self.generator).tolist():
            start = block_idx * self.block_size
            end = min(start + self.block_size, n)
            local_order = torch.randperm(end - start, generator=self.generator).tolist()
            for offset in local_order:
                yield start + offset


def resolve_base_dataset(dataset):
    """Unwrap Subset-like wrappers to the dataset holding pipeline metadata."""

    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def prepare_clutter_inputs(
    inputs: torch.Tensor,
    *,
    device,
    cast_mode: str,
    frame_layout: str,
    chan_num: int,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Cast, transfer, and expand one Clutter batch without changing pixel values."""

    if cast_mode not in CAST_MODES:
        raise ValueError(f"unknown cast_mode={cast_mode!r}")
    if frame_layout not in FRAME_LAYOUTS:
        raise ValueError(f"unknown frame_layout={frame_layout!r}")

    if cast_mode == "batch_cpu":
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    elif cast_mode == "device":
        inputs = inputs.to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
    else:
        if inputs.dtype == torch.float64:
            inputs = inputs.float()
        if inputs.dtype != torch.float32:
            raise TypeError(
                "sample cast mode expects Dataset to emit float32, "
                f"got {inputs.dtype}"
            )
        inputs = inputs.to(device=device, non_blocking=non_blocking)

    if frame_layout == "compact":
        if inputs.ndim != 4:
            raise ValueError(f"compact layout expects (B,L,H,W), got {tuple(inputs.shape)}")
        if chan_num <= 0 or inputs.shape[1] < chan_num:
            raise ValueError(
                f"invalid compact window length={inputs.shape[1]} for chan_num={chan_num}"
            )
        # unfold appends C: (B,T,H,W,C) -> (B,T,C,H,W). Materialize on
        # the target device because the CNN flattens B*T with view().
        inputs = inputs.unfold(1, chan_num, 1).permute(0, 1, 4, 2, 3).contiguous()
    elif inputs.ndim != 5:
        raise ValueError(f"stacked layout expects (B,T,C,H,W), got {tuple(inputs.shape)}")

    return inputs
