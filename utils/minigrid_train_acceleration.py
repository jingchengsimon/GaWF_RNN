"""Configure semantics-preserving CUDA acceleration for MiniGrid PPO.

The policy controls CUDA math precision, autocast, gradient scaling, optional
``torch.compile``, and logging. It does not change environment samples, PPO
losses, rollout length, update cadence, or model structure.

Outputs:
- ``MiniGridAcceleration``: runtime contexts and callable compilation helpers.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
from typing import Any, Callable, ContextManager, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class MiniGridAcceleration:
    """Runtime acceleration policy for one MiniGrid PPO process."""

    device: torch.device
    amp_dtype_name: str = "none"
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    @property
    def amp_dtype(self) -> torch.dtype | None:
        """Return the configured CUDA autocast dtype, or ``None`` when disabled."""

        if self.device.type != "cuda" or self.amp_dtype_name == "none":
            return None
        if self.amp_dtype_name == "bfloat16":
            return torch.bfloat16
        if self.amp_dtype_name == "float16":
            return torch.float16
        raise ValueError(f"Unsupported amp dtype: {self.amp_dtype_name}")

    def autocast(self) -> ContextManager[Any]:
        """Create a fresh autocast context for a model forward/loss block."""

        dtype = self.amp_dtype
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def build_grad_scaler(self) -> Any:
        """Build an FP16 gradient scaler; BF16 and full precision do not need one."""

        enabled = self.amp_dtype == torch.float16
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def compile_callable(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Compile ``fn`` on CUDA when requested, otherwise return it unchanged."""

        if not self.compile_model or self.device.type != "cuda":
            return fn
        return torch.compile(fn, mode=self.compile_mode, dynamic=False)


def configure_minigrid_acceleration(
    acceleration: MiniGridAcceleration,
    logger: logging.Logger,
) -> None:
    """Apply process-wide CUDA math settings and log the active policy."""

    if acceleration.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = acceleration.allow_tf32
        torch.backends.cudnn.allow_tf32 = acceleration.allow_tf32
        torch.backends.cudnn.benchmark = acceleration.cudnn_benchmark
        if acceleration.allow_tf32:
            torch.set_float32_matmul_precision("high")
    logger.info(
        "MiniGrid acceleration: device=%s amp=%s tf32=%s cudnn_benchmark=%s "
        "compile=%s compile_mode=%s",
        acceleration.device,
        acceleration.amp_dtype_name if acceleration.amp_dtype is not None else "off",
        acceleration.allow_tf32 and acceleration.device.type == "cuda",
        acceleration.cudnn_benchmark and acceleration.device.type == "cuda",
        acceleration.compile_model and acceleration.device.type == "cuda",
        acceleration.compile_mode,
    )
