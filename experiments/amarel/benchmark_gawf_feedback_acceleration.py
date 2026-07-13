"""Benchmark semantics-equivalent GaWF feedback/gate implementations on CUDA.

The benchmark uses the parameter-matched single-layer Atari shape by default and
measures forward plus backward time for the historical split-transform formula,
the shared-core combined transform, and its pure-tensor ``torch.compile`` path.

Outputs (in ``--output_dir``):
- ``gawf_feedback_benchmark.json`` — shape, timings, speedups, numerical deltas,
  peak CUDA memory, PyTorch version, and GPU identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.recurrent_cores.gawf import _gawf_layer_preactivation


TensorFn = Callable[..., torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=3136)
    parser.add_argument("--hidden_size", type=int, default=1577)
    parser.add_argument("--feedback_dim", type=int, default=6)
    parser.add_argument("--gate_tau", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--output_dir", default="results/benchmarks/gawf_feedback")
    return parser.parse_args()


def _legacy_preactivation(
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    feedback: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    gate_tau: float,
) -> torch.Tensor:
    input_size = x_t.size(-1)
    fb_t = feedback.clamp(-10, 10).unsqueeze(2)
    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    trans_ih = torch.matmul(scaled_u, V[:, :input_size])
    trans_hh = torch.matmul(scaled_u, V[:, input_size:])
    gate_ih = torch.sigmoid(trans_ih / gate_tau)
    gate_hh = torch.sigmoid(trans_hh / gate_tau)
    ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
    hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
    return ih + hh + bias_ih.unsqueeze(0) + bias_hh.unsqueeze(0)


def _build_tensors(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    torch.manual_seed(42)
    batch, input_size, hidden, feedback_dim = (
        args.batch_size,
        args.input_size,
        args.hidden_size,
        args.feedback_dim,
    )
    shapes = (
        (batch, input_size),
        (batch, hidden),
        (batch, feedback_dim),
        (hidden, feedback_dim),
        (feedback_dim, input_size + hidden),
        (hidden, input_size),
        (hidden, hidden),
        (hidden,),
        (hidden,),
    )
    return [torch.randn(shape, device=device, requires_grad=True) for shape in shapes]


def _clear_gradients(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _run_step(
    fn: TensorFn,
    tensors: list[torch.Tensor],
    gate_tau: float,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = fn(*tensors, gate_tau)
        loss = output.float().square().mean()
    loss.backward()
    return output


def _benchmark_variant(
    name: str,
    fn: TensorFn,
    tensors: list[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, float], torch.Tensor]:
    for _ in range(args.warmup):
        _clear_gradients(tensors)
        output = _run_step(fn, tensors, args.gate_tau)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    for _ in range(args.iterations):
        _clear_gradients(tensors)
        output = _run_step(fn, tensors, args.gate_tau)
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0 / args.iterations
    event_ms = start_event.elapsed_time(end_event) / args.iterations
    result = {
        "cuda_event_ms": float(event_ms),
        "wall_ms": float(wall_ms),
        "peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
    }
    print(f"{name}: {result}")
    return result, output.detach().float().cpu()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    tensors = _build_tensors(args, device)

    compiled = torch.compile(
        _gawf_layer_preactivation,
        mode=args.compile_mode,
        fullgraph=True,
        dynamic=False,
    )
    variants: list[tuple[str, TensorFn]] = [
        ("legacy_split_eager", _legacy_preactivation),
        ("combined_eager", _gawf_layer_preactivation),
        ("combined_compiled", compiled),
    ]
    timings: dict[str, dict[str, float]] = {}
    outputs: dict[str, torch.Tensor] = {}
    for name, fn in variants:
        timings[name], outputs[name] = _benchmark_variant(name, fn, tensors, args)

    baseline_ms = timings["legacy_split_eager"]["cuda_event_ms"]
    for name, values in timings.items():
        values["speedup_vs_legacy"] = baseline_ms / values["cuda_event_ms"]
    numerical_delta = {
        name: float((output - outputs["legacy_split_eager"]).abs().max())
        for name, output in outputs.items()
    }
    payload: dict[str, Any] = {
        "shape": {
            "batch_size": args.batch_size,
            "input_size": args.input_size,
            "hidden_size": args.hidden_size,
            "feedback_dim": args.feedback_dim,
        },
        "dtype": "bfloat16 autocast with float32 parameters",
        "gate_tau": args.gate_tau,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "compile_mode": args.compile_mode,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "timings": timings,
        "max_abs_output_delta_vs_legacy": numerical_delta,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "gawf_feedback_benchmark.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
