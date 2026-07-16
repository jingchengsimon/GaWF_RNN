"""Benchmark semantics-equivalent GaWF feedback/gate implementations on CUDA.

The benchmark uses the parameter-matched single-layer Atari shape by default and
measures forward plus backward time for the historical split-transform formula,
the shared-core combined transform, and its pure-tensor ``torch.compile`` path.

Outputs (in ``--output_dir``):
- ``gawf_feedback_benchmark_validated.json`` — shape, timings, speedups, forward/backward
  numerical checks, peak CUDA memory, PyTorch version, and GPU identity.
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
    parser.add_argument("--max_output_relative_rmse", type=float, default=0.02)
    parser.add_argument("--max_gradient_relative_l2", type=float, default=0.05)
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
    bound = hidden**-0.5
    tensors = [
        torch.randn((batch, input_size), device=device),
        torch.randn((batch, hidden), device=device),
        torch.randn((batch, feedback_dim), device=device),
        torch.randn((hidden, feedback_dim), device=device) * 0.01,
        torch.randn((feedback_dim, input_size + hidden), device=device) * 0.01,
        torch.empty((hidden, input_size), device=device).uniform_(-bound, bound),
        torch.empty((hidden, hidden), device=device).uniform_(-bound, bound),
        torch.empty((hidden,), device=device).uniform_(-bound, bound),
        torch.empty((hidden,), device=device).uniform_(-bound, bound),
    ]
    for tensor in tensors:
        tensor.requires_grad_(True)
    return tensors


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
) -> dict[str, float]:
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
    return result


def _capture_variant(
    fn: TensorFn,
    tensors: list[torch.Tensor],
    gate_tau: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Capture one synchronized forward/backward result for numerical validation."""

    _clear_gradients(tensors)
    output = _run_step(fn, tensors, gate_tau)
    torch.cuda.synchronize()
    gradients = [tensor.grad.detach().float().clone() for tensor in tensors]
    return output.detach().float().clone(), gradients


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff_rms = (actual - expected).square().mean().sqrt()
    expected_rms = expected.square().mean().sqrt().clamp_min(1e-12)
    return float((diff_rms / expected_rms).item())


def _gradient_relative_l2(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
) -> float:
    diff_sq = sum((a - e).double().square().sum() for a, e in zip(actual, expected))
    expected_sq = sum(e.double().square().sum() for e in expected)
    return float((diff_sq.sqrt() / expected_sq.sqrt().clamp_min(1e-12)).item())


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
    for name, fn in variants:
        timings[name] = _benchmark_variant(name, fn, tensors, args)

    captures = {
        name: _capture_variant(fn, tensors, args.gate_tau) for name, fn in variants
    }

    baseline_ms = timings["legacy_split_eager"]["cuda_event_ms"]
    for name, values in timings.items():
        values["speedup_vs_legacy"] = baseline_ms / values["cuda_event_ms"]
    baseline_output, baseline_gradients = captures["legacy_split_eager"]
    numerical_checks = {}
    for name, (output, gradients) in captures.items():
        numerical_checks[name] = {
            "output_max_abs_delta": float((output - baseline_output).abs().max().item()),
            "output_relative_rmse": _relative_rmse(output, baseline_output),
            "gradient_relative_l2": _gradient_relative_l2(gradients, baseline_gradients),
        }
    compiled_checks = numerical_checks["combined_compiled"]
    validation_passed = (
        compiled_checks["output_relative_rmse"] <= args.max_output_relative_rmse
        and compiled_checks["gradient_relative_l2"] <= args.max_gradient_relative_l2
    )
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
        "initialization": (
            "torch.nn.RNN-style recurrent parameters; GaWF U/V normal std=0.01; "
            "standard-normal inputs, hidden state, and feedback"
        ),
        "numerical_checks_vs_legacy": numerical_checks,
        "validation_thresholds": {
            "max_output_relative_rmse": args.max_output_relative_rmse,
            "max_gradient_relative_l2": args.max_gradient_relative_l2,
        },
        "validation_passed": validation_passed,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "gawf_feedback_benchmark_validated.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {output_path}")
    if not validation_passed:
        raise RuntimeError(f"compiled GaWF numerical validation failed: {compiled_checks}")


if __name__ == "__main__":
    main()
