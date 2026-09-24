"""Dependency-free protocol checks for the Clutter feedback-control models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Type

import torch

from experiments.clutter.feedback_control_sanity import MODEL_SPECS
from utils.training.clutter.clutter_task_models import GaWFRNNConv
from utils.training.recurrent_cores.additive_feedback import (
    AdditiveFeedbackRNNCore,
    ConcatenatedFeedbackCellCore,
)
from utils.training.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore


def parse_args() -> argparse.Namespace:
    """Parse output path for the protocol-check record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _check_parameter_counts() -> dict[str, int]:
    counts: dict[str, int] = {"gawf": 586_067}
    gawf = GaWFRNNConv(10, 9, kernel_size=5, hidden_size=256, device="cpu")
    actual_gawf = sum(parameter.numel() for parameter in gawf.parameters())
    if actual_gawf != counts["gawf"]:
        raise RuntimeError(f"GaWF parameter count {actual_gawf} != {counts['gawf']}")
    for name, model_class, width, _lr, _wd, expected_count in MODEL_SPECS:
        model = model_class(10, 9, kernel_size=5, hidden_size=width, device="cpu")
        actual = sum(parameter.numel() for parameter in model.parameters())
        if actual != expected_count:
            raise RuntimeError(f"{name} parameter count {actual} != {expected_count}")
        if model_class._compute_feedback is not GaWFRNNConv._compute_feedback:
            raise RuntimeError(f"{name} does not reuse GaWFRNNConv._compute_feedback")
        counts[name] = actual
    return counts


def _check_feedback_runtime() -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for name, model_class, _width, _lr, _wd, _expected_count in MODEL_SPECS:
        torch.manual_seed(7)
        model = model_class(
            10,
            9,
            kernel_size=5,
            hidden_size=8,
            rnn_dropout=0.0,
            device="cpu",
        )
        seen_feedback: list[torch.Tensor] = []
        original_step = model.core.step

        def traced_step(x_t: torch.Tensor, state, feedback: torch.Tensor):
            seen_feedback.append(feedback)
            return original_step(x_t, state, feedback)

        model.core.step = traced_step
        inputs = torch.randn(2, 4, 2, 96, 96)
        char_out, pos_out = model(inputs, reset_feedback=True)
        (char_out.square().mean() + pos_out.square().mean()).backward()
        first_zero = torch.count_nonzero(seen_feedback[0]).item() == 0
        later_nonzero = any(
            torch.count_nonzero(feedback).item() > 0 for feedback in seen_feedback[1:]
        )
        all_detached = all(not feedback.requires_grad for feedback in seen_feedback)
        if not (first_zero and later_nonzero and all_detached):
            raise RuntimeError(
                f"{name} feedback protocol failed: first_zero={first_zero}, "
                f"later_nonzero={later_nonzero}, all_detached={all_detached}"
            )
        reports[name] = {
            "first_feedback_exactly_zero": first_zero,
            "later_feedback_nonzero": later_nonzero,
            "all_feedback_requires_grad_false": all_detached,
        }
    return reports


def _copy_open_loop_weights(
    sequence_core: torch.nn.Module,
    feedback_core: ConcatenatedFeedbackCellCore,
    input_size: int,
) -> None:
    with torch.no_grad():
        feedback_core.cell.weight_ih[:, :input_size].copy_(sequence_core.rnn.weight_ih_l0)
        feedback_core.cell.weight_ih[:, input_size:].zero_()
        feedback_core.cell.weight_hh.copy_(sequence_core.rnn.weight_hh_l0)
        feedback_core.cell.bias_ih.copy_(sequence_core.rnn.bias_ih_l0)
        feedback_core.cell.bias_hh.copy_(sequence_core.rnn.bias_hh_l0)
        feedback_core.norm.load_state_dict(sequence_core.norm.state_dict())


def _check_open_loop_equivalence() -> dict[str, float]:
    reports: dict[str, float] = {}
    specs: tuple[tuple[str, Type[torch.nn.Module]], ...] = (
        ("rnn", RNNCore),
        ("gru", GRUCore),
        ("lstm", LSTMCore),
    )
    for cell_type, sequence_core_class in specs:
        torch.manual_seed(11)
        input_size, hidden_size, feedback_dim = 7, 5, 3
        sequence_core = sequence_core_class(input_size, hidden_size, dropout=0.0)
        feedback_core = ConcatenatedFeedbackCellCore(
            input_size,
            hidden_size,
            feedback_dim,
            cell_type=cell_type,
            dropout=0.0,
        )
        _copy_open_loop_weights(sequence_core, feedback_core, input_size)
        inputs = torch.randn(3, 6, input_size)
        expected, _ = sequence_core(inputs)
        actual, _ = feedback_core.forward_no_feedback(inputs)
        max_abs_error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        reports[cell_type] = max_abs_error
    return reports


def _check_additive_initialization() -> dict[str, object]:
    torch.manual_seed(19)
    unscaled = AdditiveFeedbackRNNCore(7, 5, 3, initial_weight_scale=1.0)
    torch.manual_seed(19)
    scaled = AdditiveFeedbackRNNCore(7, 5, 3, initial_weight_scale=0.5)
    torch.testing.assert_close(
        scaled.rnn.weight_ih_l0,
        0.5 * unscaled.rnn.weight_ih_l0,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scaled.rnn.weight_hh_l0,
        0.5 * unscaled.rnn.weight_hh_l0,
        rtol=0,
        atol=0,
    )
    bias_zero = torch.count_nonzero(scaled.feedback_linear.bias).item() == 0
    if not bias_zero:
        raise RuntimeError("Additive feedback bias must initialize to zero")
    return {"recurrent_weight_scale": 0.5, "feedback_bias_zero": bias_zero}


def main() -> None:
    """Run all checks and write a machine-readable evidence record."""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol-check output: {args.output}")
    report = {
        "parameter_counts": _check_parameter_counts(),
        "feedback_runtime": _check_feedback_runtime(),
        "zero_feedback_open_loop_max_abs_error": _check_open_loop_equivalence(),
        "additive_initialization": _check_additive_initialization(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
