"""Regression and protocol checks for Clutter non-multiplicative feedback controls."""

from __future__ import annotations

from typing import Type

import pytest
import torch

from utils.training.clutter.clutter_task_models import (
    GaWFAdditiveConv,
    GaWFRNNConv,
    GRUFeedbackConv,
    LSTMFeedbackConv,
    RNNFeedbackConv,
)
from utils.training.recurrent_cores.additive_feedback import (
    AdditiveFeedbackRNNCore,
    ConcatenatedFeedbackCellCore,
)
from utils.training.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore


@pytest.mark.parametrize(
    ("model_class", "hidden_size", "expected_count"),
    [
        (GaWFRNNConv, 256, 586_067),
        (GaWFAdditiveConv, 271, 585_401),
        (RNNFeedbackConv, 272, 586_867),
        (GRUFeedbackConv, 103, 584_562),
        (LSTMFeedbackConv, 79, 585_406),
    ],
)
def test_formal_parameter_counts(
    model_class: Type[torch.nn.Module],
    hidden_size: int,
    expected_count: int,
) -> None:
    model = model_class(
        10,
        9,
        kernel_size=5,
        hidden_size=hidden_size,
        rnn_dropout=0.5,
        device="cpu",
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count


@pytest.mark.parametrize(
    "model_class",
    [GaWFAdditiveConv, RNNFeedbackConv, GRUFeedbackConv, LSTMFeedbackConv],
)
def test_feedback_controls_reuse_gawf_feedback_function(
    model_class: Type[torch.nn.Module],
) -> None:
    assert model_class._compute_feedback is GaWFRNNConv._compute_feedback


@pytest.mark.parametrize(
    "model_class",
    [GaWFAdditiveConv, RNNFeedbackConv, GRUFeedbackConv, LSTMFeedbackConv],
)
def test_feedback_is_zero_then_detached_and_nonzero(
    model_class: Type[torch.nn.Module],
) -> None:
    torch.manual_seed(7)
    model = model_class(10, 9, kernel_size=5, hidden_size=8, rnn_dropout=0.0, device="cpu")
    seen_feedback: list[torch.Tensor] = []
    original_step = model.core.step

    def traced_step(x_t: torch.Tensor, state, feedback: torch.Tensor):
        seen_feedback.append(feedback)
        return original_step(x_t, state, feedback)

    model.core.step = traced_step
    inputs = torch.randn(2, 4, 2, 96, 96)
    char_out, pos_out = model(inputs, reset_feedback=True)
    (char_out.square().mean() + pos_out.square().mean()).backward()

    assert torch.count_nonzero(seen_feedback[0]).item() == 0
    assert all(not feedback.requires_grad for feedback in seen_feedback)
    assert any(torch.count_nonzero(feedback).item() > 0 for feedback in seen_feedback[1:])


@pytest.mark.parametrize(
    ("sequence_core_class", "cell_type"),
    [(RNNCore, "rnn"), (GRUCore, "gru"), (LSTMCore, "lstm")],
)
def test_zero_feedback_cell_matches_open_loop_sequence_core(
    sequence_core_class: Type[torch.nn.Module],
    cell_type: str,
) -> None:
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
    with torch.no_grad():
        feedback_core.cell.weight_ih[:, :input_size].copy_(sequence_core.rnn.weight_ih_l0)
        feedback_core.cell.weight_ih[:, input_size:].zero_()
        feedback_core.cell.weight_hh.copy_(sequence_core.rnn.weight_hh_l0)
        feedback_core.cell.bias_ih.copy_(sequence_core.rnn.bias_ih_l0)
        feedback_core.cell.bias_hh.copy_(sequence_core.rnn.bias_hh_l0)
        feedback_core.norm.load_state_dict(sequence_core.norm.state_dict())

    inputs = torch.randn(3, 6, input_size)
    expected, _ = sequence_core(inputs)
    actual, _ = feedback_core.forward_no_feedback(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_additive_initial_weights_match_zero_feedback_gawf_scale() -> None:
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
    assert torch.count_nonzero(scaled.feedback_linear.bias).item() == 0
