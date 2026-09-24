"""Checks that every GaWF core in the project follows the RNN-aligned contract."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.training.recurrent_cores.additive_feedback import (
    AdditiveFeedbackRNNCore,
    ConcatenatedFeedbackCellCore,
)
from utils.training.recurrent_cores.gawf import GaWFCore
from utils.training.clutter.clutter_task_models import MultiLayerGaWFRNNConv


def _saturate(core: GaWFCore) -> None:
    """Force every gate of a core to exactly one so the gated step equals the ungated step."""

    with torch.no_grad():
        gate_params = (
            [(core.U, core.V)]
            if core.num_layers == 1
            else list(zip(core.U_layers, core.V_layers))
        )
        for u_param, v_param in gate_params:
            u_param.fill_(3.0)
            v_param.fill_(3.0)


def _assert_unit_gates(core: GaWFCore, feedbacks: list[torch.Tensor]) -> None:
    """Assert that the configured gates are exactly one for the supplied feedback vectors."""

    with torch.no_grad():
        for layer_idx, feedback in enumerate(feedbacks):
            u_param, v_param = core._layer_gate_params(layer_idx)
            logits = torch.matmul(
                u_param.unsqueeze(0) * feedback.unsqueeze(2).transpose(1, 2), v_param
            ) / core.gate_tau
            assert torch.equal(torch.sigmoid(logits), torch.ones_like(logits))


def test_multilayer_gated_core_matches_stacked_nn_rnn_with_unit_gates() -> None:
    """Two aligned GaWF layers with unit gates equal two stacked nn.RNN layers plus the wrap."""

    torch.manual_seed(0)
    input_size, hidden_size, feedback_dim, num_layers = 6, 4, 3, 2
    core = GaWFCore(
        input_size=input_size,
        hidden_size=hidden_size,
        feedback_dim=feedback_dim,
        dropout=0.0,
        num_layers=num_layers,
        layer_feedback_dims=[hidden_size, feedback_dim],
    ).eval()
    _saturate(core)

    x = torch.randn(2, 5, input_size)
    feedbacks = [torch.ones(2, hidden_size), torch.ones(2, feedback_dim)]
    _assert_unit_gates(core, feedbacks)

    with torch.no_grad():
        states = core.initial_state(2, x.device, x.dtype)
        steps = []
        for time_idx in range(x.shape[1]):
            top, states = core.step(x[:, time_idx, :], states, feedbacks)
            steps.append(top)
        gated = torch.stack(steps, dim=1)

        reference = x
        for layer_idx in range(num_layers):
            layer_rnn = nn.RNN(core._layer_rnn(layer_idx).input_size, hidden_size, batch_first=True)
            layer_rnn.load_state_dict(core._layer_rnn(layer_idx).state_dict())
            layer_out, _ = layer_rnn(reference)
            reference = F.relu(core._layer_norm(layer_idx)(layer_out))

    assert torch.allclose(gated, reference, atol=2e-5)
    # Layer states are raw activations, not the wrapped readouts.
    assert not torch.allclose(states[0], F.relu(core._layer_norm(0)(states[0])), atol=1e-6)


def test_multilayer_wrapper_uses_the_aligned_core() -> None:
    """The multi-layer clutter wrapper builds the aligned core and stays finite on CPU."""

    torch.manual_seed(0)
    model = MultiLayerGaWFRNNConv(
        10,
        9,
        hidden_size=6,
        num_layers=2,
        kernel_size=5,
        device="cpu",
        rnn_dropout=0.0,
    ).eval()
    assert model.gawf_core == "rnn_aligned"
    assert model.core.num_layers == 2
    with torch.no_grad():
        char, pos = model(torch.randn(2, 4, 2, 96, 96))
    assert char.shape == (2, 4, 10) and pos.shape == (2, 4, 9)
    assert torch.isfinite(char).all() and torch.isfinite(pos).all()


def test_additive_core_carries_raw_state_and_wraps_the_readout() -> None:
    """The additive feedback control now feeds back tanh(preactivation) and wraps the readout."""

    torch.manual_seed(0)
    core = AdditiveFeedbackRNNCore(5, 4, 3, dropout=0.0).eval()
    x = torch.randn(2, 5)
    state = torch.randn(2, 4)
    feedback = torch.randn(2, 3)
    with torch.no_grad():
        readout, next_state = core.step(x, state, feedback)
        preactivation = F.linear(x, core.rnn.weight_ih_l0, core.rnn.bias_ih_l0)
        preactivation = preactivation + F.linear(
            state, core.rnn.weight_hh_l0, core.rnn.bias_hh_l0
        )
        preactivation = preactivation + core.feedback_linear(feedback.clamp(-10, 10))
        expected_state = torch.tanh(preactivation)

    assert torch.allclose(next_state, expected_state, atol=1e-6)
    assert torch.allclose(readout, F.relu(core.norm(expected_state)), atol=1e-6)
    assert not torch.allclose(readout, next_state, atol=1e-6)


def test_additive_and_concatenated_controls_share_one_function() -> None:
    """With matched weights the two feedback controls differ only by parameterization."""

    torch.manual_seed(0)
    hidden_size, input_size, feedback_dim = 6, 4, 3
    additive = AdditiveFeedbackRNNCore(input_size, hidden_size, feedback_dim, dropout=0.0).eval()
    concatenated = ConcatenatedFeedbackCellCore(
        input_size, hidden_size, feedback_dim, cell_type="rnn", dropout=0.0
    ).eval()
    with torch.no_grad():
        concatenated.cell.weight_ih[:, :input_size].copy_(additive.rnn.weight_ih_l0)
        concatenated.cell.weight_ih[:, input_size:].copy_(additive.feedback_linear.weight)
        concatenated.cell.weight_hh.copy_(additive.rnn.weight_hh_l0)
        concatenated.cell.bias_ih.copy_(additive.rnn.bias_ih_l0 + additive.feedback_linear.bias)
        concatenated.cell.bias_hh.copy_(additive.rnn.bias_hh_l0)
        concatenated.norm.load_state_dict(additive.norm.state_dict())

    x = torch.randn(2, input_size)
    state = torch.randn(2, hidden_size)
    feedback = torch.randn(2, feedback_dim)
    with torch.no_grad():
        additive_readout, additive_state = additive.step(x, state, feedback)
        concat_readout, concat_state = concatenated.step(x, state, feedback)

    assert torch.allclose(additive_state, concat_state, atol=1e-6)
    assert torch.allclose(additive_readout, concat_readout, atol=1e-6)


def test_additive_legacy_semantics_remain_available_for_old_checkpoints() -> None:
    """The additive core can still reproduce the pre-alignment in-loop-wrap behavior."""

    torch.manual_seed(0)
    legacy = AdditiveFeedbackRNNCore(
        5, 4, 3, dropout=0.0, state_semantics="legacy"
    ).eval()
    aligned = AdditiveFeedbackRNNCore(
        5, 4, 3, dropout=0.0, state_semantics="aligned"
    ).eval()
    with torch.no_grad():
        aligned.load_state_dict(legacy.state_dict())
    x = torch.randn(2, 5)
    state = torch.randn(2, 4)
    feedback = torch.randn(2, 3)
    with torch.no_grad():
        legacy_readout, legacy_state = legacy.step(x, state, feedback)
        aligned_readout, aligned_state = aligned.step(x, state, feedback)

    assert torch.allclose(legacy_readout, aligned_readout, atol=1e-6)
    assert torch.equal(legacy_state, legacy_readout)
    assert not torch.allclose(aligned_state, aligned_readout, atol=1e-6)
