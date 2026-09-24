"""Regression checks for the six canonical recurrent/state-space definitions."""

from __future__ import annotations

import numpy as np
import torch

from utils.training.clutter.clutter_train_helpers import build_arg_parser, get_model_classes
from utils.training.clutter.clutter_task_models import (
    GaWFRNNConv,
    GRUConv,
    LSTMConv,
    MambaConv,
    RNNConv,
    S5Conv,
)
from utils.training.recurrent_cores.gawf import GaWFCore
from utils.training.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

_ = np.__version__  # Load NumPy before PyTorch in the macOS test environment.


def test_gawf_and_rnn_differ_only_by_feedback_gates() -> None:
    """Zero gate logits give 0.5 gates, matching a half-scaled ungated RNN."""

    torch.manual_seed(5)
    gawf = GaWFCore(3, 4, feedback_dim=2, dropout=0.0).eval()
    rnn = RNNCore(3, 4, dropout=0.0).eval()
    with torch.no_grad():
        gawf.U.zero_()
        gawf.V.zero_()
        rnn.rnn.weight_ih_l0.copy_(0.5 * gawf.rnn.weight_ih_l0)
        rnn.rnn.weight_hh_l0.copy_(0.5 * gawf.rnn.weight_hh_l0)
        rnn.rnn.bias_ih_l0.copy_(gawf.rnn.bias_ih_l0)
        rnn.rnn.bias_hh_l0.copy_(gawf.rnn.bias_hh_l0)
        rnn.norm.load_state_dict(gawf.norm.state_dict())

    x_t = torch.randn(2, 3)
    state = torch.randn(2, 4)
    feedback = torch.randn(2, 2)
    gawf_next = gawf(x_t, state, feedback)
    rnn_next, _state = rnn.step(x_t, state)
    torch.testing.assert_close(gawf_next, rnn_next)


def test_native_gru_and_lstm_have_no_external_wrap() -> None:
    """Canonical GRU/LSTM delegates directly to the corresponding PyTorch module."""

    x = torch.randn(2, 3, 5)
    for core_class in (GRUCore, LSTMCore):
        core = core_class(5, 7, dropout=0.0).eval()
        actual = core(x)
        expected = core.rnn(x)
        torch.testing.assert_close(actual[0], expected[0])
        assert core.output_wrap == "none"


def test_public_clutter_registry_contains_only_six_models() -> None:
    """Historical ablations are not public model choices after archival."""

    classes = get_model_classes(
        RNNConv, LSTMConv, GRUConv, GaWFRNNConv, MambaConv, S5Conv
    )
    assert tuple(classes) == ("rnn", "lstm", "gru", "gawf", "mamba", "s5")
    choices = next(
        action.choices
        for action in build_arg_parser()._actions
        if action.dest == "model_types"
    )
    assert choices == ["rnn", "lstm", "gru", "gawf", "mamba", "s5"]
