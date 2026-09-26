"""Check additive feedback equations, clean recurrent state, and public Clutter wiring."""

from __future__ import annotations

import numpy as np
import torch

from utils.training.clutter.clutter_train_helpers import build_arg_parser, get_model_classes
from utils.training.clutter.clutter_task_models import (
    GaWFRNNConv,
    GRUAdditiveFeedbackConv,
    GRUConv,
    LSTMAdditiveFeedbackConv,
    LSTMConv,
    MambaConv,
    RNNAdditiveFeedbackConv,
    RNNConv,
    S5Conv,
)
from utils.training.recurrent_cores.additive_feedback import AdditiveFeedbackCellCore
from utils.training.recurrent_cores.rnn import RNNCore

_ = np.__version__  # Initialize NumPy before PyTorch in the macOS test environment.


def _states_equal(left: object, right: object) -> bool:
    if isinstance(left, tuple):
        return all(torch.equal(a, b) for a, b in zip(left, right))
    return torch.equal(left, right)


def test_additive_feedback_dropout_preserves_clean_state() -> None:
    """The same exogenous feedback gives exactly the same recurrence for p=0 and p=.5."""

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(23)
    for cell_type in ("rnn", "gru", "lstm"):
        clean = AdditiveFeedbackCellCore(5, 7, 3, cell_type, dropout=0.0)
        dropped = AdditiveFeedbackCellCore(5, 7, 3, cell_type, dropout=0.5)
        dropped.load_state_dict(clean.state_dict())
        clean.train()
        dropped.train()
        state_clean = clean.initial_state(4, "cpu", torch.float32)
        state_dropped = dropped.initial_state(4, "cpu", torch.float32)
        changed = False
        for _time_idx in range(4):
            x_t = torch.randn(4, 5)
            feedback = torch.randn(4, 3)
            output_clean, state_clean = clean.step(x_t, state_clean, feedback)
            output_dropped, state_dropped = dropped.step(x_t, state_dropped, feedback)
            assert _states_equal(state_clean, state_dropped)
            changed |= not torch.equal(output_clean, output_dropped)
        assert changed
        clean.eval()
        dropped.eval()
        output_clean, state_clean = clean.step(x_t, state_clean, feedback)
        output_dropped, state_dropped = dropped.step(x_t, state_dropped, feedback)
        assert torch.equal(output_clean, output_dropped)
        assert _states_equal(state_clean, state_dropped)


def test_rnn_feedback_matches_gawf_aligned_rnn_when_affine_zero() -> None:
    """With zero feedback affine, RNN-FB uses the canonical in-loop LN/ReLU equations."""

    torch.manual_seed(29)
    plain = RNNCore(5, 7, dropout=0.0)
    feedback_core = AdditiveFeedbackCellCore(5, 7, 3, "rnn", dropout=0.0)
    with torch.no_grad():
        for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(feedback_core.cell, suffix).copy_(
                getattr(plain.rnn, f"{suffix}_l0")
            )
        feedback_core.norm.load_state_dict(plain.norm.state_dict())
        feedback_core.feedback_linear.weight.zero_()
        feedback_core.feedback_linear.bias.zero_()
    x_t = torch.randn(3, 5)
    h = torch.randn(3, 7)
    expected_output, expected_state = plain.step(x_t, h)
    actual_output, actual_state = feedback_core.step(x_t, h, torch.randn(3, 3))
    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state.squeeze(0))
    assert feedback_core.rnn_activation == "relu"


def test_feedback_affine_bias_and_native_gru_lstm_equations() -> None:
    """Bias acts at zero feedback; zero affine recovers native GRU/LSTM cells."""

    torch.manual_seed(31)
    x_t = torch.randn(2, 5)
    for cell_type in ("gru", "lstm"):
        core = AdditiveFeedbackCellCore(5, 7, 3, cell_type, dropout=0.0).eval()
        state = core.initial_state(2, "cpu", torch.float32)
        with torch.no_grad():
            core.feedback_linear.weight.zero_()
            core.feedback_linear.bias.zero_()
        output, next_state = core.step(x_t, state, torch.zeros(2, 3))
        expected = core.cell(x_t, state)
        assert _states_equal(next_state, expected)
        assert torch.equal(output, expected[0] if cell_type == "lstm" else expected)
        with torch.no_grad():
            core.feedback_linear.bias.fill_(0.5)
        _output_with_bias, state_with_bias = core.step(x_t, state, torch.zeros(2, 3))
        assert not _states_equal(state_with_bias, expected)


def test_additive_feedback_models_are_trainable_choices() -> None:
    """Public parser, registry, and task wrapper expose all three additive controls."""

    choices = next(
        action.choices for action in build_arg_parser()._actions if action.dest == "model_types"
    )
    expected_keys = ("rnn_fb_add", "gru_fb_add", "lstm_fb_add")
    assert all(key in choices for key in expected_keys)
    classes = get_model_classes(
        RNNConv, LSTMConv, GRUConv, GaWFRNNConv, MambaConv, S5Conv,
        RNNAdditiveFeedbackConv, GRUAdditiveFeedbackConv, LSTMAdditiveFeedbackConv,
    )
    for key, expected_class in zip(
        expected_keys,
        (RNNAdditiveFeedbackConv, GRUAdditiveFeedbackConv, LSTMAdditiveFeedbackConv),
    ):
        assert classes[key] is expected_class
        model = expected_class(10, 9, device="cpu", hidden_size=8, rnn_dropout=0.5)
        assert model.feedback_dim == 19
        assert model.core.output_dropout.p == 0.5
        assert "prev_feedback" not in model.state_dict()


def test_additive_feedback_wrapper_runs_detached_logit_loop() -> None:
    """A two-frame Clutter forward produces heads and caches detached task feedback."""

    torch.manual_seed(37)
    model = RNNAdditiveFeedbackConv(10, 9, device="cpu", hidden_size=8, rnn_dropout=0.5)
    model.train()
    frames = torch.randn(1, 2, 2, 96, 96)
    char_logits, pos_logits = model(frames, reset_feedback=True)
    assert char_logits.shape == (1, 2, 10)
    assert pos_logits.shape == (1, 2, 9)
    expected_feedback = torch.cat((char_logits[:, -1], pos_logits[:, -1]), dim=-1)
    assert torch.equal(model.prev_feedback, expected_feedback.detach())
    assert not model.prev_feedback.requires_grad
    model.reset_sequence_state()
    assert model.prev_feedback is None


def test_additive_feedback_checkpoint_uses_canonical_analysis_loader(tmp_path) -> None:
    """New additive checkpoint stems resolve to the active wrappers."""

    from utils.analysis.anal_helpers import build_model_from_ckpt

    model = RNNAdditiveFeedbackConv(
        10, 9, kernel_size=5, device="cpu", hidden_size=8, rnn_dropout=0.5
    )
    checkpoint = tmp_path / "rnn_fb_add_sector_h8_lr0.001_wd0.00001_cdo0.0_rdo0.5.pth"
    torch.save(model.state_dict(), checkpoint)
    loaded = build_model_from_ckpt(str(checkpoint), num_pos=9, device=torch.device("cpu"))
    assert isinstance(loaded, RNNAdditiveFeedbackConv)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
