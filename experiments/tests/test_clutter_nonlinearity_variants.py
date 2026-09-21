"""Unit checks for the Clutter nonlinearity-placement ablation variants."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from utils.training.recurrent_cores.gawf import GaWFCore
from utils.training.recurrent_cores.rnn import RNNCore
from utils.training.clutter.clutter_task_models import (
    GaWFNoTanhConv,
    GaWFNoWrapConv,
    GaWFRNNConv,
    GRUConv,
    GRUNoWrapConv,
    LSTMConv,
    LSTMNoWrapConv,
    RNNConv,
    RNNNoTanhConv,
    RNNNoWrapConv,
)

INPUT_SIZE = 1152


def _count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build(model_class, width: int):
    return model_class(
        10,
        9,
        kernel_size=5,
        hidden_size=width,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("base_class", "variant_class"),
    (
        (GaWFRNNConv, GaWFNoWrapConv),
        (RNNConv, RNNNoWrapConv),
        (GRUConv, GRUNoWrapConv),
        (LSTMConv, LSTMNoWrapConv),
    ),
)
def test_nowrap_variants_drop_exactly_the_layernorm_parameters(base_class, variant_class) -> None:
    width = 32
    base = _build(base_class, width)
    variant = _build(variant_class, width)

    assert _count(base) - _count(variant) == 2 * width
    assert base.core.norm is not None
    assert variant.core.norm is None
    assert variant.core.output_wrap == "none"


def test_gawf_state_is_raw_activation_and_readout_is_wrapped() -> None:
    torch.manual_seed(0)
    cores = {}
    for name, activation in (("default", "tanh"), ("identity", "identity")):
        core = GaWFCore(
            input_size=INPUT_SIZE,
            hidden_size=16,
            feedback_dim=19,
            dropout=0.0,
            rnn_activation=activation,
        ).eval()
        # Zero the gate transform so the multiplicative gate is the constant sigma(0) = 0.5.
        with torch.no_grad():
            core.V.zero_()
        cores[name] = core
    with torch.no_grad():
        cores["identity"].U.copy_(cores["default"].U)
        cores["identity"].rnn.load_state_dict(cores["default"].rnn.state_dict())
        cores["identity"].norm.load_state_dict(cores["default"].norm.state_dict())

    x = torch.randn(2, INPUT_SIZE)
    h_prev = torch.randn(2, 16)
    feedback = torch.randn(2, 19)

    with torch.no_grad():
        preactivation = 0.5 * F.linear(
            x, cores["default"].rnn.weight_ih_l0, None
        ) + 0.5 * F.linear(h_prev, cores["default"].rnn.weight_hh_l0, None)
        preactivation = preactivation + cores["default"].rnn.bias_ih_l0
        preactivation = preactivation + cores["default"].rnn.bias_hh_l0

        default_state = cores["default"].step(x, h_prev, feedback)
        identity_state = cores["identity"].step(x, h_prev, feedback)

    # The state is the raw in-recurrence activation of the gated preactivation ...
    assert torch.allclose(
        default_state, torch.tanh(preactivation), atol=1e-5
    )
    assert torch.allclose(
        identity_state, preactivation, atol=1e-5
    )
    assert not torch.allclose(default_state, identity_state, atol=1e-3)
    # ... and the external wrap is applied to the readout only.
    assert torch.allclose(
        cores["default"].project_readout(default_state),
        F.relu(cores["default"].norm(default_state)),
        atol=1e-6,
    )


def test_gated_core_with_unit_gate_reproduces_nn_rnn() -> None:
    torch.manual_seed(0)
    core = GaWFCore(input_size=16, hidden_size=8, feedback_dim=5, dropout=0.0).eval()
    reference = torch.nn.RNN(16, 8, num_layers=1, batch_first=True)
    with torch.no_grad():
        reference.load_state_dict(core.rnn.state_dict())
        # Saturate the sigmoid gate to exactly 1.0, i.e. "nn.RNN with the original weights".
        core.U.fill_(3.0)
        core.V.fill_(3.0)

    feedback = torch.ones(3, 5)
    x = torch.randn(3, 6, 16)
    with torch.no_grad():
        gate = torch.sigmoid(
            torch.matmul(
                core.U.unsqueeze(0) * feedback.unsqueeze(2).transpose(1, 2), core.V
            )
            / core.gate_tau
        )
        assert torch.equal(gate, torch.ones_like(gate))
        reference_out, _ = reference(x)
        state = core.initial_state(x.shape[0], x.device, x.dtype)
        states = []
        for step in range(x.shape[1]):
            state = core.step(x[:, step, :], state, feedback)
            states.append(state)
        gated_out = torch.stack(states, dim=1)
        readout = core.project_readout(gated_out)

    assert torch.allclose(gated_out, reference_out, atol=1e-5)
    assert torch.allclose(readout, F.relu(core.norm(reference_out)), atol=1e-5)


def test_legacy_core_still_wraps_inside_the_recurrence() -> None:
    from utils.training.recurrent_cores.gawf_legacy import GaWFCoreLegacy

    torch.manual_seed(0)
    legacy = GaWFCoreLegacy(
        input_size=8, hidden_size=4, feedback_dim=3, dropout=0.0
    ).eval()
    x = torch.randn(2, 8)
    h_prev = torch.randn(2, 4)
    feedback = torch.randn(2, 3)
    with torch.no_grad():
        state = legacy.step(x, h_prev, feedback)
        assert torch.equal(legacy.project_readout(state), state)
        gate = torch.sigmoid(
            torch.matmul(legacy.U.unsqueeze(0) * feedback.unsqueeze(2).transpose(1, 2), legacy.V)
            / legacy.gate_tau
        )
        pre = torch.einsum("bi,bhi,hi->bh", x, gate[..., :8], legacy.rnn.weight_ih_l0)
        pre = pre + torch.einsum(
            "bi,bhi,hi->bh", h_prev, gate[..., 8:], legacy.rnn.weight_hh_l0
        )
        pre = pre + legacy.rnn.bias_ih_l0 + legacy.rnn.bias_hh_l0
    assert torch.allclose(state, F.relu(legacy.norm(torch.tanh(pre))), atol=1e-6)


def test_rnn_notanh_recurrence_is_linear_and_default_is_not() -> None:
    torch.manual_seed(0)
    linear = RNNCore(INPUT_SIZE, 8, dropout=0.0, output_wrap="none", rnn_activation="identity")
    linear.eval()
    default = RNNCore(INPUT_SIZE, 8, dropout=0.0)
    default.eval()

    x1 = torch.randn(2, 5, INPUT_SIZE)
    x2 = torch.randn(2, 5, INPUT_SIZE)
    # The recurrence keeps its biases, so it is affine rather than strictly linear: the affine
    # identity holds exactly when the mixing weights sum to one.
    with torch.no_grad():
        left, _ = linear(0.7 * x1 + 0.3 * x2)
        right = 0.7 * linear(x1)[0] + 0.3 * linear(x2)[0]
        assert torch.allclose(left, right, atol=1e-5)

        default_left, _ = default(0.7 * x1 + 0.3 * x2)
        default_right = 0.7 * default(x1)[0] + 0.3 * default(x2)[0]
        assert not torch.allclose(default_left, default_right, atol=1e-3)


def test_default_rnn_core_keeps_the_ln_relu_dropout_wrap() -> None:
    torch.manual_seed(0)
    core = RNNCore(INPUT_SIZE, 8, dropout=0.0)
    core.eval()
    x = torch.randn(2, 4, INPUT_SIZE)

    with torch.no_grad():
        wrapped, _ = core(x)
        raw, _ = core.rnn(x)
        expected = F.relu(core.norm(raw))

    assert torch.allclose(wrapped, expected, atol=1e-6)


@pytest.mark.parametrize("variant_class", (GaWFNoWrapConv, RNNNoWrapConv, GaWFNoTanhConv, RNNNoTanhConv))
def test_variant_wrappers_run_on_cpu(variant_class) -> None:
    torch.manual_seed(0)
    model = variant_class(
        10,
        9,
        kernel_size=5,
        hidden_size=8,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
        device="cpu",
    )
    model.eval()
    x = torch.randn(2, 4, 2, 96, 96)
    with torch.no_grad():
        char, pos = model(x)
    assert char.shape == (2, 4, 10)
    assert pos.shape == (2, 4, 9)
    assert torch.isfinite(char).all() and torch.isfinite(pos).all()


def test_rnn_aligned_gawf_equals_legacy_gawf_once_the_wrap_is_disabled() -> None:
    """The two GaWF generations share one map; only the placement of the wrap differs."""

    from utils.training.recurrent_cores.gawf_legacy import GaWFCoreLegacy

    torch.manual_seed(0)
    aligned = GaWFCore(input_size=8, hidden_size=4, feedback_dim=3, dropout=0.0).eval()
    legacy = GaWFCoreLegacy(input_size=8, hidden_size=4, feedback_dim=3, dropout=0.0).eval()
    unwrapped = GaWFCoreLegacy(
        input_size=8, hidden_size=4, feedback_dim=3, dropout=0.0, output_wrap="none"
    ).eval()
    with torch.no_grad():
        legacy.load_state_dict(aligned.state_dict())
        unwrapped.load_state_dict(aligned.state_dict(), strict=False)

    x = torch.randn(2, 8)
    h_prev = torch.randn(2, 4)
    feedback = torch.randn(2, 3)
    with torch.no_grad():
        aligned_state = aligned.step(x, h_prev, feedback)
        unwrapped_state = unwrapped.step(x, h_prev, feedback)
        legacy_state = legacy.step(x, h_prev, feedback)
        aligned_readout = aligned.project_readout(aligned_state)

    # Same gated recurrence, bitwise.
    assert torch.equal(aligned_state, unwrapped_state)
    # Same readout function, bitwise.
    assert torch.equal(aligned_readout, legacy_state)
    # The only difference is which value is carried into the next step.
    assert not torch.equal(aligned_state, legacy_state)


def test_gated_step_equals_nn_rnn_with_modulated_weights() -> None:
    """Apart from the element-wise gate, GaWF is exactly nn.RNN."""

    torch.manual_seed(0)
    input_size, hidden_size, feedback_dim = 6, 4, 3
    core = GaWFCore(
        input_size=input_size,
        hidden_size=hidden_size,
        feedback_dim=feedback_dim,
        dropout=0.0,
    ).eval()
    reference = torch.nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
    with torch.no_grad():
        reference.load_state_dict(core.rnn.state_dict())
        # Only the gate parameters are extra; the external wrap is the shared readout contract.
        assert set(core.state_dict()) - {"rnn." + key for key in reference.state_dict()} == {
            "U",
            "V",
            "norm.weight",
            "norm.bias",
        }

    # Batch size one: the gate is input-dependent, so a single modulated weight matrix can only
    # stand in for one sample's gate.
    x = torch.randn(1, input_size)
    h_prev = torch.randn(1, hidden_size)
    feedback = torch.randn(1, feedback_dim)
    with torch.no_grad():
        logits = torch.matmul(
            core.U.unsqueeze(0) * feedback.clamp(-10, 10).unsqueeze(2).transpose(1, 2), core.V
        ) / core.gate_tau
        gate = torch.sigmoid(logits)[0]
        modulated = {key: value.clone() for key, value in core.rnn.state_dict().items()}
        modulated["weight_ih_l0"] = gate[:, :input_size] * modulated["weight_ih_l0"]
        modulated["weight_hh_l0"] = gate[:, input_size:] * modulated["weight_hh_l0"]
        gated_rnn = torch.nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        gated_rnn.load_state_dict(modulated)
        expected, _ = gated_rnn(x.unsqueeze(1), h_prev.unsqueeze(0))
        state = core.step(x, h_prev, feedback)

    assert torch.allclose(state, expected[:, 0], atol=1e-6)


def test_ungated_gawf_reproduces_nn_rnn_bitwise() -> None:
    """With the gate removed the trajectory is nn.RNN itself, not merely close to it."""

    torch.manual_seed(0)
    input_size, hidden_size, feedback_dim = 6, 4, 3
    core = GaWFCore(
        input_size=input_size,
        hidden_size=hidden_size,
        feedback_dim=feedback_dim,
        dropout=0.0,
    ).eval()
    reference = torch.nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
    with torch.no_grad():
        reference.load_state_dict(core.rnn.state_dict())

    x = torch.randn(2, 5, input_size)
    with torch.no_grad():
        reference_out, _ = reference(x)
        state = core.initial_state(x.shape[0], x.device, x.dtype)
        states = []
        for step in range(x.shape[1]):
            state = core.step_no_feedback(x[:, step, :], state)
            states.append(state)
        ungated = torch.stack(states, dim=1)
        readout, _ = core.forward_no_feedback(x)

    assert torch.equal(ungated, reference_out)
    assert torch.equal(readout, core.project_readout(reference_out))
