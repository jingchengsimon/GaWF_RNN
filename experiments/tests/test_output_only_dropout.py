"""Verify output dropout without directly masking single-layer recurrent states."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from utils.training.clutter.clutter_task_models import GaWFRNNConv
from utils.training.recurrent_cores.gawf import GaWFCore
from utils.training.recurrent_cores.mamba import MambaCore
from utils.training.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore
from utils.training.recurrent_cores.s5 import S5Core

_ = np.__version__  # Keep macOS NumPy/OpenMP import order stable.


@pytest.fixture(autouse=True)
def deterministic_cpu() -> Iterator[None]:
    """Use a fixed seed and deterministic CPU kernels for each check."""

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(31)
    yield
    torch.use_deterministic_algorithms(previous)


def _paired(core_type: type[nn.Module], *args: int) -> tuple[nn.Module, nn.Module]:
    dry = core_type(*args, dropout=0.0)
    dropped = core_type(*args, dropout=0.5)
    dropped.load_state_dict(dry.state_dict())
    return dry, dropped


@pytest.mark.parametrize("core_type", [RNNCore, GRUCore, LSTMCore])
def test_single_layer_recurrent_state_is_clean(core_type: type[nn.Module]) -> None:
    """The masked readout changes while h (and c for LSTM) stays bitwise equal."""

    dry, dropped = _paired(core_type, 5, 12)
    x = torch.randn(3, 7, 5)
    dry.train()
    dropped.train()
    if core_type is RNNCore:
        dry_states = []
        dropped_states = []
        dry_state = dry.initial_state(3, "cpu", x.dtype)
        drop_state = dropped.initial_state(3, "cpu", x.dtype)
        for time_idx in range(x.size(1)):
            dry_out, dry_state = dry.step(x[:, time_idx], dry_state)
            drop_out, drop_state = dropped.step(x[:, time_idx], drop_state)
            assert torch.equal(dry_state, drop_state)
            dry_states.append(dry_state)
            dropped_states.append(drop_state)
            if time_idx == 0:
                assert not torch.equal(dry_out, drop_out)
    else:
        dry_state = None
        drop_state = None
        for time_idx in range(x.size(1)):
            dry_out, dry_state = dry(x[:, time_idx : time_idx + 1], dry_state)
            drop_out, drop_state = dropped(x[:, time_idx : time_idx + 1], drop_state)
            if core_type is LSTMCore:
                assert all(torch.equal(a, b) for a, b in zip(dry_state, drop_state))
            else:
                assert torch.equal(dry_state, drop_state)
            if time_idx == 0:
                assert not torch.equal(dry_out, drop_out)
    dry.eval()
    dropped.eval()
    assert torch.equal(dry(x)[0], dropped(x)[0])


def test_gawf_clean_state_with_fixed_external_feedback() -> None:
    """Fix task feedback so this isolates direct recurrent-state masking."""

    dry, dropped = _paired(GaWFCore, 5, 12, 4)
    x = torch.randn(3, 7, 5)
    feedback = torch.randn(3, 4)
    dry_state = dry.initial_state(3, "cpu", x.dtype)
    drop_state = dropped.initial_state(3, "cpu", x.dtype)
    dry.train()
    dropped.train()
    for time_idx in range(x.size(1)):
        dry_out, dry_state = dry.step_with_state(x[:, time_idx], dry_state, feedback)
        drop_out, drop_state = dropped.step_with_state(x[:, time_idx], drop_state, feedback)
        assert torch.equal(dry_state, drop_state)
        if time_idx == 0:
            assert not torch.equal(dry_out, drop_out)
    dry.eval()
    dropped.eval()
    state = dry.initial_state(3, "cpu", x.dtype)
    assert torch.equal(
        dry.step_with_state(x[:, 0], state, feedback)[0],
        dropped.step_with_state(x[:, 0], state, feedback)[0],
    )


def test_clutter_gawf_carries_clean_state_and_uses_output_logits_feedback() -> None:
    """The Clutter loop carries clean h while preserving the requested logits feedback."""

    model = GaWFRNNConv(10, 9, device="cpu", hidden_size=8, rnn_dropout=0.5)
    encoded = torch.randn(2, 4, model.encoder_flatten_size)
    recorded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original = model.core.step_with_state

    def record_step(
        x_t: torch.Tensor, state: torch.Tensor, feedback: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_state = original(x_t, state, feedback)
        recorded.append((state.detach().clone(), feedback.detach().clone(), next_state))
        return output, next_state

    with patch.object(model, "encode_frames", return_value=encoded), patch.object(
        model.core, "step_with_state", side_effect=record_step
    ):
        chars, positions = model(torch.empty(0))
    for time_idx in range(1, encoded.size(1)):
        assert torch.equal(recorded[time_idx][0], recorded[time_idx - 1][2])
        expected_feedback = torch.cat(
            [chars[:, time_idx - 1], positions[:, time_idx - 1]], dim=-1
        )
        assert torch.equal(recorded[time_idx][1], expected_feedback)


@pytest.mark.parametrize("core_type", [GRUCore, LSTMCore])
def test_stacked_native_layers_drop_every_output(core_type: type[nn.Module]) -> None:
    """Explicit dropout includes the last layer; native interlayer dropout is disabled."""

    core = core_type(5, 8, dropout=0.5, num_layers=2)
    seen: list[int] = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, _output, index=index: seen.append(index)
        )
        for index, layer in enumerate(core.output_dropouts)
    ]
    try:
        core(torch.randn(2, 4, 5))
        assert seen == [0, 1]
        assert all(layer.dropout == 0.0 for layer in core.rnns)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    ("core_type", "native_type"),
    [(GRUCore, nn.GRU), (LSTMCore, nn.LSTM)],
)
def test_stacked_native_path_is_unchanged_at_zero_dropout(
    core_type: type[nn.Module], native_type: type[nn.Module]
) -> None:
    """Splitting native layers retains initialization and exact p=0 computation."""

    torch.manual_seed(3)
    native = native_type(5, 8, num_layers=2, batch_first=True, dropout=0).eval()
    torch.manual_seed(3)
    core = core_type(5, 8, num_layers=2, dropout=0).eval()
    for index, layer in enumerate(core.rnns):
        for name, value in layer.named_parameters():
            assert torch.equal(value, getattr(native, name.replace("l0", f"l{index}")))
    x = torch.randn(2, 4, 5)
    output, state = core(x)
    expected_output, expected_state = native(x)
    assert torch.equal(output, expected_output)
    if core_type is LSTMCore:
        assert all(torch.equal(a, b) for a, b in zip(state, expected_state))
    else:
        assert torch.equal(state, expected_state)


def test_s5_branch_dropout_preserves_native_state() -> None:
    """Read the real S5 layer's returned state and its undropped branch output."""

    dry, dropped = _paired(S5Core, 5, 8, 4)
    x = torch.randn(2, 6, 5)
    dry.train()
    dropped.train()
    for time_idx in range(1, x.size(1) + 1):
        _, state_dry = dry.layers[0](dry.input_proj(x[:, :time_idx]), return_state=True)
        _, state_drop = dropped.layers[0](
            dropped.input_proj(x[:, :time_idx]), return_state=True
        )
        assert torch.equal(state_dry, state_drop)
    native_dry = dry.layers[0](dry.input_proj(x))
    native_drop = dropped.layers[0](dropped.input_proj(x))
    assert torch.equal(native_dry, native_drop)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    handle = dropped.output_dropouts[0].register_forward_hook(
        lambda _module, inputs, output: seen.append((inputs[0], output))
    )
    dropped_output = dropped(x)[0]
    handle.remove()
    assert torch.equal(seen[0][0], native_drop)
    assert torch.equal(dropped_output, dropped.input_proj(x) + seen[0][1])
    assert not torch.equal(dry(x)[0], dropped_output)
    dry.eval()
    dropped.eval()
    assert torch.equal(dry(x)[0], dropped(x)[0])


class _FakeMamba(nn.Module):
    """CPU mixer standing in for the unavailable mamba_ssm package."""

    def __init__(self, d_model: int, **_kwargs: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(x)


def test_mamba_wrapper_drops_only_mixer_branch() -> None:
    """Forward hooks compare the mixer output before the wrapper's dropout."""

    with patch("utils.training.recurrent_cores.mamba._mamba_class", return_value=_FakeMamba):
        dry, dropped = _paired(MambaCore, 5, 8)
    x = torch.randn(2, 6, 5)
    branches: list[torch.Tensor] = []
    handles = [
        core.layers[0].register_forward_hook(
            lambda _module, _inputs, output: branches.append(output.detach().clone())
        )
        for core in (dry, dropped)
    ]
    dropped_branch: list[torch.Tensor] = []
    handles.append(
        dropped.output_dropouts[0].register_forward_hook(
            lambda _module, _inputs, output: dropped_branch.append(output)
        )
    )
    try:
        dry.train()
        dropped.train()
        dropped_output = dropped(x)[0]
        assert not torch.equal(dry(x)[0], dropped_output)
        assert torch.equal(branches[0], branches[1])
        assert torch.equal(dropped_output, dropped.input_proj(x) + dropped_branch[0])
        dry.eval()
        dropped.eval()
        assert torch.equal(dry(x)[0], dropped(x)[0])
    finally:
        for handle in handles:
            handle.remove()
