"""Protocol checks for open-loop dynamic/input-conditioned recurrent baselines."""

from __future__ import annotations

from typing import Callable

import pytest
import torch
import torch.nn.functional as F

from experiments.clutter.dynamic_weight_param_match import find_matches
from utils.analysis.model_train_single_result import parse_hparams_from_filename
from utils.training.clutter.clutter_task_models import (
    BRIMsConv,
    GaWFRNNConv,
    HyperLSTMConv,
    MLSTMConv,
)
from utils.training.clutter.clutter_train_helpers import build_arg_parser
from utils.training.recurrent_cores import BRIMsCore, HyperLSTMCore, LSTMCore, MLSTMCore


def _core_factories() -> list[Callable[[], torch.nn.Module]]:
    return [
        lambda: MLSTMCore(12, 12, dropout=0.0),
        lambda: HyperLSTMCore(12, 12, hyper_hidden_size=4, dropout=0.0),
        lambda: BRIMsCore(12, 12, attention_dropout=0.0, dropout=0.0),
    ]


@pytest.mark.parametrize("factory", _core_factories())
def test_forward_shape_matches_lstm(factory: Callable[[], torch.nn.Module]) -> None:
    inputs = torch.randn(3, 5, 12)
    expected, _ = LSTMCore(12, 12, dropout=0.0)(inputs)
    actual, _ = factory()(inputs)
    assert actual.shape == expected.shape == (3, 5, 12)


@pytest.mark.parametrize("factory", _core_factories())
def test_none_state_resets_and_explicit_state_continues(
    factory: Callable[[], torch.nn.Module],
) -> None:
    torch.manual_seed(17)
    core = factory().eval()
    inputs = torch.randn(2, 4, 12)
    reset_first, state = core(inputs)
    reset_second, _ = core(inputs)
    continued, _ = core(inputs, state)
    torch.testing.assert_close(reset_first, reset_second)
    assert not torch.allclose(reset_first, continued)


def test_mlstm_reduces_to_lstm_when_input_multiplier_is_one() -> None:
    torch.manual_seed(23)
    input_size, hidden_size = 5, 7
    core = MLSTMCore(input_size, hidden_size, dropout=0.0)
    with torch.no_grad():
        core.mx.weight.fill_(1.0 / input_size)

    x_t = torch.ones(3, input_size)
    h_prev = torch.randn(3, hidden_size)
    c_prev = torch.randn(3, hidden_size)
    actual_h, actual_c = core.cell_step(x_t, h_prev, c_prev)

    multiplier = core.mx(x_t)
    torch.testing.assert_close(multiplier, torch.ones_like(multiplier))
    effective_weight_hh = core.m_gates.weight @ core.mh.weight
    i, f, g, o = (core.x_gates(x_t) + F.linear(h_prev, effective_weight_hh)).chunk(4, dim=-1)
    expected_c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
    expected_h = torch.sigmoid(o) * torch.tanh(expected_c)
    torch.testing.assert_close(actual_c, expected_c)
    torch.testing.assert_close(actual_h, expected_h)


@pytest.mark.parametrize(
    ("model_class", "kwargs", "expected_core", "expected_total"),
    [
        (GaWFRNNConv, {"hidden_size": 256}, 393_088, 586_067),
        (MLSTMConv, {"hidden_size": 65}, 395_915, 585_265),
        (
            HyperLSTMConv,
            {"hidden_size": 70, "hyper_hidden_size": 10, "hyper_embedding_size": 4},
            396_572,
            586_017,
        ),
        (BRIMsConv, {"hidden_size": 84}, 386_532, 576_243),
    ],
)
def test_formal_parameter_counts(
    model_class: type[torch.nn.Module],
    kwargs: dict[str, int],
    expected_core: int,
    expected_total: int,
) -> None:
    model = model_class(10, 9, kernel_size=5, device="cpu", **kwargs)
    assert sum(parameter.numel() for parameter in model.core.parameters()) == expected_core
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_total


def test_parameter_search_reproduces_formal_choices() -> None:
    matches = {match.model: match for match in find_matches()}
    assert matches["mLSTM"].architecture == "H=65"
    assert matches["HyperLSTM"].architecture == "H=70, hyper_H=10, n_z=4"
    assert matches["BRIMs"].architecture == "H=84, blocks=(6,3), topk=(4,2)"


def test_brims_keeps_official_mnist_structure() -> None:
    core = BRIMsCore(12, 12)
    assert core.num_layers == 2
    assert core.num_blocks == (6, 3)
    assert core.topk == (4, 2)


def test_cli_and_checkpoint_names_record_dynamic_architecture() -> None:
    args = build_arg_parser().parse_args(
        [
            "--model_types",
            "mlstm",
            "hyperlstm",
            "brims",
            "--hidden_sizes",
            "70",
            "--hyper_hidden_sizes",
            "10",
        ]
    )
    assert args.model_types == ["mlstm", "hyperlstm", "brims"]
    assert args.hyper_hidden_sizes == [10]
    parsed = parse_hparams_from_filename(
        "hyperlstm_sector_acc_h70_hh10_nz4_lr0.001_wd0.001_cdo0.0_rdo0.5_model.pth"
    )
    assert parsed["model_type"] == "HyperLSTM"
    assert parsed["hidden_size"] == 70
    assert parsed["hyper_hidden_size"] == 10
    assert parsed["hyper_embedding_size"] == 4


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (MLSTMConv, {"hidden_size": 12}),
        (
            HyperLSTMConv,
            {"hidden_size": 12, "hyper_hidden_size": 4, "hyper_embedding_size": 4},
        ),
        (BRIMsConv, {"hidden_size": 12}),
    ],
)
def test_clutter_adapters_keep_encoder_and_head_shapes(
    model_class: type[torch.nn.Module],
    kwargs: dict[str, int],
) -> None:
    model = model_class(
        10,
        9,
        kernel_size=5,
        device="cpu",
        rnn_dropout=0.0,
        **kwargs,
    ).eval()
    char_logits, pos_logits = model(torch.randn(2, 3, 2, 96, 96))
    assert char_logits.shape == (2, 3, 10)
    assert pos_logits.shape == (2, 3, 9)
