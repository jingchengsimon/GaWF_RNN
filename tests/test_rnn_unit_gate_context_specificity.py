"""Tests for exact LSTM/GRU unit-gate extraction and compact decomposition."""

from __future__ import annotations

import numpy as np
import torch

from utils_anal.rnn_unit_gate_context_specificity import (
    UnitGateAggregate,
    _summarize_gate,
    gru_unit_gates,
    lstm_unit_gates,
)


def test_lstm_manual_recurrence_matches_pytorch() -> None:
    """Manual LSTM gate equations must reproduce PyTorch hidden outputs."""

    torch.manual_seed(3)
    rnn = torch.nn.LSTM(7, 5, batch_first=True).double()
    encoded = torch.randn(4, 6, 7, dtype=torch.float64)
    _gates, manual = lstm_unit_gates(encoded, rnn)
    native, _state = rnn(encoded)
    assert torch.max(torch.abs(manual - native)).item() < 1e-12


def test_gru_manual_recurrence_matches_pytorch() -> None:
    """Manual GRU gate equations must reproduce PyTorch hidden outputs."""

    torch.manual_seed(5)
    rnn = torch.nn.GRU(7, 5, batch_first=True).double()
    encoded = torch.randn(4, 6, 7, dtype=torch.float64)
    _gates, manual = gru_unit_gates(encoded, rnn)
    native, _state = rnn(encoded)
    assert torch.max(torch.abs(manual - native)).item() < 1e-12


def test_balanced_variance_summaries_sum_to_100_percent() -> None:
    """Both decomposition views must account for their full variance totals."""

    rng = np.random.default_rng(11)
    equal_n = 8
    values = rng.normal(size=(90, equal_n, 4))
    aggregate = UnitGateAggregate(
        joint_sum=values.sum(axis=1),
        joint_sumsq=np.square(values).sum(axis=(0, 1)),
    )
    report, cell_mean = _summarize_gate(aggregate, equal_n)
    condition = report["equal_cell_condition_mean"]
    trial = report["equal_cell_trial_total"]
    assert cell_mean.shape == (9, 10, 4)
    assert np.isclose(sum(condition["fractions"].values()), 1.0)
    assert np.isclose(sum(trial["percent"].values()), 100.0)

