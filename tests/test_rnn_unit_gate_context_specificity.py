"""Tests for exact GaWF/LSTM/GRU unit-gate extraction and compact decomposition."""

from __future__ import annotations

import numpy as np
import torch

from utils_anal.rnn_unit_gate_context_specificity import (
    UnitGateAggregate,
    _accumulate_destination_unit_gate,
    _accumulate_reconstructed_gawf_unit_gates,
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


def test_gawf_destination_unit_gate_averages_incoming_synapses(tmp_path) -> None:
    """GaWF projection must reduce only the incoming-source axis before pooling SS."""

    hidden_size = 2
    source_size = 3
    equal_n = 2
    labels = np.asarray(
        [
            (digit, sector)
            for sector in range(9)
            for digit in range(10)
            for _repeat in range(equal_n)
        ],
        dtype=np.int64,
    )
    gates = np.arange(
        labels.shape[0] * hidden_size * source_size, dtype=np.float32
    ).reshape(labels.shape[0], hidden_size, source_size)
    gate_path = tmp_path / "gate.npy"
    np.save(gate_path, gates)
    aggregate = _accumulate_destination_unit_gate(
        gate_path,
        hidden_size=hidden_size,
        source_size=source_size,
        reference_labels=labels,
        equal_joint_mask=np.ones(labels.shape[0], dtype=bool),
        batch_size=17,
    )
    expected = gates.mean(axis=2, dtype=np.float64).reshape(90, equal_n, hidden_size)
    np.testing.assert_allclose(aggregate.joint_sum, expected.sum(axis=1))
    np.testing.assert_allclose(
        aggregate.joint_sumsq,
        np.square(expected).sum(axis=(0, 1)),
    )


def test_reconstructed_gawf_projection_matches_explicit_float32_gates(tmp_path) -> None:
    """Compact feedback/U/V reconstruction must match explicit destination means."""

    rng = np.random.default_rng(23)
    hidden_size = 2
    input_size = 3
    feedback_size = 19
    equal_n = 2
    labels = np.asarray(
        [
            (digit, sector)
            for sector in range(9)
            for digit in range(10)
            for _repeat in range(equal_n)
        ],
        dtype=np.int64,
    )
    feedback = rng.normal(size=(labels.shape[0], feedback_size)).astype(np.float32)
    u = rng.normal(size=(hidden_size, feedback_size)).astype(np.float32)
    v = rng.normal(size=(feedback_size, input_size + hidden_size)).astype(np.float32)
    trajectory_path = tmp_path / "trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        feedback=feedback,
        labels=labels,
        U=u,
        V=v,
        weight_ih=np.zeros((hidden_size, input_size), dtype=np.float32),
        weight_hh=np.zeros((hidden_size, hidden_size), dtype=np.float32),
    )
    aggregates, _ih_shape, _hh_shape = _accumulate_reconstructed_gawf_unit_gates(
        trajectory_path,
        reference_labels=labels,
        equal_joint_mask=np.ones(labels.shape[0], dtype=bool),
        batch_size=17,
        gate_tau=0.5,
        device=torch.device("cpu"),
    )
    feedback_tensor = torch.from_numpy(feedback)
    scaled_u = torch.from_numpy(u).unsqueeze(0) * feedback_tensor.clamp(-10, 10).unsqueeze(1)
    gates = torch.sigmoid(torch.matmul(scaled_u, torch.from_numpy(v)) / 0.5).numpy()
    expected = {
        "input_mean": gates[..., :input_size].mean(axis=2, dtype=np.float64),
        "recurrent_mean": gates[..., input_size:].mean(axis=2, dtype=np.float64),
    }
    for gate_name, values in expected.items():
        cells = values.reshape(90, equal_n, hidden_size)
        np.testing.assert_allclose(
            aggregates[gate_name].joint_sum,
            cells.sum(axis=1),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            aggregates[gate_name].joint_sumsq,
            np.square(cells).sum(axis=(0, 1)),
            rtol=1e-6,
            atol=1e-8,
        )
