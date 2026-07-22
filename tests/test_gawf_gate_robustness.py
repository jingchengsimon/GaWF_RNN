"""Regression tests for corrected GaWF gate robustness reductions."""

from __future__ import annotations

import numpy as np

from utils_anal.gawf_gate_robustness import (
    _crossings,
    _decomposition,
    _group_deltas,
)


def test_group_deltas_use_trial_weighted_per_synapse_grand() -> None:
    counts = np.arange(1, 91, dtype=np.int64).reshape(9, 10)
    group_value = np.arange(90, dtype=np.float64).reshape(9, 10, 1)
    sums = group_value * counts[..., None]
    result = _group_deltas(sums.reshape(90, 1), counts)
    grand = float(sums.sum() / counts.sum())
    expected_sector = sums.sum(axis=1)[:, 0] / counts.sum(axis=1) - grand
    expected_digit = sums.sum(axis=0)[:, 0] / counts.sum(axis=0) - grand
    np.testing.assert_allclose(result["sector"][:, 0], expected_sector)
    np.testing.assert_allclose(result["digit"][:, 0], expected_digit)


def test_two_way_decomposition_recovers_pure_main_effects() -> None:
    sector = np.arange(9, dtype=np.float64)[:, None, None]
    digit = 2.0 * np.arange(10, dtype=np.float64)[None, :, None]
    result = _decomposition(sector + digit)
    assert result["fractions"]["interaction"] < 1e-12
    assert np.isclose(sum(result["fractions"].values()), 1.0)


def test_crossing_is_linearly_interpolated() -> None:
    x = np.asarray([0.0, 1.0, 2.0])
    first = np.asarray([0.0, 0.75, 1.0])
    second = np.asarray([1.0, 0.25, 0.0])
    np.testing.assert_allclose(_crossings(x, first, second), [2.0 / 3.0])
