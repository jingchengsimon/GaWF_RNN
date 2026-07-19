"""Tests for saved tensor sources used by the unified decomposition runner."""

from __future__ import annotations

import gc
import weakref

import numpy as np

from utils_anal.run_unified_variance_decomposition import ArraySource, WeightedSource, _summary_only
from utils_anal.variance_decomposition import RepeatedDecomposition


def test_effective_source_multiplies_saved_gate_by_matching_static_weight(tmp_path) -> None:
    gates = np.arange(3 * 12, dtype=np.float32).reshape(3, 12) / 10.0
    gate_path = tmp_path / "saved_gate.npy"
    np.save(gate_path, gates)
    gate = ArraySource(gate_path, num_units=12, num_trials=3)
    weights = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    effective = WeightedSource(gate, weights)
    trials = np.asarray([0, 2], dtype=np.int64)
    unit_slice = slice(3, 9)
    expected = gates[trials, unit_slice] * weights[unit_slice]
    np.testing.assert_allclose(effective.read(trials, unit_slice), expected)


def test_array_source_accepts_saved_spatial_representation(tmp_path) -> None:
    values = np.arange(2 * 3 * 32 * 6 * 6, dtype=np.float32).reshape(2, 3, 32, 6, 6)
    path = tmp_path / "encoder.npy"
    np.save(path, values)
    source = ArraySource(path, num_units=1152, num_trials=6)
    actual = source.read(np.asarray([1, 5]), slice(1148, 1152))
    expected = values.reshape(6, 1152)[[1, 5], 1148:1152]
    np.testing.assert_array_equal(actual, expected)


def test_summary_copy_does_not_retain_large_per_unit_arrays() -> None:
    heavy = np.ones((2, 1000), dtype=np.float32)
    reference = weakref.ref(heavy)
    result = RepeatedDecomposition(
        aggregate_cm={"sector": np.ones(2)},
        aggregate_trial={"sector": np.ones(2)},
        per_unit_cm={"sector": heavy},
        per_unit_trial={"sector": heavy},
        unweighted_per_unit_mean_cm={"sector": np.ones(2)},
        unweighted_per_unit_mean_trial={"sector": np.ones(2)},
        consistency={"check": np.zeros(2)},
    )
    summary = _summary_only(result)
    del result, heavy
    gc.collect()
    assert reference() is None
    assert summary.per_unit_cm == {}
    assert summary.per_unit_trial == {}
