"""Regression tests for group-mean GaWF gate deviations."""

from __future__ import annotations

import numpy as np

from utils_anal.gawf_gate_distribution import _group_mean_delta


def test_group_mean_delta_uses_trial_weighted_per_synapse_grand_mean() -> None:
    values = np.asarray(
        [
            [[0.0, 2.0], [4.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float64,
    )
    counts = np.asarray([2, 1], dtype=np.int64)
    group_sums = values * counts[:, None, None]

    means, grand, delta = _group_mean_delta(group_sums, counts)

    expected_grand = (values[0] * 2 + values[1]) / 3
    np.testing.assert_allclose(means, values)
    np.testing.assert_allclose(grand, expected_grand)
    np.testing.assert_allclose(delta, values - expected_grand)
    np.testing.assert_allclose(
        (delta * counts[:, None, None]).sum(axis=0),
        0.0,
        atol=1e-12,
    )


def test_distinct_groupings_produce_distinct_group_mean_deltas() -> None:
    trial_values = np.arange(24, dtype=np.float64).reshape(6, 2, 2)
    sector = np.asarray([0, 0, 0, 1, 1, 1])
    digit = np.asarray([0, 1, 2, 0, 1, 2])

    def aggregate(labels: np.ndarray, levels: int) -> np.ndarray:
        sums = np.stack([trial_values[labels == level].sum(axis=0) for level in range(levels)])
        counts = np.bincount(labels, minlength=levels)
        return _group_mean_delta(sums, counts)[2]

    sector_delta = aggregate(sector, 2)
    digit_delta = aggregate(digit, 3)

    assert sector_delta.shape == (2, 2, 2)
    assert digit_delta.shape == (3, 2, 2)
    assert not np.array_equal(sector_delta, digit_delta)
