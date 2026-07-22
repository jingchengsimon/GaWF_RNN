"""Tests for the shared balanced streaming variance decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from utils_anal.variance_decomposition import (
    StreamingMoments,
    balanced_subsample_indices,
    decompose_array,
    decompose_repeated_blocks,
    unbalanced_condition_mean_bridge,
)


def _synthetic(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(
        [(digit, sector) for sector in range(9) for digit in range(10) for _ in range(5)],
        dtype=np.int64,
    )
    sector = labels[:, 1:2].astype(np.float64)
    digit = labels[:, 0:1].astype(np.float64)
    noise = rng.normal(scale=0.3, size=(labels.shape[0], 3))
    values = np.concatenate(
        [sector + noise[:, :1], digit + noise[:, 1:2], sector * digit + noise[:, 2:3]],
        axis=1,
    )
    return values, labels


def test_four_cells_and_consistency_checks_hold() -> None:
    values, labels = _synthetic()
    result = decompose_array(values, labels, batch_size=17)
    assert sum(result.aggregate_cm.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(result.aggregate_trial.values()) == pytest.approx(1.0, abs=1e-12)
    assert result.aggregate_cm["sector"] > result.aggregate_cm["digit"]
    assert result.consistency["aggregate_weighted_per_unit_max_abs_deviation"] < 1e-12
    assert (
        result.consistency["condition_mean_trial_renormalization_max_abs_deviation_from_one"]
        < 1e-12
    )
    assert result.consistency["zero_sum_max_abs_violation"] < 1e-12


def test_centering_each_unit_does_not_change_any_fraction() -> None:
    values, labels = _synthetic()
    raw = decompose_array(values, labels)
    centered = decompose_array(values - values.mean(axis=0), labels)
    assert raw.aggregate_cm == pytest.approx(centered.aggregate_cm, abs=1e-12)
    assert raw.aggregate_trial == pytest.approx(centered.aggregate_trial, abs=1e-12)


def test_balanced_draws_use_common_minimum_and_fixed_seed() -> None:
    values, labels = _synthetic()
    extra = np.repeat(labels[:1], 3, axis=0)
    labels = np.concatenate([labels, extra], axis=0)
    first, report = balanced_subsample_indices(labels, repeats=3, seed=91)
    second, _ = balanced_subsample_indices(labels, repeats=3, seed=91)
    assert report.n_per_cell == 5
    assert report.trials_retained_per_draw == 450
    assert report.trials_discarded_per_draw == 3
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    for indices in first:
        counts = np.bincount(labels[indices, 1] * 10 + labels[indices, 0], minlength=90)
        assert np.all(counts == 5)


def test_streaming_accumulator_rejects_unbalanced_input() -> None:
    values, labels = _synthetic()
    accumulator = StreamingMoments(values.shape[1])
    accumulator.update(values[:-1], labels[:-1])
    with pytest.raises(RuntimeError, match="Balanced decomposition"):
        accumulator.finalize()


def test_memory_budget_fails_before_accumulator_allocation() -> None:
    with pytest.raises(MemoryError, match="memory budget"):
        StreamingMoments(1_179_648, memory_budget_bytes=100_000)


def test_repeated_block_path_matches_individual_decompositions() -> None:
    values, labels = _synthetic()
    draws, _ = balanced_subsample_indices(labels, repeats=2, seed=12)

    def read_block(indices: np.ndarray, unit_slice: slice) -> np.ndarray:
        return values[indices, unit_slice]

    repeated = decompose_repeated_blocks(
        read_block,
        labels,
        draws,
        num_units=values.shape[1],
        unit_block_size=2,
        trial_batch_size=19,
    )
    for repeat, indices in enumerate(draws):
        expected = decompose_array(values, labels, selected_indices=indices, batch_size=19)
        for factor, value in expected.aggregate_cm.items():
            assert repeated.aggregate_cm[factor][repeat] == pytest.approx(value, abs=1e-12)
        for factor, value in expected.aggregate_trial.items():
            assert repeated.aggregate_trial[factor][repeat] == pytest.approx(value, abs=1e-12)
        for factor in expected.per_unit_cm:
            assert repeated.per_unit_cm[factor][repeat] == pytest.approx(
                expected.per_unit_cm[factor], abs=1e-7
            )


def test_unbalanced_bridge_is_explicitly_not_canonical() -> None:
    values, labels = _synthetic()
    values = np.concatenate([values, values[:3]], axis=0)
    labels = np.concatenate([labels, labels[:3]], axis=0)
    accumulator = StreamingMoments(values.shape[1])
    accumulator.update(values, labels)
    bridge = unbalanced_condition_mean_bridge(
        accumulator.total_sum, accumulator.cell_sum, accumulator.cell_count
    )
    assert set(bridge) == {"sector", "digit", "interaction", "sum"}
    assert np.isfinite(list(bridge.values())).all()


def test_unbalanced_bridge_matches_historical_condition_mean_formula() -> None:
    values, labels = _synthetic()
    values = np.concatenate([values, values[:3]], axis=0)
    labels = np.concatenate([labels, labels[:3]], axis=0)
    accumulator = StreamingMoments(values.shape[1])
    accumulator.update(values, labels)
    actual = unbalanced_condition_mean_bridge(
        accumulator.total_sum, accumulator.cell_sum, accumulator.cell_count
    )

    means = accumulator.cell_sum / accumulator.cell_count[:, None]
    historical = means.reshape(9, 10, -1).transpose(2, 1, 0)
    grand = np.nanmean(historical, axis=(1, 2))
    digit = np.nanmean(historical, axis=2) - grand[:, None]
    sector = np.nanmean(historical, axis=1) - grand[:, None]
    digit_bc = np.broadcast_to(digit[:, :, None], historical.shape)
    sector_bc = np.broadcast_to(sector[:, None, :], historical.shape)
    interaction = historical - grand[:, None, None] - digit_bc - sector_bc
    total = np.square(historical - grand[:, None, None]).sum()
    expected = {
        "digit": np.square(digit_bc).sum() / total,
        "sector": np.square(sector_bc).sum() / total,
        "interaction": np.square(interaction).sum() / total,
    }
    assert actual == pytest.approx({**expected, "sum": sum(expected.values())}, abs=1e-12)
