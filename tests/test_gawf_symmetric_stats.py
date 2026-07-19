"""Synthetic checks for symmetric GaWF selectivity, relevance, and timing statistics."""

from __future__ import annotations

import numpy as np

from utils_anal.gawf_symmetric_stats import (
    architecture_axis_variance,
    benjamini_hochberg,
    bootstrap_d,
    first_crossing,
    first_negative_to_nonnegative,
    interaction_dominant,
    joint_design,
    paired_lead_test,
    relevance_masks,
    trial_relevance_moments,
    two_way_decomposition,
)


def _balanced_design(repeats: int = 5) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    values = []
    rng = np.random.default_rng(7)
    for sector in range(9):
        for digit in range(10):
            for _ in range(repeats):
                labels.append([digit, sector])
                values.append(
                    [
                        3.0 * sector + rng.normal(0, 0.02),
                        2.0 * digit + rng.normal(0, 0.02),
                        4.0 * ((sector + digit) % 2) + rng.normal(0, 0.02),
                    ]
                )
    return np.asarray(values, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def test_two_way_decomposition_recovers_known_effects() -> None:
    values, labels = _balanced_design()
    result = two_way_decomposition(values, labels)
    assert result.eta_sector[0] > 0.99
    assert result.eta_digit[1] > 0.99
    assert result.eta_interaction[2] > result.eta_sector[2]
    assert result.eta_interaction[2] > result.eta_digit[2]
    assert interaction_dominant(result).tolist() == [False, False, True]
    total = result.eta_sector + result.eta_digit + result.eta_interaction + result.eta_residual
    np.testing.assert_allclose(total, 1.0, atol=1e-8)
    assert result.tuning_sector.shape == (9, 3)
    assert result.tuning_digit.shape == (10, 3)


def test_joint_design_and_bh_fdr() -> None:
    _values, labels = _balanced_design(repeats=2)
    design = joint_design(labels)
    assert design["n_trials"] == 180
    assert design["independent_at_alpha_0_05"]
    np.testing.assert_array_equal(
        np.asarray(design["joint_frequency_sector_rows_digit_columns"]),
        np.full((9, 10), 2),
    )
    passed, q_values = benjamini_hochberg(np.asarray([0.001, 0.02, 0.8]))
    assert passed.tolist() == [True, True, False]
    assert np.all((q_values >= 0) & (q_values <= 1))


def test_architecture_axis_variance_uses_both_axes() -> None:
    values, labels = _balanced_design(repeats=2)
    base = two_way_decomposition(values, labels)
    spatial = np.tile(np.arange(36, dtype=np.float64), 32).reshape(32, 6, 6)
    channel = np.arange(32, dtype=np.float64)[:, None, None] * np.ones((1, 6, 6))
    expanded = type(base)(
        eta_sector=spatial.reshape(-1),
        eta_digit=channel.reshape(-1),
        eta_interaction=np.zeros(1152),
        eta_residual=np.zeros(1152),
        tuning_sector=np.zeros((9, 1152)),
        tuning_digit=np.zeros((10, 1152)),
    )
    audit = architecture_axis_variance(expanded)
    assert audit["sector_spatial_to_channel_variance_ratio"] == float("inf")
    assert audit["digit_spatial_to_channel_variance_ratio"] == 0.0


def test_relevance_effect_bootstrap_is_positive() -> None:
    rng = np.random.default_rng(11)
    tuning = np.vstack([np.arange(10), -np.arange(10)]).astype(np.float64)
    eligible = np.ones(10, dtype=bool)
    masks = relevance_masks(tuning, eligible, 0.2)
    contexts = np.repeat([0, 1], 40)
    gates = rng.normal(0.2, 0.02, size=(80, 10))
    for trial, context in enumerate(contexts):
        gates[trial, masks[context]] += 0.5
    moments = trial_relevance_moments(gates, contexts, masks, eligible)
    point, draws = bootstrap_d(moments, resamples=100, seed=4)
    assert point > 2.0
    assert np.quantile(draws, 0.025) > 0


def test_first_crossing_and_paired_lead_direction() -> None:
    gate = np.asarray([[-1.0, 0.2, 0.5], [-0.5, -0.1, 0.3]])
    readout = np.asarray([[0, 0, 1], [0, 0, 1]])
    gate_frame = first_crossing(gate)
    readout_frame = first_crossing(readout, threshold=1.0)
    np.testing.assert_array_equal(gate_frame, np.asarray([2.0, 3.0]))
    paired = paired_lead_test(gate_frame, readout_frame)
    assert paired["difference_definition"].startswith("readout_frame_minus_gate_frame")
    assert paired["mean_difference"] == 0.5
    assert paired["fraction_gate_leads"] == 0.5


def test_directional_crossing_rejects_initial_positive_value() -> None:
    values = np.asarray(
        [
            [-1.0, -0.2, 0.3, 0.5],
            [0.2, -0.4, -0.1, 0.1],
            [0.3, 0.2, 0.1, 0.4],
            [-0.2, -0.1, -0.3, -0.4],
        ]
    )
    crossing = first_negative_to_nonnegative(values)
    np.testing.assert_allclose(crossing[:2], [3.0, 4.0])
    assert np.isnan(crossing[2:]).all()
