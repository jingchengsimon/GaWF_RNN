"""Regression checks for equal-n Sector/Digit pattern selection."""

from __future__ import annotations

import numpy as np

from utils.analysis.clutter.fig6_encoder_sector_patterns import (
    CONDITIONS,
    ENCODER_SHAPE,
    _equal_n_condition_mask,
    _load_patterns,
    _spatial_activation_limits,
)


def test_equal_n_condition_mask_balances_all_digit_values() -> None:
    """The generic selector retains the minimum count for every requested class value."""

    labels = np.repeat(np.arange(10), np.arange(1, 11))
    selected, target, original_counts = _equal_n_condition_mask(labels, 10, seed=7)

    assert target == 1
    np.testing.assert_array_equal(original_counts, np.arange(1, 11))
    np.testing.assert_array_equal(np.bincount(labels[selected], minlength=10), np.ones(10))


def test_figure6_condition_grids_match_the_requested_combined_layouts() -> None:
    """Sector and Digit keep their prescribed internal spatial and channel grids."""

    assert CONDITIONS["sector"].spatial_grid == (3, 3)
    assert CONDITIONS["sector"].channel_grid == (2, 5)
    assert CONDITIONS["digit"].spatial_grid == (2, 5)
    assert CONDITIONS["digit"].channel_grid == (3, 4)


def test_spatial_plot_uses_the_spatial_activation_range_only() -> None:
    """The standalone grid must not inherit the combined plot's channel scale."""

    maps = np.linspace(0.02, 0.18, 9 * 6 * 6, dtype=np.float32).reshape(9, 6, 6)

    assert np.allclose(_spatial_activation_limits(maps, CONDITIONS["sector"]), (0.02, 0.18))


def test_load_patterns_accepts_refresh_seed_directories(tmp_path) -> None:
    """The no-tanh refresh collector's ``seedNN`` layout must aggregate directly."""

    config = CONDITIONS["sector"]
    expected = []
    for seed in range(1, 11):
        patterns = np.full((config.count, *ENCODER_SHAPE), seed, dtype=np.float32)
        expected.append(patterns)
        destination = tmp_path / f"seed{seed:02d}"
        destination.mkdir()
        np.savez_compressed(destination / "encoder_sector_patterns.npz", patterns=patterns)

    np.testing.assert_array_equal(_load_patterns(tmp_path, config), np.stack(expected))
