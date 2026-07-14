"""Tests for foreground-switch-aligned population-activity trajectories."""
from __future__ import annotations

import numpy as np

from utils_anal.pop_act_switch_trajectory import (
    extract_trials,
    fit_shared_pca,
    select_switch_events,
)


def test_select_switch_events_applies_next_fg_and_bg_filters() -> None:
    """Eligible trials have 20 stable post frames; filtered trials have no bg switch."""

    fg = np.zeros(100, dtype=np.int64)
    fg[[10, 35, 50, 80]] = 1
    bg = np.zeros(100, dtype=np.int64)
    bg[29] = 1
    selected = select_switch_events(fg, bg, pre_frames=8, post_frames=20)

    np.testing.assert_array_equal(selected["eligible_unfiltered"], [10, 50])
    np.testing.assert_array_equal(selected["eligible_bg_filtered"], [50])
    assert int(selected["rejected_short_next_fg"][0]) == 1
    assert int(selected["rejected_bg_in_window"][0]) == 1


def test_extract_trials_uses_half_open_offsets() -> None:
    """A [-2, 3) window extracts five ordered timepoints around each event."""

    pop = np.arange(40, dtype=np.float32).reshape(20, 2)
    offsets = np.arange(-2, 3, dtype=np.int64)
    trials = extract_trials(pop, np.asarray([5, 10], dtype=np.int64), offsets)
    assert trials.shape == (2, 5, 2)
    np.testing.assert_array_equal(trials[0], pop[3:8])
    np.testing.assert_array_equal(trials[1], pop[8:13])


def test_shared_pca_uses_one_basis_for_both_mean_trajectories() -> None:
    """Both versions produce 3D coordinates under one deterministic component basis."""

    time = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    filtered = np.stack([time, time**2, np.sin(time), np.cos(time)], axis=1)
    unfiltered = filtered + np.asarray([0.1, -0.2, 0.05, 0.0], dtype=np.float32)
    result = fit_shared_pca(filtered, unfiltered, n_components=3)
    assert result["coords_bg_filtered"].shape == (12, 3)
    assert result["coords_unfiltered"].shape == (12, 3)
    assert result["components"].shape == (3, 4)
    assert result["explained_variance_ratio"].shape == (3,)
    np.testing.assert_allclose(
        result["components"] @ result["components"].T,
        np.eye(3),
        atol=1e-5,
    )
