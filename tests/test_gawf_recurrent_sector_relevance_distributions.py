"""Tests for recurrent sector relevance-distribution analysis and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_anal.gawf_recurrent_sector_relevance_distributions import (
    accumulate_context_group_distributions,
    accumulate_sector_group_distributions,
    summarize_group_moments,
)
import utils_viz.gawf_recurrent_sector_relevance_distributions as distribution_viz


def test_distribution_accumulator_separates_sector_specific_masks() -> None:
    sectors = np.repeat(np.arange(9, dtype=np.int64), 2)
    values = np.tile(
        np.asarray([[0.1, 0.3, 0.7, 0.85], [0.1, 0.3, 0.7, 0.95]]),
        (9, 1),
    ).astype(np.float32)
    relevant = np.zeros((9, 4), dtype=bool)
    relevant[:, 3] = True
    eligible = np.ones(4, dtype=bool)
    chunks = ((0, 8, values[:8]), (8, 18, values[8:]))
    hist, sums, sums_sq, counts = accumulate_sector_group_distributions(
        chunks,
        sectors,
        relevant,
        eligible,
        np.asarray([0.0, 0.5, 1.0]),
    )
    np.testing.assert_array_equal(counts[:, 0], 2)
    np.testing.assert_array_equal(counts[:, 1], 6)
    np.testing.assert_array_equal(hist[:, 0], np.asarray([[0, 2]] * 9))
    means, _stds, sector_d, global_d = summarize_group_moments(sums, sums_sq, counts)
    np.testing.assert_allclose(means[:, 0], 0.9)
    np.testing.assert_allclose(means[:, 1], 1.1 / 3.0)
    assert np.all(sector_d > 0)
    assert global_d > 0


def test_sector_plot_has_two_groups_open_spines_and_stable_filename(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def capture(figure, path: Path, dpi: int) -> None:
        captured.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(distribution_viz, "_save", capture)
    destination = distribution_viz.plot_sector_distribution(
        3,
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([[2, 8], [7, 3]]),
        np.asarray([0.72, 0.41]),
        np.asarray([10, 10]),
        -0.65,
        24,
        215,
        tmp_path,
        180,
        density_limit=2.0,
    )
    axis = captured["figure"].axes[0]
    assert destination.name == "recurrent_sector_3_top10_vs_remaining_distribution.png"
    assert captured["path"] == destination
    assert captured["dpi"] == 180
    assert len(axis.patches) == 2
    assert len(axis.lines) == 2
    assert axis.get_xlim() == (0.0, 1.0)
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    distribution_viz.plt.close(captured["figure"])


def test_context_accumulator_supports_ten_digit_levels() -> None:
    contexts = np.repeat(np.arange(10, dtype=np.int64), 2)
    values = np.tile(np.asarray([[0.2, 0.8], [0.3, 0.9]], dtype=np.float32), (10, 1))
    relevant = np.zeros((10, 2), dtype=bool)
    relevant[:, 1] = True
    hist, sums, sums_sq, counts = accumulate_context_group_distributions(
        ((0, 20, values),),
        contexts,
        relevant,
        np.ones(2, dtype=bool),
        np.asarray([0.0, 0.5, 1.0]),
    )
    assert hist.shape == (10, 2, 2)
    np.testing.assert_array_equal(counts, np.full((10, 2), 2))
    means, _stds, context_d, global_d = summarize_group_moments(sums, sums_sq, counts)
    np.testing.assert_allclose(means[:, 0], 0.85)
    np.testing.assert_allclose(means[:, 1], 0.25)
    assert np.all(context_d > 0)
    assert global_d > 0


def test_generic_plot_uses_gate_factor_and_context_in_labels(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def capture(figure, path: Path, dpi: int) -> None:
        captured.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(distribution_viz, "_save", capture)
    destination = distribution_viz.plot_context_distribution(
        "input",
        "digit",
        7,
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([[2, 8], [7, 3]]),
        np.asarray([0.72, 0.41]),
        np.asarray([10, 10]),
        0.35,
        44,
        393,
        tmp_path,
        150,
        density_limit=2.0,
    )
    axis = captured["figure"].axes[0]
    assert destination.name == "input_digit_7_top10_vs_remaining_distribution.png"
    assert "Digit 7" in axis.get_title()
    assert "input SOURCE-gate" in axis.get_title()
    assert axis.get_xlabel().startswith("Raw input gate")
    distribution_viz.plt.close(captured["figure"])
