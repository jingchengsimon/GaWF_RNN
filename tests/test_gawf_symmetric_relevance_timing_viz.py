"""Tests for the symmetric GaWF relevance and alignment plotting contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba

import utils_viz.gawf_symmetric_relevance_timing as relevance_viz


def _top10_report() -> dict:
    """Build the minimal saved-report structure needed by the compact effect plot."""

    values = {
        "input_sector": (1.1, [1.0, 1.2]),
        "input_digit": (0.3, [0.2, 0.4]),
        "recurrent_sector": (-0.7, [-0.8, -0.6]),
        "recurrent_digit": (-0.4, [-0.5, -0.3]),
    }
    cells = {
        name: {
            "top_percent": {
                "10": {"cohens_d": point, "bootstrap_ci95": interval},
            }
        }
        for name, (point, interval) in values.items()
    }
    return {"primary_validation_estimate_test_effect": {"interaction_excluded": {"cells": cells}}}


def test_top10_excluded_plot_uses_gate_categories_and_solid_factor_colors(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def capture(figure, path: str, dpi: int) -> None:
        captured.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(relevance_viz, "_save", capture)
    relevance_viz.plot_part2_top10_excluded_effects(_top10_report(), str(tmp_path), 180)

    figure = captured["figure"]
    axis = figure.axes[0]
    assert Path(captured["path"]).name == "part2_relevance_effects_top10_excluded.png"
    assert captured["dpi"] == 180
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Input gate",
        "Recurrent gate",
    ]
    assert len(axis.patches) == 4
    assert all(not patch.get_hatch() for patch in axis.patches)
    expected_colors = [
        to_rgba(relevance_viz.COLORS["sector"], alpha=0.82),
        to_rgba(relevance_viz.COLORS["sector"], alpha=0.82),
        to_rgba(relevance_viz.COLORS["digit"], alpha=0.82),
        to_rgba(relevance_viz.COLORS["digit"], alpha=0.82),
    ]
    np.testing.assert_allclose([patch.get_facecolor() for patch in axis.patches], expected_colors)
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert sorted(text.get_text() for text in axis.texts) == ["-0.40", "-0.70", "0.30", "1.10"]
    relevance_viz.plt.close(figure)


def test_alignment_uses_fixed_point_six_scale_and_writes_pdf_copy(
    tmp_path: Path, monkeypatch
) -> None:
    data = {
        "primary_interaction_excluded_input_sector_alignment_matrix": np.asarray(
            [[-0.2, 0.4], [0.1, 0.3]]
        ),
        "primary_interaction_excluded_input_digit_alignment_matrix": np.asarray(
            [[-0.8, 0.5], [0.2, 0.1]]
        ),
        "primary_interaction_excluded_recurrent_sector_alignment_matrix": np.asarray(
            [[-1.2, 0.7], [0.4, 0.2]]
        ),
        "primary_interaction_excluded_recurrent_digit_alignment_matrix": np.asarray(
            [[-0.6, 0.9], [0.3, 0.1]]
        ),
    }
    captured = {}

    def capture(figure, path: str, dpi: int) -> None:
        captured.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(relevance_viz, "_save", capture)
    relevance_viz.plot_part2_alignment(data, str(tmp_path), 150)

    figure = captured["figure"]
    heatmap_axes = [axis for axis in figure.axes if axis.images]
    assert len(heatmap_axes) == 4
    assert [axis.images[0].get_clim() for axis in heatmap_axes] == [(-0.6, 0.6)] * 4
    expected_titles = [
        "input gate / sector\ndiag-offdiag = -0.200",
        "input gate / digit\ndiag-offdiag = -0.700",
        "recurrent gate / sector\ndiag-offdiag = -1.050",
        "recurrent gate / digit\ndiag-offdiag = -0.850",
    ]
    assert [axis.get_title() for axis in heatmap_axes] == expected_titles
    pdf_path = tmp_path / "part2_continuous_alignment.pdf"
    assert pdf_path.is_file()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    relevance_viz.plt.close(figure)
