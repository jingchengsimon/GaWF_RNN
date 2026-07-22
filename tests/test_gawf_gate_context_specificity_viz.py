"""Tests for compact GaWF gate context-specificity figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import utils_viz.gawf_gate_context_specificity as gate_viz


def test_excluded_spatial_figure_is_sector_only_three_by_three_with_pdf(
    tmp_path: Path, monkeypatch
) -> None:
    sector = np.linspace(-0.3, 0.3, 9 * 6 * 6, dtype=np.float32).reshape(9, 6, 6)
    digit = np.ones((10, 6, 6), dtype=np.float32)
    captured = {}

    def capture(figure, path: str, dpi: int) -> None:
        captured.update(figure=figure, path=path, dpi=dpi)

    monkeypatch.setattr(gate_viz, "_save", capture)
    gate_viz.plot_spatial_maps(
        {
            "spatial_sector_point_excluded": sector,
            "spatial_digit_point_excluded": digit,
        },
        "point_excluded",
        str(tmp_path),
        180,
    )

    figure = captured["figure"]
    heatmap_axes = [axis for axis in figure.axes if axis.images]
    assert Path(captured["path"]).name == "02_input_spatial_maps_point_excluded.png"
    assert captured["dpi"] == 180
    assert len(heatmap_axes) == 9
    assert [axis.get_title() for axis in heatmap_axes] == [f"Sector {index}" for index in range(9)]
    assert all(axis.get_xticks().size == 0 for axis in heatmap_axes)
    assert all(axis.get_yticks().size == 0 for axis in heatmap_axes)
    assert figure.axes[-1].get_ylabel() == ""
    np.testing.assert_allclose(
        [axis.images[0].get_clim() for axis in heatmap_axes],
        [(-0.3, 0.3)] * 9,
    )
    pdf_path = tmp_path / "02_input_spatial_maps_point_excluded.pdf"
    assert pdf_path.is_file()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    gate_viz.plt.close(figure)
