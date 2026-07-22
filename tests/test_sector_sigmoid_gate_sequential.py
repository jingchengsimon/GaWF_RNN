"""Tests for sequential equal-n sector input-gate means and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils_anal.sector_sigmoid_gate_sequential import (
    accumulate_equal_n_input_gates,
    equal_n_sector_mask,
    spatial_gate_means,
)
import utils_viz.sector_sigmoid_gate_sequential as gate_viz


def test_equal_n_mask_selects_minimum_count_per_sector() -> None:
    sectors = np.concatenate([np.full(sector + 2, sector, dtype=np.int64) for sector in range(9)])
    selected, target, original = equal_n_sector_mask(sectors, seed=11)
    assert target == 2
    np.testing.assert_array_equal(original, np.arange(2, 11))
    np.testing.assert_array_equal(np.bincount(sectors[selected], minlength=9), np.full(9, 2))


def test_accumulation_excludes_point_mass_per_synapse_and_spatializes() -> None:
    sectors = np.repeat(np.arange(9, dtype=np.int64), 2)
    selected = np.ones(sectors.size, dtype=bool)
    gates = np.empty((sectors.size, 1, 1152), dtype=np.float32)
    for frame, sector in enumerate(sectors):
        gates[frame] = 0.1 + 0.04 * sector
    gates[::2, 0, 0] = 0.5
    chunks = ((0, 7, gates[:7]), (7, sectors.size, gates[7:]))
    accumulators = accumulate_equal_n_input_gates(
        chunks,
        sectors,
        selected,
        (1, 1152),
        1e-6,
        progress_every=0,
    )
    included, excluded = spatial_gate_means(*accumulators)
    assert included.shape == (9, 6, 6)
    assert excluded.shape == (9, 6, 6)
    assert excluded[0, 0, 0] < included[0, 0, 0]
    np.testing.assert_allclose(excluded[:, 1:, 1:], included[:, 1:, 1:])


def test_plot_writes_tickless_png_and_pdf(tmp_path: Path) -> None:
    maps = np.linspace(0.2, 0.8, 9 * 6 * 6, dtype=np.float32).reshape(9, 6, 6)
    png_path, pdf_path = gate_viz.plot_sector_grid(
        maps,
        "point_excluded",
        str(tmp_path),
        150,
    )
    assert Path(png_path).is_file()
    assert Path(pdf_path).read_bytes().startswith(b"%PDF")
