"""Tests for the compact GaWF gate-distribution histogram figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils_viz import gawf_gate_histogram_summary as summary


def _synthetic_inputs() -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray]]:
    edges = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    base = np.asarray([1, 2, 3, 4], dtype=np.int64)
    raw = {
        "gate_edges": edges,
        "hist_input_all": base,
        "hist_recurrent_all": base[::-1],
        "hist_input_sign": np.stack([base, base[::-1]]),
        "hist_recurrent_sign": np.stack([base[::-1], base]),
        "hist_input_context": np.tile(base, (9, 1)),
        "hist_recurrent_context": np.tile(base[::-1], (9, 1)),
    }
    digit = {
        "gate_edges": edges,
        "hist_input_digit": np.tile(base, (10, 1)),
        "hist_recurrent_digit": np.tile(base[::-1], (10, 1)),
    }
    metadata = {
        "distribution": {
            "input": {"mean": 0.3, "median": 0.2},
            "recurrent": {"mean": 0.7, "median": 0.8},
        }
    }
    return raw, metadata, digit


def test_histogram_summary_has_four_rows_and_two_columns(tmp_path: Path, monkeypatch) -> None:
    raw, metadata, digit = _synthetic_inputs()
    monkeypatch.setattr(summary.plt, "close", lambda figure: None)
    output = summary.plot_histogram_summary(
        raw,
        metadata,
        digit,
        tmp_path / "summary.png",
        tmp_path / "summary.pdf",
    )
    figure = summary.plt.gcf()
    assert output.is_file()
    assert (tmp_path / "summary.pdf").is_file()
    assert len(figure.axes) == 8
    assert all(
        not tick.get_text()
        for axis in figure.axes[:6]
        for tick in axis.get_xticklabels()
    )
    assert any(text.get_text() == "Probability (%)" for text in figure.texts)
    assert all(axis.get_xlim() == (-0.05, 1.05) for axis in figure.axes)
    assert len(figure.axes[0].lines) == 4
    assert len(figure.axes[1].lines) == 4
    assert len(figure.axes[2].lines) == 7
    assert len(figure.axes[3].lines) == 7
    assert len(figure.axes[4].lines) == 27
    assert len(figure.axes[5].lines) == 27
    assert len(figure.axes[6].lines) == 30
    assert len(figure.axes[7].lines) == 30
    assert len(figure.axes[0].child_axes) == 1
    assert len(figure.axes[1].child_axes) == 1
    assert len(figure.axes[2].child_axes) == 2
    assert len(figure.axes[3].child_axes) == 2
    assert all(
        child.get_xlim() == summary.ZOOM_XLIM
        for axis in figure.axes[:4]
        for child in axis.child_axes
    )
    assert all(axis.spines["top"].get_visible() is False for axis in figure.axes)
    assert all(axis.spines["right"].get_visible() is False for axis in figure.axes)
    plt.close(figure)


def test_all_gate_distribution_combines_input_and_recurrent(tmp_path: Path, monkeypatch) -> None:
    raw, _, _ = _synthetic_inputs()
    monkeypatch.setattr(summary.plt, "close", lambda figure: None)
    output = summary.plot_all_gate_distribution(
        raw,
        tmp_path / "all_gate.png",
        tmp_path / "all_gate.pdf",
    )
    figure = summary.plt.gcf()
    assert output.is_file()
    assert (tmp_path / "all_gate.pdf").is_file()
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].get_ylabel() == "Probability (%)"
    assert figure.axes[0].get_title() == "GaWF gate distribution"
    assert figure.axes[0].get_xlim() == (-0.05, 1.05)
    assert np.array_equal(figure.axes[0].get_xticks(), np.linspace(0.0, 1.0, 6))
    assert figure.axes[0].get_legend() is None
    assert len(figure.axes[0].child_axes) == 1
    assert figure.axes[0].child_axes[0].get_xlim() == summary.ZOOM_XLIM
    plt.close(figure)


def test_probability_percent_sums_to_one_hundred() -> None:
    counts = np.asarray([1, 2, 3, 4])
    probability_percent = summary._probability_percent(counts)
    assert np.isclose(np.sum(probability_percent), 100.0)
