"""Checks for compact ten-seed recurrent-gate analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors
from utils.analysis.clutter.fig7_recurrent_gate_multiseed import (
    GROUP_NAMES,
    RESULT_NAME,
    VARIABLES,
    _compact_paths,
    _group_variable_interaction,
    _recurrent_gate_chunks,
    _sign_magnitude_seed_metrics,
)


def test_compact_paths_use_canonical_subdirectory(tmp_path) -> None:
    expected = []
    for seed in (1, 2):
        path = tmp_path / f"seed{seed:02d}" / "compact" / RESULT_NAME
        path.parent.mkdir(parents=True)
        path.touch()
        expected.append(path)
    assert _compact_paths(tmp_path) == expected


def test_recurrent_only_chunks_match_full_gate_reconstruction() -> None:
    rng = np.random.default_rng(4)
    feedback = rng.normal(size=(2, 3, 2)).astype(np.float32)
    u = rng.normal(size=(3, 2)).astype(np.float32)
    v = rng.normal(size=(2, 7)).astype(np.float32)
    reconstructed = np.concatenate(
        [gate for _start, _end, gate in _recurrent_gate_chunks(
            feedback, u, v, 4, 0.5, 2, "cpu"
        )]
    )
    with torch.no_grad():
        _input_gate, expected = _gate_tensors(
            torch.from_numpy(feedback.reshape(-1, 2)),
            torch.from_numpy(u),
            torch.from_numpy(v),
            4,
            0.5,
        )
    np.testing.assert_allclose(reconstructed, expected.numpy(), rtol=0.0, atol=0.0)


def test_sign_magnitude_metrics_split_slopes_and_keep_overall_level() -> None:
    rows = []
    for signpos, slope in ((1, 2.0), (0, -1.0)):
        for abs_weight in (1.0, 2.0, 3.0):
            rows.append(
                {
                    "absW": abs_weight,
                    "signpos": signpos,
                    "delta_of": 0.5 + slope * abs_weight,
                }
            )
    frame = pd.DataFrame(rows)
    metrics = _sign_magnitude_seed_metrics(
        {group: frame.copy() for group in ("TT", "TR", "RT", "RR")}
    )
    for group in metrics.values():
        assert group["positive_overlap_slope"] == pytest.approx(2.0)
        assert group["negative_overlap_slope"] == pytest.approx(-1.0)
        assert group["overall_delta_level"] == pytest.approx(1.5)


def test_group_variable_interaction_reads_the_requested_summary_directory(tmp_path) -> None:
    """Supplementary statistics must reuse the summary written by the preceding plot step."""

    rng = np.random.default_rng(17)
    values = rng.normal(size=(10, len(GROUP_NAMES), len(VARIABLES)))
    payload = {
        f"{variable}_{group}_gap_seed_values": values[:, group_index, variable_index]
        for group_index, group in enumerate(GROUP_NAMES)
        for variable_index, variable in enumerate(VARIABLES)
    }
    np.savez_compressed(tmp_path / "fig7_seed_level_summary.npz", **payload)

    result = _group_variable_interaction(tmp_path)

    assert result["df_num"] == 3
    assert result["df_den"] == 27
    assert np.isfinite(result["f_statistic"])
