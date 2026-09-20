"""Regression checks for condition-balanced recurrent-gate bar aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.analysis.clutter.fig7_recurrent_gate_disinhibition import (
    _condition_means,
    overlap_band_sign_stats,
)
from utils.analysis.clutter.supple2_input_gate_sign_magnitude_sector import (
    GROUPS as INPUT_GROUPS,
    _seed_level_stats as input_gate_seed_level_stats,
)


def test_condition_means_do_not_weight_conditions_by_connection_count() -> None:
    frame = pd.DataFrame(
        {
            "context": [0, 1, 1, 1],
            "delta_of": [0.0, 10.0, 10.0, 10.0],
        }
    )

    np.testing.assert_allclose(_condition_means(frame, "delta_of"), [0.0, 10.0])


def test_overlap_bar_averages_condition_means_equally() -> None:
    rows = []
    for sign in (0, 1):
        rows.append({"context": 0, "signpos": sign, "absW": 1.0, "delta_of": 0.0})
        rows.extend(
            {"context": 1, "signpos": sign, "absW": 1.0, "delta_of": 10.0}
            for _ in range(3)
        )
    frame = pd.DataFrame(rows)
    groups = ("TT", "TR", "RT", "RR")
    stats = overlap_band_sign_stats(
        {group: frame for group in groups},
        {group: {"overlap_low": 0.5, "overlap_high": 1.5} for group in groups},
        y_col="delta_of",
    )

    assert stats["TT"]["+"]["mean"] == 5.0
    assert stats["TT"]["-"]["mean"] == 5.0
    assert stats["TT"]["+"]["n_conditions"] == 2


def test_input_gate_summary_averages_sector_means_equally() -> None:
    rows = []
    for sign in (0, 1):
        rows.extend(
            {"context": 0, "signpos": sign, "absW": abs_w, "delta_gate": 0.0}
            for abs_w in (1.0, 2.0)
        )
        rows.extend(
            {"context": 1, "signpos": sign, "absW": abs_w, "delta_gate": 10.0}
            for abs_w in (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
        )
    frame = pd.DataFrame(rows)
    stats = input_gate_seed_level_stats(
        {group: [frame.copy() for _ in range(10)] for group in INPUT_GROUPS}
    )

    assert stats["sector0_sources"]["positive_overlap_mean"]["mean"] == 5.0
    assert stats["sector0_sources"]["negative_overlap_mean"]["mean"] == 5.0
    assert stats["sector0_sources"]["overall_delta_level"]["mean"] == 5.0
