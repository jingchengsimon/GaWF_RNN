"""Tests for the presentation tick contract of switch-recovery figures."""

from __future__ import annotations

import numpy as np

from utils_viz.fg_switch_offset_acc import select_key_recovery_ticks


def test_select_key_recovery_ticks_uses_requested_four_labels() -> None:
    offsets = np.asarray(
        [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    indices, labels = select_key_recovery_ticks(offsets)

    assert indices.tolist() == [0, 10, 13, 19]
    assert labels == ["pre10", "switch", "post4", "post10"]
