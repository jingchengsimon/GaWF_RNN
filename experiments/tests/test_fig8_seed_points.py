"""Checks that Figure 8 bars and points use the same seed-level observations."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
import numpy as np

from utils.analysis.clutter.fig6_net_recurrent_current import (
    FULL_SIGNS,
    GROUPS,
    _plot_fig8_bars,
)


def test_fig8_points_and_records_are_seed_averages() -> None:
    values = np.arange(10 * 3 * len(GROUPS) * len(FULL_SIGNS), dtype=np.float64).reshape(
        10, 3, len(GROUPS), len(FULL_SIGNS)
    )
    figure, axis = plt.subplots()
    records = _plot_fig8_bars(
        axis,
        {"gate_component": values},
        "digit",
        "connection",
        show_legend=False,
    )
    expected = values.mean(axis=1)
    point_collections = [
        collection for collection in axis.collections
        if isinstance(collection, PathCollection)
    ]

    assert len(records) == len(GROUPS) * len(FULL_SIGNS)
    assert len(point_collections) == len(records)
    for record, collection in zip(records, point_collections):
        group_idx = GROUPS.index(record["group"])
        sign_idx = FULL_SIGNS.index(record["sign"])
        np.testing.assert_allclose(record["seed_values"], expected[:, group_idx, sign_idx])
        np.testing.assert_allclose(
            np.sort(collection.get_offsets()[:, 1]),
            np.sort(expected[:, group_idx, sign_idx]),
        )
        assert "condition_means" not in record
    plt.close(figure)
