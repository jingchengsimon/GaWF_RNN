"""Unit checks for the feedback-control target-switch comparison."""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.analysis.clutter.fig1b_feedback_controls_target_switch import (
    CONDITION_ORDER,
    build_curves,
    load_units,
)


OFFSETS = [-10, -9, -1, 1, 2, 10]


def _write_unit(root: Path, name: str, *, offset_scale: float, exclude_initial: bool = True) -> None:
    """Write one synthetic unit export with a deterministic offset ramp."""

    directory = root / name
    directory.mkdir(parents=True)
    conditions = {}
    for index, condition in enumerate(CONDITION_ORDER):
        values = [offset_scale * (offset + 10) - index for offset in OFFSETS]
        conditions[condition] = {
            "switch_char_acc": [value + 50.0 for value in values],
            "switch_sector_acc": [value + 70.0 for value in values],
        }
    payload = {
        "switch_offsets": OFFSETS,
        "exclude_window_initial_frame": exclude_initial,
        "conditions": conditions,
    }
    (directory / "ablation_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_units_groups_models_and_aggregates_seeds(tmp_path: Path) -> None:
    curves_root = tmp_path / "curves"
    units_root = tmp_path / "units"
    _write_unit(curves_root, "gawf_additive-seed01", offset_scale=1.0)
    _write_unit(curves_root, "gawf_additive-seed02", offset_scale=3.0)
    _write_unit(curves_root, "rnn_fb-seed01", offset_scale=2.0)
    _write_unit(curves_root, "gru_fb-seed01", offset_scale=1.5)
    _write_unit(curves_root, "lstm_fb-seed01", offset_scale=2.5)
    for name, char in (
        ("gawf_additive-seed01", 80.0),
        ("gawf_additive-seed02", 84.0),
        ("rnn_fb-seed01", 79.0),
    ):
        directory = units_root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reset_excluded_test_accuracy.json").write_text(
            json.dumps({"char_acc": char, "sector_acc": char + 5.0}), encoding="utf-8"
        )

    grouped = load_units(units_root, curves_root)
    curves = build_curves(grouped)

    gawf = curves["models"]["gawf_additive"]
    assert gawf["n_seeds"] == 2
    assert gawf["seeds"] == [1, 2]
    baseline = np.asarray(gawf["baseline"]["char"]["mean"], dtype=np.float64)
    expected = 50.0 + np.asarray([(1.0 + 3.0) / 2 * (offset + 10) for offset in OFFSETS])
    assert np.allclose(baseline, expected)
    sem = np.asarray(gawf["baseline"]["char"]["sem"], dtype=np.float64)
    assert np.allclose(sem, np.asarray(OFFSETS) + 10)
    assert curves["models"]["rnn_fb"]["n_seeds"] == 1
    assert np.allclose(curves["models"]["rnn_fb"]["baseline"]["char"]["sem"], 0.0)


def test_loader_rejects_an_export_that_keeps_the_reset_frame(tmp_path: Path) -> None:
    curves_root = tmp_path / "curves"
    _write_unit(curves_root, "gawf_additive-seed01", offset_scale=1.0, exclude_initial=False)

    with pytest.raises(RuntimeError, match="reset frame"):
        load_units(tmp_path / "units", curves_root)


def test_loader_rejects_mismatched_offset_grids(tmp_path: Path) -> None:
    curves_root = tmp_path / "curves"
    _write_unit(curves_root, "gawf_additive-seed01", offset_scale=1.0)
    _write_unit(curves_root, "gawf_additive-seed02", offset_scale=1.0)
    path = curves_root / "gawf_additive-seed02" / "ablation_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["switch_offsets"] = OFFSETS[:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Offset grid differs"):
        load_units(tmp_path / "units", curves_root)
