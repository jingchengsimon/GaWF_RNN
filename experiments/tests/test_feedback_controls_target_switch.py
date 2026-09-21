"""Unit checks for the feedback-control target-switch comparison."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from utils.analysis.clutter.fig1b_feedback_controls_target_switch import (
    CONDITION_ORDER,
    build_curves,
    load_units,
)
from utils.analysis.clutter.fig1b_feedback_controls_behavior_2x4 import (
    REFERENCE_MODEL,
    build_panels,
    load_loss_histories,
    load_reference_gawf,
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


def _write_reference(tmp_path: Path, *, exclude_initial: bool = True) -> dict[str, Path]:
    """Write a synthetic strict-GaWF reference bundle with ten seeds."""

    test_csv = tmp_path / "reference_test.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "model", "seed", "char_acc", "sector_acc"])
        for seed in range(1, 11):
            writer.writerow(
                ["test_reset_excluded", "gawf", seed, 84.0 + seed, 91.0 + seed]
            )
    histories_root = tmp_path / "reference_histories"
    shuffle_root = tmp_path / "reference_shuffle"
    for seed in range(1, 11):
        history_dir = histories_root / f"gawf-seed{seed:02d}"
        history_dir.mkdir(parents=True)
        payload = {
            "val_loss_pos": np.linspace(1.0, 0.2 + 0.01 * seed, 150),
            "val_loss_char": np.linspace(2.0, 1.0 + 0.01 * seed, 150),
        }
        (history_dir / "gawf_history.pkl").write_bytes(pickle.dumps(payload))
        shuffle_dir = shuffle_root / f"gawf-seed{seed:02d}"
        shuffle_dir.mkdir(parents=True)
        conditions = {
            condition: {
                "switch_char_acc": [50.0 + seed + offset for offset in OFFSETS],
                "switch_sector_acc": [70.0 + seed + offset for offset in OFFSETS],
            }
            for condition in CONDITION_ORDER
        }
        (shuffle_dir / "ablation_metrics.json").write_text(
            json.dumps(
                {
                    "switch_offsets": OFFSETS,
                    "exclude_window_initial_frame": exclude_initial,
                    "conditions": conditions,
                }
            ),
            encoding="utf-8",
        )
    return {
        "test_csv": test_csv,
        "histories_root": histories_root,
        "shuffle_root": shuffle_root,
    }


def test_reference_gawf_merges_into_the_behavior_panels(tmp_path: Path) -> None:
    curves_root = tmp_path / "curves"
    units_root = tmp_path / "units"
    histories_root = tmp_path / "histories"
    for model in ("gawf_additive", "rnn_fb", "gru_fb", "lstm_fb"):
        _write_unit(curves_root, f"{model}-seed01", offset_scale=1.0)
        directory = units_root / f"{model}-seed01"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reset_excluded_test_accuracy.json").write_text(
            json.dumps({"char_acc": 80.0, "sector_acc": 90.0}), encoding="utf-8"
        )
        history_dir = histories_root / f"{model}-seed01"
        history_dir.mkdir(parents=True)
        (history_dir / f"{model}.pkl").write_bytes(
            pickle.dumps(
                {
                    "val_loss_pos": np.linspace(1.0, 0.3, 150),
                    "val_loss_char": np.linspace(2.0, 1.0, 150),
                }
            )
        )
    paths = _write_reference(tmp_path)

    grouped = load_units(units_root, curves_root)
    curves = build_curves(grouped)
    histories = load_loss_histories(histories_root)
    reference = load_reference_gawf(
        paths["test_csv"], paths["histories_root"], paths["shuffle_root"]
    )
    panels = build_panels(curves, grouped, histories, reference)

    assert REFERENCE_MODEL in panels["models"]
    assert panels["seeds"][REFERENCE_MODEL] == 10
    assert panels["test"][REFERENCE_MODEL]["sector"][0] == pytest.approx(92.0)
    assert panels["test"][REFERENCE_MODEL]["char"][-1] == pytest.approx(94.0)
    expected_recovery = np.mean(
        [[70.0 + seed + offset for offset in OFFSETS] for seed in range(1, 11)], axis=0
    )
    assert np.allclose(
        panels["recovery"][REFERENCE_MODEL]["sector"][0], expected_recovery
    )
    assert np.allclose(
        panels["ablation"][REFERENCE_MODEL]["shuffle_sector"]["sector"],
        np.mean([[70.0 + seed + offset for offset in OFFSETS] for seed in range(1, 11)], axis=1),
    )
    assert panels["seeds"]["gawf_additive"] == 1


def test_reference_loader_rejects_a_wrong_reset_policy(tmp_path: Path) -> None:
    paths = _write_reference(tmp_path, exclude_initial=False)

    with pytest.raises(RuntimeError, match="reset frame"):
        load_reference_gawf(paths["test_csv"], paths["histories_root"], paths["shuffle_root"])
