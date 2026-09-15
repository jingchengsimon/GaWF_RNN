"""Regression checks for ten-seed recovery aggregation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from utils.analysis.clutter.clutter_multiseed_summary import (
    _recovery_mean_sem,
    load_recovery_curves,
    load_test_metrics,
)
from utils.analysis.clutter.supple1_feedback_ablation import _load_metrics, _mean_sem


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _argument_default(module_path: Path, option: str) -> object:
    """Read one argparse default without importing CUDA/PyTorch analysis modules."""

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    raise AssertionError(f"No constant default found for {option} in {module_path}")


def test_recovery_helpers_compute_offsetwise_sem() -> None:
    """Both figure paths use training-seed mean and SEM at every offset."""

    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    expected_mean = values.mean(axis=0)
    expected_half = values.std(axis=0, ddof=1) / np.sqrt(10.0)
    for helper in (_recovery_mean_sem, _mean_sem):
        mean, half = helper(values)
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(half, expected_half)


def test_recovery_loaders_find_ten_seed_subdirectories(tmp_path: Path) -> None:
    """Both loaders preserve ten independent seed curves from nested result leaves."""

    offsets = np.asarray([-1, 1], dtype=np.int64)
    for seed in range(1, 11):
        fig1 = tmp_path / "fig1" / f"gawf-seed{seed:02d}"
        fig1.mkdir(parents=True)
        np.savez(
            fig1 / "fg_switch_offset_acc_gawf_sector_acc_h256.npz",
            offset_order=offsets,
            char_acc=np.asarray([seed, seed + 1], dtype=np.float32),
            sector_acc=np.asarray([seed + 2, seed + 3], dtype=np.float32),
        )
        (fig1 / "fg_switch_offset_meta_gawf_sector_acc_h256.json").write_text(
            json.dumps({"exclude_window_initial_frame": True}),
            encoding="utf-8",
        )
        supple1 = tmp_path / "supple1" / f"gawf-seed{seed:02d}"
        supple1.mkdir(parents=True)
        (supple1 / "ablation_metrics.json").write_text(
            json.dumps(
                {
                    "exclude_window_initial_frame": True,
                    "conditions": {"baseline": {"switch_offsets": offsets.tolist()}},
                }
            ),
            encoding="utf-8",
        )

    loaded_offsets, curves = load_recovery_curves(tmp_path / "fig1")
    np.testing.assert_array_equal(loaded_offsets, offsets)
    assert curves["gawf"]["char"].shape == (10, 2)
    assert len(_load_metrics(str(tmp_path / "supple1"))) == 10


def test_recovery_cli_defaults_exclude_window_initial_frame() -> None:
    """Both recovery collectors exclude recurrent-window initial frames by default."""

    for script in ("fig1_target_switch_recovery.py", "fig2_feedback_ablation.py"):
        path = PROJECT_ROOT / "utils" / "analysis" / "clutter" / script
        assert _argument_default(path, "--exclude_window_initial_frame") is True


def test_recovery_loaders_reject_results_without_reset_exclusion(tmp_path: Path) -> None:
    """Formal renderers refuse legacy recovery files that include or omit t=0 provenance."""

    fig1 = tmp_path / "fig1" / "gawf-seed01"
    fig1.mkdir(parents=True)
    np.savez(
        fig1 / "fg_switch_offset_acc_gawf_sector_acc_h256.npz",
        offset_order=np.asarray([-1, 1], dtype=np.int64),
        char_acc=np.asarray([1.0, 2.0], dtype=np.float32),
        sector_acc=np.asarray([3.0, 4.0], dtype=np.float32),
    )
    (fig1 / "fg_switch_offset_meta_gawf_sector_acc_h256.json").write_text(
        json.dumps({"exclude_window_initial_frame": False}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="rollout t=0"):
        load_recovery_curves(tmp_path / "fig1")

    supple1 = tmp_path / "supple1"
    for seed in range(1, 11):
        leaf = supple1 / f"gawf-seed{seed:02d}"
        leaf.mkdir(parents=True)
        (leaf / "ablation_metrics.json").write_text(
            json.dumps({"exclude_window_initial_frame": seed != 1}),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="rollout t=0"):
        _load_metrics(str(supple1))


def test_test_metrics_rejects_incomplete_seed_set(tmp_path: Path) -> None:
    """Formal test bars must not silently render fewer than ten training seeds."""

    path = tmp_path / "metrics.csv"
    rows = ["model,seed,char_acc,sector_acc"]
    rows.extend(f"gawf,{seed},{80 + seed},{90 + seed}" for seed in range(1, 10))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly ten"):
        load_test_metrics(path)
