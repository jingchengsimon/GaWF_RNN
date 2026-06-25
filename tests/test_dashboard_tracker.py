"""Tests for result-evidence dashboard progress tracking."""
from __future__ import annotations

import json
from pathlib import Path

from dashboard.tracker_core import evaluate_task


def _write_triplet(root: Path, stem: str, metrics: dict[str, object]) -> None:
    path = root / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + "_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    path.with_suffix(".pkl").write_bytes(b"pkl")
    path.with_name(path.name + "_model.pth").write_bytes(b"model")


def test_metrics_grid_uses_content_and_companions(tmp_path: Path) -> None:
    _write_triplet(
        tmp_path,
        "results/task_1/rnn_run",
        {"model_type": "rnn", "hidden_size": 531, "lr": 0.001, "weight_decay": 0.0},
    )
    _write_triplet(
        tmp_path,
        "results/task_2/gru_run",
        {"model_type": "gru", "hidden_size": 219, "lr": 0.001, "weight_decay": 0.0},
    )
    task = {
        "id": "grid",
        "remote_root": str(tmp_path),
        "tracker": {
            "type": "metrics_grid",
            "expected_total": 2,
            "result_glob": "results/task_*/*_metrics.json",
            "match": {"lr": {"equals": 0.001}},
            "allowed_combinations": [
                {"model_type": "rnn", "hidden_size": 531},
                {"model_type": "gru", "hidden_size": 219},
            ],
            "uniqueness_fields": ["model_type", "hidden_size", "lr", "weight_decay"],
        },
    }
    assert evaluate_task(task)["valid_count"] == 2


def test_explicit_units_require_every_file(tmp_path: Path) -> None:
    for name in ["a.json", "a.pkl", "a.pth", "b.json", "b.pkl"]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    task = {
        "id": "runs",
        "remote_root": str(tmp_path),
        "tracker": {
            "type": "explicit_units",
            "expected_total": 2,
            "units": [
                {"required_files": ["a.json", "a.pkl", "a.pth"]},
                {"required_files": ["b.json", "b.pkl", "b.pth"]},
            ],
        },
    }
    assert evaluate_task(task)["valid_count"] == 1

