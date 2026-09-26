"""Bounded planner and quota checks for the Clutter 330-unit rolling campaign."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from experiments.clutter.amarel.rolling330 import (
    TAG, quota_slots, reconcile, unit_for_task,
)


def test_plan_has_330_unique_protocol_units() -> None:
    """The two distinct 4h campaigns and all six model families remain separate."""

    units = [unit_for_task(task_id) for task_id in range(330)]
    assert len({unit.suffix for unit in units}) == 330
    assert Counter(unit.group for unit in units) == {
        "matched40": 60, "feedback40": 30, "equal4": 60,
        "scale4": 60, "scale10": 60, "scale20": 60,
    }
    assert units[0].model == "rnn" and units[0].scale == "40h"
    assert units[60].model == "rnn_fb_add" and units[60].width == 271
    assert units[90].model == "rnn" and units[90].width == 128
    assert units[150].model == "rnn" and units[150].width == 275
    assert units[329].model == "s5" and units[329].scale == "20h"
    assert all(unit.suffix.startswith(f"{TAG}/") for unit in units)


def test_quota_counts_pending_tasks_and_other_user_jobs() -> None:
    """A 32-task unrelated queue leaves 66 slots after one current and one next controller."""

    assert quota_slots(all_active=34, campaign_active=0) == 65
    assert quota_slots(all_active=90, campaign_active=80) == 9
    assert quota_slots(all_active=99, campaign_active=50) == 0
    assert quota_slots(all_active=20, campaign_active=90) == 0


def test_first_controller_cycle_submits_sparse_capacity_without_duplicates(
    tmp_path, monkeypatch
) -> None:
    """One cycle creates a durable 65-task array with 32 other jobs and itself active."""

    from experiments.clutter.amarel import rolling330

    root = tmp_path / "repo"
    results = tmp_path / "results"
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    for directory in (root, results, data):
        directory.mkdir()
    monkeypatch.setenv("USER", "test-user")
    monkeypatch.setattr(
        rolling330, "_run",
        lambda *args: "\n".join(str(index) for index in range(33))
        if args[0] == "squeue" else "unexpected",
    )
    submitted: list[list[int]] = []
    monkeypatch.setattr(
        rolling330, "_submit_successor", lambda *_args: "controller-next"
    )
    monkeypatch.setattr(
        rolling330, "_submit_array",
        lambda ids, *_args: submitted.append(ids) or "array-first",
    )
    state = reconcile(root, results, data, artifacts, "a" * 40)
    assert len(submitted) == 1
    assert submitted[0] == list(range(65))
    assert len(state["submitted"]) == 65
    assert state["campaign_active"] == 0
    saved = json.loads((artifacts / "rolling_state.json").read_text())
    assert saved["batches"] == [{"job_id": "array-first", "task_ids": list(range(65))}]
    assert "intent" not in saved


def test_ambiguous_submission_intent_stops_reconciliation(tmp_path) -> None:
    """A crash between sbatch and job-ID recording cannot duplicate a task."""

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    state = {
        "source_commit": "a" * 40, "submitted": {}, "completed": [], "blocked": {},
        "batches": [], "controllers": [], "intent": [0, 1],
    }
    (artifacts / "rolling_state.json").write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="Ambiguous prior sbatch intent"):
        reconcile(tmp_path, tmp_path, tmp_path, artifacts, "a" * 40)
