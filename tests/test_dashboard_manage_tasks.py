"""Tests for Dashboard registration metadata and human-confirmed retention."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dashboard.collector import load_registry
from dashboard.manage_tasks import (
    RETENTION_POLICY,
    _write_registry,
    normalize_spec,
    register_task,
    remove_task,
)


def _task_spec() -> dict[str, object]:
    return {
        "id": "remote-run",
        "description": "Remote training run",
        "machine": "amarel",
        "job_ids": [12345],
        "remote_root": "~/FAW_RNN",
        "tracker": {
            "type": "explicit_units",
            "expected_total": 1,
            "units": [{"required_files": ["results/run_metrics.json"]}],
        },
    }


class DashboardTaskLifecycleTests(unittest.TestCase):
    def test_registration_adds_retention_and_string_job_ids(self) -> None:
        registry = {"version": 1, "tasks": []}
        updated = register_task(registry, _task_spec())
        task = updated["tasks"][0]
        self.assertEqual(task["retention_policy"], RETENTION_POLICY)
        self.assertEqual(task["job_ids"], ["12345"])

    def test_remove_requires_human_confirmation(self) -> None:
        registry = {"version": 1, "tasks": [normalize_spec(_task_spec())]}
        with self.assertRaisesRegex(ValueError, "human confirmation"):
            remove_task(registry, "remote-run", human_confirmed=False)
        self.assertEqual(len(registry["tasks"]), 1)

    def test_confirmed_remove_deletes_requested_task(self) -> None:
        registry = {"version": 1, "tasks": [normalize_spec(_task_spec())]}
        updated = remove_task(registry, "remote-run", human_confirmed=True)
        self.assertEqual(updated["tasks"], [])

    def test_registry_rejects_task_without_retention_policy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            path.write_text(
                json.dumps({"version": 1, "tasks": [_task_spec()]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "retention policy"):
                load_registry(path)

    def test_custom_registry_write_does_not_create_runtime_mirror(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            registry = {"version": 1, "tasks": []}
            _write_registry(path, registry)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), registry)


if __name__ == "__main__":
    unittest.main()
