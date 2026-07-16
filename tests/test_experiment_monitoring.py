"""Tests for the project-local remote experiment registry and exact-path probe."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.monitoring.job_registry import (
    RegistryError,
    find_job,
    load_jobs,
    remove_job,
    save_job,
    update_status,
)
from experiments.monitoring.remote_probe import collect


def _minimal_manifest(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": "example-job-123",
        "description": "example remote job",
        "host": "sjc-remote",
        "status": "running",
        "remote_root": str(root),
        "environment": {"name": "aim3_rnn", "conda_init": "/tmp/conda.sh"},
        "scheduler": {
            "type": "none",
            "job_ids": ["123"],
            "run_ids": ["example-run"],
        },
        "paths": {"log_globs": [], "result_paths": ["results/train_data/example"]},
        "tracking": {"expected_units": 0, "units": []},
        "notes": [],
    }


class ExperimentMonitoringTests(unittest.TestCase):
    def test_registry_retains_terminal_jobs_and_rebuilds_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = _minimal_manifest(root / "remote")
            save_job(manifest, base_dir=root)

            active = json.loads((root / "active_jobs.json").read_text(encoding="utf-8"))
            self.assertEqual([job["id"] for job in active["jobs"]], ["example-job-123"])
            self.assertEqual(find_job("123", load_jobs(root))["id"], "example-job-123")

            update_status("example-run", "completed", base_dir=root)
            jobs = load_jobs(root)
            self.assertEqual(jobs[0]["status"], "completed")
            active = json.loads((root / "active_jobs.json").read_text(encoding="utf-8"))
            self.assertEqual(active["jobs"], [])
            history = (root / "JOBS.md").read_text(encoding="utf-8")
            self.assertIn("example-job-123", history)

    def test_registry_deletion_requires_human_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_job(_minimal_manifest(root / "remote"), base_dir=root)
            with self.assertRaisesRegex(RegistryError, "human-confirmed"):
                remove_job("example-job-123", human_confirmed=False, base_dir=root)
            self.assertEqual(len(load_jobs(root)), 1)

            remove_job("example-job-123", human_confirmed=True, base_dir=root)
            self.assertEqual(load_jobs(root), [])

    def test_exact_unit_probe_validates_metrics_done_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "results" / "train_data" / "pong_gawf_seed42"
            status_dir = root / "status"
            result_dir.mkdir(parents=True)
            status_dir.mkdir()
            metrics = {
                "global_step": 1000000,
                "frame_skip": 1,
                "frame_stack": 1,
                "num_layers": 1,
                "model_type": "gawf",
            }
            (result_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            history = {"global_step": 1000000, "fps": 42, "episodic_return_100": -18.0}
            history_path = result_dir / "metrics_history.jsonl"
            history_path.write_text(json.dumps(history) + "\n", encoding="utf-8")
            (result_dir / "model.pth").write_bytes(b"checkpoint")
            (status_dir / "pong_gawf_seed42.done").touch()

            manifest = _minimal_manifest(root)
            manifest["tracking"] = {
                "expected_units": 1,
                "defaults": {"checkpoint_glob": "*.pth", "checkpoint_count": 1},
                "units": [
                    {
                        "id": "gawf-seed42",
                        "result_dir": "results/train_data/pong_gawf_seed42",
                        "done_file": "status/pong_gawf_seed42.done",
                        "fail_file": "status/pong_gawf_seed42.fail",
                        "expected": metrics,
                    }
                ],
            }
            report = collect(manifest)
            self.assertEqual(report["valid_units"], 1)
            self.assertEqual(report["done_units"], 1)
            self.assertEqual(report["failed_units"], 0)
            self.assertEqual(report["units"][0]["fps"], 42)

            mismatched = {**metrics, "frame_skip": 4}
            (result_dir / "metrics.json").write_text(
                json.dumps(mismatched), encoding="utf-8"
            )
            report = collect(manifest)
            self.assertEqual(report["valid_units"], 0)
            self.assertEqual(
                report["units"][0]["mismatches"]["frame_skip"],
                {"expected": 1, "actual": 4},
            )


if __name__ == "__main__":
    unittest.main()
