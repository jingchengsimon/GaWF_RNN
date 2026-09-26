"""Selection coverage for isolated experiment-progress manifest lookups."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.monitoring.job_registry import RegistryError
from experiments.monitoring.progress import (
    _PROBE_JSON_MARKER,
    _local_ssh_commands,
    collect_remote_jobs,
    format_report,
    select_job,
)
from experiments.monitoring.remote_probe import _scheduler_state, collect


def _manifest(job_id: str, *, status: str = "running") -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": job_id,
        "description": f"{job_id} description",
        "host": "sjc-remote",
        "status": status,
        "remote_root": "/remote/project",
        "environment": {"name": "aim3_rnn", "conda_init": "/remote/conda.sh"},
    }


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    jobs_dir = root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / f"{manifest['id']}.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_exact_id_does_not_read_unrelated_historical_manifest(tmp_path: Path) -> None:
    target = _manifest("sjc-exact-target")
    _write_manifest(tmp_path, target)
    (tmp_path / "jobs" / "broken-history.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "active_jobs.json").write_text("{not json", encoding="utf-8")

    selected = select_job("sjc-exact-target", base_dir=tmp_path)

    assert selected["id"] == "sjc-exact-target"


def test_nonexistent_id_does_not_fall_back_to_historical_scan(tmp_path: Path) -> None:
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "broken-history.json").write_text("{not json", encoding="utf-8")

    try:
        select_job("historical", base_dir=tmp_path)
    except RegistryError as exc:
        assert "No retained job has exact ID 'historical'" in str(exc)
    else:
        raise AssertionError("A nonexistent complete ID must fail")


def test_socket_check_failure_does_not_open_a_remote_ssh_connection() -> None:
    job = _manifest("sjc-exact-target")

    with patch(
        "experiments.monitoring.progress.subprocess.run",
        return_value=CompletedProcess(["ssh"], 255, "", "Control socket missing"),
    ) as run:
        reports = collect_remote_jobs([job], timeout=30)

    assert len(reports) == 1
    assert reports[0]["probe_error"] == "SSH socket unavailable: Control socket missing"
    assert run.call_count == 1
    assert run.call_args.args[0] == ["ssh", "-O", "check", "sjc-remote"]


def test_local_full_ssh_command_is_loaded_as_direct_endpoint(tmp_path: Path) -> None:
    config = tmp_path / "local.md"
    config.write_text(
        "## SSH aliases\n"
        "- sjc remote: `sjc-remote`\n"
        "- DSW 5000: `ssh -o IdentitiesOnly=yes -i ~/.ssh/key -p 5000 root@example`\n",
        encoding="utf-8",
    )

    commands = _local_ssh_commands(config)

    assert "sjc-remote" not in commands
    assert commands["dsw-5000"][0] == "ssh"
    assert commands["dsw-5000"][-1] == "root@example"
    assert commands["dsw-5000"][4] == str(Path.home() / ".ssh" / "key")


def test_explicit_direct_endpoint_skips_control_socket_check() -> None:
    job = _manifest("dsw-direct-target")
    job["host"] = "dsw-5000"
    response = "remote banner\n" + _PROBE_JSON_MARKER + json.dumps(
        [{"id": job["id"], "host": job["host"]}]
    )
    direct = ["ssh", "-p", "5000", "root@example"]

    with patch(
        "experiments.monitoring.progress.subprocess.run",
        return_value=CompletedProcess(["ssh"], 0, response, ""),
    ) as run:
        reports = collect_remote_jobs(
            [job], direct_commands={"dsw-5000": direct}, timeout=30
        )

    assert reports == [{"id": "dsw-direct-target", "host": "dsw-5000"}]
    assert run.call_count == 1
    command = run.call_args.args[0]
    assert command[:3] == ["ssh", "-p", "5000"]
    assert "BatchMode=yes" in command
    assert command[-2] == "root@example"


def test_file_result_glob_counts_complete_analysis_units() -> None:
    """An NPZ result glob is valid completion evidence without a metrics.json sidecar."""

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for seed in ("seed01", "seed02"):
            destination = root / "results" / seed
            destination.mkdir(parents=True)
            (destination / "summary.npz").write_bytes(b"npz")
        job = _manifest("analysis-files")
        job["remote_root"] = str(root)
        job["tracking"] = {
            "expected_units": 2,
            "result_globs": ["results/seed*/summary.npz"],
        }

        report = collect(job)

    assert report["discovered_units"] == 2
    assert report["valid_units"] == 2
    assert all(unit["artifact_exists"] for unit in report["units"])
    assert "status=completed (verified)" in format_report(report, job)


def test_rolling_scheduler_reads_later_array_ids(tmp_path: Path) -> None:
    state_file = tmp_path / "rolling_state.json"
    state_file.write_text(
        json.dumps({
            "submitted": {"0": "100", "1": "101"},
            "completed": [0],
            "blocked": {},
            "batches": [{"job_id": "100"}, {"job_id": "101"}],
            "controllers": ["102"],
        }),
        encoding="utf-8",
    )
    manifest = {
        "scheduler": {
            "type": "slurm",
            "job_ids": ["100"],
            "rolling_state_file": str(state_file),
        }
    }

    with patch(
        "experiments.monitoring.remote_probe._run",
        return_value={"returncode": 0, "stdout": "101_1|PENDING|0:00|Priority", "stderr": ""},
    ) as run:
        scheduler = _scheduler_state(manifest, tmp_path)

    assert scheduler["rolling"] == {
        "submitted": 2, "completed": 1, "blocked": 0, "batches": 2,
    }
    assert run.call_args_list[0].args[0] == [
        "squeue", "-r", "-h", "-j", "100,101,102", "-o", "%i|%T|%M|%R",
    ]
