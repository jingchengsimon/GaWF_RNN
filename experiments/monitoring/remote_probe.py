"""Collect scheduler, process, log, and result evidence for one remote-job manifest.

The module uses only the Python standard library so ``progress.py`` can send its source through
one SSH session after activating the remote ``aim3_rnn`` environment. The returned dictionary is
JSON serializable and contains scheduler state, GPU state, valid/pending units, and recent errors.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ERROR_PATTERNS = (
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"\bRuntimeError\b",
    r"\bFAILED\b",
    r"\bError:\s",
)


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _last_history_record(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = bytearray()
            while position > 0 and buffer.count(b"\n") < 3:
                read_size = min(8192, position)
                position -= read_size
                handle.seek(position)
                buffer[:0] = handle.read(read_size)
        for raw_line in reversed(buffer.splitlines()):
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                return value
    except OSError:
        pass
    return {}


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _unit_from_result_dir(path: Path, root: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    unit = dict(defaults)
    unit.setdefault("id", path.name)
    try:
        unit["result_dir"] = str(path.relative_to(root))
    except ValueError:
        unit["result_dir"] = str(path)
    return unit


def _expand_units(root: Path, tracking: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(tracking.get("defaults", {}))
    explicit = tracking.get("units", [])
    if explicit:
        return [{**defaults, **unit} for unit in explicit]
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in tracking.get("result_globs", []):
        absolute_pattern = str(_resolve(root, pattern))
        for value in sorted(glob.glob(absolute_pattern)):
            path = Path(value)
            if not path.is_dir() or str(path) in seen:
                continue
            seen.add(str(path))
            units.append(_unit_from_result_dir(path, root, defaults))
    return units


def _check_unit(root: Path, unit: dict[str, Any]) -> dict[str, Any]:
    result_dir = _resolve(root, unit.get("result_dir"))
    if result_dir is None:
        result_dir = root
    metrics_path = result_dir / unit.get("metrics_file", "metrics.json")
    history_path = result_dir / unit.get("history_file", "metrics_history.jsonl")
    metrics = _read_json(metrics_path)
    history = _last_history_record(history_path)
    done_path = _resolve(root, unit.get("done_file"))
    fail_path = _resolve(root, unit.get("fail_file"))
    done = done_path.is_file() if done_path is not None else None
    failed = fail_path.is_file() if fail_path is not None else False
    checkpoint_glob = unit.get("checkpoint_glob")
    checkpoints = sorted(result_dir.glob(checkpoint_glob)) if checkpoint_glob else []
    expected = dict(unit.get("expected", {}))
    mismatches = {
        key: {"expected": expected_value, "actual": metrics.get(key)}
        for key, expected_value in expected.items()
        if metrics.get(key) != expected_value
    }
    if done_path is not None and not done:
        mismatches["done_file"] = {"expected": True, "actual": False}
    checkpoint_count = unit.get("checkpoint_count")
    if checkpoint_count is not None and len(checkpoints) != checkpoint_count:
        mismatches["checkpoint_count"] = {
            "expected": checkpoint_count,
            "actual": len(checkpoints),
        }
    valid = metrics_path.is_file() and not failed and not mismatches
    return {
        "id": unit.get("id", result_dir.name),
        "result_dir": str(result_dir),
        "metrics_exists": metrics_path.is_file(),
        "history_exists": history_path.is_file(),
        "done": done,
        "failed": failed,
        "valid": valid,
        "global_step": _finite(history.get("global_step", metrics.get("global_step"))),
        "fps": _finite(history.get("fps", metrics.get("fps"))),
        "episodic_return_100": _finite(
            history.get("episodic_return_100", metrics.get("episodic_return_100"))
        ),
        "checkpoint_count": len(checkpoints),
        "mismatches": mismatches,
    }


def _scheduler_state(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    scheduler = manifest.get("scheduler", {})
    scheduler_type = scheduler.get("type", "none")
    state: dict[str, Any] = {"type": scheduler_type}
    job_ids = [str(value) for value in scheduler.get("job_ids", [])]
    if scheduler_type == "slurm" and job_ids:
        joined = ",".join(job_ids)
        state["squeue"] = _run(
            ["squeue", "-h", "-j", joined, "-o", "%i|%T|%M|%R"], cwd=root
        )
        state["sacct"] = _run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                joined,
                "--format=JobIDRaw,State,Elapsed,ExitCode",
            ],
            cwd=root,
        )
    tmux_session = scheduler.get("tmux_session")
    if tmux_session:
        result = _run(["tmux", "has-session", "-t", str(tmux_session)], cwd=root)
        state["tmux_session"] = tmux_session
        state["tmux_active"] = result["returncode"] == 0
    process_matches: list[str] = []
    for pattern in scheduler.get("process_patterns", []):
        result = _run(["pgrep", "-af", str(pattern)], cwd=root)
        if result["stdout"]:
            process_matches.extend(result["stdout"].splitlines())
    state["process_matches"] = process_matches[:30]
    return state


def _gpu_state(manifest: dict[str, Any], root: Path) -> dict[str, Any] | None:
    if not manifest.get("scheduler", {}).get("collect_gpu", manifest.get("host") == "sjc-remote"):
        return None
    return _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=root,
    )


def _scan_errors(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    paths = manifest.get("paths", {})
    patterns = manifest.get("error_patterns", DEFAULT_ERROR_PATTERNS)
    regex = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
    log_paths: list[Path] = []
    for pattern in paths.get("log_globs", []):
        absolute_pattern = str(_resolve(root, pattern))
        log_paths.extend(Path(value) for value in glob.glob(absolute_pattern, recursive=True))
    unique_paths = sorted(
        {path for path in log_paths if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:20]
    hits: list[dict[str, Any]] = []
    for path in unique_paths:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 262_144))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if regex.search(line):
                hits.append({"path": str(path), "line": line[-500:]})
    return hits[-20:]


def collect(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect one complete progress snapshot from the manifest's exact paths."""

    root = Path(manifest["remote_root"])
    tracking = manifest.get("tracking", {})
    units = [_check_unit(root, unit) for unit in _expand_units(root, tracking)]
    done_glob = tracking.get("done_glob")
    fail_glob = tracking.get("fail_glob")
    done_count = len(glob.glob(str(_resolve(root, done_glob)))) if done_glob else 0
    fail_count = len(glob.glob(str(_resolve(root, fail_glob)))) if fail_glob else 0
    explicit_done = sum(unit["done"] is True for unit in units)
    explicit_failed = sum(unit["failed"] is True for unit in units)
    return {
        "id": manifest["id"],
        "host": manifest["host"],
        "remote_root": str(root),
        "root_exists": root.is_dir(),
        "scheduler": _scheduler_state(manifest, root),
        "gpu": _gpu_state(manifest, root),
        "expected_units": tracking.get("expected_units", len(units)),
        "discovered_units": len(units),
        "valid_units": sum(unit["valid"] for unit in units),
        "done_units": explicit_done if tracking.get("units") else done_count,
        "failed_units": explicit_failed if tracking.get("units") else fail_count,
        "units": units,
        "errors": _scan_errors(manifest, root),
    }
