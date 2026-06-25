"""Collect registered experiment progress through short-lived SSH connections."""
from __future__ import annotations

import base64
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any


DASHBOARD_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = Path(
    os.environ.get("FAW_RNN_DASHBOARD_REGISTRY", DASHBOARD_DIR / "tasks.json")
)
DEFAULT_CACHE = Path(
    os.environ.get("FAW_RNN_DASHBOARD_CACHE", DASHBOARD_DIR / "status_cache.json")
)
RETENTION_POLICY = "human_confirmation_required"


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and minimally validate the persistent task registry."""
    with path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    seen: set[str] = set()
    for task in registry.get("tasks", []):
        required = {"id", "description", "machine", "job_ids", "remote_root", "tracker"}
        missing = required - set(task)
        if missing:
            raise ValueError(f"Task {task.get('id', '<unknown>')} is missing {sorted(missing)}")
        if task["id"] in seen:
            raise ValueError(f"Duplicate task id: {task['id']}")
        if task["machine"] not in {"amarel", "sjc-remote"}:
            raise ValueError(f"Unsupported machine for task {task['id']}: {task['machine']}")
        if not isinstance(task["job_ids"], list) or not task["job_ids"]:
            raise ValueError(f"Task {task['id']} must have at least one job id")
        if task.get("retention_policy") != RETENTION_POLICY:
            raise ValueError(
                f"Task {task['id']} must use retention policy {RETENTION_POLICY!r}"
            )
        seen.add(task["id"])
    return registry


def _remote_script(tasks: list[dict[str, Any]]) -> str:
    core_source = (DASHBOARD_DIR / "tracker_core.py").read_text(encoding="utf-8")
    core_b64 = base64.b64encode(core_source.encode()).decode()
    tasks_b64 = base64.b64encode(json.dumps(tasks).encode()).decode()
    return f"""set -euo pipefail
python3 - <<'PY'
import base64
import json
namespace = {{}}
source = base64.b64decode({core_b64!r}).decode()
exec(compile(source, 'dashboard_tracker_core.py', 'exec'), namespace)
tasks = json.loads(base64.b64decode({tasks_b64!r}).decode())
results = []
for task in tasks:
    try:
        results.append(namespace['evaluate_task'](task))
    except Exception as exc:
        results.append({{'task_id': task['id'], 'error': f'{{type(exc).__name__}}: {{exc}}'}})
print(json.dumps(results))
PY
"""


def collect_machine(
    machine: str,
    tasks: list[dict[str, Any]],
    *,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """Evaluate all tasks for one host using one fail-fast SSH connection."""
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=12",
        "-o",
        "ServerAliveInterval=10",
        machine,
        "bash -s",
    ]
    completed = subprocess.run(
        command,
        input=_remote_script(tasks),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else f"ssh exited with {completed.returncode}"
        raise RuntimeError(message)
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("remote collector returned no JSON")
    return json.loads(output_lines[-1])


class DashboardState:
    """Thread-safe cached state with resilient periodic refreshes."""

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY,
        cache_path: Path = DEFAULT_CACHE,
    ) -> None:
        self.registry_path = registry_path
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        try:
            with self.cache_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"tasks": [], "updated_at": None, "machines": {}}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def refresh(self) -> dict[str, Any]:
        registry = load_registry(self.registry_path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for task in registry["tasks"]:
            grouped[task["machine"]].append(task)

        previous = {item["id"]: item for item in self.snapshot().get("tasks", [])}
        machine_state: dict[str, Any] = {}
        results_by_id: dict[str, dict[str, Any]] = {}
        now = datetime.now().astimezone().isoformat(timespec="seconds")

        for machine, tasks in grouped.items():
            try:
                results = collect_machine(machine, tasks)
                results_by_id.update({item["task_id"]: item for item in results})
                machine_state[machine] = {"online": True, "last_success": now}
            except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                old_machine = self.snapshot().get("machines", {}).get(machine, {})
                machine_state[machine] = {
                    "online": False,
                    "last_success": old_machine.get("last_success"),
                    "error": str(exc),
                }

        display_tasks: list[dict[str, Any]] = []
        for task in registry["tasks"]:
            result = results_by_id.get(task["id"])
            stale = False
            if result is None or result.get("error"):
                old = previous.get(task["id"], {})
                valid_count = int(old.get("valid_count", 0))
                expected_total = int(
                    old.get("expected_total", task["tracker"].get("expected_total", 0))
                )
                stale = True
            else:
                valid_count = int(result["valid_count"])
                expected_total = int(result["expected_total"])
            display_tasks.append(
                {
                    "id": task["id"],
                    "description": task["description"],
                    "machine": task["machine"],
                    "job_ids": list(task["job_ids"]),
                    "valid_count": valid_count,
                    "expected_total": expected_total,
                    "stale": stale,
                }
            )

        state = {"tasks": display_tasks, "updated_at": now, "machines": machine_state}
        with self._lock:
            self._state = state
            temp_path = self.cache_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_path, self.cache_path)
        return state

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                registry = load_registry(self.registry_path)
                interval = max(30, int(registry.get("refresh_seconds", 120)))
                self.refresh()
            except Exception:
                interval = 120
            self._wake.wait(interval)
            self._wake.clear()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="dashboard-collector", daemon=True)
        self._thread.start()

    def request_refresh(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=3)
