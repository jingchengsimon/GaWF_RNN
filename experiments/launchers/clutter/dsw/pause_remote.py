"""Remote stdlib worker for checkpoint-aligned Clutter campaign pauses.

Input is one base64-encoded campaign configuration. ``inspect`` is read-only. ``pause`` first
barriers coordinator shells, then stops each root trainer only after a valid periodic checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import shutil
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPLETED_PATTERN = re.compile(r"Epoch (\d+)/(\d+) wall_time_sec=")
CHECKPOINT_PATTERN = re.compile(r"Saved resumable checkpoint: epoch=(\d+)")


def _utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _safe_stamp() -> str:
    """Return a filename-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a small text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small JSON file."""
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve(run_root: Path, value: str) -> Path:
    """Resolve a config path relative to the campaign run root."""
    path = Path(value)
    return path if path.is_absolute() else run_root / path


def _read_cmdline(pid: int) -> str:
    """Read one Linux process command line."""
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def _read_ppid(pid: int) -> int | None:
    """Read one Linux parent PID from /proc."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[3])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


def _process_state(pid: int) -> str | None:
    """Return a Linux process state code, or None when absent."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return fields[2]
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
        return None


def _is_alive(pid: int) -> bool:
    """Return whether a process exists and is not a zombie."""
    state = _process_state(pid)
    return state is not None and state != "Z"


def _matching_root_pids(token: str) -> list[int]:
    """Find top-level processes whose command line contains an exact campaign token."""
    matches: set[int] = set()
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        pid = int(proc_path.name)
        if token in _read_cmdline(pid):
            matches.add(pid)
    return sorted(pid for pid in matches if _read_ppid(pid) not in matches)


def _parse_progress(log_path: Path) -> tuple[int, int]:
    """Return latest completed epoch and latest checkpoint epoch from one log."""
    completed_epoch = 0
    checkpoint_epoch = 0
    if not log_path.is_file():
        return completed_epoch, checkpoint_epoch
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            completed_match = COMPLETED_PATTERN.search(line)
            if completed_match:
                completed_epoch = max(completed_epoch, int(completed_match.group(1)))
            checkpoint_match = CHECKPOINT_PATTERN.search(line)
            if checkpoint_match:
                checkpoint_epoch = max(checkpoint_epoch, int(checkpoint_match.group(1)))
    return completed_epoch, checkpoint_epoch


def _target_checkpoint(
    completed_epoch: int,
    checkpoint_epoch: int,
    interval: int,
    total_epochs: int,
) -> int:
    """Choose the nearest safe checkpoint at or after observed completed progress."""
    if checkpoint_epoch > 0 and checkpoint_epoch >= completed_epoch:
        return checkpoint_epoch
    target = ((completed_epoch // interval) + 1) * interval
    return min(max(interval, target), total_epochs)


def _result_artifacts(result_leaf: Path) -> dict[str, list[str]]:
    """List resumable and final artifacts for one result leaf."""
    return {
        "checkpoint": sorted(glob.glob(str(result_leaf / "*_train_state.pth"))),
        "metrics": sorted(glob.glob(str(result_leaf / "*_metrics.json"))),
        "model": sorted(glob.glob(str(result_leaf / "*_model.pth"))),
        "history": sorted(glob.glob(str(result_leaf / "*.pkl"))),
    }


def _is_complete(artifacts: dict[str, list[str]]) -> bool:
    """Return whether one unit has the exact final training artifact contract."""
    return all(len(artifacts[key]) == 1 for key in ("metrics", "model", "history"))


def _unit_snapshot(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build a read-only snapshot for one campaign seed."""
    run_root = Path(config["run_root"])
    result_root = Path(config["result_root"])
    log_path = run_root / config["log_template"].format(seed=seed)
    result_leaf = result_root / config["result_leaf_template"].format(seed=seed)
    token = config["process_token_template"].format(seed=seed)
    completed_epoch, checkpoint_epoch = _parse_progress(log_path)
    artifacts = _result_artifacts(result_leaf)
    roots = _matching_root_pids(token)
    target = _target_checkpoint(
        completed_epoch,
        checkpoint_epoch,
        int(config["checkpoint_interval_epochs"]),
        int(config["total_epochs"]),
    )
    return {
        "seed": seed,
        "log": str(log_path),
        "result_leaf": str(result_leaf),
        "root_pids": roots,
        "completed_epoch": completed_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "target_checkpoint_epoch": target,
        "checkpoint_paths": artifacts["checkpoint"],
        "complete": _is_complete(artifacts),
        "final_artifact_counts": {
            key: len(artifacts[key]) for key in ("metrics", "model", "history")
        },
    }


def inspect_campaign(config: dict[str, Any]) -> dict[str, Any]:
    """Return a read-only campaign snapshot and duplicate-writer validation."""
    units = [_unit_snapshot(config, int(seed)) for seed in config["seeds"]]
    duplicates = [unit["seed"] for unit in units if len(unit["root_pids"]) > 1]
    missing_roots = (
        not Path(config["run_root"]).is_dir() or not Path(config["result_root"]).parent.is_dir()
    )
    return {
        "ok": not duplicates and not missing_roots,
        "campaign": config["name"],
        "host": os.uname().nodename,
        "timestamp": _utc_now(),
        "duplicate_root_writer_seeds": duplicates,
        "missing_campaign_paths": missing_roots,
        "units": units,
    }


def _controller_pids(config: dict[str, Any]) -> list[int]:
    """Resolve controller PIDs from configured run-root files."""
    run_root = Path(config["run_root"])
    pids: set[int] = set()
    for source in config.get("controller_pid_sources", []):
        path = _resolve(run_root, source["path"])
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if source["format"] == "single":
            values = lines[:1]
        elif source["format"] == "last_column":
            values = [line.split()[-1] for line in lines if line.split()]
        else:
            raise ValueError(f"Unsupported PID source format: {source['format']}")
        for value in values:
            try:
                pid = int(value.strip())
            except ValueError:
                continue
            if _is_alive(pid):
                pids.add(pid)
    return sorted(pids)


def _archive_if_present(path: Path, stamp: str) -> str | None:
    """Move one prior marker aside without deleting it."""
    if not path.exists():
        return None
    archived = path.with_name(f"{path.name}.prepause-{stamp}")
    shutil.move(str(path), str(archived))
    return str(archived)


def _write_unit_paused_status(
    config: dict[str, Any], unit: dict[str, Any], stamp: str
) -> list[str]:
    """Preserve prior status evidence and write one explicit paused marker."""
    run_root = Path(config["run_root"])
    seed = int(unit["seed"])
    archived: list[str] = []
    if config["status_style"] == "seed_status":
        status_path = run_root / config["status_template"].format(seed=seed)
        previous = _archive_if_present(status_path, stamp)
        if previous:
            archived.append(previous)
        _atomic_write_text(
            status_path,
            "state=paused "
            f"seed={seed} checkpoint_epoch={unit['checkpoint_epoch']} "
            f"timestamp={_utc_now()}\n",
        )
    elif config["status_style"] == "task_markers":
        task = seed - 1
        status_dir = _resolve(run_root, config["status_dir"])
        for suffix in ("running", "fail"):
            previous = _archive_if_present(status_dir / f"task_{task}.{suffix}", stamp)
            if previous:
                archived.append(previous)
        _atomic_write_text(
            status_dir / f"task_{task}.paused",
            "status=paused "
            f"task={task} seed={seed} checkpoint_epoch={unit['checkpoint_epoch']} "
            f"timestamp={_utc_now()}\n",
        )
    else:
        raise ValueError(f"Unsupported status style: {config['status_style']}")
    return archived


def _write_campaign_paused_status(
    config: dict[str, Any], stamp: str, receipt_path: Path
) -> list[str]:
    """Preserve the campaign status and write the paused state."""
    run_root = Path(config["run_root"])
    path = _resolve(run_root, config["campaign_status"])
    archived: list[str] = []
    previous = _archive_if_present(path, stamp)
    if previous:
        archived.append(previous)
    _atomic_write_text(
        path,
        f"state=paused receipt={receipt_path} timestamp={_utc_now()}\n",
    )
    return archived


def pause_campaign(
    config: dict[str, Any],
    poll_seconds: int,
    checkpoint_wait_seconds: int,
    exit_wait_seconds: int,
) -> dict[str, Any]:
    """Checkpoint and stop every active writer while holding queue controllers."""
    preflight = inspect_campaign(config)
    if not preflight["ok"]:
        return {"ok": False, "phase": "preflight", "preflight": preflight}

    run_root = Path(config["run_root"])
    stamp = _safe_stamp()
    pause_request = run_root / "pause.requested"
    _atomic_write_text(
        pause_request,
        f"state=requested timestamp={_utc_now()} campaign={config['name']}\n",
    )

    controllers = _controller_pids(config)
    stopped_controllers: list[int] = []
    for pid in controllers:
        os.kill(pid, signal.SIGSTOP)
        stopped_controllers.append(pid)

    units = {int(unit["seed"]): unit for unit in preflight["units"]}
    active = {seed for seed, unit in units.items() if unit["root_pids"] and not unit["complete"]}
    signaled: dict[int, dict[str, Any]] = {}
    checkpoint_deadline = time.monotonic() + checkpoint_wait_seconds

    while active - signaled.keys():
        for seed in sorted(active - signaled.keys()):
            unit = _unit_snapshot(config, seed)
            units[seed] = unit
            roots = unit["root_pids"]
            if unit["complete"]:
                signaled[seed] = {"signal": None, "reason": "completed_naturally"}
                continue
            if not roots:
                return {
                    "ok": False,
                    "phase": "checkpoint_wait",
                    "error": f"seed {seed} exited before a valid checkpoint/final result",
                    "stopped_controller_pids": stopped_controllers,
                    "units": list(units.values()),
                }
            checkpoint_ready = (
                unit["checkpoint_epoch"] >= unit["target_checkpoint_epoch"]
                and len(unit["checkpoint_paths"]) == 1
            )
            if checkpoint_ready:
                for pid in roots:
                    os.kill(pid, signal.SIGINT)
                signaled[seed] = {
                    "signal": "SIGINT",
                    "root_pids": roots,
                    "checkpoint_epoch": unit["checkpoint_epoch"],
                }
        if active - signaled.keys():
            if time.monotonic() >= checkpoint_deadline:
                return {
                    "ok": False,
                    "phase": "checkpoint_wait",
                    "error": "checkpoint wait timeout",
                    "pending_seeds": sorted(active - signaled.keys()),
                    "stopped_controller_pids": stopped_controllers,
                    "units": list(units.values()),
                }
            time.sleep(poll_seconds)

    exit_deadline = time.monotonic() + exit_wait_seconds
    while True:
        remaining: dict[int, list[int]] = {}
        for seed in sorted(active):
            roots = [pid for pid in signaled[seed].get("root_pids", []) if _is_alive(pid)]
            if roots:
                remaining[seed] = roots
        if not remaining:
            break
        if time.monotonic() >= exit_deadline:
            return {
                "ok": False,
                "phase": "exit_wait",
                "error": "trainer exit timeout; no force-kill was attempted",
                "remaining": remaining,
                "stopped_controller_pids": stopped_controllers,
            }
        time.sleep(min(poll_seconds, 10))

    time.sleep(5)
    archived_status: list[str] = []
    final_units: list[dict[str, Any]] = []
    for seed in sorted(units):
        unit = _unit_snapshot(config, seed)
        if not unit["complete"]:
            checkpoint_required = bool(signaled.get(seed, {}).get("signal"))
            if checkpoint_required and len(unit["checkpoint_paths"]) != 1:
                return {
                    "ok": False,
                    "phase": "verification",
                    "error": f"seed {seed} lacks exactly one resumable checkpoint",
                    "unit": unit,
                    "stopped_controller_pids": stopped_controllers,
                }
            archived_status.extend(_write_unit_paused_status(config, unit, stamp))
        final_units.append(unit)

    receipt_path = run_root / f"pause_receipt_{stamp}.json"
    archived_status.extend(_write_campaign_paused_status(config, stamp, receipt_path))
    receipt = {
        "schema_version": 1,
        "campaign": config["name"],
        "host": os.uname().nodename,
        "paused_at": _utc_now(),
        "pause_request": str(pause_request),
        "stopped_controller_pids": stopped_controllers,
        "signaled_units": signaled,
        "archived_status_paths": archived_status,
        "units": final_units,
    }
    _atomic_write_json(receipt_path, receipt)
    return {"ok": True, "phase": "paused", "receipt": str(receipt_path), **receipt}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "pause"), required=True)
    parser.add_argument("--config-b64", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--checkpoint-wait-seconds", type=int, default=3600)
    parser.add_argument("--exit-wait-seconds", type=int, default=300)
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def main() -> int:
    """Run the requested remote worker mode."""
    args = _parse_args()
    config = json.loads(base64.b64decode(args.config_b64).decode("utf-8"))
    if args.mode == "inspect":
        payload = inspect_campaign(config)
    else:
        if args.confirm != "PAUSE-CHECKPOINTED-CAMPAIGNS":
            payload = {
                "ok": False,
                "phase": "authorization",
                "error": "exact pause confirmation token is required",
            }
        else:
            payload = pause_campaign(
                config,
                poll_seconds=args.poll_seconds,
                checkpoint_wait_seconds=args.checkpoint_wait_seconds,
                exit_wait_seconds=args.exit_wait_seconds,
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
