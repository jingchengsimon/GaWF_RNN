"""Coordinate a checkpoint-aligned pause across configured Clutter DSW campaigns.

The default invocation is read-only. Mutation requires ``--execute`` and the exact confirmation
token. Each host receives the stdlib-only remote worker over SSH stdin; no remote code copy or
background SSH session is created.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


CONFIRMATION = "PAUSE-CHECKPOINTED-CAMPAIGNS"


def _load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate one local pause configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("pause config schema_version must be 1")
    campaigns = config.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) < 2:
        raise ValueError("pause config must contain at least two campaigns")
    names = [campaign.get("name") for campaign in campaigns]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("campaign names must be present and unique")
    return config


def _parse_remote_json(stdout: str) -> dict[str, Any]:
    """Parse JSON after an optional DSW login banner."""
    start = stdout.find("{")
    if start < 0:
        raise ValueError("remote worker returned no JSON object")
    return json.loads(stdout[start:])


def _run_remote(
    campaign: dict[str, Any],
    worker: bytes,
    mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one foreground remote worker and return its structured result."""
    ssh_command = [os.path.expanduser(str(part)) for part in campaign["ssh_command"]]
    encoded = base64.b64encode(json.dumps(campaign).encode("utf-8")).decode("ascii")
    remote_parts = [
        "python3",
        "-B",
        "-",
        "--mode",
        mode,
        "--config-b64",
        encoded,
        "--poll-seconds",
        str(config["poll_seconds"]),
        "--checkpoint-wait-seconds",
        str(config["checkpoint_wait_seconds"]),
        "--exit-wait-seconds",
        str(config["exit_wait_seconds"]),
    ]
    if mode == "pause":
        remote_parts.extend(("--confirm", CONFIRMATION))
    remote_command = " ".join(shlex.quote(part) for part in remote_parts)
    timeout = int(config["checkpoint_wait_seconds"]) + int(config["exit_wait_seconds"]) + 180
    completed = subprocess.run(
        [*ssh_command, remote_command],
        input=worker,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    try:
        payload = _parse_remote_json(stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "campaign": campaign["name"],
            "phase": "transport",
            "error": str(exc),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    payload["ssh_returncode"] = completed.returncode
    if stderr.strip():
        payload["ssh_stderr"] = stderr.strip()
    return payload


def _run_all(
    campaigns: list[dict[str, Any]],
    worker: bytes,
    mode: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one foreground SSH operation per host concurrently and wait for all."""
    with ThreadPoolExecutor(max_workers=len(campaigns)) as executor:
        futures = [
            executor.submit(_run_remote, campaign, worker, mode, config) for campaign in campaigns
        ]
        return [future.result() for future in futures]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def main() -> int:
    """Inspect both campaigns or execute the explicitly confirmed coordinated pause."""
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    repo_root = Path(__file__).resolve().parents[4]
    worker_path = repo_root / config["remote_worker"]
    worker = worker_path.read_bytes()
    campaigns = config["campaigns"]

    if args.execute and args.confirm != CONFIRMATION:
        raise SystemExit(f"--execute requires --confirm {CONFIRMATION}; no pause action was taken")

    preflight = _run_all(campaigns, worker, "inspect", config)
    if not args.execute:
        print(json.dumps({"mode": "plan", "campaigns": preflight}, indent=2))
        return 0 if all(item.get("ok") for item in preflight) else 1

    if not all(item.get("ok") for item in preflight):
        print(json.dumps({"mode": "preflight_failed", "campaigns": preflight}, indent=2))
        return 1

    paused = _run_all(campaigns, worker, "pause", config)
    print(json.dumps({"mode": "pause", "preflight": preflight, "campaigns": paused}, indent=2))
    return 0 if all(item.get("ok") for item in paused) else 2


if __name__ == "__main__":
    raise SystemExit(main())
