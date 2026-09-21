"""Resubmit array elements that were killed without writing a done or fail marker.

Purpose
-------
The CM-MNIST nonlinearity-placement array runs with a 12 h wall limit, which the gated GaWF
variants can exceed. This cluster does not requeue a job that is killed by its time limit, so such
an element simply disappears from the queue leaving neither ``task_<id>.done`` nor
``task_<id>.fail``. This checker finds exactly those elements and resubmits them as a new array
built from the same execution snapshot and the same ``run_*.sh`` launcher, so each one resumes
from its last 5-epoch checkpoint (``--auto_resume``) without human intervention.

Inputs
------
- ``--experiment-id``: id of the manifest in ``experiments/monitoring/jobs`` (the run's fact
  source), or ``--manifest`` with an explicit path.
- ``--max-rounds``: maximum resubmissions per task before the checker reports a human decision
  instead of resubmitting again (default 3).
- ``--dry-run``: report the plan without submitting anything.

Outputs
-------
- Prints a one-line status plus the task lists.
- Writes/extended ``experiments/monitoring/state/<id>_resubmit.json`` with per-task rounds.
- On resubmission, appends the new Slurm array id to the manifest and rebuilds the registry.

Exit codes: 0 = nothing to do, 2 = resubmitted, 3 = needs a human decision, 1 = error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = PROJECT_ROOT / "experiments" / "monitoring" / "jobs"
STATE_DIR = PROJECT_ROOT / "experiments" / "monitoring" / "state"


def parse_args() -> argparse.Namespace:
    """Parse the manifest selection, the per-task round cap, and the dry-run switch."""

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-id", type=str)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expand_slurm_tasks(lines: list[str]) -> dict[int, str]:
    """Expand ``squeue`` output such as ``61735045_[13-39%30] PD`` into task id -> state."""

    tasks: dict[int, str] = {}
    pattern = re.compile(r"^(?P<job>\d+)(?:_(?P<task>\[[^\]]+\]|\d+))?\s+(?P<state>\S+)")
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        task_field = match.group("task")
        state = match.group("state")
        if task_field is None:
            continue
        if task_field.isdigit():
            tasks[int(task_field)] = state
            continue
        body = task_field.strip("[]").split("%")[0]
        for chunk in body.split(","):
            if "-" in chunk:
                start_text, _, stop_text = chunk.partition("-")
                start, stop = int(start_text), int(stop_text)
                for task in range(start, stop + 1):
                    tasks[task] = state
            elif chunk.isdigit():
                tasks[int(chunk)] = state
    return tasks


def ssh_run(alias: str, script: str, *, timeout: int = 120) -> str:
    """Run one bounded script over the existing SSH ControlMaster socket."""

    completed = subprocess.run(
        ["ssh", alias, "bash -s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return completed.stdout


def check_socket(alias: str) -> None:
    """Fail fast unless the shared SSH ControlMaster socket is already up."""

    completed = subprocess.run(
        ["ssh", "-O", "check", alias], capture_output=True, text=True, timeout=30
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SSH ControlMaster socket for {alias} is not usable: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )


def remote_state(alias: str, array_id: str, status_dir: str, expected: int) -> dict[str, object]:
    """Return the live, done, fail, and still-running task sets in one SSH round trip."""

    script = f"""
set -u
echo "##SQUEUE"
squeue -j {array_id} -h -o "%i|%t" 2>/dev/null || true
echo "##DONE"
ls {status_dir}/task_*.done 2>/dev/null | sed -E 's/.*task_([0-9]+)\\.done/\\1/' || true
echo "##FAIL"
ls {status_dir}/task_*.fail 2>/dev/null | sed -E 's/.*task_([0-9]+)\\.fail/\\1/' || true
echo "##RUNNING_MARKERS"
ls {status_dir}/task_*.running 2>/dev/null | sed -E 's/.*task_([0-9]+)\\.running/\\1/' || true
"""
    output = ssh_run(alias, script)
    sections: dict[str, list[str]] = {"SQUEUE": [], "DONE": [], "FAIL": [], "RUNNING_MARKERS": []}
    current = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            current = stripped[2:]
            continue
        if current and stripped:
            sections[current].append(stripped)

    live = expand_slurm_tasks([entry.replace("|", " ") for entry in sections["SQUEUE"]])
    done = {int(value) for value in sections["DONE"] if value.isdigit()}
    failed = {int(value) for value in sections["FAIL"] if value.isdigit()}
    started = {int(value) for value in sections["RUNNING_MARKERS"] if value.isdigit()}
    return {
        "live": live,
        "done": done,
        "failed": failed,
        "started": started,
        "expected": expected,
    }


def load_state(path: Path) -> dict[str, object]:
    """Load the per-task resubmission ledger, creating an empty one when absent."""

    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "tasks": {}, "history": []}


def save_state(path: Path, state: dict[str, object]) -> None:
    """Persist the resubmission ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Inspect the array, resubmit vanished elements, and record every action."""

    args = parse_args()
    manifest_path = args.manifest or (JOBS_DIR / f"{args.experiment_id}.json")
    if not manifest_path.is_file():
        print(f"error: missing manifest {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_id = manifest["id"]
    host = manifest["host"]
    job_ids = list(manifest["scheduler"]["job_ids"])
    expected = int(manifest["tracking"]["expected_units"])
    status_dir = manifest["paths"]["status_dir"]
    remote_root = manifest["remote_root"]

    try:
        check_socket(host)
    except Exception as exc:  # noqa: BLE001 - report the raw socket error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        state = remote_state(host, ",".join(job_ids), status_dir, expected)
    except Exception as exc:  # noqa: BLE001 - report the raw remote error
        print(f"error: remote state query failed: {exc}", file=sys.stderr)
        return 1

    live: dict[int, str] = state["live"]  # type: ignore[assignment]
    done: set[int] = state["done"]  # type: ignore[assignment]
    failed: set[int] = state["failed"]  # type: ignore[assignment]
    active = {task for task, task_state in live.items() if task_state in ("R", "PD", "CG")}
    vanished = sorted(set(range(expected)) - done - failed - active)
    print(
        f"{experiment_id}: done={len(done)} failed={len(failed)} active={len(active)} "
        f"vanished={len(vanished)}"
    )
    if vanished:
        print(f"vanished tasks: {vanished}")
    if failed:
        print(f"failed tasks (need a human look): {sorted(failed)}")

    ledger_path = STATE_DIR / f"{experiment_id}_resubmit.json"
    ledger = load_state(ledger_path)
    rounds: dict[str, int] = ledger.get("tasks", {})  # type: ignore[assignment]
    over_cap = [task for task in vanished if rounds.get(str(task), 0) >= args.max_rounds]
    if over_cap:
        print(f"needs human decision (round cap reached): {over_cap}")
    resubmit = [task for task in vanished if task not in over_cap]

    if not resubmit:
        if not vanished and not failed:
            print("nothing to do")
        return 3 if (over_cap or failed) else 0
    if args.dry_run:
        print(f"dry-run: would resubmit tasks {resubmit}")
        return 2

    exports = (
        f"ALL,AIM3_ROOT={remote_root},AIM3_RESULTS_PATH=/scratch/js3269/results"
        ",AIM3_CLUTTER_DATA_DIR=/scratch/js3269/stimuli"
        f",AIM3_STATUS_DIR={status_dir}"
        f",AIM3_SOURCE_COMMIT={manifest.get('source_commit', '')}"
        ",AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"
    )
    artifact_root = str(Path(status_dir).parent)
    array_spec = ",".join(str(task) for task in resubmit)
    submit_script = f"""
set -euo pipefail
cd {remote_root}
sbatch --parsable --chdir={remote_root} --array={array_spec} \
  --output={artifact_root}/%A_%a.retry.out --error={artifact_root}/%A_%a.retry.err \
  --export='{exports}' {remote_root}/experiments/clutter/amarel/run_clutter_nonlinearity_ablation.sh
"""
    try:
        raw = ssh_run(host, submit_script).strip().splitlines()[-1]
        new_id = raw.split(";")[0]
    except Exception as exc:  # noqa: BLE001 - report the raw submission error
        print(f"error: resubmission failed: {exc}", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc).isoformat()
    for task in resubmit:
        rounds[str(task)] = rounds.get(str(task), 0) + 1
    ledger["tasks"] = rounds
    ledger["history"] = list(ledger.get("history", [])) + [
        {"at": now, "tasks": resubmit, "array_id": new_id}
    ]
    save_state(ledger_path, ledger)

    if new_id not in job_ids:
        job_ids.append(new_id)
    manifest["scheduler"]["job_ids"] = job_ids
    manifest["notes"] = list(manifest.get("notes", [])) + [
        f"Auto-resubmitted killed elements {resubmit} as array {new_id} at {now}; each element "
        "resumes from its last 5-epoch checkpoint through --auto_resume."
    ]
    manifest["updated_at"] = now
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-B", "-m", "experiments.monitoring.job_registry", "rebuild"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    print(f"resubmitted tasks {resubmit} as array {new_id}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
