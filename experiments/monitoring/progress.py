"""Resolve retained jobs and inspect their exact remote scheduler, log, and result paths.

With no selector, active jobs are checked; if there are no active records, the newest retained
job is checked. Jobs sharing a host and Conda environment are combined into one foreground SSH
session. Output is a concise Chinese progress summary or JSON for downstream automation.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.monitoring import remote_probe
from experiments.monitoring.job_registry import (
    ACTIVE_STATUSES,
    RegistryError,
    find_job,
    load_jobs,
    update_status,
)


def _ssh_alias_map(values: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RegistryError("--ssh-alias must use HOST=ALIAS format.")
        host, alias = value.split("=", 1)
        aliases[host.strip()] = alias.strip()
    return aliases


def select_jobs(
    selectors: list[str],
    *,
    active: bool = False,
    all_jobs: bool = False,
    host: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve CLI selectors, defaulting to active jobs and then the newest retained job."""

    jobs = load_jobs()
    if host:
        jobs = [job for job in jobs if job["host"] == host]
    if selectors:
        selected = [find_job(selector, jobs) for selector in selectors]
    elif all_jobs:
        selected = jobs
    else:
        selected = [job for job in jobs if job["status"] in ACTIVE_STATUSES]
        if not selected and jobs and not active:
            selected = jobs[:1]
    unique: dict[str, dict[str, Any]] = {job["id"]: job for job in selected}
    return list(unique.values())


def _probe_source(manifests: list[dict[str, Any]]) -> str:
    clean_manifests = []
    for manifest in manifests:
        clean = {key: value for key, value in manifest.items() if not key.startswith("_")}
        clean_manifests.append(clean)
    encoded = base64.b64encode(json.dumps(clean_manifests).encode("utf-8")).decode("ascii")
    module_source = Path(remote_probe.__file__).read_text(encoding="utf-8")
    trailer = f"""
import base64 as _base64
_manifests = json.loads(_base64.b64decode({encoded!r}).decode('utf-8'))
print(json.dumps([collect(item) for item in _manifests], ensure_ascii=False, allow_nan=False))
"""
    return module_source + trailer


def collect_remote_jobs(
    jobs: list[dict[str, Any]],
    *,
    alias_overrides: dict[str, str] | None = None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Probe jobs with one SSH session per host/environment/Conda-initialization tuple."""

    alias_overrides = alias_overrides or {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        environment = job["environment"]
        groups[(job["host"], environment["name"], environment["conda_init"])].append(job)

    reports: list[dict[str, Any]] = []
    for (host, environment_name, conda_init), group in groups.items():
        alias = alias_overrides.get(host, host)
        activation = (
            f"source {shlex.quote(conda_init)} && "
            f"conda activate {shlex.quote(environment_name)} && python -"
        )
        command = [
            "ssh",
            "-o",
            "ConnectTimeout=15",
            alias,
            f"bash -lc {shlex.quote(activation)}",
        ]
        try:
            completed = subprocess.run(
                command,
                input=_probe_source(group),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            reports.extend(
                {
                    "id": job["id"],
                    "host": host,
                    "probe_error": str(exc),
                }
                for job in group
            )
            continue
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            reports.extend(
                {"id": job["id"], "host": host, "probe_error": error} for job in group
            )
            continue
        try:
            values = json.loads(completed.stdout)
        except json.JSONDecodeError:
            error = f"Remote probe returned invalid JSON: {completed.stdout[-1000:]}"
            reports.extend(
                {"id": job["id"], "host": host, "probe_error": error} for job in group
            )
            continue
        reports.extend(values)
    return reports


def _scheduler_lines(scheduler: dict[str, Any]) -> list[str]:
    def summarize_states(output: str) -> str:
        counts: dict[str, int] = {}
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) < 2 or "." in fields[0]:
                continue
            state = fields[1].split()[0].split("+")[0]
            counts[state] = counts.get(state, 0) + 1
        return ", ".join(f"{state}:{count}" for state, count in sorted(counts.items()))

    lines: list[str] = []
    if scheduler.get("tmux_session"):
        state = "active" if scheduler.get("tmux_active") else "inactive"
        lines.append(f"tmux={scheduler['tmux_session']} ({state})")
    squeue = scheduler.get("squeue", {}).get("stdout")
    if squeue:
        lines.append(f"squeue={summarize_states(squeue)}")
    elif scheduler.get("type") == "slurm":
        sacct = scheduler.get("sacct", {}).get("stdout", "")
        summary = summarize_states(sacct) or "no data"
        lines.append(f"Slurm queue empty; sacct(all attempts)={summary}")
    if scheduler.get("process_matches"):
        lines.append(f"processes={len(scheduler['process_matches'])}")
    return lines


def format_report(report: dict[str, Any], job: dict[str, Any]) -> str:
    """Format one remote snapshot as a compact Chinese progress report."""

    header = f"[{job['id']}] {job['description']}"
    if report.get("probe_error"):
        return f"{header}\n  {job['host']}: 连接/检查失败：{report['probe_error']}"
    lines = [
        header,
        f"  host={job['host']} status={job['status']} root={report['remote_root']}",
        "  progress={}/{} valid, {} done, {} failed, {} discovered".format(
            report["valid_units"],
            report["expected_units"],
            report["done_units"],
            report["failed_units"],
            report["discovered_units"],
        ),
    ]
    lines.extend(f"  {line}" for line in _scheduler_lines(report.get("scheduler", {})))
    gpu = report.get("gpu")
    if gpu and gpu.get("stdout"):
        lines.append(f"  gpu={gpu['stdout'].replace(chr(10), '; ')}")
    active_units = [
        unit
        for unit in report.get("units", [])
        if not unit["valid"] and (unit["history_exists"] or unit["metrics_exists"])
    ]
    for unit in active_units[:12]:
        lines.append(
            "  unit={} step={} fps={} return={}{}".format(
                unit["id"],
                unit["global_step"],
                unit["fps"],
                unit["episodic_return_100"],
                " FAILED" if unit["failed"] else "",
            )
        )
    errors = report.get("errors", [])
    if errors:
        lines.append(f"  recent_errors={len(errors)}")
        for error in errors[-5:]:
            lines.append(f"    {error['path']}: {error['line']}")
    else:
        lines.append("  recent_errors=0")
    return "\n".join(lines)


def _update_completed_jobs(
    reports: list[dict[str, Any]], jobs_by_id: dict[str, dict[str, Any]]
) -> None:
    for report in reports:
        job = jobs_by_id.get(report.get("id", ""))
        if not job or report.get("probe_error"):
            continue
        if not job.get("tracking", {}).get("auto_complete", False):
            continue
        expected = report.get("expected_units", 0)
        if expected > 0 and report.get("valid_units") == expected and job["status"] != "completed":
            update_status(job["id"], "completed")
            job["status"] = "completed"


def parse_args() -> argparse.Namespace:
    """Parse the progress command line."""

    parser = argparse.ArgumentParser(description="Check retained remote jobs by exact paths.")
    parser.add_argument("selectors", nargs="*", help="Job ID, Slurm ID, run ID, or name fragment.")
    parser.add_argument("--active", action="store_true", help="Check active records only.")
    parser.add_argument("--all", action="store_true", help="Check all retained records.")
    parser.add_argument("--host", help="Restrict to one logical host name.")
    parser.add_argument(
        "--ssh-alias",
        action="append",
        default=[],
        metavar="HOST=ALIAS",
        help="Override an SSH alias without editing tracked manifests.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--json", action="store_true", help="Print raw JSON snapshots.")
    parser.add_argument(
        "--no-update", action="store_true", help="Do not mark verified jobs completed."
    )
    return parser.parse_args()


def main() -> None:
    """Resolve jobs, probe each host once, and print progress."""

    args = parse_args()
    try:
        jobs = select_jobs(
            args.selectors,
            active=args.active,
            all_jobs=args.all,
            host=args.host,
        )
        if not jobs:
            raise RegistryError("No retained jobs match the requested scope.")
        reports = collect_remote_jobs(
            jobs,
            alias_overrides=_ssh_alias_map(args.ssh_alias),
            timeout=args.timeout,
        )
        jobs_by_id = {job["id"]: job for job in jobs}
        if not args.no_update:
            _update_completed_jobs(reports, jobs_by_id)
        if args.json:
            print(json.dumps(reports, indent=2, ensure_ascii=False))
        else:
            formatted = (format_report(report, jobs_by_id[report["id"]]) for report in reports)
            print("\n\n".join(formatted))
    except RegistryError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
