"""Resolve retained jobs and inspect their exact remote scheduler, log, and result paths.

A complete experiment ID reads only its matching manifest. The manifest supplies the host and
exact remote paths for one consolidated foreground SSH session. Output is a concise Chinese
progress summary or JSON for downstream automation.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.monitoring import remote_probe
from experiments.monitoring.job_registry import (
    RegistryError,
    load_job,
    update_status,
)


_PROBE_JSON_MARKER = "__AIM3_PROGRESS_JSON__="


def _ssh_alias_map(values: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RegistryError("--ssh-alias must use HOST=ALIAS format.")
        host, alias = value.split("=", 1)
        aliases[host.strip()] = alias.strip()
    return aliases


def _local_ssh_commands(path: Path | None = None) -> dict[str, list[str]]:
    """Read explicit full SSH commands from the ignored ``.agents/local.md`` file.

    A full command is an intentional direct endpoint, not a fallback after a failed control-socket
    check. Human-readable labels are normalized so ``DSW 5000`` maps to logical host
    ``dsw-5000``. Alias-only entries continue to require an existing ControlMaster socket.
    """

    config_path = path or Path(__file__).resolve().parents[2] / ".agents" / "local.md"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    commands: dict[str, list[str]] = {}
    for line in lines:
        match = re.fullmatch(r"\s*-\s+([^:]+):\s+`([^`]+)`\s*", line)
        if not match:
            continue
        label, value = match.groups()
        argv = shlex.split(value)
        if not argv or argv[0] != "ssh":
            continue
        if len(argv) < 2 or any(token in {";", "&&", "||", "|"} for token in argv):
            raise RegistryError(f"Unsafe SSH command in {config_path}: {value!r}")
        host = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        commands[host] = [
            str(Path(token).expanduser()) if token.startswith("~/") else token for token in argv
        ]
    return commands


def _direct_ssh_command(base: list[str], remote_command: str) -> list[str]:
    """Append one remote command to a validated full SSH endpoint command."""

    if not base or base[0] != "ssh" or len(base) < 2 or base[-1].startswith("-"):
        raise RegistryError(f"Invalid direct SSH command: {base!r}")
    return [
        *base[:-1],
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        base[-1],
        remote_command,
    ]


def select_job(experiment_id: str, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Load one manifest by its complete experiment ID."""

    return load_job(experiment_id, base_dir)


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
print({_PROBE_JSON_MARKER!r} + json.dumps(
    [collect(item) for item in _manifests], ensure_ascii=False, allow_nan=False
))
"""
    return module_source + trailer


def collect_remote_jobs(
    jobs: list[dict[str, Any]],
    *,
    alias_overrides: dict[str, str] | None = None,
    direct_commands: dict[str, list[str]] | None = None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Probe jobs with one SSH session per host/environment/Conda-initialization tuple."""

    alias_overrides = alias_overrides or {}
    direct_commands = direct_commands or {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        environment = job["environment"]
        groups[(job["host"], environment["name"], environment["conda_init"])].append(job)

    reports: list[dict[str, Any]] = []
    for (host, environment_name, conda_init), group in groups.items():
        alias = alias_overrides.get(host, host)
        direct = direct_commands.get(host) if host not in alias_overrides else None
        if direct is None:
            try:
                socket_check = subprocess.run(
                    ["ssh", "-O", "check", alias],
                    text=True,
                    capture_output=True,
                    timeout=min(timeout, 15),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                reports.extend(
                    {
                        "id": job["id"],
                        "host": host,
                        "probe_error": f"SSH socket check failed: {exc}",
                    }
                    for job in group
                )
                continue
            if socket_check.returncode != 0:
                error = socket_check.stderr.strip() or socket_check.stdout.strip()
                if not error:
                    error = (
                        f"ssh -O check {alias} failed with exit code "
                        f"{socket_check.returncode}."
                    )
                reports.extend(
                    {
                        "id": job["id"],
                        "host": host,
                        "probe_error": f"SSH socket unavailable: {error}",
                    }
                    for job in group
                )
                continue
        activation = (
            f"source {shlex.quote(conda_init)} && "
            f"conda activate {shlex.quote(environment_name)} && python -"
        )
        remote_command = f"bash -lc {shlex.quote(activation)}"
        command = (
            _direct_ssh_command(direct, remote_command)
            if direct is not None
            else ["ssh", "-o", "ConnectTimeout=15", alias, remote_command]
        )
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
        payload = completed.stdout.rsplit(_PROBE_JSON_MARKER, 1)[-1].strip()
        try:
            values = json.loads(payload)
        except json.JSONDecodeError:
            output = completed.stdout[-1000:] or "<empty stdout>"
            error = f"Remote probe returned invalid JSON: {output}"
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
    rolling = scheduler.get("rolling")
    if rolling:
        lines.append(
            "rolling={} submitted, {} completed, {} blocked, {} batches".format(
                rolling["submitted"], rolling["completed"],
                rolling["blocked"], rolling["batches"],
            )
        )
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
    expected = report.get("expected_units", 0)
    verified_complete = expected > 0 and report.get("valid_units") == expected
    observed_status = "completed (verified)" if verified_complete else job["status"]
    lines = [
        header,
        f"  host={job['host']} status={observed_status} root={report['remote_root']}",
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
    parser.add_argument(
        "experiment_id",
        help="Complete experiment ID from JOBS.md.",
    )
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
        jobs = [select_job(args.experiment_id)]
        reports = collect_remote_jobs(
            jobs,
            alias_overrides=_ssh_alias_map(args.ssh_alias),
            direct_commands=_local_ssh_commands(),
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
        print(f"progress_error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
