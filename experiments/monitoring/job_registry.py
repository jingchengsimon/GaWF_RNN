"""Maintain persistent remote-job manifests and generated human/machine indexes.

Inputs are one JSON manifest per submitted experiment under ``jobs/``. Outputs are the
human-readable ``JOBS.md`` history and the machine-readable ``active_jobs.json`` index. Records
are retained after completion or failure; deletion requires an explicit human-confirmation flag.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MONITORING_DIR = Path(__file__).resolve().parent
JOBS_DIR = MONITORING_DIR / "jobs"
ACTIVE_INDEX_PATH = MONITORING_DIR / "active_jobs.json"
HISTORY_PATH = MONITORING_DIR / "JOBS.md"
SCHEMA_VERSION = 1
ACTIVE_STATUSES = frozenset({"queued", "running", "recovering", "unknown"})
VALID_STATUSES = frozenset(
    {"queued", "running", "recovering", "unknown", "completed", "failed", "cancelled"}
)


class RegistryError(RuntimeError):
    """Raised when a job manifest or registry operation is invalid."""


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp without fractional seconds."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    """Convert a human identifier into a stable filename-safe job ID."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_.").lower()
    if not slug:
        raise RegistryError("Job ID must contain at least one letter or number.")
    return slug


def _registry_paths(base_dir: Path | None = None) -> tuple[Path, Path, Path]:
    root = Path(base_dir) if base_dir is not None else MONITORING_DIR
    return root / "jobs", root / "active_jobs.json", root / "JOBS.md"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the fields required to locate and inspect a remote experiment."""

    required = ("id", "description", "host", "status", "remote_root", "environment")
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        raise RegistryError(f"Manifest is missing required fields: {', '.join(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(
            f"Unsupported schema_version={manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    if manifest["status"] not in VALID_STATUSES:
        raise RegistryError(
            f"Invalid status {manifest['status']!r}; choose from {sorted(VALID_STATUSES)}."
        )
    environment = manifest["environment"]
    if not isinstance(environment, dict) or not environment.get("name"):
        raise RegistryError("environment.name is required.")
    if not environment.get("conda_init"):
        raise RegistryError("environment.conda_init is required for remote diagnostics.")
    tracking = manifest.get("tracking", {})
    if not isinstance(tracking, dict):
        raise RegistryError("tracking must be a JSON object.")
    validation_mode = tracking.get("validation_mode", "artifacts")
    if validation_mode not in {"artifacts", "strict"}:
        raise RegistryError("tracking.validation_mode must be 'artifacts' or 'strict'.")
    expected = tracking.get("expected_units", 0)
    if not isinstance(expected, int) or expected < 0:
        raise RegistryError("tracking.expected_units must be a non-negative integer.")
    units = tracking.get("units", [])
    if not isinstance(units, list):
        raise RegistryError("tracking.units must be a list.")
    if units and expected and len(units) != expected:
        raise RegistryError(
            "tracking.units length must equal tracking.expected_units when explicit units are used."
        )
    if tracking.get("auto_complete"):
        if expected <= 0:
            raise RegistryError("tracking.auto_complete requires expected_units > 0.")
        if not units and not tracking.get("result_globs"):
            raise RegistryError("tracking.auto_complete requires units or result_globs.")


def normalize_manifest(
    manifest: dict[str, Any], *, preserve_updated: bool = False
) -> dict[str, Any]:
    """Return a normalized copy with stable IDs and timestamps."""

    normalized = json.loads(json.dumps(manifest))
    normalized["schema_version"] = int(normalized.get("schema_version", SCHEMA_VERSION))
    normalized["id"] = slugify(str(normalized.get("id", "")))
    now = utc_now()
    normalized.setdefault("created_at", now)
    if not preserve_updated:
        normalized["updated_at"] = now
    else:
        normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("scheduler", {"type": "none", "job_ids": [], "run_ids": []})
    normalized.setdefault("paths", {"log_globs": [], "result_paths": []})
    normalized.setdefault("tracking", {"expected_units": 0, "units": []})
    normalized.setdefault("notes", [])
    validate_manifest(normalized)
    return normalized


def load_jobs(base_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load every retained job manifest, newest first."""

    jobs_dir, _, _ = _registry_paths(base_dir)
    jobs: list[dict[str, Any]] = []
    if not jobs_dir.exists():
        return jobs
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            validate_manifest(manifest)
        except (json.JSONDecodeError, OSError, RegistryError) as exc:
            raise RegistryError(f"Cannot load {path}: {exc}") from exc
        manifest["_manifest_path"] = str(path)
        jobs.append(manifest)
    return sorted(jobs, key=lambda job: (job.get("created_at", ""), job["id"]), reverse=True)


def save_job(
    manifest: dict[str, Any],
    *,
    overwrite: bool = False,
    base_dir: Path | None = None,
    preserve_updated: bool = False,
) -> Path:
    """Persist one manifest and rebuild both generated indexes."""

    jobs_dir, _, _ = _registry_paths(base_dir)
    normalized = normalize_manifest(manifest, preserve_updated=preserve_updated)
    path = jobs_dir / f"{normalized['id']}.json"
    if path.exists() and not overwrite:
        raise RegistryError(
            f"Job {normalized['id']!r} already exists; use --overwrite to replace it."
        )
    _atomic_write(path, json.dumps(normalized, indent=2, ensure_ascii=False) + "\n")
    rebuild_indexes(base_dir=base_dir)
    return path


def _flatten_identifiers(job: dict[str, Any]) -> set[str]:
    scheduler = job.get("scheduler", {})
    values: set[str] = {str(job["id"]).lower()}
    for field in ("job_ids", "run_ids"):
        values.update(str(value).lower() for value in scheduler.get(field, []))
    if scheduler.get("tmux_session"):
        values.add(str(scheduler["tmux_session"]).lower())
    for value in job.get("paths", {}).get("result_paths", []):
        values.add(str(value).lower())
    return values


def find_job(selector: str, jobs: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Resolve an ID, scheduler ID, run ID, tmux name, result path, or description fragment."""

    candidates = list(jobs) if jobs is not None else load_jobs()
    needle = selector.strip().lower()
    exact = [job for job in candidates if needle in _flatten_identifiers(job)]
    if len(exact) == 1:
        return exact[0]
    partial = [
        job
        for job in candidates
        if needle in job["id"].lower()
        or needle in str(job.get("description", "")).lower()
        or any(needle in value for value in _flatten_identifiers(job))
    ]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise RegistryError(f"No retained job matches {selector!r}.")
    matches = ", ".join(job["id"] for job in partial[:10])
    raise RegistryError(f"Selector {selector!r} is ambiguous; matches: {matches}")


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def rebuild_indexes(base_dir: Path | None = None) -> None:
    """Regenerate the append-preserving job history and active-job index."""

    _, active_path, history_path = _registry_paths(base_dir)
    jobs = load_jobs(base_dir)
    active_jobs = [job for job in jobs if job["status"] in ACTIVE_STATUSES]
    active_payload = {
        "schema_version": SCHEMA_VERSION,
        "active_statuses": sorted(ACTIVE_STATUSES),
        "jobs": [
            {
                "id": job["id"],
                "description": job["description"],
                "host": job["host"],
                "status": job["status"],
                "job_ids": job.get("scheduler", {}).get("job_ids", []),
                "run_ids": job.get("scheduler", {}).get("run_ids", []),
                "remote_root": job["remote_root"],
                "updated_at": job.get("updated_at"),
            }
            for job in active_jobs
        ],
    }
    _atomic_write(active_path, json.dumps(active_payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Remote Experiment Job History",
        "",
        "该文档由 `python -m experiments.monitoring.job_registry rebuild` 从 `jobs/*.json`",
        "生成。记录默认永久保留；只有人类明确确认后才能删除。它只服务于实验定位和",
        "检测，不是实验协议或项目方法定义。",
        "",
        "| Job | Status | Host | Scheduler / run IDs | Units | Remote root | "
        "Description |",
        "|---|---|---|---|---:|---|---|",
    ]
    for job in jobs:
        scheduler = job.get("scheduler", {})
        identifiers = [*scheduler.get("job_ids", []), *scheduler.get("run_ids", [])]
        if scheduler.get("tmux_session"):
            identifiers.append(scheduler["tmux_session"])
        lines.append(
            "| `{}` | {} | `{}` | {} | {} | `{}` | {} |".format(
                _markdown_escape(job["id"]),
                _markdown_escape(job["status"]),
                _markdown_escape(job["host"]),
                _markdown_escape(", ".join(map(str, identifiers)) or "—"),
                job.get("tracking", {}).get("expected_units", 0),
                _markdown_escape(job["remote_root"]),
                _markdown_escape(job["description"]),
            )
        )
    lines.extend(
        [
            "",
            "单个 job 的精确日志、结果路径、完成条件和备注位于对应的",
            "`jobs/<id>.json`。",
            "",
        ]
    )
    _atomic_write(history_path, "\n".join(lines))


def update_status(
    selector: str,
    status: str,
    *,
    base_dir: Path | None = None,
    completed_at: str | None = None,
) -> Path:
    """Update a retained job status without removing its history."""

    if status not in VALID_STATUSES:
        raise RegistryError(f"Invalid status {status!r}.")
    job = find_job(selector, load_jobs(base_dir))
    job.pop("_manifest_path", None)
    job["status"] = status
    if status == "completed":
        job["completed_at"] = completed_at or utc_now()
    return save_job(job, overwrite=True, base_dir=base_dir)


def remove_job(selector: str, *, human_confirmed: bool, base_dir: Path | None = None) -> None:
    """Remove one record only after explicit human confirmation."""

    if not human_confirmed:
        raise RegistryError("Deletion requires --human-confirmed.")
    job = find_job(selector, load_jobs(base_dir))
    path = Path(job["_manifest_path"])
    path.unlink()
    rebuild_indexes(base_dir=base_dir)


def _new_manifest_from_args(args: argparse.Namespace) -> dict[str, Any]:
    expected = {}
    if args.expected_global_step is not None:
        expected["global_step"] = args.expected_global_step
    defaults: dict[str, Any] = {"expected": expected}
    if args.checkpoint_glob:
        defaults["checkpoint_glob"] = args.checkpoint_glob
    if args.checkpoint_count is not None:
        defaults["checkpoint_count"] = args.checkpoint_count
    return {
        "schema_version": SCHEMA_VERSION,
        "id": args.id,
        "description": args.description,
        "host": args.host,
        "status": args.status,
        "remote_root": args.remote_root,
        "environment": {"name": args.environment, "conda_init": args.conda_init},
        "scheduler": {
            "type": args.scheduler_type,
            "job_ids": args.job_id,
            "run_ids": args.run_id,
            "tmux_session": args.tmux,
        },
        "paths": {
            "log_globs": args.log_glob,
            "status_dir": args.status_dir,
            "result_paths": args.result_path,
        },
        "tracking": {
            "expected_units": args.expected_units,
            "auto_complete": args.auto_complete,
            "result_globs": args.result_path,
            "done_glob": args.done_glob,
            "fail_glob": args.fail_glob,
            "defaults": defaults,
            "units": [],
        },
        "notes": args.note,
    }


def parse_args() -> argparse.Namespace:
    """Parse the registry command line."""

    parser = argparse.ArgumentParser(description="Maintain persistent remote experiment records.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register a complete JSON manifest.")
    register.add_argument("manifest", type=Path)
    register.add_argument("--overwrite", action="store_true")

    new = subparsers.add_parser("new", help="Register a minimal job from command-line fields.")
    new.add_argument("--id", required=True)
    new.add_argument("--description", required=True)
    new.add_argument("--host", required=True)
    new.add_argument("--remote-root", required=True)
    new.add_argument("--conda-init", required=True)
    new.add_argument("--environment", default="aim3_rnn")
    new.add_argument("--status", choices=sorted(VALID_STATUSES), default="queued")
    new.add_argument(
        "--scheduler-type",
        choices=("slurm", "tmux", "process", "none"),
        default="none",
    )
    new.add_argument("--job-id", action="append", default=[])
    new.add_argument("--run-id", action="append", default=[])
    new.add_argument("--tmux")
    new.add_argument("--log-glob", action="append", default=[])
    new.add_argument("--status-dir")
    new.add_argument("--result-path", action="append", default=[])
    new.add_argument("--expected-units", type=int, default=0)
    new.add_argument("--done-glob")
    new.add_argument("--fail-glob")
    new.add_argument("--expected-global-step", type=int)
    new.add_argument("--checkpoint-glob")
    new.add_argument("--checkpoint-count", type=int)
    new.add_argument("--auto-complete", action="store_true")
    new.add_argument("--note", action="append", default=[])

    show = subparsers.add_parser("show", help="Print one retained manifest.")
    show.add_argument("selector")

    listing = subparsers.add_parser("list", help="List retained jobs.")
    listing.add_argument("--active", action="store_true")

    status = subparsers.add_parser("set-status", help="Update status without deleting history.")
    status.add_argument("selector")
    status.add_argument("status", choices=sorted(VALID_STATUSES))

    subparsers.add_parser("rebuild", help="Rebuild JOBS.md and active_jobs.json.")

    remove = subparsers.add_parser("remove", help="Delete only with explicit human confirmation.")
    remove.add_argument("selector")
    remove.add_argument("--human-confirmed", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested registry operation."""

    args = parse_args()
    try:
        if args.command == "register":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            path = save_job(manifest, overwrite=args.overwrite)
            print(f"registered {path}")
        elif args.command == "new":
            path = save_job(_new_manifest_from_args(args))
            print(f"registered {path}")
        elif args.command == "show":
            job = find_job(args.selector)
            job.pop("_manifest_path", None)
            print(json.dumps(job, indent=2, ensure_ascii=False))
        elif args.command == "list":
            jobs = load_jobs()
            if args.active:
                jobs = [job for job in jobs if job["status"] in ACTIVE_STATUSES]
            for job in jobs:
                scheduler = job.get("scheduler", {})
                ids = [*scheduler.get("job_ids", []), *scheduler.get("run_ids", [])]
                print(f"{job['id']}\t{job['status']}\t{job['host']}\t{','.join(map(str, ids))}")
        elif args.command == "set-status":
            print(f"updated {update_status(args.selector, args.status)}")
        elif args.command == "rebuild":
            rebuild_indexes()
            print(f"rebuilt {HISTORY_PATH} and {ACTIVE_INDEX_PATH}")
        elif args.command == "remove":
            remove_job(args.selector, human_confirmed=args.human_confirmed)
            print(f"removed {args.selector}")
    except (RegistryError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
