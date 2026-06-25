"""Register future remote jobs in the persistent dashboard task registry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dashboard.collector import DEFAULT_REGISTRY, load_registry


RETENTION_POLICY = "human_confirmation_required"
INSTALLED_REGISTRY = (
    Path.home() / "Library" / "Application Support" / "FAW_RNN Dashboard" / "tasks.json"
)


def _write_registry(path: Path, registry: dict[str, Any]) -> None:
    _write_json(path, registry)
    if path.resolve() == DEFAULT_REGISTRY.resolve() and INSTALLED_REGISTRY.exists():
        _write_json(INSTALLED_REGISTRY, registry)


def _write_json(path: Path, registry: dict[str, Any]) -> None:
    """Atomically write one registry copy."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered human-readable task descriptions")

    register = subparsers.add_parser("register", help="Register or replace one JSON task spec")
    register.add_argument(
        "--spec",
        required=True,
        help="JSON object containing id, description, machine, job_ids, remote_root, and tracker",
    )

    remove = subparsers.add_parser("remove", help="Remove one registered task")
    remove.add_argument("task_id")
    remove.add_argument(
        "--human-confirmed",
        action="store_true",
        help="Required acknowledgement that a human said this task is no longer needed",
    )
    return parser.parse_args()


def _validate_spec(spec: dict[str, Any]) -> None:
    required = {"id", "description", "machine", "job_ids", "remote_root", "tracker"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Task spec is missing {sorted(missing)}")
    if spec["machine"] not in {"amarel", "sjc-remote"}:
        raise ValueError("machine must be amarel or sjc-remote")
    if not spec["description"].strip():
        raise ValueError("description must be human-readable and non-empty")
    if not isinstance(spec["job_ids"], list) or not spec["job_ids"]:
        raise ValueError("job_ids must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in spec["job_ids"]):
        raise ValueError("every job id must be a non-empty string")
    if not isinstance(spec["tracker"], dict):
        raise ValueError("tracker must be an object")
    if spec["tracker"].get("type") not in {"metrics_grid", "explicit_units"}:
        raise ValueError("tracker.type must be metrics_grid or explicit_units")
    if int(spec["tracker"].get("expected_total", 0)) <= 0:
        raise ValueError("tracker.expected_total must be positive")
    if spec.get("retention_policy") != RETENTION_POLICY:
        raise ValueError(f"retention_policy must be {RETENTION_POLICY!r}")


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a validated task spec with the mandatory retention policy."""
    normalized = dict(spec)
    normalized.setdefault("retention_policy", RETENTION_POLICY)
    normalized["job_ids"] = [str(item) for item in normalized.get("job_ids", [])]
    _validate_spec(normalized)
    return normalized


def register_task(registry: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Register or replace one task while preserving the human-confirmation policy."""
    normalized = normalize_spec(spec)
    updated = dict(registry)
    updated["tasks"] = [
        task for task in registry.get("tasks", []) if task["id"] != normalized["id"]
    ]
    updated["tasks"].append(normalized)
    return updated


def remove_task(
    registry: dict[str, Any],
    task_id: str,
    *,
    human_confirmed: bool,
) -> dict[str, Any]:
    """Remove a task only after explicit human confirmation."""
    if not human_confirmed:
        raise ValueError(
            "Refusing to remove dashboard task without explicit human confirmation"
        )
    updated = dict(registry)
    updated["tasks"] = [
        task for task in registry.get("tasks", []) if task["id"] != task_id
    ]
    return updated


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    if args.command == "list":
        for task in registry["tasks"]:
            print(f"{task['id']}: {task['description']}")
        return
    if args.command == "register":
        spec = normalize_spec(json.loads(args.spec))
        registry = register_task(registry, spec)
        _write_registry(args.registry, registry)
        print(f"Registered: {spec['description']}")
        return
    try:
        registry = remove_task(
            registry,
            args.task_id,
            human_confirmed=args.human_confirmed,
        )
    except ValueError as exc:
        parser = argparse.ArgumentParser(prog="dashboard.manage_tasks remove")
        parser.error(str(exc))
    _write_registry(args.registry, registry)
    print(f"Removed: {args.task_id}")


if __name__ == "__main__":
    main()
