"""Evaluate dashboard task progress from result files on a remote filesystem.

This module intentionally uses only the Python standard library so the collector can
send it to Amarel or sjc-remote and execute it without installing dashboard packages.
"""
from __future__ import annotations

from glob import glob
import json
import math
import os
from typing import Any


def _matches_value(actual: Any, condition: dict[str, Any]) -> bool:
    if "equals" in condition:
        expected = condition["equals"]
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
        return actual == expected
    if "in" in condition:
        return any(_matches_value(actual, {"equals": value}) for value in condition["in"])
    raise ValueError(f"Unsupported match condition: {condition}")


def _matches_metrics(metrics: dict[str, Any], tracker: dict[str, Any]) -> bool:
    for key, condition in tracker.get("match", {}).items():
        if key not in metrics or not _matches_value(metrics[key], condition):
            return False

    combinations = tracker.get("allowed_combinations", [])
    if combinations:
        if not any(
            all(_matches_value(metrics.get(key), {"equals": value}) for key, value in row.items())
            for row in combinations
        ):
            return False
    return True


def _companion_path(metrics_path: str, template: str) -> str:
    suffix = "_metrics.json"
    stem = metrics_path[: -len(suffix)] if metrics_path.endswith(suffix) else metrics_path
    return template.format(stem=stem, metrics=metrics_path)


def _evaluate_metrics_grid(remote_root: str, tracker: dict[str, Any]) -> dict[str, Any]:
    pattern = os.path.join(remote_root, tracker["result_glob"])
    unique_fields = tracker.get("uniqueness_fields", [])
    valid_keys: set[tuple[Any, ...]] = set()
    invalid_files = 0

    for metrics_path in sorted(glob(pattern, recursive=True)):
        try:
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid_files += 1
            continue
        if not _matches_metrics(metrics, tracker):
            continue
        companion_templates = tracker.get(
            "companion_files", ["{stem}.pkl", "{stem}_model.pth"]
        )
        if not all(os.path.isfile(_companion_path(metrics_path, item)) for item in companion_templates):
            invalid_files += 1
            continue
        key = tuple(metrics.get(field) for field in unique_fields) if unique_fields else (metrics_path,)
        valid_keys.add(key)

    return {
        "valid_count": min(len(valid_keys), int(tracker["expected_total"])),
        "expected_total": int(tracker["expected_total"]),
        "invalid_files": invalid_files,
    }


def _evaluate_explicit_units(remote_root: str, tracker: dict[str, Any]) -> dict[str, Any]:
    units = tracker["units"]
    valid_count = 0
    for unit in units:
        required = [os.path.join(remote_root, path) for path in unit["required_files"]]
        if all(os.path.isfile(path) for path in required):
            valid_count += 1
    return {
        "valid_count": valid_count,
        "expected_total": int(tracker.get("expected_total", len(units))),
        "invalid_files": 0,
    }


def evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return valid/expected counts for one registered task."""
    remote_root = os.path.expanduser(task["remote_root"])
    tracker = task["tracker"]
    tracker_type = tracker["type"]
    if tracker_type == "metrics_grid":
        result = _evaluate_metrics_grid(remote_root, tracker)
    elif tracker_type == "explicit_units":
        result = _evaluate_explicit_units(remote_root, tracker)
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")
    result["task_id"] = task["id"]
    return result

