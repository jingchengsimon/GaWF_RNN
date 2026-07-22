"""Define the category-indexed analysis figure and data locations.

Inputs are an analysis category, a producing script name, and ``figs`` or ``data``.
Figures are written directly below ``results/anal_figs/<CATEGORY>/`` and data below the
parallel ``results/anal_data/<CATEGORY>/<script_name>/`` tree.  A run manifest is written
in the data tree and records both roots.  A script may call :func:`output_dir` for more than
one category.
"""

from __future__ import annotations

import atexit
import datetime as dt
import inspect
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


CATEGORIES = (
    "A_raw_gate",
    "B_gate_by_context",
    "C_delta_gate",
    "D_variance_decomposition",
    "E_relevance_alignment",
    "F_timing",
    "G_behaviour",
    "H_controls",
)
KINDS = ("figs", "data")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _RunRecord:
    script_path: str
    started_at: str
    roots: set[Path] = field(default_factory=set)
    initial_files: dict[Path, dict[str, tuple[int, int]]] = field(default_factory=dict)


_RUNS: dict[str, _RunRecord] = {}


def _caller_script() -> str:
    this_file = Path(__file__).resolve()
    for frame in inspect.stack()[2:]:
        candidate = Path(frame.filename).resolve()
        if candidate != this_file:
            try:
                return candidate.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return candidate.as_posix()
    return "unknown"


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _flatten_numbers(value: object, prefix: str, output: dict[str, float]) -> None:
    if len(output) >= 500:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numbers(child, child_prefix, output)


def _key_results(root: Path, files: list[str], existing: object) -> dict[str, float]:
    output = {
        str(key): float(value)
        for key, value in (existing.items() if isinstance(existing, dict) else ())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    for relative in files:
        if not relative.endswith(".json") or len(output) >= 500:
            continue
        try:
            payload = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        _flatten_numbers(payload, Path(relative).stem, output)
    return output


def _output_root(kind: str) -> Path:
    """Return the split analysis root for one output kind."""

    return PROJECT_ROOT / "results" / ("anal_figs" if kind == "figs" else "anal_data")


def _run_identity(root: Path) -> tuple[str, str, str | None]:
    """Return ``(kind, category, script_name)`` for a canonical output root."""

    for kind in KINDS:
        try:
            relative = root.relative_to(_output_root(kind))
        except ValueError:
            continue
        if kind == "data" and len(relative.parts) == 2:
            return kind, relative.parts[0], relative.parts[1]
        if kind == "figs" and len(relative.parts) == 1:
            return kind, relative.parts[0], None
    raise ValueError(f"Not a canonical analysis run root: {root}")


def _write_manifests() -> None:
    commit = _git_commit()
    for record in _RUNS.values():
        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for root in record.roots:
            kind, category, script_name = _run_identity(root)
            if script_name is None:
                script_name = Path(record.script_path).stem
            initial = record.initial_files.get(root, {})
            changed: list[str] = []
            for path in root.rglob("*"):
                if not path.is_file() or path.name == "manifest.json":
                    continue
                relative = path.relative_to(root).as_posix()
                stat = path.stat()
                if initial.get(relative) != (stat.st_mtime_ns, stat.st_size):
                    changed.append(relative)
            grouped.setdefault((category, script_name), {})[kind] = sorted(changed)

        for (category, script_name), changed_by_kind in grouped.items():
            data_root = _output_root("data") / category / script_name
            figure_root = _output_root("figs") / category
            manifest_path = data_root / "manifest.json"
            existing: dict[str, object] = {}
            if manifest_path.is_file():
                try:
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            data_files = changed_by_kind.get("data", [])
            figure_files = changed_by_kind.get("figs", [])
            files = [f"data/{path}" for path in data_files]
            files.extend(f"figs/{path}" for path in figure_files)
            if not files and not manifest_path.is_file():
                continue
            key_results = _key_results(
                data_root,
                data_files,
                existing.get("key_numerical_results", {}),
            )
            payload = {
                "script_path": record.script_path,
                "git_commit": commit,
                "timestamp": record.started_at,
                "category": category,
                "data_root": str(data_root.relative_to(PROJECT_ROOT)),
                "figure_root": str(figure_root.relative_to(PROJECT_ROOT)),
                "data_files": data_files,
                "figure_files": figure_files,
                "files_written": sorted(files),
                "key_numerical_results": key_results,
            }
            data_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


atexit.register(_write_manifests)


def output_dir(category: str, script_name: str, kind: str) -> Path:
    """Return and create the canonical output directory for one analysis script.

    ``category`` must be one of the eight declared analysis categories and ``kind`` must be
    either ``figs`` or ``data``.  ``script_name`` is a single safe path component, normally
    the producing module's basename without ``.py``.
    """

    if category not in CATEGORIES:
        raise ValueError(f"Unknown analysis category {category!r}; expected one of {CATEGORIES}")
    if kind not in KINDS:
        raise ValueError(f"Unknown analysis output kind {kind!r}; expected one of {KINDS}")
    if not script_name or Path(script_name).name != script_name or script_name in {".", ".."}:
        raise ValueError("script_name must be one non-empty path component")

    root = (
        _output_root("figs") / category
        if kind == "figs"
        else _output_root("data") / category / script_name
    )
    destination = root
    destination.mkdir(parents=True, exist_ok=True)
    script_path = _caller_script()
    if Path(script_path).stem != script_name:
        return destination
    record = _RUNS.setdefault(
        script_path,
        _RunRecord(
            script_path=script_path,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    if root not in record.roots:
        record.initial_files[root] = {
            path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
            for path in root.rglob("*")
            if path.is_file()
        }
    record.roots.add(root)
    return destination
