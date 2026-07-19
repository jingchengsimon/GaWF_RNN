"""Tests for the category-indexed analysis output path contract."""

from __future__ import annotations

import json

import pytest

import utils_anal.anal_paths as anal_paths
from utils_anal.anal_paths import output_dir


def test_output_dir_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="Unknown analysis category"):
        output_dir("Z_misc", "probe", "data")


def test_output_dir_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown analysis output kind"):
        output_dir("H_controls", "probe", "cache")


def test_output_dir_rejects_nested_script_name() -> None:
    with pytest.raises(ValueError, match="one non-empty path component"):
        output_dir("H_controls", "nested/probe", "data")


def test_output_dir_writes_manifest_for_files_changed_during_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(anal_paths, "PROJECT_ROOT", tmp_path)
    anal_paths._RUNS.clear()
    data_dir = output_dir("H_controls", "test_anal_paths", "data")
    (data_dir / "result.json").write_text('{"score": 1}\n', encoding="utf-8")
    anal_paths._write_manifests()
    manifest = json.loads((data_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["category"] == "H_controls"
    assert manifest["files_written"] == ["data/result.json"]
    assert manifest["script_path"].endswith("tests/test_anal_paths.py")
    assert manifest["key_numerical_results"] == {"result.score": 1.0}
    anal_paths._RUNS.clear()
