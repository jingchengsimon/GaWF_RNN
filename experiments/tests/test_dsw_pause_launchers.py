"""Tests for the coordinated DSW checkpoint-pause launchers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DSW_DIR = ROOT / "experiments" / "launchers" / "clutter" / "dsw"


def _load_remote_module():
    spec = importlib.util.spec_from_file_location("pause_remote", DSW_DIR / "pause_remote.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_target_uses_current_or_next_periodic_checkpoint(tmp_path: Path) -> None:
    remote = _load_remote_module()
    log_path = tmp_path / "seed.log"
    log_path.write_text(
        "Epoch 74/150 wall_time_sec=140.0\n"
        "Saved resumable checkpoint: epoch=70 path=/tmp/state.pth\n",
        encoding="utf-8",
    )
    completed, checkpoint = remote._parse_progress(log_path)
    assert (completed, checkpoint) == (74, 70)
    assert remote._target_checkpoint(completed, checkpoint, 5, 150) == 75
    assert remote._target_checkpoint(75, 75, 5, 150) == 75


def test_example_config_has_two_distinct_campaigns() -> None:
    config = json.loads((DSW_DIR / "pause_campaigns.example.json").read_text(encoding="utf-8"))
    assert config["schema_version"] == 1
    assert len(config["campaigns"]) == 2
    assert len({campaign["name"] for campaign in config["campaigns"]}) == 2


def test_execute_requires_confirmation_before_ssh() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(DSW_DIR / "pause_campaigns.py"),
            "--config",
            str(DSW_DIR / "pause_campaigns.example.json"),
            "--execute",
            "--confirm",
            "WRONG",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "no pause action was taken" in result.stderr
