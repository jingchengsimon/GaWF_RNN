"""Verify smoke defaults and the explicitly authorized no-smoke submission path."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("skip_smoke", [False, True])
def test_submission_count_and_smoke_dependency(tmp_path: Path, skip_smoke: bool) -> None:
    sources = tmp_path / "sources"
    for model in ("lstm", "gru", "gawf"):
        leaf = sources / model
        leaf.mkdir(parents=True)
        (leaf / "model.pth").write_text("fixture")
        (leaf / "metrics.json").write_text("{}")
    (sources / "SHA256SUMS").write_text("fixture")
    calls = tmp_path / "calls"
    binary = tmp_path / "bin"
    binary.mkdir()
    sbatch = binary / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho 12345\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, AIM3_ROOT=str(ROOT), AIM3_RESULTS_PATH=str(tmp_path / "results"),
               SKIING_SOURCE_ROOT=str(sources), AIM3_CONDA_SH=str(tmp_path / "conda.sh"),
               CALLS=str(calls), PATH=str(binary) + os.pathsep + os.environ["PATH"])
    command = ["bash", str(ROOT / "experiments/rl/atari/amarel/"
                           "submit_atari_skiing_unclipped_l3.sh")]
    if skip_smoke:
        command.append("--skip-smoke")
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    submitted = calls.read_text().splitlines()
    assert len(submitted) == (1 if skip_smoke else 2)
    assert "RUN_PHASE=formal" in submitted[-1]
    assert ("--dependency=afterok:12345" in submitted[-1]) is (not skip_smoke)
    assert (any("RUN_PHASE=smoke" in call for call in submitted)) is (not skip_smoke)
