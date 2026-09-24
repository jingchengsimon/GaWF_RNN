"""CPU regression checks for GaWF runtime state and formal submission guards."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.training.train_scripts.clutter import (
    GaWFRNNConv,
    _load_clutter_checkpoint,
    _load_clutter_model_state,
    _save_clutter_checkpoint,
)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_legacy_feedback_resume(num_layers: int, tmp_path: Path) -> None:
    model = GaWFRNNConv(10, 9, hidden_size=8, device="cpu", num_layers=num_layers)
    model.prev_feedback = torch.ones(3, 19)
    assert "prev_feedback" not in model.state_dict()
    optimizer = torch.optim.AdamW(model.parameters())
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    components = {"optim": optimizer, "train_acc_char": np.arange(150, dtype=np.float32)}
    path = str(tmp_path / "train_state.pth")
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _save_clutter_checkpoint(
        path, mdl=model, components=components, metadata={"seed": 1}, completed_epochs=20,
        best_val_acc=0.8, best_epoch_idx=19, best_state=expected,
        epochs_without_improvement=0, stopped_by_patience=False,
    )
    payload = torch.load(path, weights_only=False)
    assert "prev_feedback" not in payload["model"]
    # Reproduce pre-fix checkpoints, including legacy best-validation state.
    payload["model"]["prev_feedback"] = torch.ones(3, 19)
    payload["best_state"]["prev_feedback"] = torch.ones(7, 19)
    torch.save(payload, path)
    target = GaWFRNNConv(10, 9, hidden_size=8, device="cpu", num_layers=num_layers)
    target.prev_feedback = torch.zeros(2, 19)
    restored = {
        "optim": torch.optim.AdamW(target.parameters()),
        "train_acc_char": np.zeros(150, dtype=np.float32),
    }
    loaded = _load_clutter_checkpoint(
        path, mdl=target, components=restored, expected_metadata={"seed": 1},
    )
    assert loaded["completed_epochs"] == 20
    assert target.prev_feedback is None
    assert restored["optim"].state_dict()["state"]
    np.testing.assert_array_equal(restored["train_acc_char"], components["train_acc_char"])
    for key, value in expected.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    _load_clutter_model_state(target, loaded["best_state"])
    broken = dict(expected)
    del broken[next(iter(broken))]
    with pytest.raises(RuntimeError, match="learned-state mismatch"):
        _load_clutter_model_state(target, broken)


def test_submit_dependencies_and_recovery_guards(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    submitter = root / "experiments/clutter/amarel/submit_clutter_data_scale_formal.sh"
    args = ["bash", str(submitter), "--scale", "20h", "--array-tasks", "30"]
    env = dict(os.environ, AIM3_ROOT=str(tmp_path), AIM3_RESULTS_PATH=str(tmp_path / "results"),
               AIM3_CLUTTER_DATA_DIR=str(tmp_path / "data"))
    data = tmp_path / "data"
    data.mkdir()
    for name in ("stimulus_reg-train-20h-uint8.npy", "stimulus_reg-train-20h-uint8.tsv",
                 "stimulus_reg-validation-40h-uint8.npy",
                 "stimulus_reg-validation-40h-uint8.tsv", "generation-20h-uint8.json"):
        (data / name).write_text("test")
    target = tmp_path / "results/data/clutter/runs/data_scale/clutter_formal_4scale_ep150"
    target = target / "20h/gawf-seed01"
    target.mkdir(parents=True)
    (target / "gawf_train_state.pth").write_text("checkpoint")
    bins = tmp_path / "bin"
    bins.mkdir()
    sbatch = bins / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$AIM3_ROOT/call.txt"\necho 12345\n')
    sbatch.chmod(0o755)
    env["PATH"] = str(bins) + os.pathsep + env["PATH"]
    assert subprocess.run(args, env=env, capture_output=True).returncode != 0
    resume = args + ["--resume-existing", "--status-tag", "resume-test",
                     "--dependency", "afterok:123_50:456"]
    result = subprocess.run(resume, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--dependency=afterok:123_50:456" in (tmp_path / "call.txt").read_text()
    assert subprocess.run(resume, env=env, capture_output=True).returncode != 0
    (target / "gawf_model.pth").write_text("final")
    assert subprocess.run(resume, env=env, capture_output=True).returncode != 0
    assert subprocess.run(args + ["--dry-run", "--dependency", "afterany:1"],
                          env=env, capture_output=True).returncode != 0
