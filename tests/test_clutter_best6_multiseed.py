"""Tests for the fixed-best six-model Clutter multi-seed task mapping."""
from __future__ import annotations

import json

from experiments.generalization.clutter_best6_multiseed import (
    MODEL_SPECS,
    NUM_EPOCHS,
    all_task_configs,
    build_manifest,
    task_config,
    validate_task_output,
)


def test_task_mapping_has_six_models_for_each_of_ten_seeds() -> None:
    """The Slurm array must map exactly 60 unique seed/model units."""

    tasks = all_task_configs()
    assert len(tasks) == 60
    assert len({task.unit_id for task in tasks}) == 60
    assert task_config(0).unit_id == "gawf-seed01"
    assert task_config(5).unit_id == "s5-seed01"
    assert task_config(59).unit_id == "s5-seed10"


def test_frozen_model_hyperparameters_match_selected_checkpoints() -> None:
    """The frozen widths and optimizer values must not drift from model selection."""

    assert (MODEL_SPECS["gawf"].width, MODEL_SPECS["gawf"].lr) == (256, 5e-3)
    assert (MODEL_SPECS["rnn"].width, MODEL_SPECS["rnn"].weight_decay) == (275, 1e-5)
    assert MODEL_SPECS["lstm"].width == 80
    assert MODEL_SPECS["gru"].width == 105
    assert MODEL_SPECS["mamba"].width == 170
    assert (MODEL_SPECS["s5"].width, MODEL_SPECS["s5"].state_size) == (256, 128)


def test_manifest_tracks_every_unit_with_strict_epoch_and_seed_evidence() -> None:
    """Monitoring must require all 60 full-epoch checkpoints and matching seeds."""

    manifest = build_manifest("12345678", "/remote/root", "/conda/init.sh")
    tracking = manifest["tracking"]
    assert tracking["expected_units"] == 60
    assert tracking["auto_complete"] is True
    assert len(tracking["units"]) == 60
    first = tracking["units"][0]
    assert first["checkpoint_count"] == 1
    assert first["expected"]["seed"] == 1
    assert first["expected"]["actual_epochs"] == NUM_EPOCHS
    assert first["expected"]["stopped_by_patience"] is False


def test_validator_requires_matching_seed_and_full_epoch_completion(tmp_path) -> None:
    """A complete file trio is valid only when its metrics match the exact protocol."""

    config = task_config(0)
    metrics_path = tmp_path / config.metrics_relpath
    checkpoint_path = tmp_path / config.checkpoint_relpath
    pickle_path = tmp_path / config.pickle_relpath
    metrics_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    pickle_path.touch()
    metrics = {
        "model_type": "gawf",
        "seed": 1,
        "num_epochs": 150,
        "actual_epochs": 150,
        "patience": 0,
        "stopped_by_patience": False,
        "dataset_suffix": "40h-float32",
        "eval_dataset_suffix": "40h-float32",
        "dataset_mode": "sector",
        "use_acceleration": True,
        "use_mmap": True,
        "hidden_size": 256,
        "lr": 0.005,
        "weight_decay": 0.001,
        "cnn_dropout": 0.0,
        "rnn_dropout": 0.5,
        "num_layers": 1,
        "val_acc_at_best": 90.0,
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert validate_task_output(config, str(tmp_path))["valid"] is True

    metrics["actual_epochs"] = 149
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report = validate_task_output(config, str(tmp_path))
    assert report["valid"] is False
    assert "actual_epochs" in report["failed_checks"]
