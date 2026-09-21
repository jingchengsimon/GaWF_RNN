"""Unit checks for the IMDB ten-seed behaviour comparison."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from utils.analysis.text.imdb_best10seed_behavior import (
    EXPECTED_EPOCHS,
    MODEL_ORDER,
    load_series,
    write_structured_outputs,
)


def _write_series(root: Path, *, seeds: int = 10, models: tuple[str, ...] = MODEL_ORDER) -> None:
    """Write a synthetic five-model series with deterministic metrics and histories."""

    for model_index, model in enumerate(models):
        for seed in range(1, seeds + 1):
            directory = root / model / f"seed_{seed:02d}"
            directory.mkdir(parents=True)
            train_acc = np.linspace(0.6, 0.99, EXPECTED_EPOCHS)
            val_acc = np.linspace(0.6, 0.85 + 0.01 * seed, EXPECTED_EPOCHS)
            train_loss = np.linspace(0.7, 0.05, EXPECTED_EPOCHS)
            val_loss = np.linspace(0.7, 0.30 + 0.01 * model_index, EXPECTED_EPOCHS)
            history = {
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            (directory / f"{model}_imdb_history.pkl").write_bytes(pickle.dumps(history))
            metrics = {
                "model_type": model,
                "seed": seed,
                "num_epochs": EXPECTED_EPOCHS,
                "actual_epochs": EXPECTED_EPOCHS,
                "best_epoch_val_acc_1based": 5,
                "test_acc_at_best": 0.80 + 0.01 * seed,
                "test_loss_at_best": 0.30 + 0.01 * model_index,
                "core_param_count": 1000 + model_index,
                "total_param_count": 10_000 + model_index,
            }
            (directory / f"{model}_imdb_metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )


def test_load_series_and_outputs(tmp_path: Path) -> None:
    root = tmp_path / "series"
    _write_series(root)

    series = load_series(root)
    assert sorted(series) == sorted(MODEL_ORDER)
    assert all(len(seeds) == 10 for seeds in series.values())

    data_dir = tmp_path / "out"
    written = write_structured_outputs(series, data_dir)
    assert written["npz"].is_file() and written["csv"].is_file() and written["key"].is_file()

    key = json.loads(written["key"].read_text(encoding="utf-8"))
    assert key["gawf.test_acc_mean"] == pytest.approx(0.855)
    assert key["gawf.n_seeds"] == 10
    train_at_best = np.linspace(0.6, 0.99, EXPECTED_EPOCHS)[4]
    val_at_best = np.mean(
        [
            np.linspace(0.6, 0.85 + 0.01 * seed, EXPECTED_EPOCHS)[4]
            for seed in range(1, 11)
        ]
    )
    expected_gap = (train_at_best - val_at_best) * 100.0
    assert key["gawf.accuracy_gap_mean"] == pytest.approx(expected_gap)

    with np.load(written["npz"]) as payload:
        assert payload["gawf__val_loss"].shape == (10, EXPECTED_EPOCHS)
        assert payload["epochs"].shape == (EXPECTED_EPOCHS,)


def test_load_series_rejects_an_incomplete_campaign(tmp_path: Path) -> None:
    root = tmp_path / "series"
    _write_series(root, seeds=9)

    with pytest.raises(RuntimeError, match="Expected 10 seeds per model"):
        load_series(root)


def test_load_series_rejects_an_unknown_model_directory(tmp_path: Path) -> None:
    root = tmp_path / "series"
    _write_series(root)
    (root / "mamba" / "seed_01").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Unrecognized IMDB model directory"):
        load_series(root)
