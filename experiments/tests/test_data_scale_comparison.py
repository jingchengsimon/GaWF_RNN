"""Regression check for the formal ten-seed Clutter data-scale comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math

from utils.analysis.clutter.data_scale_comparison import (
    ACTUAL_EPOCHS,
    FORMAL_HPARAMS,
    MODELS,
    SCALES,
    SEEDS,
    _panel_limits,
    run,
)


def test_workflow_aggregates_ten_seeds_and_separates_outputs(tmp_path) -> None:
    """The workflow writes analysis data separately from the curated PDF."""

    input_root = tmp_path / "input"
    for scale_index, scale in enumerate(SCALES):
        for model_index, model in enumerate(MODELS):
            width, lr, weight_decay = FORMAL_HPARAMS[model]
            for seed in SEEDS:
                leaf = input_root / scale / f"{model}-seed{seed:02d}"
                leaf.mkdir(parents=True)
                val_char = 50.0 + scale_index + model_index + seed
                metrics = {
                    "model_type": model,
                    "seed": seed,
                    "dataset_suffix": f"{scale}-uint8",
                    "eval_dataset_suffix": "40h-uint8",
                    "hidden_size": width,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "actual_epochs": ACTUAL_EPOCHS,
                    "patience": 0,
                    "stopped_by_patience": False,
                    "train_acc_at_best_val": val_char + 10.0,
                    "val_acc_at_best": val_char,
                    "overfit_gap": 10.0,
                    "train_acc_sector_at_best_val_sector": 92.0,
                    "val_acc_sector_at_best": 88.0,
                    "overfit_gap_sector": 4.0,
                    "best_epoch_val_acc_1based": 90,
                    "best_epoch_val_acc_sector_1based": 80,
                }
                (leaf / f"{model}_metrics.json").write_text(json.dumps(metrics))
    data_dir = tmp_path / "data"
    output_pdf = tmp_path / "save" / "data_scale_performance_2x3_10seed.pdf"

    data_files, figure_files = run(
        argparse.Namespace(
            input_root=input_root,
            source_root=None,
            data_dir=data_dir,
            output_pdf=output_pdf,
        )
    )

    assert len(data_files) == 3
    assert all(path.parent == data_dir for path in data_files)
    assert len(figure_files) == 1
    assert figure_files == [output_pdf]
    assert output_pdf.is_file()
    with (data_dir / "data_scale_seed_runs_10seed.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 240
    with (data_dir / "data_scale_mean_sem_10seed.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    selected = next(row for row in summary if row["scale"] == "4h" and row["model"] == "rnn")
    assert int(float(selected["n_seeds"])) == 10
    assert math.isclose(float(selected["mean_val_acc_char"]), 55.5)
    assert float(selected["sem_val_acc_char"]) > 0.0
    for field in ("train_acc_sector", "val_acc_sector", "train_acc_char", "val_acc_char"):
        lower, upper = _panel_limits(summary, field)
        assert lower > 0.0
        assert all(
            lower <= float(row[f"mean_{field}"]) - float(row[f"sem_{field}"])
            and float(row[f"mean_{field}"]) + float(row[f"sem_{field}"]) <= upper
            for row in summary
        )
