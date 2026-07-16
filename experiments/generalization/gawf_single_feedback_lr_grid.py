#!/usr/bin/env python3
"""Grid utility for single-layer GaWF feedback learning-rate scale search.

This is a narrow replacement for the withdrawn Task A. It keeps the same
40h single-layer GaWF fine-search grid and adds ``gawf_feedback_lr_scale``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from dataclasses import asdict, dataclass
from glob import glob
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence

MODEL = "gawf"
SCALE = "40h"
DATA_SUFFIX = "40h-float32"
EVAL_DATA_SUFFIX = "40h-float32"
HIDDENS = [192, 256, 320, 384]
LRS = [0.002, 0.003, 0.005, 0.007]
WDS = [0.0003, 0.001, 0.003]
GAWF_FEEDBACK_LR_SCALES = [0.1, 0.3, 1.0]
CNN_DROPOUT = 0.0
RNN_DROPOUT = 0.5
NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42
RESULT_ROOT_SUFFIX = "gawf_single_fblr_finesearch_40h"
TOTAL_TASKS = len(HIDDENS) * len(LRS) * len(WDS) * len(GAWF_FEEDBACK_LR_SCALES)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    hidden_size: int
    lr: float
    weight_decay: float
    gawf_feedback_lr_scale: float

    @property
    def result_suffix(self) -> str:
        return f"{RESULT_ROOT_SUFFIX}/task_{self.task_id:04d}"

    @property
    def result_stem(self) -> str:
        fblr = (
            ""
            if math.isclose(self.gawf_feedback_lr_scale, 1.0, rel_tol=0, abs_tol=1e-12)
            else f"_fblr{self.gawf_feedback_lr_scale}"
        )
        return (
            f"{MODEL}_sector_acc_h{self.hidden_size}_lr{self.lr}_wd{self.weight_decay}"
            f"_cdo{CNN_DROPOUT}_rdo{RNN_DROPOUT}{fblr}"
        )

    @property
    def metrics_relpath(self) -> str:
        return f"results/train_data/{self.result_suffix}/{self.result_stem}_metrics.json"

    @property
    def pkl_relpath(self) -> str:
        return f"results/train_data/{self.result_suffix}/{self.result_stem}.pkl"

    @property
    def model_relpath(self) -> str:
        return f"results/train_data/{self.result_suffix}/{self.result_stem}_model.pth"


def iter_task_configs() -> Iterable[TaskConfig]:
    task_id = 0
    for hidden, lr, wd, fblr in product(HIDDENS, LRS, WDS, GAWF_FEEDBACK_LR_SCALES):
        yield TaskConfig(
            task_id=task_id,
            hidden_size=hidden,
            lr=lr,
            weight_decay=wd,
            gawf_feedback_lr_scale=fblr,
        )
        task_id += 1


def all_task_configs() -> List[TaskConfig]:
    return list(iter_task_configs())


def task_config(task_id: int) -> TaskConfig:
    if task_id < 0 or task_id >= TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}], got {task_id}")
    return all_task_configs()[task_id]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _isclose(a: Any, b: float) -> bool:
    try:
        return math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def metrics_matches_task(metrics: Dict[str, Any], cfg: TaskConfig) -> bool:
    return all(
        [
            metrics.get("model_type") == MODEL,
            metrics.get("dataset_suffix") == DATA_SUFFIX,
            metrics.get("eval_dataset_suffix") == EVAL_DATA_SUFFIX,
            int(metrics.get("hidden_size", -1)) == cfg.hidden_size,
            _isclose(metrics.get("lr"), cfg.lr),
            _isclose(metrics.get("weight_decay"), cfg.weight_decay),
            _isclose(metrics.get("cnn_dropout"), CNN_DROPOUT),
            _isclose(metrics.get("rnn_dropout"), RNN_DROPOUT),
            _isclose(metrics.get("gawf_feedback_lr_scale"), cfg.gawf_feedback_lr_scale),
            int(metrics.get("num_epochs", -1)) == NUM_EPOCHS,
        ]
    )


def validate_task_output(cfg: TaskConfig, root: str) -> Dict[str, Any]:
    metrics_path = os.path.join(root, cfg.metrics_relpath)
    pkl_path = os.path.join(root, cfg.pkl_relpath)
    model_path = os.path.join(root, cfg.model_relpath)
    row: Dict[str, Any] = {
        "task_id": cfg.task_id,
        "model_type": MODEL,
        "hidden_size": cfg.hidden_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "gawf_feedback_lr_scale": cfg.gawf_feedback_lr_scale,
        "metrics_path": metrics_path,
        "pkl_path": pkl_path,
        "model_path": model_path,
        "metrics_exists": os.path.isfile(metrics_path),
        "pkl_exists": os.path.isfile(pkl_path),
        "model_exists": os.path.isfile(model_path),
        "valid": False,
        "reason": "",
    }
    if not row["metrics_exists"]:
        row["reason"] = "missing_metrics"
        return row
    try:
        metrics = read_json(metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        row["reason"] = f"bad_metrics_json:{exc}"
        return row
    if not metrics_matches_task(metrics, cfg):
        row["reason"] = "metrics_mismatch"
        return row
    if not row["pkl_exists"]:
        row["reason"] = "missing_pkl"
        return row
    if not row["model_exists"]:
        row["reason"] = "missing_model"
        return row
    row["valid"] = True
    row["reason"] = "ok"
    row["val_acc_at_best"] = metrics.get("val_acc_at_best")
    row["train_acc_at_best_val"] = metrics.get("train_acc_at_best_val")
    row["actual_epochs"] = metrics.get("actual_epochs")
    row["stopped_by_patience"] = metrics.get("stopped_by_patience")
    return row


def shell_assignments(cfg: TaskConfig, root: str) -> str:
    values = {
        "TASK_ID": str(cfg.task_id),
        "MODEL_TYPE": MODEL,
        "SCALE": SCALE,
        "DATA_SUFFIX": DATA_SUFFIX,
        "EVAL_DATA_SUFFIX": EVAL_DATA_SUFFIX,
        "HIDDEN_SIZE": str(cfg.hidden_size),
        "LR": repr(cfg.lr),
        "WD": repr(cfg.weight_decay),
        "CNN_DROPOUT": repr(CNN_DROPOUT),
        "RNN_DROPOUT": repr(RNN_DROPOUT),
        "GAWF_FEEDBACK_LR_SCALE": repr(cfg.gawf_feedback_lr_scale),
        "NUM_EPOCHS": str(NUM_EPOCHS),
        "PATIENCE": str(PATIENCE),
        "SEED": str(SEED),
        "RESULT_SUFFIX": cfg.result_suffix,
        "RESULT_STEM": cfg.result_stem,
        "METRICS_PATH": os.path.join(root, cfg.metrics_relpath),
        "PKL_PATH": os.path.join(root, cfg.pkl_relpath),
        "MODEL_PATH": os.path.join(root, cfg.model_relpath),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status(root: str) -> Dict[str, Any]:
    rows = [validate_task_output(cfg, root) for cfg in iter_task_configs()]
    valid = sum(1 for row in rows if row["valid"])
    out_dir = os.path.join(
        os.path.dirname(__file__), "artifacts", RESULT_ROOT_SUFFIX
    )
    write_json(
        os.path.join(out_dir, "gawf_single_feedback_lr_status.json"),
        {"expected_total": TOTAL_TASKS, "valid": valid, "rows": rows},
    )
    write_csv(
        os.path.join(out_dir, "gawf_single_feedback_lr_status.csv"),
        rows,
        [
            "task_id",
            "model_type",
            "hidden_size",
            "lr",
            "weight_decay",
            "gawf_feedback_lr_scale",
            "valid",
            "reason",
            "val_acc_at_best",
            "train_acc_at_best_val",
            "actual_epochs",
            "stopped_by_patience",
            "metrics_path",
        ],
    )
    return {"expected_total": TOTAL_TASKS, "valid": valid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit-task")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    val = sub.add_parser("validate")
    val.add_argument("--task-id", type=int, required=True)
    val.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    val.add_argument("--json", action="store_true")
    sub.add_parser("list-task-ids")
    stat = sub.add_parser("status")
    stat.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    args = parser.parse_args()

    if args.cmd == "emit-task":
        print(shell_assignments(task_config(args.task_id), args.root))
    elif args.cmd == "validate":
        row = validate_task_output(task_config(args.task_id), args.root)
        if args.json:
            print(json.dumps(row, indent=2))
        raise SystemExit(0 if row["valid"] else 1)
    elif args.cmd == "list-task-ids":
        for cfg in iter_task_configs():
            print(cfg.task_id)
    elif args.cmd == "status":
        print(json.dumps(status(args.root), indent=2))


if __name__ == "__main__":
    main()
