#!/usr/bin/env python3
"""Hyperparameter grid for the unified IMDB 2-layer GaWF search.

Task C searches only ``hidden x lr x weight_decay`` for the unified
``gawf_logits --num_layers 2`` model. Lower layers receive detached adjacent
hidden feedback; the final layer receives detached classifier logits. U/V use
``base_lr * gawf_feedback_lr_scale`` with scale 0.1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from dataclasses import dataclass
from glob import glob
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence

MODELS = ["gawf_logits"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3]
WDS = [0.0, 1e-5, 1e-4, 1e-3]
HIDDENS = [64, 96, 128, 192]
EMBED_DIM = 128
EMBED_DROPOUT = 0.0
RNN_DROPOUT = 0.5
POOLING = "last"
OPTIM = "adam"
NUM_EPOCHS = 50
PATIENCE = 10
SEED = 42
BATCH_SIZE = 64
NUM_LAYERS = 2
GAWF_FEEDBACK_LR_SCALE = 0.1
RESULT_ROOT_SUFFIX = "imdb_gawf_depth2_grid"
TOTAL_TASKS = len(MODELS) * len(LRS) * len(WDS) * len(HIDDENS)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    model: str
    hidden: int
    lr: float
    weight_decay: float

    @property
    def result_suffix(self) -> str:
        return f"{RESULT_ROOT_SUFFIX}/task_{self.task_id:04d}"

    @property
    def result_stem(self) -> str:
        return (
            f"{self.model}_imdb_h{self.hidden}_L{NUM_LAYERS}_emb{EMBED_DIM}"
            f"_lr{self.lr}_wd{self.weight_decay}"
            f"_edo{EMBED_DROPOUT}_rdo{RNN_DROPOUT}"
        )

    @property
    def metrics_relpath(self) -> str:
        return f"train_data/{self.result_suffix}/{self.result_stem}_metrics.json"

    @property
    def pkl_relpath(self) -> str:
        return f"train_data/{self.result_suffix}/{self.result_stem}.pkl"

    @property
    def model_relpath(self) -> str:
        return f"train_data/{self.result_suffix}/{self.result_stem}_model.pth"


def iter_task_configs() -> Iterable[TaskConfig]:
    task_id = 0
    for model, hidden, lr, wd in product(MODELS, HIDDENS, LRS, WDS):
        yield TaskConfig(task_id=task_id, model=model, hidden=hidden, lr=lr, weight_decay=wd)
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


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _isclose(a: Any, b: float) -> bool:
    try:
        return math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def metrics_matches_task(metrics: Dict[str, Any], cfg: TaskConfig) -> bool:
    return all(
        [
            metrics.get("model_type") == cfg.model,
            metrics.get("dataset") == "imdb",
            int(metrics.get("hidden_size", -1)) == cfg.hidden,
            int(metrics.get("embed_dim", -1)) == EMBED_DIM,
            _isclose(metrics.get("lr"), cfg.lr),
            _isclose(metrics.get("weight_decay"), cfg.weight_decay),
            _isclose(metrics.get("embed_dropout"), EMBED_DROPOUT),
            _isclose(metrics.get("rnn_dropout"), RNN_DROPOUT),
            int(metrics.get("num_epochs", -1)) == NUM_EPOCHS,
            int(metrics.get("num_layers", -1)) == NUM_LAYERS,
            metrics.get("feedback_mode") == "logits",
            int(metrics.get("feedback_dim", -1)) == 2,
            list(metrics.get("layer_feedback_dims", [])) == [cfg.hidden, 2],
            _isclose(metrics.get("gawf_feedback_lr_scale"), GAWF_FEEDBACK_LR_SCALE),
        ]
    )


def validate_task_output(cfg: TaskConfig, root: str) -> Dict[str, Any]:
    metrics_path = os.path.join(root, cfg.metrics_relpath)
    pkl_path = os.path.join(root, cfg.pkl_relpath)
    model_path = os.path.join(root, cfg.model_relpath)
    row: Dict[str, Any] = {
        "task_id": cfg.task_id,
        "model": cfg.model,
        "hidden": cfg.hidden,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
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
    row["test_acc_at_best"] = metrics.get("test_acc_at_best")
    row["core_param_count"] = metrics.get("core_param_count")
    row["actual_epochs"] = metrics.get("actual_epochs")
    row["stopped_by_patience"] = metrics.get("stopped_by_patience")
    return row


def shell_assignments(cfg: TaskConfig, root: str) -> str:
    values = {
        "TASK_ID": str(cfg.task_id),
        "MODEL_TYPE": cfg.model,
        "HIDDEN": str(cfg.hidden),
        "EMBED_DIM": str(EMBED_DIM),
        "LR": repr(cfg.lr),
        "WD": repr(cfg.weight_decay),
        "EMBED_DROPOUT": repr(EMBED_DROPOUT),
        "RNN_DROPOUT": repr(RNN_DROPOUT),
        "POOLING": POOLING,
        "OPTIM": OPTIM,
        "NUM_EPOCHS": str(NUM_EPOCHS),
        "PATIENCE": str(PATIENCE),
        "SEED": str(SEED),
        "BATCH_SIZE": str(BATCH_SIZE),
        "NUM_LAYERS": str(NUM_LAYERS),
        "GAWF_FEEDBACK_LR_SCALE": repr(GAWF_FEEDBACK_LR_SCALE),
        "RESULT_SUFFIX": cfg.result_suffix,
        "RESULT_STEM": cfg.result_stem,
        "METRICS_PATH": os.path.join(root, cfg.metrics_relpath),
        "PKL_PATH": os.path.join(root, cfg.pkl_relpath),
        "MODEL_PATH": os.path.join(root, cfg.model_relpath),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def status(root: str) -> Dict[str, Any]:
    rows = [validate_task_output(cfg, root) for cfg in iter_task_configs()]
    valid = sum(1 for row in rows if row["valid"])
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts", RESULT_ROOT_SUFFIX)
    write_json(
        os.path.join(out_dir, "imdb_gawf_depth_grid_status.json"),
        {"expected_total": TOTAL_TASKS, "valid": valid, "rows": rows},
    )
    write_csv(
        os.path.join(out_dir, "imdb_gawf_depth_grid_status.csv"),
        rows,
        [
            "task_id",
            "model",
            "hidden",
            "lr",
            "weight_decay",
            "valid",
            "reason",
            "val_acc_at_best",
            "test_acc_at_best",
            "core_param_count",
            "actual_epochs",
            "stopped_by_patience",
            "metrics_path",
        ],
    )
    return {"expected_total": TOTAL_TASKS, "valid": valid}


def load_all_metrics(result_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob(os.path.join(result_root, "task_*", "*_metrics.json"))):
        row = read_json(path)
        row["_path"] = path
        rows.append(row)
    return rows


def default_results_root() -> str:
    """Resolve the physical results root without importing the training stack."""
    for env_name in ("AIM3_RESULTS_PATH", "FAW_RNN_RESULTS_PATH"):
        value = os.environ.get(env_name)
        if value:
            return os.path.abspath(os.path.expanduser(value))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit-task")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=default_results_root())
    val = sub.add_parser("validate")
    val.add_argument("--task-id", type=int, required=True)
    val.add_argument("--root", default=default_results_root())
    val.add_argument("--json", action="store_true")
    sub.add_parser("list-task-ids")
    stat = sub.add_parser("status")
    stat.add_argument("--root", default=default_results_root())
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
