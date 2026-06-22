#!/usr/bin/env python3
"""Param-count matched RNN/GRU/LSTM hyperparameter grid utilities.

The task grid runs one GaWF reference size at a time. With the default
``--gawf-ref-hidden 512``, each scale contains 3 models x 4 lr x 4 wd = 48
tasks, using RNN/GRU/LSTM hidden sizes matched to GaWF h=512 parameter counts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Sequence

SCALES = ["4h", "10h", "20h", "40h"]
MODELS = ["rnn", "lstm", "gru"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3]
WDS = [0.0, 1e-5, 1e-4, 1e-3]
PARAM_MATCH_HIDDENS = {
    64: {"rnn": 82, "lstm": 22, "gru": 29},
    128: {"rnn": 146, "lstm": 40, "gru": 53},
    256: {"rnn": 275, "lstm": 80, "gru": 105},
    512: {"rnn": 531, "lstm": 170, "gru": 219},
}
CNN_DROPOUT = 0.0
RNN_DROPOUT = 0.5
NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42
TOTAL_TASKS = len(SCALES) * len(MODELS) * len(LRS) * len(WDS)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    scale_task_id: int
    scale: str
    model: str
    hidden_size: int
    gawf_ref_hidden: int
    lr: float
    weight_decay: float
    cnn_dropout: float = CNN_DROPOUT
    rnn_dropout: float = RNN_DROPOUT
    num_epochs: int = NUM_EPOCHS
    patience: int = PATIENCE
    seed: int = SEED

    @property
    def data_suffix(self) -> str:
        return f"{self.scale}-float32"

    @property
    def eval_data_suffix(self) -> str:
        return "40h-float32"

    @property
    def result_root_suffix(self) -> str:
        return f"gen_hparam_{self.scale}_param_match"

    @property
    def result_suffix(self) -> str:
        return f"{self.result_root_suffix}/task_{self.scale_task_id:04d}"

    @property
    def result_stem(self) -> str:
        # ``train_model.py`` derives the saved filename from the model width and
        # optimizer values.  Keep the validator aligned with that real filename;
        # the GaWF reference size is already encoded by the matched hidden size.
        return (
            f"{self.model}_sector_acc_h{self.hidden_size}"
            f"_lr{self.lr}_wd{self.weight_decay}_cdo{self.cnn_dropout}_rdo{self.rnn_dropout}"
        )

    @property
    def metrics_relpath(self) -> str:
        return (
            f"results/train_data/{self.result_suffix}/"
            f"{self.result_stem}_metrics.json"
        )

    @property
    def pkl_relpath(self) -> str:
        return f"results/train_data/{self.result_suffix}/{self.result_stem}.pkl"

    @property
    def model_relpath(self) -> str:
        return (
            f"results/train_data/{self.result_suffix}/"
            f"{self.result_stem}_model.pth"
        )


def normalize_scale(scale: str) -> str:
    aliases = {"4": "4h", "10": "10h", "20": "20h", "40": "40h"}
    return aliases.get(scale, scale)


def iter_task_configs(gawf_ref_hidden: int = 512) -> Iterable[TaskConfig]:
    if gawf_ref_hidden not in PARAM_MATCH_HIDDENS:
        raise ValueError(f"Unsupported gawf_ref_hidden: {gawf_ref_hidden}")
    task_id = 0
    for scale, model, lr, wd in product(SCALES, MODELS, LRS, WDS):
        scale_offset = SCALES.index(scale) * len(MODELS) * len(LRS) * len(WDS)
        yield TaskConfig(
            task_id=task_id,
            scale_task_id=task_id - scale_offset,
            scale=scale,
            model=model,
            hidden_size=PARAM_MATCH_HIDDENS[gawf_ref_hidden][model],
            gawf_ref_hidden=gawf_ref_hidden,
            lr=lr,
            weight_decay=wd,
        )
        task_id += 1


def all_task_configs(gawf_ref_hidden: int = 512) -> List[TaskConfig]:
    return list(iter_task_configs(gawf_ref_hidden))


def task_config(task_id: int, gawf_ref_hidden: int = 512) -> TaskConfig:
    if task_id < 0 or task_id >= TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}], got {task_id}")
    return all_task_configs(gawf_ref_hidden)[task_id]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_matches_task(metrics: Dict[str, Any], cfg: TaskConfig) -> bool:
    checks = [
        metrics.get("model_type") == cfg.model,
        int(metrics.get("hidden_size", -1)) == cfg.hidden_size,
        math.isclose(float(metrics.get("lr", float("nan"))), cfg.lr, rel_tol=0, abs_tol=1e-12),
        math.isclose(
            float(metrics.get("weight_decay", float("nan"))),
            cfg.weight_decay,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        math.isclose(
            float(metrics.get("cnn_dropout", float("nan"))),
            cfg.cnn_dropout,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        math.isclose(
            float(metrics.get("rnn_dropout", float("nan"))),
            cfg.rnn_dropout,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        int(metrics.get("num_epochs", -1)) == cfg.num_epochs,
        metrics.get("dataset_suffix") == cfg.data_suffix,
        metrics.get("eval_dataset_suffix") == cfg.eval_data_suffix,
    ]
    return all(checks)


def validate_task_output(cfg: TaskConfig, root: str) -> Dict[str, Any]:
    metrics_path = os.path.join(root, cfg.metrics_relpath)
    pkl_path = os.path.join(root, cfg.pkl_relpath)
    model_path = os.path.join(root, cfg.model_relpath)
    row: Dict[str, Any] = {
        "task_id": cfg.task_id,
        "scale_task_id": cfg.scale_task_id,
        "scale": cfg.scale,
        "model": cfg.model,
        "hidden_size": cfg.hidden_size,
        "gawf_ref_hidden": cfg.gawf_ref_hidden,
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
    row["actual_epochs"] = metrics.get("actual_epochs")
    row["stopped_by_patience"] = metrics.get("stopped_by_patience")
    return row


def shell_assignments(cfg: TaskConfig, root: str) -> str:
    values = {
        "TASK_ID": str(cfg.task_id),
        "SCALE_TASK_ID": str(cfg.scale_task_id),
        "SCALE": cfg.scale,
        "MODEL_TYPE": cfg.model,
        "HIDDEN_SIZE": str(cfg.hidden_size),
        "GAWF_REF_HIDDEN": str(cfg.gawf_ref_hidden),
        "LR": repr(cfg.lr),
        "WD": repr(cfg.weight_decay),
        "CNN_DROPOUT": repr(cfg.cnn_dropout),
        "RNN_DROPOUT": repr(cfg.rnn_dropout),
        "NUM_EPOCHS": str(cfg.num_epochs),
        "PATIENCE": str(cfg.patience),
        "SEED": str(cfg.seed),
        "DATA_SUFFIX": cfg.data_suffix,
        "EVAL_DATA_SUFFIX": cfg.eval_data_suffix,
        "RESULT_SUFFIX": cfg.result_suffix,
        "RESULT_STEM": cfg.result_stem,
        "METRICS_PATH": os.path.join(root, cfg.metrics_relpath),
        "PKL_PATH": os.path.join(root, cfg.pkl_relpath),
        "MODEL_PATH": os.path.join(root, cfg.model_relpath),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def selected_task_ids(
    scales: Sequence[str],
    models: Sequence[str],
    gawf_ref_hidden: int,
) -> List[int]:
    wanted_scales = {normalize_scale(s) for s in scales}
    wanted_models = set(models)
    return [
        cfg.task_id
        for cfg in iter_task_configs(gawf_ref_hidden)
        if cfg.scale in wanted_scales and cfg.model in wanted_models
    ]


def cmd_emit_task(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    cfg = task_config(args.task_id, args.gawf_ref_hidden)
    if args.format == "json":
        obj = asdict(cfg)
        obj.update(
            {
                "data_suffix": cfg.data_suffix,
                "eval_data_suffix": cfg.eval_data_suffix,
                "result_suffix": cfg.result_suffix,
                "result_stem": cfg.result_stem,
                "metrics_path": os.path.join(root, cfg.metrics_relpath),
                "pkl_path": os.path.join(root, cfg.pkl_relpath),
                "model_path": os.path.join(root, cfg.model_relpath),
            }
        )
        print(json.dumps(obj, indent=2))
    else:
        print(shell_assignments(cfg, root))


def cmd_validate(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    cfg = task_config(args.task_id, args.gawf_ref_hidden)
    row = validate_task_output(cfg, root)
    if args.json:
        print(json.dumps(row, indent=2))
    if not row["valid"]:
        raise SystemExit(f"Task {args.task_id} invalid: {row['reason']}")


def cmd_list_task_ids(args: argparse.Namespace) -> None:
    scales = SCALES if args.scales == ["all"] else [normalize_scale(s) for s in args.scales]
    models = MODELS if args.models == ["all"] else args.models
    task_ids = selected_task_ids(scales, models, args.gawf_ref_hidden)
    if args.format == "json":
        print(json.dumps(task_ids))
    else:
        for task_id in task_ids:
            print(task_id)


def cmd_status(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    rows = [validate_task_output(cfg, root) for cfg in iter_task_configs(args.gawf_ref_hidden)]
    failed = [r for r in rows if not r["valid"]]
    summary = {
        "total": TOTAL_TASKS,
        "valid": len(rows) - len(failed),
        "failed": len(failed),
        "gawf_ref_hidden": args.gawf_ref_hidden,
        "failed_task_ids": [r["task_id"] for r in failed],
    }
    print(json.dumps(summary, indent=2))
    if failed and args.fail_on_missing:
        raise SystemExit(1)


def add_gawf_ref_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gawf-ref-hidden",
        type=int,
        default=512,
        choices=sorted(PARAM_MATCH_HIDDENS),
        help="GaWF hidden size whose parameter count is used for matching.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit-task", help="Emit config for one task id")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=".")
    emit.add_argument("--format", choices=["shell", "json"], default="shell")
    add_gawf_ref_arg(emit)
    emit.set_defaults(func=cmd_emit_task)

    val = sub.add_parser("validate", help="Validate one task output")
    val.add_argument("--task-id", type=int, required=True)
    val.add_argument("--root", default=".")
    val.add_argument("--json", action="store_true")
    add_gawf_ref_arg(val)
    val.set_defaults(func=cmd_validate)

    list_ids = sub.add_parser("list-task-ids", help="Print selected task ids")
    list_ids.add_argument("--scales", nargs="+", default=["all"])
    list_ids.add_argument("--models", nargs="+", default=["all"])
    list_ids.add_argument("--format", choices=["lines", "json"], default="lines")
    add_gawf_ref_arg(list_ids)
    list_ids.set_defaults(func=cmd_list_task_ids)

    status = sub.add_parser("status", help="Check all expected task outputs")
    status.add_argument("--root", default=".")
    status.add_argument("--fail-on-missing", action="store_true")
    add_gawf_ref_arg(status)
    status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
