#!/usr/bin/env python3
"""Single-stage full-grid generalization hyperparameter utilities.

This script owns the task-id mapping for the 4 scale x 4 model x 4 hidden-size
x 4 lr x 4 wd sweep, validates task outputs, and aggregates the completed
search into best-hparam summaries plus Phase-3-compatible CSVs for plotting.
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

SCALES = ["4h", "10h", "20h", "40h"]
MODELS = ["rnn", "lstm", "gru", "gawf"]
HIDDEN_SIZES = [64, 128, 256, 512]
LRS = [1e-4, 5e-4, 1e-3, 5e-3]
WDS = [0.0, 1e-5, 1e-4, 1e-3]
CNN_DROPOUT = 0.0
RNN_DROPOUT = 0.5
NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42
RESULT_ROOT_SUFFIX = "gen_hparam_full_grid"
CSV_TAG = "_hparam_full_grid"
TOTAL_TASKS = len(SCALES) * len(MODELS) * len(HIDDEN_SIZES) * len(LRS) * len(WDS)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    scale: str
    model: str
    hidden_size: int
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
        return "40h-uint8"

    @property
    def result_suffix(self) -> str:
        return f"{RESULT_ROOT_SUFFIX}/task_{self.task_id:04d}"

    @property
    def result_stem(self) -> str:
        return (
            f"{self.model}_sector_acc_h{self.hidden_size}_lr{self.lr}"
            f"_wd{self.weight_decay}_cdo{self.cnn_dropout}_rdo{self.rnn_dropout}"
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


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def artifact_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "artifacts", RESULT_ROOT_SUFFIX)


def iter_task_configs() -> Iterable[TaskConfig]:
    task_id = 0
    for scale, model, hidden_size, lr, wd in product(
        SCALES, MODELS, HIDDEN_SIZES, LRS, WDS
    ):
        yield TaskConfig(
            task_id=task_id,
            scale=scale,
            model=model,
            hidden_size=hidden_size,
            lr=lr,
            weight_decay=wd,
        )
        task_id += 1


def all_task_configs() -> List[TaskConfig]:
    return list(iter_task_configs())


def task_config(task_id: int) -> TaskConfig:
    if task_id < 0 or task_id >= TOTAL_TASKS:
        raise ValueError(f"task_id must be in [0, {TOTAL_TASKS - 1}], got {task_id}")
    return all_task_configs()[task_id]


def score_row(m: Dict[str, Any]) -> float:
    if m.get("val_acc_at_best") is not None:
        return float(m["val_acc_at_best"])
    return float(m.get("best_val_acc_char") or 0.0)


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "scale": cfg.scale,
        "model": cfg.model,
        "hidden_size": cfg.hidden_size,
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
        "SCALE": cfg.scale,
        "MODEL_TYPE": cfg.model,
        "HIDDEN_SIZE": str(cfg.hidden_size),
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


def phase3_csv_row(scale: str, m: Dict[str, Any]) -> Dict[str, Any]:
    train_acc_char = m.get("train_acc_at_best_val")
    if train_acc_char is None:
        train_acc_char = m.get("best_train_acc_char")
    val_acc_char = m.get("val_acc_at_best")
    if val_acc_char is None:
        val_acc_char = m.get("best_val_acc_char")
    overfit_gap_char = m.get("overfit_gap")
    if overfit_gap_char is None and train_acc_char is not None and val_acc_char is not None:
        overfit_gap_char = float(train_acc_char) - float(val_acc_char)

    train_acc_sector = m.get("train_acc_sector_at_best_val_sector")
    if train_acc_sector is None:
        train_acc_sector = m.get("best_train_acc_pos")
    val_acc_sector = m.get("val_acc_sector_at_best")
    if val_acc_sector is None:
        val_acc_sector = m.get("best_val_acc_pos")
    overfit_gap_sector = m.get("overfit_gap_sector")
    if overfit_gap_sector is None and train_acc_sector is not None and val_acc_sector is not None:
        overfit_gap_sector = float(train_acc_sector) - float(val_acc_sector)

    return {
        "scale": scale,
        "model": m.get("model_type"),
        "lr": m.get("lr"),
        "wd": m.get("weight_decay"),
        "hidden_size": m.get("hidden_size"),
        "early_stop_epoch": m.get("early_stop_epoch_1based", m.get("actual_epochs")),
        "train_acc": train_acc_char,
        "val_acc": val_acc_char,
        "overfit_gap": overfit_gap_char,
        "train_acc_char": train_acc_char,
        "val_acc_char": val_acc_char,
        "overfit_gap_char": overfit_gap_char,
        "train_acc_sector": train_acc_sector,
        "val_acc_sector": val_acc_sector,
        "overfit_gap_sector": overfit_gap_sector,
        "best_epoch_val_acc_1based": m.get("best_epoch_val_acc_1based"),
        "best_epoch_val_acc_sector_1based": m.get("best_epoch_val_acc_sector_1based"),
        "stopped_by_patience": m.get("stopped_by_patience"),
        "metrics_path": m.get("_path"),
    }


def load_all_metrics(result_root: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(result_root, "task_*", "*_metrics.json")
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob(pattern)):
        row = read_json(path)
        row["_path"] = path
        rows.append(row)
    return rows


def best_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for scale in SCALES:
        for model in MODELS:
            candidates = [
                r
                for r in rows
                if r.get("dataset_suffix") == f"{scale}-float32"
                and r.get("model_type") == model
            ]
            if not candidates:
                continue
            best = max(candidates, key=score_row)
            phase_row = phase3_csv_row(scale, best)
            out.append(
                {
                    "scale": scale,
                    "model": model,
                    "hidden_size": best.get("hidden_size"),
                    "lr": best.get("lr"),
                    "weight_decay": best.get("weight_decay"),
                    "val_acc_at_best": score_row(best),
                    "train_acc_at_best_val": best.get("train_acc_at_best_val"),
                    "overfit_gap": best.get("overfit_gap"),
                    "val_acc_sector_at_best": best.get("val_acc_sector_at_best"),
                    "train_acc_sector_at_best_val_sector": best.get(
                        "train_acc_sector_at_best_val_sector"
                    ),
                    "overfit_gap_sector": best.get("overfit_gap_sector"),
                    "best_epoch_val_acc_1based": best.get("best_epoch_val_acc_1based"),
                    "actual_epochs": best.get("actual_epochs"),
                    "stopped_by_patience": best.get("stopped_by_patience"),
                    "metrics_path": best.get("_path"),
                    "plot_train_acc_char": phase_row["train_acc_char"],
                    "plot_val_acc_char": phase_row["val_acc_char"],
                    "plot_train_acc_sector": phase_row["train_acc_sector"],
                    "plot_val_acc_sector": phase_row["val_acc_sector"],
                }
            )
    return out


def write_markdown_summary(path: str, best_rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Full-Grid Hyperparameter Search Summary\n\n")
        f.write(
            "Selection criterion: highest `val_acc_at_best` per scale and model. "
            "Char and sector plots use the same selected run.\n\n"
        )
        f.write(
            "| Scale | Model | Hidden | LR | WD | Val Char | Train Char | "
            "Gap Char | Val Sector | Train Sector | Gap Sector | Epochs |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best_rows:
            f.write(
                f"| {row['scale']} | {row['model']} | {row['hidden_size']} | "
                f"{row['lr']} | {row['weight_decay']} | "
                f"{row.get('val_acc_at_best')} | {row.get('train_acc_at_best_val')} | "
                f"{row.get('overfit_gap')} | {row.get('val_acc_sector_at_best')} | "
                f"{row.get('train_acc_sector_at_best_val_sector')} | "
                f"{row.get('overfit_gap_sector')} | {row.get('actual_epochs')} |\n"
            )


def cmd_emit_task(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    cfg = task_config(args.task_id)
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
    cfg = task_config(args.task_id)
    row = validate_task_output(cfg, root)
    if args.json:
        print(json.dumps(row, indent=2))
    if not row["valid"]:
        raise SystemExit(f"Task {args.task_id} invalid: {row['reason']}")


def cmd_status(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out_dir or artifact_dir())
    rows = [validate_task_output(cfg, root) for cfg in iter_task_configs()]
    failed = [r for r in rows if not r["valid"]]
    ok = [r for r in rows if r["valid"]]

    status_csv = os.path.join(out_dir, "hparam_full_grid_status.csv")
    fieldnames = [
        "task_id",
        "scale",
        "model",
        "hidden_size",
        "lr",
        "weight_decay",
        "valid",
        "reason",
        "metrics_exists",
        "pkl_exists",
        "model_exists",
        "val_acc_at_best",
        "actual_epochs",
        "stopped_by_patience",
        "metrics_path",
        "pkl_path",
        "model_path",
    ]
    write_csv(status_csv, rows, fieldnames)

    failed_path = os.path.join(out_dir, "failed_task_ids.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(failed_path, "w", encoding="utf-8") as f:
        for row in failed:
            f.write(f"{row['task_id']}\n")

    summary = {
        "total": TOTAL_TASKS,
        "valid": len(ok),
        "failed": len(failed),
        "failed_task_ids_path": failed_path,
        "status_csv": status_csv,
    }
    write_json(os.path.join(out_dir, "hparam_full_grid_status.json"), summary)
    print(json.dumps(summary, indent=2))
    if failed and args.fail_on_missing:
        raise SystemExit(1)


def cmd_summarize(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    result_root = os.path.join(root, "results", "train_data", RESULT_ROOT_SUFFIX)
    out_dir = os.path.abspath(args.out_dir or artifact_dir())
    rows = load_all_metrics(result_root)
    if not rows:
        raise RuntimeError(f"No metrics found under {result_root}")

    best_rows = best_summary_rows(rows)
    if len(best_rows) != len(SCALES) * len(MODELS):
        raise RuntimeError(
            f"Expected {len(SCALES) * len(MODELS)} best rows, got {len(best_rows)}"
        )

    best_json: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in best_rows:
        best_json.setdefault(row["scale"], {})[row["model"]] = row
    write_json(os.path.join(out_dir, "hparam_best.json"), best_json)
    write_csv(
        os.path.join(out_dir, "hparam_best.csv"),
        best_rows,
        list(best_rows[0].keys()),
    )
    write_markdown_summary(os.path.join(out_dir, "hparam_best_summary.md"), best_rows)

    all_rows: List[Dict[str, Any]] = []
    for m in rows:
        scale = str(m.get("dataset_suffix", "")).replace("-float32", "")
        all_rows.append(phase3_csv_row(scale, m))
    write_csv(os.path.join(out_dir, "hparam_all_trials.csv"), all_rows, list(all_rows[0].keys()))

    for scale in SCALES:
        phase_rows: List[Dict[str, Any]] = []
        for model in MODELS:
            row = best_json[scale][model]
            metrics = read_json(row["metrics_path"])
            metrics["_path"] = row["metrics_path"]
            phase_rows.append(phase3_csv_row(scale, metrics))
        phase_csv = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            f"phase3_summary_{scale}{args.csv_tag}.csv",
        )
        write_csv(phase_csv, phase_rows, list(phase_rows[0].keys()))

    print(f"Wrote best summaries under {out_dir}")
    print(f"Plot with: python utils_viz/plot_generalization.py --csv_tag {args.csv_tag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit-task", help="Emit config for one task id")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=".")
    emit.add_argument("--format", choices=["shell", "json"], default="shell")
    emit.set_defaults(func=cmd_emit_task)

    val = sub.add_parser("validate", help="Validate one task output")
    val.add_argument("--task-id", type=int, required=True)
    val.add_argument("--root", default=".")
    val.add_argument("--json", action="store_true")
    val.set_defaults(func=cmd_validate)

    status = sub.add_parser("status", help="Check all 1024 expected task outputs")
    status.add_argument("--root", default=".")
    status.add_argument("--out-dir", default="")
    status.add_argument("--fail-on-missing", action="store_true")
    status.set_defaults(func=cmd_status)

    summ = sub.add_parser("summarize", help="Aggregate best hparams and plot CSVs")
    summ.add_argument("--root", default=".")
    summ.add_argument("--out-dir", default="")
    summ.add_argument("--csv-tag", default=CSV_TAG)
    summ.set_defaults(func=cmd_summarize)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
