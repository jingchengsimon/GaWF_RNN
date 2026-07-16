#!/usr/bin/env python3
"""Hyperparameter grid for the IMDB GaWF core param-match search.

Phase E of the IMDB pilot: the LSTM anchor search (``imdb_hparam_grid.py``)
selected ``H* = 128`` (LSTM recurrent core = 132,352 params). Here ``gawf`` is
swept over ``lr x weight_decay`` at the single ``hidden`` whose recurrent core
matches that anchor within tolerance:

- ``TextGaWF`` core(H) = ``3H^2 + 2*H*embed_dim + 4H`` (rnn + U + V + LayerNorm).
- At ``embed_dim=128``: ``H=171`` -> core 132,183 (``-0.128%`` vs LSTM 132,352),
  the closest integer match (``H=170`` is ``-1.10%``, ``H=172`` is ``+0.85%``).

Dropout is fixed to the vision config (``embed_dropout=0.0``, ``rnn_dropout=0.5``;
not searched). Selection is by validation accuracy. Mirrors
``imdb_hparam_grid.py`` exactly so ``emit-task`` / ``validate`` / ``status`` /
``summarize`` and result paths behave identically.
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

MODELS = ["gawf"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3]  # same range as the LSTM anchor search
WDS = [0.0, 1e-5, 1e-4, 1e-3]
HIDDENS = [171]  # core-matched to LSTM H*=128: GaWF core(171)=132,183 vs 132,352 (-0.13%)
EMBED_DIM = 128
EMBED_DROPOUT = 0.0
RNN_DROPOUT = 0.5
POOLING = "last"
OPTIM = "adam"
NUM_EPOCHS = 50
PATIENCE = 10
SEED = 42
BATCH_SIZE = 64
RESULT_ROOT_SUFFIX = "imdb_gawf_param_match"
CSV_TAG = "_imdb_gawf_param_match"
TOTAL_TASKS = len(MODELS) * len(LRS) * len(WDS) * len(HIDDENS)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    model: str
    hidden: int
    lr: float
    weight_decay: float
    embed_dim: int = EMBED_DIM
    embed_dropout: float = EMBED_DROPOUT
    rnn_dropout: float = RNN_DROPOUT
    pooling: str = POOLING
    optim: str = OPTIM
    num_epochs: int = NUM_EPOCHS
    patience: int = PATIENCE
    seed: int = SEED
    batch_size: int = BATCH_SIZE

    @property
    def result_suffix(self) -> str:
        return f"{RESULT_ROOT_SUFFIX}/task_{self.task_id:04d}"

    @property
    def result_stem(self) -> str:
        return (
            f"{self.model}_imdb_h{self.hidden}_emb{self.embed_dim}"
            f"_lr{self.lr}_wd{self.weight_decay}"
            f"_edo{self.embed_dropout}_rdo{self.rnn_dropout}"
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


def default_results_root() -> str:
    for env_name in ("AIM3_RESULTS_PATH", "FAW_RNN_RESULTS_PATH"):
        if value := os.environ.get(env_name):
            return os.path.abspath(os.path.expanduser(value))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "results")


def artifact_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "artifacts", RESULT_ROOT_SUFFIX)


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_row(m: Dict[str, Any]) -> float:
    if m.get("val_acc_at_best") is not None:
        return float(m["val_acc_at_best"])
    return float(m.get("best_val_acc") or 0.0)


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
            int(metrics.get("embed_dim", -1)) == cfg.embed_dim,
            _isclose(metrics.get("lr"), cfg.lr),
            _isclose(metrics.get("weight_decay"), cfg.weight_decay),
            _isclose(metrics.get("embed_dropout"), cfg.embed_dropout),
            _isclose(metrics.get("rnn_dropout"), cfg.rnn_dropout),
            int(metrics.get("num_epochs", -1)) == cfg.num_epochs,
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
        "EMBED_DIM": str(cfg.embed_dim),
        "LR": repr(cfg.lr),
        "WD": repr(cfg.weight_decay),
        "EMBED_DROPOUT": repr(cfg.embed_dropout),
        "RNN_DROPOUT": repr(cfg.rnn_dropout),
        "POOLING": cfg.pooling,
        "OPTIM": cfg.optim,
        "NUM_EPOCHS": str(cfg.num_epochs),
        "PATIENCE": str(cfg.patience),
        "SEED": str(cfg.seed),
        "BATCH_SIZE": str(cfg.batch_size),
        "RESULT_SUFFIX": cfg.result_suffix,
        "RESULT_STEM": cfg.result_stem,
        "METRICS_PATH": os.path.join(root, cfg.metrics_relpath),
        "PKL_PATH": os.path.join(root, cfg.pkl_relpath),
        "MODEL_PATH": os.path.join(root, cfg.model_relpath),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def trial_csv_row(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": m.get("model_type"),
        "hidden": m.get("hidden_size"),
        "embed_dim": m.get("embed_dim"),
        "lr": m.get("lr"),
        "wd": m.get("weight_decay"),
        "core_param_count": m.get("core_param_count"),
        "total_param_count": m.get("total_param_count"),
        "val_acc": m.get("val_acc_at_best", m.get("best_val_acc")),
        "train_acc": m.get("train_acc_at_best_val"),
        "test_acc": m.get("test_acc_at_best"),
        "overfit_gap": m.get("overfit_gap"),
        "best_epoch_val_acc_1based": m.get("best_epoch_val_acc_1based"),
        "actual_epochs": m.get("actual_epochs"),
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
    for model in MODELS:
        candidates = [r for r in rows if r.get("model_type") == model]
        if not candidates:
            continue
        best = max(candidates, key=score_row)
        out.append(
            {
                "model": model,
                "hidden": best.get("hidden_size"),
                "embed_dim": best.get("embed_dim"),
                "lr": best.get("lr"),
                "weight_decay": best.get("weight_decay"),
                "core_param_count": best.get("core_param_count"),
                "val_acc_at_best": score_row(best),
                "train_acc_at_best_val": best.get("train_acc_at_best_val"),
                "test_acc_at_best": best.get("test_acc_at_best"),
                "overfit_gap": best.get("overfit_gap"),
                "actual_epochs": best.get("actual_epochs"),
                "stopped_by_patience": best.get("stopped_by_patience"),
                "metrics_path": best.get("_path"),
            }
        )
    return out


def write_markdown_summary(path: str, best_rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# IMDB GaWF Param-match Search Summary\n\n")
        f.write(
            "Selection criterion: highest `val_acc_at_best`. Dropout fixed "
            "(`embed_dropout=0.0`, `rnn_dropout=0.5`). GaWF `hidden=171` is core "
            "param-matched to the LSTM anchor `H*=128` (132,352 -> 132,183, -0.13%).\n\n"
        )
        f.write("| Model | Hidden | Embed | LR | WD | Core params | Val | Train | Test | Gap | Epochs |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in best_rows:
            f.write(
                f"| {r['model']} | {r.get('hidden')} | {r.get('embed_dim')} | {r['lr']} | "
                f"{r['weight_decay']} | {r.get('core_param_count')} | {r.get('val_acc_at_best')} | "
                f"{r.get('train_acc_at_best_val')} | {r.get('test_acc_at_best')} | "
                f"{r.get('overfit_gap')} | {r.get('actual_epochs')} |\n"
            )


def cmd_emit_task(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    cfg = task_config(args.task_id)
    if args.format == "json":
        obj = asdict(cfg)
        obj.update(
            {
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

    status_csv = os.path.join(out_dir, "imdb_gawf_param_match_status.csv")
    fieldnames = [
        "task_id", "model", "hidden", "lr", "weight_decay", "valid", "reason",
        "metrics_exists", "pkl_exists", "model_exists", "val_acc_at_best",
        "test_acc_at_best", "core_param_count", "actual_epochs", "stopped_by_patience",
        "metrics_path", "pkl_path", "model_path",
    ]
    write_csv(status_csv, rows, fieldnames)

    os.makedirs(out_dir, exist_ok=True)
    failed_path = os.path.join(out_dir, "failed_task_ids.txt")
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
    write_json(os.path.join(out_dir, "imdb_gawf_param_match_status.json"), summary)
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
    best_json = {row["model"]: row for row in best_rows}
    write_json(os.path.join(out_dir, "imdb_gawf_param_match_best.json"), best_json)
    write_csv(
        os.path.join(out_dir, "imdb_gawf_param_match_best.csv"),
        best_rows,
        list(best_rows[0].keys()),
    )
    write_markdown_summary(
        os.path.join(out_dir, "imdb_gawf_param_match_best_summary.md"), best_rows
    )

    all_rows = [trial_csv_row(m) for m in rows]
    write_csv(
        os.path.join(out_dir, "imdb_gawf_param_match_all_trials.csv"),
        all_rows,
        list(all_rows[0].keys()),
    )
    print(f"Wrote IMDB GaWF param-match summaries under {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit-task", help="Emit config for one task id")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=default_results_root())
    emit.add_argument("--format", choices=["shell", "json"], default="shell")
    emit.set_defaults(func=cmd_emit_task)

    val = sub.add_parser("validate", help="Validate one task output")
    val.add_argument("--task-id", type=int, required=True)
    val.add_argument("--root", default=default_results_root())
    val.add_argument("--json", action="store_true")
    val.set_defaults(func=cmd_validate)

    status = sub.add_parser("status", help="Check all expected task outputs")
    status.add_argument("--root", default=default_results_root())
    status.add_argument("--out-dir", default="")
    status.add_argument("--fail-on-missing", action="store_true")
    status.set_defaults(func=cmd_status)

    summ = sub.add_parser("summarize", help="Aggregate best hparams")
    summ.add_argument("--root", default=default_results_root())
    summ.add_argument("--out-dir", default="")
    summ.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
