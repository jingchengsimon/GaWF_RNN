#!/usr/bin/env python3
"""Hyperparameter grid utilities for Mamba and S5 sequence baselines.

This grid follows the RNN/LSTM/GRU scheme: fixed model size, fixed architecture
knobs, lr x weight-decay search, validation selection, and 40h validation/test
reporting through ``train_model.py``. Mamba/S5 use the wider lr range needed by
SSM-style models. S5 also carries a fixed SSM-core lr multiplier; it is not a
search dimension.
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
MODELS = ["mamba", "s5"]
LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
WDS = [0.0, 1e-5, 1e-4, 1e-3]
MAMBA_D_MODEL = 170
S5_D_MODEL = 256
# Param-matched to GaWF h=256 (~586k params); do not reuse DiagLTI state=189.
S5_STATE_SIZE = 128
S5_SSM_LR_SCALE = 0.1
GAWF_REF_HIDDEN = 256
CNN_DROPOUT = 0.0
RNN_DROPOUT = 0.5
NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42
RESULT_ROOT_SUFFIX = "gen_hparam_mamba_s5_grid"
CSV_TAG = "_mamba_s5_hparam_grid"
TOTAL_TASKS = len(SCALES) * len(MODELS) * len(LRS) * len(WDS)


@dataclass(frozen=True)
class TaskConfig:
    task_id: int
    scale: str
    model: str
    lr: float
    weight_decay: float
    cnn_dropout: float = CNN_DROPOUT
    rnn_dropout: float = RNN_DROPOUT
    num_epochs: int = NUM_EPOCHS
    patience: int = PATIENCE
    seed: int = SEED
    gawf_ref_hidden: int = GAWF_REF_HIDDEN
    mamba_d_model: int = MAMBA_D_MODEL
    s5_d_model: int = S5_D_MODEL
    s5_state_size: int = S5_STATE_SIZE
    s5_ssm_lr_scale: float = S5_SSM_LR_SCALE

    @property
    def data_suffix(self) -> str:
        return f"{self.scale}-float32"

    @property
    def eval_data_suffix(self) -> str:
        return "40h-float32"

    @property
    def result_suffix(self) -> str:
        return f"{RESULT_ROOT_SUFFIX}/task_{self.task_id:04d}"

    @property
    def result_stem(self) -> str:
        if self.model == "mamba":
            width_suffix = f"_dmodel{self.mamba_d_model}"
        elif self.model == "s5":
            width_suffix = f"_dmodel{self.s5_d_model}_state{self.s5_state_size}"
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        return (
            f"{self.model}_sector_acc{width_suffix}_lr{self.lr}"
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
    for scale, model, lr, wd in product(SCALES, MODELS, LRS, WDS):
        yield TaskConfig(
            task_id=task_id,
            scale=scale,
            model=model,
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
    return float(m.get("best_val_acc_char") or 0.0)


def metrics_matches_task(metrics: Dict[str, Any], cfg: TaskConfig) -> bool:
    checks = [
        metrics.get("model_type") == cfg.model,
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
    if cfg.model == "mamba":
        checks.append(int(metrics.get("mamba_d_model", -1)) == cfg.mamba_d_model)
    elif cfg.model == "s5":
        checks.extend(
            [
                int(metrics.get("s5_d_model", -1)) == cfg.s5_d_model,
                int(metrics.get("s5_state_size", -1)) == cfg.s5_state_size,
                math.isclose(
                    float(metrics.get("s5_ssm_lr_scale", float("nan"))),
                    cfg.s5_ssm_lr_scale,
                    rel_tol=0,
                    abs_tol=1e-12,
                ),
            ]
        )
    return all(checks)


def validate_task_output(cfg: TaskConfig, root: str) -> Dict[str, Any]:
    metrics_path = os.path.join(root, cfg.metrics_relpath)
    pkl_path = os.path.join(root, cfg.pkl_relpath)
    model_path = os.path.join(root, cfg.model_relpath)
    row: Dict[str, Any] = {
        "task_id": cfg.task_id,
        "scale": cfg.scale,
        "model": cfg.model,
        "gawf_ref_hidden": cfg.gawf_ref_hidden,
        "mamba_d_model": cfg.mamba_d_model if cfg.model == "mamba" else "",
        "s5_d_model": cfg.s5_d_model if cfg.model == "s5" else "",
        "s5_state_size": cfg.s5_state_size if cfg.model == "s5" else "",
        "s5_ssm_lr_scale": cfg.s5_ssm_lr_scale if cfg.model == "s5" else "",
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
        "MAMBA_D_MODEL": str(cfg.mamba_d_model),
        "S5_D_MODEL": str(cfg.s5_d_model),
        "S5_STATE_SIZE": str(cfg.s5_state_size),
        "S5_SSM_LR_SCALE": repr(cfg.s5_ssm_lr_scale),
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
        "gawf_ref_hidden": m.get("gawf_ref_hidden", GAWF_REF_HIDDEN),
        "d_model": m.get("mamba_d_model", m.get("s5_d_model")),
        "state_size": m.get("s5_state_size"),
        "s5_ssm_lr_scale": m.get("s5_ssm_lr_scale"),
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
    max_lr = max(LRS)
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
            best_lr = float(best.get("lr", float("nan")))
            out.append(
                {
                    "scale": scale,
                    "model": model,
                    "gawf_ref_hidden": best.get("gawf_ref_hidden", GAWF_REF_HIDDEN),
                    "d_model": best.get("mamba_d_model", best.get("s5_d_model")),
                    "state_size": best.get("s5_state_size"),
                    "s5_ssm_lr_scale": best.get("s5_ssm_lr_scale"),
                    "lr": best.get("lr"),
                    "weight_decay": best.get("weight_decay"),
                    "best_at_lr_ceiling": math.isclose(
                        best_lr, max_lr, rel_tol=0, abs_tol=1e-12
                    ),
                    "val_acc_at_best": score_row(best),
                    "train_acc_at_best_val": best.get("train_acc_at_best_val"),
                    "overfit_gap": best.get("overfit_gap"),
                    "val_acc_sector_at_best": best.get("val_acc_sector_at_best"),
                    "train_acc_sector_at_best_val_sector": best.get(
                        "train_acc_sector_at_best_val_sector"
                    ),
                    "overfit_gap_sector": best.get("overfit_gap_sector"),
                    "actual_epochs": best.get("actual_epochs"),
                    "stopped_by_patience": best.get("stopped_by_patience"),
                    "metrics_path": best.get("_path"),
                }
            )
    return out


def write_markdown_summary(path: str, best_rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Mamba/S5 Hyperparameter Search Summary\n\n")
        f.write(
            "Selection criterion: highest `val_acc_at_best` per scale and model. "
            "Mamba/S5 use a wider lr grid up to 1e-2; S5 uses fixed "
            "`s5_ssm_lr_scale=0.1`, so SSM-core lr is `lr * 0.1`.\n\n"
        )
        f.write(
            "| Scale | Model | d_model | State | LR | WD | LR ceiling? | "
            "Val Char | Train Char | Gap Char | Val Sector | Train Sector | Gap Sector | Epochs |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best_rows:
            f.write(
                f"| {row['scale']} | {row['model']} | {row.get('d_model')} | "
                f"{row.get('state_size')} | {row['lr']} | {row['weight_decay']} | "
                f"{row.get('best_at_lr_ceiling')} | {row.get('val_acc_at_best')} | "
                f"{row.get('train_acc_at_best_val')} | {row.get('overfit_gap')} | "
                f"{row.get('val_acc_sector_at_best')} | "
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

    status_csv = os.path.join(out_dir, "mamba_s5_hparam_status.csv")
    fieldnames = [
        "task_id",
        "scale",
        "model",
        "gawf_ref_hidden",
        "mamba_d_model",
        "s5_d_model",
        "s5_state_size",
        "s5_ssm_lr_scale",
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
    write_json(os.path.join(out_dir, "mamba_s5_hparam_status.json"), summary)
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
    write_json(os.path.join(out_dir, "mamba_s5_hparam_best.json"), best_json)
    write_csv(
        os.path.join(out_dir, "mamba_s5_hparam_best.csv"),
        best_rows,
        list(best_rows[0].keys()),
    )
    write_markdown_summary(
        os.path.join(out_dir, "mamba_s5_hparam_best_summary.md"), best_rows
    )

    all_rows = []
    for m in rows:
        scale = str(m.get("dataset_suffix", "")).replace("-float32", "")
        all_rows.append(phase3_csv_row(scale, m))
    write_csv(
        os.path.join(out_dir, "mamba_s5_hparam_all_trials.csv"),
        all_rows,
        list(all_rows[0].keys()),
    )
    print(f"Wrote Mamba/S5 hparam summaries under {out_dir}")


def cmd_purge_legacy_s5_state189(args: argparse.Namespace) -> None:
    """Remove legacy S5 grid artifacts that used state_size=189."""
    root = os.path.abspath(args.root)
    status_dir = os.path.join(
        root,
        "experiments",
        "generalization",
        "artifacts",
        RESULT_ROOT_SUFFIX,
        "status",
    )
    removed: List[str] = []
    for cfg in iter_task_configs():
        if cfg.model != "s5":
            continue
        task_dir = os.path.join(root, "results", "train_data", cfg.result_suffix)
        if not os.path.isdir(task_dir):
            continue
        for path in sorted(glob(os.path.join(task_dir, "*state189*"))):
            if args.dry_run:
                print(f"would_remove {path}")
            else:
                os.remove(path)
                removed.append(path)
        done_file = os.path.join(status_dir, f"task_{cfg.task_id:04d}.done")
        fail_file = os.path.join(status_dir, f"task_{cfg.task_id:04d}.fail")
        for path in (done_file, fail_file):
            if os.path.isfile(path):
                if args.dry_run:
                    print(f"would_remove {path}")
                else:
                    os.remove(path)
                    removed.append(path)
    summary = {
        "dry_run": args.dry_run,
        "removed_count": len(removed),
        "removed_paths": removed,
    }
    print(json.dumps(summary, indent=2))


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

    status = sub.add_parser("status", help="Check all expected task outputs")
    status.add_argument("--root", default=".")
    status.add_argument("--out-dir", default="")
    status.add_argument("--fail-on-missing", action="store_true")
    status.set_defaults(func=cmd_status)

    summ = sub.add_parser("summarize", help="Aggregate best hparams")
    summ.add_argument("--root", default=".")
    summ.add_argument("--out-dir", default="")
    summ.set_defaults(func=cmd_summarize)

    purge = sub.add_parser(
        "purge-legacy-s5-state189",
        help="Delete legacy S5 hparam outputs/checkpoints that used state_size=189",
    )
    purge.add_argument("--root", default=".")
    purge.add_argument("--dry-run", action="store_true")
    purge.set_defaults(func=cmd_purge_legacy_s5_state189)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
