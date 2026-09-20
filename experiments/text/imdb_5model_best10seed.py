#!/usr/bin/env python3
"""Fixed best-hyperparameter IMDB protocol for five models and seeds 1--10.

The selected hyperparameters are the validation winners from
``imdb_5model_full50_grid``.  Each task trains for the full 50 epochs and evaluates
the test split once using the validation-best checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable


MODELS = ("lstm", "rnn", "gru", "gawf", "gawf_logits")
SEEDS = tuple(range(1, 11))
BEST_CONFIGS = {
    "lstm": {"hidden": 128, "lr": 5e-3, "weight_decay": 1e-4},
    "rnn": {"hidden": 304, "lr": 5e-4, "weight_decay": 1e-3},
    "gru": {"hidden": 155, "lr": 5e-3, "weight_decay": 1e-5},
    "gawf": {"hidden": 171, "lr": 1e-3, "weight_decay": 1e-3},
    "gawf_logits": {"hidden": 301, "lr": 5e-4, "weight_decay": 1e-4},
}
EMBED_DIM = 128
EMBED_DROPOUT = 0.0
RNN_DROPOUT = 0.5
POOLING = "last"
OPTIM = "adam"
NUM_LAYERS = 1
NUM_EPOCHS = 50
PATIENCE = 999_999
BATCH_SIZE = 64
GAWF_FEEDBACK_LR_SCALE = 1.0
RESULT_ROOT_SUFFIX = "imdb_5model_best10seed"
TOTAL_TASKS = len(MODELS) * len(SEEDS)


@dataclass(frozen=True)
class TaskConfig:
    """One fixed model/seed unit in the formal IMDB array."""

    task_id: int
    model: str
    seed: int
    hidden: int
    lr: float
    weight_decay: float

    @property
    def result_suffix(self) -> str:
        """Return the collision-free result suffix for this unit."""

        return f"{RESULT_ROOT_SUFFIX}/{self.model}/seed_{self.seed:02d}"

    @property
    def result_stem(self) -> str:
        """Return the filename stem emitted by the IMDB trainer."""

        return (
            f"{self.model}_imdb_h{self.hidden}_emb{EMBED_DIM}"
            f"_lr{self.lr}_wd{self.weight_decay}"
            f"_edo{EMBED_DROPOUT}_rdo{RNN_DROPOUT}"
        )

    @property
    def result_rel_dir(self) -> str:
        """Return the result directory relative to the physical result root."""

        return f"data/text/runs/{self.result_suffix}"


def iter_task_configs() -> Iterable[TaskConfig]:
    """Yield the stable model-major task map used by the Slurm array."""

    for task_id, (model, seed) in enumerate(product(MODELS, SEEDS)):
        best = BEST_CONFIGS[model]
        yield TaskConfig(
            task_id=task_id,
            model=model,
            seed=seed,
            hidden=int(best["hidden"]),
            lr=float(best["lr"]),
            weight_decay=float(best["weight_decay"]),
        )


def task_config(task_id: int) -> TaskConfig:
    """Resolve one array task ID to its immutable protocol configuration."""

    configs = tuple(iter_task_configs())
    if task_id < 0 or task_id >= len(configs):
        raise ValueError(f"task_id must be in [0, {len(configs) - 1}], got {task_id}")
    return configs[task_id]


def _isclose(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def output_paths(cfg: TaskConfig, root: str) -> dict[str, str]:
    """Return the three required final artifact paths for one task."""

    result_dir = os.path.join(os.path.abspath(root), cfg.result_rel_dir)
    return {
        "result_dir": result_dir,
        "metrics_path": os.path.join(result_dir, f"{cfg.result_stem}_metrics.json"),
        "pkl_path": os.path.join(result_dir, f"{cfg.result_stem}.pkl"),
        "model_path": os.path.join(result_dir, f"{cfg.result_stem}_model.pth"),
    }


def validate_task_output(cfg: TaskConfig, root: str) -> dict[str, Any]:
    """Validate artifact presence and the exact fixed protocol fields."""

    paths = output_paths(cfg, root)
    row: dict[str, Any] = {"task_id": cfg.task_id, "valid": False, **asdict(cfg), **paths}
    missing = [name for name in ("metrics_path", "pkl_path", "model_path")
               if not os.path.isfile(paths[name])]
    if missing:
        row["reason"] = f"missing:{','.join(missing)}"
        return row
    try:
        with open(paths["metrics_path"], "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        row["reason"] = f"bad_metrics:{exc}"
        return row

    expected_feedback = {"gawf": "hidden", "gawf_logits": "logits"}.get(cfg.model)
    checks = (
        metrics.get("model_type") == cfg.model,
        metrics.get("dataset") == "imdb",
        int(metrics.get("hidden_size", -1)) == cfg.hidden,
        int(metrics.get("num_layers", 1)) == NUM_LAYERS,
        int(metrics.get("seed", -1)) == cfg.seed,
        _isclose(metrics.get("lr"), cfg.lr),
        _isclose(metrics.get("weight_decay"), cfg.weight_decay),
        int(metrics.get("embed_dim", -1)) == EMBED_DIM,
        _isclose(metrics.get("embed_dropout"), EMBED_DROPOUT),
        _isclose(metrics.get("rnn_dropout"), RNN_DROPOUT),
        metrics.get("pooling") == POOLING,
        metrics.get("optimizer") == OPTIM,
        int(metrics.get("num_epochs", -1)) == NUM_EPOCHS,
        int(metrics.get("actual_epochs", -1)) == NUM_EPOCHS,
        int(metrics.get("patience", -1)) == PATIENCE,
        metrics.get("stopped_by_patience") is False,
        metrics.get("feedback_mode") == expected_feedback,
        metrics.get("test_acc_at_best") is not None,
    )
    if not all(checks):
        row["reason"] = "metrics_mismatch"
        return row
    row.update(
        valid=True,
        reason="ok",
        val_acc_at_best=metrics.get("val_acc_at_best"),
        test_acc_at_best=metrics.get("test_acc_at_best"),
        best_epoch=metrics.get("best_epoch_val_acc_1based"),
    )
    return row


def shell_assignments(cfg: TaskConfig, root: str) -> str:
    """Emit shell-safe assignments consumed by the compute-node runner."""

    values: dict[str, Any] = {
        "TASK_ID": cfg.task_id,
        "MODEL_TYPE": cfg.model,
        "SEED": cfg.seed,
        "HIDDEN": cfg.hidden,
        "LR": cfg.lr,
        "WD": cfg.weight_decay,
        "EMBED_DIM": EMBED_DIM,
        "EMBED_DROPOUT": EMBED_DROPOUT,
        "RNN_DROPOUT": RNN_DROPOUT,
        "POOLING": POOLING,
        "OPTIM": OPTIM,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "GAWF_FEEDBACK_LR_SCALE": GAWF_FEEDBACK_LR_SCALE,
        "RESULT_SUFFIX": cfg.result_suffix,
        **output_paths(cfg, root),
    }
    return "\n".join(f"{key.upper()}={shlex.quote(str(value))}" for key, value in values.items())


def cmd_emit_task(args: argparse.Namespace) -> None:
    """CLI handler: emit one task as shell or JSON."""

    cfg = task_config(args.task_id)
    if args.format == "shell":
        print(shell_assignments(cfg, args.root))
        return
    print(json.dumps({**asdict(cfg), **output_paths(cfg, args.root)}, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    """CLI handler: validate one task and fail nonzero when incomplete."""

    row = validate_task_output(task_config(args.task_id), args.root)
    if args.json:
        print(json.dumps(row, indent=2))
    if not row["valid"]:
        raise SystemExit(f"Task {args.task_id} invalid: {row['reason']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the small task-map CLI used locally and on compute nodes."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit-task")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", required=True)
    emit.add_argument("--format", choices=("shell", "json"), default="shell")
    emit.set_defaults(func=cmd_emit_task)
    validate = sub.add_parser("validate")
    validate.add_argument("--task-id", type=int, required=True)
    validate.add_argument("--root", required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> None:
    """Run the requested task-map command."""

    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
