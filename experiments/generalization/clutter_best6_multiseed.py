#!/usr/bin/env python3
"""Map and validate the fixed-hyperparameter six-model Clutter multi-seed run.

Inputs are a task ID in ``[0, 59]`` and an optional repository root. Outputs are shell
assignments for one Slurm task, strict result validation, aggregate status, or a monitoring
manifest. The experiment uses six frozen best configurations, seeds 1--10, 150 epochs, and no
early stopping on the 40h training/validation dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MODELS = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
SEEDS = tuple(range(1, 11))
NUM_EPOCHS = 150
PATIENCE = 0
DATA_SUFFIX = "40h-float32"
EVAL_DATA_SUFFIX = "40h-float32"
CNN_DROPOUT = 0.0
RNN_DROPOUT = 0.5
RESULT_ROOT_SUFFIX = "clutter_best6_multiseed_40h_ep150"
ARTIFACT_TAG = "clutter_best6_10seed_ep150"


@dataclass(frozen=True)
class ModelSpec:
    """Frozen hyperparameters for one selected Clutter model."""

    model: str
    width: int
    lr: float
    weight_decay: float
    width_kind: str = "hidden"
    state_size: int | None = None
    s5_ssm_lr_scale: float = 0.1


MODEL_SPECS = {
    "gawf": ModelSpec("gawf", 256, 5e-3, 1e-3),
    "rnn": ModelSpec("rnn", 275, 1e-3, 1e-5),
    "lstm": ModelSpec("lstm", 80, 1e-3, 1e-3),
    "gru": ModelSpec("gru", 105, 5e-3, 1e-3),
    "mamba": ModelSpec("mamba", 170, 1e-3, 1e-3, width_kind="mamba"),
    "s5": ModelSpec("s5", 256, 1e-3, 0.0, width_kind="s5", state_size=128),
}


@dataclass(frozen=True)
class TaskConfig:
    """One model/seed unit in the 60-task array."""

    task_id: int
    seed: int
    spec: ModelSpec

    @property
    def unit_id(self) -> str:
        """Return the stable monitoring unit identifier."""

        return f"{self.spec.model}-seed{self.seed:02d}"

    @property
    def result_suffix(self) -> str:
        """Return the result directory suffix consumed by ``train_model.py``."""

        return f"{RESULT_ROOT_SUFFIX}/{self.unit_id}"

    @property
    def result_stem(self) -> str:
        """Return the exact checkpoint/metrics stem emitted by ``train_model.py``."""

        if self.spec.width_kind == "mamba":
            width = f"dmodel{self.spec.width}"
        elif self.spec.width_kind == "s5":
            width = f"dmodel{self.spec.width}_state{self.spec.state_size}"
        else:
            width = f"h{self.spec.width}"
        return (
            f"{self.spec.model}_sector_acc_{width}_lr{self.spec.lr}"
            f"_wd{self.spec.weight_decay}_cdo{CNN_DROPOUT}_rdo{RNN_DROPOUT}"
        )

    @property
    def result_dir_relpath(self) -> str:
        """Return the repository-relative unit result directory."""

        return f"results/train_data/{self.result_suffix}"

    @property
    def metrics_relpath(self) -> str:
        """Return the repository-relative metrics path."""

        return f"{self.result_dir_relpath}/{self.result_stem}_metrics.json"

    @property
    def checkpoint_relpath(self) -> str:
        """Return the repository-relative checkpoint path."""

        return f"{self.result_dir_relpath}/{self.result_stem}_model.pth"

    @property
    def pickle_relpath(self) -> str:
        """Return the repository-relative training pickle path."""

        return f"{self.result_dir_relpath}/{self.result_stem}.pkl"

    @property
    def done_relpath(self) -> str:
        """Return the repository-relative completion marker path."""

        return (
            f"experiments/amarel/artifacts/{ARTIFACT_TAG}/status/"
            f"task_{self.task_id:03d}.done"
        )

    @property
    def fail_relpath(self) -> str:
        """Return the repository-relative failure marker path."""

        return (
            f"experiments/amarel/artifacts/{ARTIFACT_TAG}/status/"
            f"task_{self.task_id:03d}.fail"
        )


def iter_task_configs() -> Iterable[TaskConfig]:
    """Yield the seed-major task mapping used by the Slurm array."""

    task_id = 0
    for seed in SEEDS:
        for model in MODELS:
            yield TaskConfig(task_id=task_id, seed=seed, spec=MODEL_SPECS[model])
            task_id += 1


def all_task_configs() -> list[TaskConfig]:
    """Return all 60 task configurations."""

    return list(iter_task_configs())


def task_config(task_id: int) -> TaskConfig:
    """Resolve one task ID or raise ``ValueError`` when it is outside the grid."""

    tasks = all_task_configs()
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task_id must be in [0, {len(tasks) - 1}], got {task_id}")
    return tasks[task_id]


def shell_assignments(config: TaskConfig, root: str) -> str:
    """Emit shell-safe assignments consumed by the Amarel array runner."""

    values = {
        "TASK_ID": str(config.task_id),
        "UNIT_ID": config.unit_id,
        "MODEL_TYPE": config.spec.model,
        "MODEL_WIDTH": str(config.spec.width),
        "WIDTH_KIND": config.spec.width_kind,
        "S5_STATE_SIZE": str(config.spec.state_size or 0),
        "S5_SSM_LR_SCALE": repr(config.spec.s5_ssm_lr_scale),
        "LR": repr(config.spec.lr),
        "WD": repr(config.spec.weight_decay),
        "CNN_DROPOUT": repr(CNN_DROPOUT),
        "RNN_DROPOUT": repr(RNN_DROPOUT),
        "NUM_EPOCHS": str(NUM_EPOCHS),
        "PATIENCE": str(PATIENCE),
        "SEED": str(config.seed),
        "DATA_SUFFIX": DATA_SUFFIX,
        "EVAL_DATA_SUFFIX": EVAL_DATA_SUFFIX,
        "RESULT_SUFFIX": config.result_suffix,
        "METRICS_PATH": os.path.join(root, config.metrics_relpath),
        "CHECKPOINT_PATH": os.path.join(root, config.checkpoint_relpath),
        "PICKLE_PATH": os.path.join(root, config.pickle_relpath),
        "DONE_FILE": os.path.join(root, config.done_relpath),
        "FAIL_FILE": os.path.join(root, config.fail_relpath),
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def _matches_float(metrics: dict[str, Any], key: str, expected: float) -> bool:
    """Return whether a numeric metrics field exactly matches the frozen protocol."""

    try:
        return math.isclose(float(metrics[key]), expected, rel_tol=0.0, abs_tol=1e-12)
    except (KeyError, TypeError, ValueError):
        return False


def validate_task_output(config: TaskConfig, root: str) -> dict[str, Any]:
    """Validate metrics, checkpoint, pickle, seed, and full-epoch completion for one unit."""

    paths = {
        "metrics": os.path.join(root, config.metrics_relpath),
        "checkpoint": os.path.join(root, config.checkpoint_relpath),
        "pickle": os.path.join(root, config.pickle_relpath),
    }
    report: dict[str, Any] = {
        "task_id": config.task_id,
        "unit_id": config.unit_id,
        "model": config.spec.model,
        "seed": config.seed,
        "valid": False,
        "reason": "",
        "paths": paths,
    }
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    if missing:
        report["reason"] = f"missing_{'_'.join(missing)}"
        return report
    try:
        metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["reason"] = f"bad_metrics_json:{exc}"
        return report

    checks = {
        "model_type": metrics.get("model_type") == config.spec.model,
        "seed": metrics.get("seed") == config.seed,
        "num_epochs": metrics.get("num_epochs") == NUM_EPOCHS,
        "actual_epochs": metrics.get("actual_epochs") == NUM_EPOCHS,
        "patience": metrics.get("patience") == PATIENCE,
        "stopped_by_patience": metrics.get("stopped_by_patience") is False,
        "dataset_suffix": metrics.get("dataset_suffix") == DATA_SUFFIX,
        "eval_dataset_suffix": metrics.get("eval_dataset_suffix") == EVAL_DATA_SUFFIX,
        "dataset_mode": metrics.get("dataset_mode") == "sector",
        "use_acceleration": metrics.get("use_acceleration") is True,
        "use_mmap": metrics.get("use_mmap") is True,
        "width": metrics.get("hidden_size") == config.spec.width,
        "lr": _matches_float(metrics, "lr", config.spec.lr),
        "weight_decay": _matches_float(metrics, "weight_decay", config.spec.weight_decay),
        "cnn_dropout": _matches_float(metrics, "cnn_dropout", CNN_DROPOUT),
        "rnn_dropout": _matches_float(metrics, "rnn_dropout", RNN_DROPOUT),
    }
    if config.spec.model in {"gawf", "rnn", "lstm", "gru"}:
        checks["num_layers"] = metrics.get("num_layers") == 1
    if config.spec.model == "s5":
        checks["s5_state_size"] = metrics.get("s5_state_size") == config.spec.state_size
        checks["s5_ssm_lr_scale"] = _matches_float(
            metrics, "s5_ssm_lr_scale", config.spec.s5_ssm_lr_scale
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        report["reason"] = f"metrics_mismatch:{','.join(failed)}"
        report["failed_checks"] = failed
        return report
    report["valid"] = True
    report["reason"] = "ok"
    report["val_acc_at_best"] = metrics.get("val_acc_at_best")
    return report


def build_manifest(
    job_ids: str | list[str], remote_root: str, conda_init: str
) -> dict[str, Any]:
    """Build the strict manifest for ten seed-level Slurm arrays and 60 units."""

    normalized_job_ids = [job_ids] if isinstance(job_ids, str) else list(job_ids)
    if not normalized_job_ids:
        raise ValueError("At least one Slurm job ID is required.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    units = []
    for config in all_task_configs():
        units.append(
            {
                "id": config.unit_id,
                "result_dir": config.result_dir_relpath,
                "metrics_file": f"{config.result_stem}_metrics.json",
                "done_file": config.done_relpath,
                "fail_file": config.fail_relpath,
                "checkpoint_glob": f"{config.result_stem}_model.pth",
                "checkpoint_count": 1,
                "expected": {
                    "model_type": config.spec.model,
                    "seed": config.seed,
                    "num_epochs": NUM_EPOCHS,
                    "actual_epochs": NUM_EPOCHS,
                    "patience": PATIENCE,
                    "stopped_by_patience": False,
                    "use_acceleration": True,
                    "use_mmap": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "id": f"amarel-clutter-best6-10seed-ep150-{normalized_job_ids[0]}",
        "description": (
            "Clutter six frozen best models, seeds 1-10, 150 full epochs, no early stopping"
        ),
        "host": "amarel",
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "remote_root": remote_root,
        "environment": {"name": "aim3_rnn", "conda_init": conda_init},
        "scheduler": {
            "type": "slurm",
            "job_ids": [str(job_id) for job_id in normalized_job_ids],
            "run_ids": [],
            "tmux_session": None,
            "process_patterns": [],
            "collect_gpu": False,
        },
        "paths": {
            "log_globs": [
                f"experiments/amarel/artifacts/{ARTIFACT_TAG}/{job_id}_*.out"
                for job_id in normalized_job_ids
            ],
            "status_dir": f"experiments/amarel/artifacts/{ARTIFACT_TAG}/status",
            "result_paths": [f"results/train_data/{RESULT_ROOT_SUFFIX}"],
        },
        "tracking": {
            "expected_units": len(units),
            "auto_complete": True,
            "units": units,
        },
        "notes": [
            "40h training stimulus is approximately 119 GB and is loaded with mmap.",
            "DataLoader uses num_workers=0 and pin_memory=False to preserve mmap safety.",
            "Saved checkpoints contain the best-validation state after all 150 epochs are run.",
            "Submitted as ten independent Slurm arrays, one seed and six model tasks per job.",
        ],
    }


def _write_json(value: Any, output: str | None) -> None:
    """Write JSON to a requested file or stdout."""

    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def parse_args() -> argparse.Namespace:
    """Parse the task mapping, validation, status, and manifest commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-task")
    emit.add_argument("--task-id", type=int, required=True)
    emit.add_argument("--root", default=os.getcwd())

    validate = subparsers.add_parser("validate")
    validate.add_argument("--task-id", type=int, required=True)
    validate.add_argument("--root", default=os.getcwd())
    validate.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--root", default=os.getcwd())
    status.add_argument("--json", action="store_true")

    manifest = subparsers.add_parser("emit-manifest")
    manifest.add_argument("--job-id", action="append", required=True)
    manifest.add_argument("--remote-root", required=True)
    manifest.add_argument("--conda-init", required=True)
    manifest.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    """Execute the selected utility command."""

    args = parse_args()
    if args.command == "emit-task":
        print(shell_assignments(task_config(args.task_id), os.path.abspath(args.root)))
        return
    if args.command == "validate":
        report = validate_task_output(task_config(args.task_id), os.path.abspath(args.root))
        if args.json:
            _write_json(report, None)
        elif report["valid"]:
            print(f"VALID {report['unit_id']}")
        else:
            print(f"INVALID {report['unit_id']}: {report['reason']}")
        raise SystemExit(0 if report["valid"] else 1)
    if args.command == "status":
        reports = [
            validate_task_output(config, os.path.abspath(args.root))
            for config in all_task_configs()
        ]
        valid = sum(report["valid"] for report in reports)
        payload = {"expected": len(reports), "valid": valid, "reports": reports}
        if args.json:
            _write_json(payload, None)
        else:
            print(f"valid={valid}/{len(reports)}")
            for report in reports:
                if not report["valid"]:
                    print(f"{report['task_id']:03d} {report['unit_id']} {report['reason']}")
        raise SystemExit(0 if valid == len(reports) else 1)
    manifest = build_manifest(args.job_id, args.remote_root, args.conda_init)
    _write_json(manifest, args.output)


if __name__ == "__main__":
    main()
