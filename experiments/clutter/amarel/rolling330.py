"""Plan and reconcile the 330 output-only Clutter model/seed training units.

The controller runs only inside a short Slurm CPU job. It never imports project training code.
Its input is a fixed campaign root and the live Slurm queue; output is an atomic state ledger
and, when quota permits, one sparse GPU job array plus one delayed successor controller.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


TAG = "clutter_output_only_330_v1"
MODELS = ("rnn", "lstm", "gru", "gawf", "mamba", "s5")
WIDTHS = (275, 80, 105, 256, 170, 256)
LRS = (0.001, 0.001, 0.005, 0.005, 0.001, 0.001)
WDS = (0.00001, 0.001, 0.001, 0.001, 0.001, 0.0)
FB_MODELS = ("rnn_fb_add", "gru_fb_add", "lstm_fb_add")
FB_WIDTHS = (271, 103, 79)
FB_LRS = (0.001, 0.005, 0.001)
FB_WDS = (0.00001, 0.001, 0.001)
GROUPS = ("matched40", "feedback40", "equal4", "scale4", "scale10", "scale20")
EXPECTED_DROPOUT = "per_layer_output_only_clean_state_v1"


@dataclass(frozen=True)
class Unit:
    """Immutable specification for one model/seed training run."""

    task_id: int
    group: str
    scale: str
    model: str
    seed: int
    width: int
    lr: float
    weight_decay: float

    @property
    def suffix(self) -> str:
        """Return the isolated result suffix consumed by the Clutter training CLI."""

        return f"{TAG}/{self.group}/{self.scale}/{self.model}-seed{self.seed:02d}"


def unit_for_task(task_id: int) -> Unit:
    """Map every array index in `[0, 329]` to exactly one campaign unit."""

    if not 0 <= task_id < 330:
        raise ValueError(f"Task ID must be in [0, 329], got {task_id}")
    if task_id < 60:
        group, scale, offset = "matched40", "40h", task_id
    elif task_id < 90:
        index, seed_idx = divmod(task_id - 60, 10)
        return Unit(
            task_id, "feedback40", "40h", FB_MODELS[index], seed_idx + 1,
            FB_WIDTHS[index], FB_LRS[index], FB_WDS[index],
        )
    elif task_id < 150:
        group, scale, offset = "equal4", "4h", task_id - 90
    else:
        scale_idx, offset = divmod(task_id - 150, 60)
        group = ("scale4", "scale10", "scale20")[scale_idx]
        scale = ("4h", "10h", "20h")[scale_idx]
    model_idx, seed_idx = divmod(offset, 10)
    width = 128 if group == "equal4" else WIDTHS[model_idx]
    return Unit(
        task_id, group, scale, MODELS[model_idx], seed_idx + 1,
        width, LRS[model_idx], WDS[model_idx],
    )


def _atomic_json(path: Path, value: dict) -> None:
    """Replace the campaign ledger atomically within its filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".state-", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _run(*args: str) -> str:
    """Run one bounded Slurm control-plane query or submission."""

    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def valid_training_files(results: Path, unit: Unit) -> bool:
    """Check the final checkpoint, history, and protocol-tagged metrics."""

    result_dir = results / "data" / "clutter" / "runs" / unit.suffix
    if not result_dir.is_dir():
        return False
    try:
        metrics_files = list(result_dir.glob("*_metrics.json"))
        if len(metrics_files) != 1 or len(list(result_dir.glob("*_model.pth"))) != 1:
            return False
        if len(list(result_dir.glob("*.pkl"))) != 1:
            return False
        metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
        return (
            metrics.get("model_type") == unit.model
            and metrics.get("seed") == unit.seed
            and metrics.get("dataset_suffix") == f"{unit.scale}-uint8"
            and metrics.get("eval_dataset_suffix") == "40h-uint8"
            and metrics.get("dropout_protocol") == EXPECTED_DROPOUT
            and metrics.get("actual_epochs") == 150
        )
    except (OSError, ValueError, TypeError):
        return False


def _valid_output(results: Path, artifacts: Path, unit: Unit, source_commit: str) -> bool:
    """Require an exact matching marker plus valid final training files."""

    marker = artifacts / "status" / f"task_{unit.task_id}.done"
    if not marker.is_file():
        return False
    try:
        receipt = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return receipt == {"task_id": unit.task_id, "source_commit": source_commit} and (
        valid_training_files(results, unit)
    )


def quota_slots(all_active: int, campaign_active: int, *, window: int = 90) -> int:
    """Leave one user-wide slot below the asserted 100-task submission limit."""

    return max(0, min(window - campaign_active, 99 - all_active))


def _submit_array(ids: list[int], root: Path, results: Path, data: Path,
                  artifacts: Path, source_commit: str) -> str:
    """Submit one sparse array of at most the currently free quota slots."""

    exports = ",".join((
        "ALL", f"AIM3_ROOT={root}", f"AIM3_RESULTS_PATH={results}",
        f"AIM3_CLUTTER_DATA_DIR={data}", f"AIM3_ARTIFACT_ROOT={artifacts}",
        f"AIM3_SOURCE_COMMIT={source_commit}", "AIM3_NUM_WORKERS=2", "AIM3_PIN_MEMORY=1",
    ))
    result = _run(
        "sbatch", "--parsable", f"--chdir={root}",
        f"--array={','.join(map(str, ids))}%90",
        f"--output={artifacts}/%A_%a.out", f"--error={artifacts}/%A_%a.err",
        f"--export={exports}",
        str(root / "experiments/clutter/amarel/run_clutter_330_unit.sh"),
    )
    return result.split(";", 1)[0]


def _submit_successor(root: Path, results: Path, data: Path, artifacts: Path,
                      source_commit: str) -> str:
    """Schedule the next short CPU reconciliation without a login-node daemon."""

    exports = ",".join((
        "ALL", f"AIM3_ROOT={root}", f"AIM3_RESULTS_PATH={results}",
        f"AIM3_CLUTTER_DATA_DIR={data}", f"AIM3_ARTIFACT_ROOT={artifacts}",
        f"AIM3_SOURCE_COMMIT={source_commit}",
    ))
    result = _run(
        "sbatch", "--parsable", "--begin=now+15minutes", f"--chdir={root}",
        f"--output={artifacts}/%j.controller.out",
        f"--error={artifacts}/%j.controller.err", f"--export={exports}",
        str(root / "experiments/clutter/amarel/run_clutter_330_controller.sh"),
    )
    return result.split(";", 1)[0]


def reconcile(root: Path, results: Path, data: Path, artifacts: Path,
              source_commit: str) -> dict:
    """Fill available Slurm slots once and persist exact per-unit submission identity."""

    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = artifacts / "rolling_state.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {"source_commit": source_commit, "submitted": {}, "completed": [],
                  "blocked": {}, "batches": [], "controllers": []}
        )
        if state["source_commit"] != source_commit:
            raise RuntimeError("Campaign source commit changed; refusing to mix model definitions")
        if state.get("intent"):
            raise RuntimeError(
                "Ambiguous prior sbatch intent; reconcile its exact task IDs before continuing"
            )
        queued = _run("squeue", "-r", "-h", "-u", os.environ["USER"], "-o", "%i")
        active_ids = set(queued.splitlines()) if queued else set()
        completed = set(state["completed"])
        blocked = state["blocked"]
        for task_text, array_id in state["submitted"].items():
            task_id = int(task_text)
            if task_id in completed or task_text in blocked:
                continue
            if f"{array_id}_{task_id}" in active_ids:
                continue
            if _valid_output(results, artifacts, unit_for_task(task_id), source_commit):
                completed.add(task_id)
            else:
                blocked[task_text] = "Submitted task left Slurm without validated final artifacts"
        state["completed"] = sorted(completed)
        pending = [
            task_id for task_id in range(330)
            if str(task_id) not in state["submitted"] and task_id not in completed
            and str(task_id) not in blocked
        ]
        for task_id in pending[:]:
            unit = unit_for_task(task_id)
            result_dir = results / "data" / "clutter" / "runs" / unit.suffix
            if result_dir.exists() or (artifacts / "status" / f"task_{task_id}.done").exists():
                blocked[str(task_id)] = "Unregistered result or marker already exists"
                pending.remove(task_id)
        campaign_active = sum(
            f"{array_id}_{task_text}" in active_ids
            for task_text, array_id in state["submitted"].items()
        )
        state["checked_at"] = datetime.now(timezone.utc).isoformat()
        state["all_user_active"] = len(active_ids)
        state["campaign_active"] = campaign_active
        if pending or campaign_active:
            # Record the successor first so a failed GPU submission can be retried next cycle.
            successor = _submit_successor(root, results, data, artifacts, source_commit)
            state["controllers"].append(successor)
            state["all_user_active"] += 1
        slots = quota_slots(state["all_user_active"], campaign_active)
        ids = pending[:slots]
        if ids:
            # An intent written before sbatch prevents duplicate submissions on an ambiguous crash.
            state["intent"] = ids
            _atomic_json(state_path, state)
            array_id = _submit_array(ids, root, results, data, artifacts, source_commit)
            state["submitted"].update({str(task_id): array_id for task_id in ids})
            state["batches"].append({"job_id": array_id, "task_ids": ids})
            state.pop("intent", None)
        _atomic_json(state_path, state)
        return state


def main() -> None:
    """Print a bounded plan locally or reconcile one controller cycle on Amarel."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "reconcile", "spec", "shell-spec", "validate")
    )
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.command in ("spec", "shell-spec"):
        if args.task_id is None:
            parser.error(f"{args.command} requires --task-id")
        unit = unit_for_task(args.task_id)
        if args.command == "spec":
            print(json.dumps(asdict(unit), sort_keys=True))
        else:
            print("\n".join(map(str, (
                unit.group, unit.scale, unit.model, unit.seed, unit.width,
                unit.lr, unit.weight_decay, unit.suffix,
            ))))
    elif args.command == "plan":
        print(json.dumps({
            "tag": TAG, "expected_units": 330,
            "groups": {group: sum(unit_for_task(i).group == group for i in range(330))
                       for group in GROUPS},
        }, sort_keys=True))
    elif args.command == "validate":
        if args.task_id is None:
            parser.error("validate requires --task-id")
        valid = valid_training_files(
            Path(os.environ["AIM3_RESULTS_PATH"]), unit_for_task(args.task_id)
        )
        print("valid" if valid else "invalid")
        if not valid:
            raise SystemExit(1)
    else:
        root = Path(os.environ["AIM3_ROOT"])
        results = Path(os.environ["AIM3_RESULTS_PATH"])
        data = Path(os.environ["AIM3_CLUTTER_DATA_DIR"])
        artifacts = Path(os.environ["AIM3_ARTIFACT_ROOT"])
        source_commit = os.environ["AIM3_SOURCE_COMMIT"]
        print(json.dumps(reconcile(root, results, data, artifacts, source_commit), sort_keys=True))


if __name__ == "__main__":
    main()
