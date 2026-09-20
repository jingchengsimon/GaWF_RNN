"""Run one model's smoke-gated Seaquest A/B/C/F sweep serially on one CUDA device.

Inputs: model, physical CUDA device, and explicit result/artifact parents. Outputs: isolated
run bundles, commands, logs, status markers, and retained smoke acceptance evidence.
Activate aim3_rnn before invoking this module; no training implementation is changed here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# end epsilon, fixed exploration steps, LR decay step, gamma; F uses B as its control.
VARIANTS = {
    "a": (0.01, 300_000, 1_000_000, 0.99),
    "b": (0.05, 1_000_000, 1_000_000, 0.99),
    "c": (0.05, 1_000_000, 2_000_000, 0.99),
    "f995": (0.05, 1_000_000, 1_000_000, 0.995),
    "f999": (0.05, 1_000_000, 1_000_000, 0.999),
}
# Keep the Breakout L3 widths; only the action head/feedback dimension changes to 18.
HIDDEN = {"lstm": 373, "gawf": 605}
LOG = logging.getLogger(__name__)


def command(model: str, variant: str, result: Path, smoke: bool = False) -> list[str]:
    """Build the explicit single-task protocol without changing global CLI defaults."""
    epsilon, exploration, decay, gamma = VARIANTS[variant]
    values = {
        "env_id": "ALE/Seaquest-v5", "action_space_mode": "full18",
        "atari_env_protocol": "baseline", "model_type": model, "num_layers": 3,
        "hidden_size": HIDDEN[model], "feedback_mode": "qvalues" if model == "gawf" else "none",
        "gawf_feedback_lr_scale": 1.0, "frame_skip": 4, "frame_stack": 4,
        "flicker_prob": 0.0, "total_timesteps": 25_000 if smoke else 3_000_000,
        "num_envs": 1, "seed": 2, "learning_rate": 1e-4,
        "learning_rate_decay_step": decay, "learning_rate_decay_scale": 0.1,
        "learning_rate_decay_per_task_steps": 0, "gamma": gamma,
        "start_epsilon": 1.0, "end_epsilon": epsilon, "exploration_steps": exploration,
        "buffer_size": 1_000_000, "replay_backing": "mmap", "replay_layout": "shared",
        "learning_starts": 20_000, "learning_starts_per_task": 0,
        "train_frequency": 4, "target_network_frequency": 1000,
        "batch_size": 32, "seq_len": 16, "sequences_per_batch": 8,
        "max_grad_norm": 10.0, "checkpoint_interval_steps": 5000 if smoke else 50_000,
        "amp_dtype": "bfloat16", "device": "cuda",
        "result_suffix": result.name, "save_dir": str(result),
    }
    args = [sys.executable, "-B", "run_task.py", "atari-dqn"]
    for key, value in values.items():
        args.extend((f"--{key}", str(value)))
    return args + ["--allow_tf32", "--cudnn_benchmark", "--fused_optimizer"]


def validate(result: Path, model: str, variant: str, smoke: bool) -> dict:
    """Require completed budget, immutable protocol, finite trained outputs, and artifacts."""
    metrics = json.loads((result / "metrics.json").read_text())
    epsilon, exploration, decay, _ = VARIANTS[variant]
    steps = 25_000 if smoke else 3_000_000
    expected = {
        "global_step": steps, "model_type": model, "num_layers": 3,
        "hidden_size": HIDDEN[model], "env_id": "ALE/Seaquest-v5",
        "num_actions": 18, "action_space_mode": "full18", "frame_skip": 4,
        "frame_stack": 4, "atari_env_protocol": "baseline", "buffer_size": 1_000_000,
        "seq_len": 16, "learning_rate": 1e-4, "learning_rate_decay_step": decay,
        "exploration_steps": exploration,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"{result.name}: {key}={metrics.get(key)!r}, expected {value!r}")
    expected_epsilon = 1.0 + (epsilon - 1.0) * min(1.0, steps / exploration)
    if not math.isclose(metrics["epsilon"], expected_epsilon):
        raise RuntimeError("Final epsilon does not match the fixed-step schedule")
    if not math.isclose(metrics["effective_learning_rate"], 1e-4 if smoke else 1e-5):
        raise RuntimeError("Final LR does not match the decay schedule")
    for key in ("loss", "q_values_mean", "episodic_return_100"):
        if not math.isfinite(float(metrics.get(key, float("nan")))):
            raise RuntimeError(f"Non-finite or missing final {key}")
    if not (result / "metrics_history.jsonl").stat().st_size:
        raise RuntimeError("Empty metrics history")
    checkpoints = list(result.glob("*.pth"))
    if not checkpoints or not all(path.stat().st_size for path in checkpoints):
        raise RuntimeError("Missing/empty final or resumable checkpoint")
    return metrics


def run_unit(model: str, variant: str, result: Path, artifact: Path, smoke: bool) -> dict:
    """Run or resume one exact leaf; never advance the queue after a failed/paused unit."""
    if (result / "metrics.json").exists():
        return validate(result, model, variant, smoke)
    args = command(model, variant, result, smoke)
    if (result / "checkpoint.pth").exists():
        args += ["--resume_from", str(result / "checkpoint.pth")]
    elif result.exists() and any(result.iterdir()):
        raise RuntimeError(f"Refusing non-empty result without checkpoint: {result}")
    result.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    with (artifact / "command.json").open("w") as stream:
        json.dump({"argv": args, "gamma": VARIANTS[variant][3], "seed": 2}, stream, indent=2)
    LOG.info("Starting %s: %s", result.name, shlex.join(args))
    with (artifact / "train.log").open("a") as stream:
        completed = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        (artifact / "fail").write_text(f"training exit_code={completed.returncode}\n")
        raise RuntimeError(f"Training failed: {result}; see {artifact / 'train.log'}")
    if not (result / "metrics.json").exists():
        (artifact / "paused").write_text(f"Preserve checkpoint/replay: {result}\n")
        raise RuntimeError(f"Paused without final metrics: {result}; queue stopped")
    try:
        metrics = validate(result, model, variant, smoke)
        log = (artifact / "train.log").read_text(errors="replace")
        if re.search(r"Traceback \(most recent call last\)|No space left|Disk quota exceeded"
                     r"|non[- ]finite|CUDA out of memory", log, re.IGNORECASE):
            raise RuntimeError("Explicit training/storage/numeric error in log")
    except Exception as exc:
        (artifact / "fail").write_text(f"validation: {exc}\n")
        raise
    (artifact / "done").write_text(f"completed global_step={metrics['global_step']}\n")
    LOG.info("Completed %s", result.name)
    return metrics


def accept_smoke(result: Path, artifact: Path, acceptance: Path, metrics: dict) -> None:
    """Record accepted smoke evidence, then delete only its two exact temporary leaves."""
    expected_name = f"{metrics['model_type']}_smoke25k_seed2"
    inventory = []
    for leaf in (result, artifact):
        if leaf.is_symlink() or leaf.name != expected_name or not leaf.is_dir():
            raise RuntimeError(f"Unsafe smoke cleanup target: {leaf}")
        for path in sorted(leaf.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"Unexpected smoke symlink: {path}")
            if path.is_file():
                inventory.append({"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    acceptance.write_text(json.dumps({"metrics": metrics, "removed_files": inventory}, indent=2))
    LOG.info("Accepted smoke cleanup inventory: %s", json.dumps(inventory))
    for leaf in (result, artifact):
        shutil.rmtree(leaf)


def main() -> None:
    """Run one smoke-gated five-unit queue or print all commands without writing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=HIDDEN, required=True)
    parser.add_argument("--cuda-device", choices=("0", "1"), required=True)
    parser.add_argument("--result-parent", type=Path, required=True)
    parser.add_argument("--artifact-parent", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for parent in (args.result_parent, args.artifact_parent):
        if not parent.is_absolute():
            parser.error("Result/artifact parents must be absolute")
    if args.result_parent.resolve() == args.artifact_parent.resolve():
        parser.error("Result and artifact parents must differ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        for variant in VARIANTS:
            result = args.result_parent / f"{args.model}_{variant}_seed2"
            print(shlex.join(command(args.model, variant, result)))
        return
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=args.cuda_device,
                      DISABLE_TQDM="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    args.artifact_parent.mkdir(parents=True, exist_ok=True)
    with (args.artifact_parent / f"{args.model}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acceptance = args.artifact_parent / f"{args.model}_smoke_accepted.json"
        if not acceptance.exists():
            name = f"{args.model}_smoke25k_seed2"
            result, artifact = args.result_parent / name, args.artifact_parent / name
            metrics = run_unit(args.model, "a", result, artifact, True)
            accept_smoke(result, artifact, acceptance, metrics)
        for variant in VARIANTS:
            name = f"{args.model}_{variant}_seed2"
            run_unit(args.model, variant, args.result_parent / name,
                     args.artifact_parent / name, False)
        LOG.info("All five %s units completed", args.model)


if __name__ == "__main__":
    main()
