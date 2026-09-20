"""Run one Seaquest A/C model/seed cell with isolated smoke and resumable 3M training.

Inputs are a cell index and exact result/artifact parents. Outputs retain commands, logs,
smoke evidence, replay, final model and metrics, using the existing SJC diagnostic protocol.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import subprocess

from experiments.remote.run_sjc_seaquest_diagnostics import command, validate


def cell_identity(cell: int) -> tuple[str, str, int]:
    """Map 0..15 to LSTM/GaWF, A/C and seeds 1..4 without serial dependencies."""
    if cell not in range(16):
        raise ValueError("Cell must be in 0..15")
    return ("lstm", "gawf")[cell // 8], ("a", "c")[(cell % 8) // 4], cell % 4 + 1


def training_command(cell: int, result: Path, smoke: bool) -> list[str]:
    """Reuse SJC arguments verbatim except seed, paths and replay retention."""
    model, variant, seed = cell_identity(cell)
    argv = command(model, variant, result, smoke)
    argv[argv.index("--seed") + 1] = str(seed)
    argv.append("--keep_replay_on_success")
    return argv


def main() -> None:
    """Validate smoke then run/resume the same cell from its own checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, required=True, choices=range(16))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model, variant, seed = cell_identity(args.cell)
    unit = f"{model}_{variant}_seed{seed}"
    artifact = args.artifacts / unit
    logging.basicConfig(level=logging.INFO)
    for smoke in (True, False):
        phase = "smoke" if smoke else "formal"
        result = args.results / phase / unit
        argv = training_command(args.cell, result, smoke)
        if args.dry_run:
            print(json.dumps(argv))
            continue
        artifact.mkdir(parents=True, exist_ok=True)
        checkpoint = result / "checkpoint.pth"
        if not (result / "metrics.json").exists():
            if checkpoint.exists():
                argv += ["--resume_from", str(checkpoint)]
            elif result.exists() and any(result.iterdir()):
                raise RuntimeError(f"Non-empty result without checkpoint: {result}")
            (artifact / f"command_{phase}.json").write_text(json.dumps(argv, indent=2) + "\n")
            with (artifact / f"{phase}.log").open("a") as stream:
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)

                def pause(signum: int, frame: object) -> None:
                    process.send_signal(signal.SIGUSR1)

                previous = {s: signal.signal(s, pause) for s in (signal.SIGUSR1, signal.SIGTERM)}
                try:
                    rc = process.wait()
                finally:
                    for s, handler in previous.items():
                        signal.signal(s, handler)
            if rc:
                (artifact / "fail").write_text(f"{phase} training exit={rc}\n")
                raise RuntimeError(f"Training exit={rc}; see {artifact / (phase + '.log')}")
            if not (result / "metrics.json").exists():
                assert checkpoint.stat().st_size > 0
                job = os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
                subprocess.run(["scontrol", "requeue", job], check=True)
                return
        metrics = validate(result, model, variant, smoke)
        if metrics.get("gamma") != .99 or metrics.get("reward_clip") is not True:
            raise RuntimeError("Seaquest gamma/reward clipping mismatch")
        marker = "smoke_accepted.json" if smoke else "done"
        (artifact / marker).write_text(json.dumps(dict(
            seed=seed, variant=variant, global_step=metrics["global_step"], result=str(result),
            artifacts_retained=True)) + "\n")


if __name__ == "__main__":
    main()
