"""Run disjoint GaWF queues after an authorized checkpoint handoff of the original queue.

Inputs: immutable training snapshot and explicit result/artifact/control directories.
Outputs: original training bundles plus lane logs/status. GPU1 runs A/B/C; GPU0 waits for
all five LSTM units, then runs F995/F999. Training commands come from the original snapshot.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

LANES = {"gpu1": ("a", "b", "c"), "gpu0": ("f995", "f999")}
ORIGINAL_SHA256 = "6395ba1ad20b692e90afd1deb1f8e43d27264c08277483faedc0cfca9dec6310"
LOG = logging.getLogger(__name__)


def process_arguments() -> list[list[str]]:
    """Read process arguments without inspecting anything outside exact process metadata."""
    commands = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            commands.append((process / "cmdline").read_bytes().decode().strip("\0").split("\0"))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
    return commands


def assert_no_writer(result: Path, commands: list[list[str]]) -> None:
    """Reject both the obsolete GaWF queue and any trainer writing the target result leaf."""
    for argv in commands:
        if ("experiments.remote.run_sjc_seaquest_diagnostics" in argv
                and "--model" in argv and argv[argv.index("--model") + 1] == "gawf"):
            raise RuntimeError("Original GaWF queue is still active; checkpoint handoff incomplete")
        if "--save_dir" in argv and argv[argv.index("--save_dir") + 1] == str(result):
            raise RuntimeError(f"Existing writer for {result}")


def lstm_ready(original: object, results: Path, artifacts: Path) -> bool:
    """Require five successful LSTM results; fail safely on an explicitly paused/failed unit."""
    for variant in original.VARIANTS:
        name = f"lstm_{variant}_seed2"
        metrics, status = results / name / "metrics.json", artifacts / name
        if metrics.exists() and (status / "done").exists():
            original.validate(results / name, "lstm", variant, False)
            continue
        if (status / "fail").exists() or (status / "paused").exists():
            raise RuntimeError(f"LSTM dependency is failed/paused: {name}")
        return False
    return True


def gpu_idle(gpu: str) -> bool:
    """Treat a GPU as available only when its compute-process query is empty."""
    return not subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()


def main() -> None:
    """Execute a locked fixed lane, leaving model/replay/checkpoint behavior unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--result-parent", type=Path, required=True)
    parser.add_argument("--artifact-parent", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (args.training_root, args.result_parent, args.artifact_parent, args.control_dir):
        if not path.is_absolute():
            parser.error("All paths must be absolute")
    helper = args.training_root / "experiments/remote/run_sjc_seaquest_diagnostics.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("Original training launcher changed; refusing protocol drift")
    sys.path.insert(0, str(args.training_root))
    from experiments.remote import run_sjc_seaquest_diagnostics as original

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        for variant in LANES[args.lane]:
            result = args.result_parent / f"gawf_{variant}_seed2"
            print(shlex.join(original.command("gawf", variant, result)))
        return
    if not (args.artifact_parent / "gawf_smoke_accepted.json").is_file():
        raise RuntimeError("GaWF smoke acceptance is missing")
    if not (args.control_dir / "handoff.json").is_file():
        raise RuntimeError("Verified checkpoint handoff evidence is missing")
    gpu = args.lane[-1]
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=gpu,
                      DISABLE_TQDM="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    with (args.artifact_parent / f"gawf_split_{args.lane}.lock").open("a") as lane_lock:
        fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.lane == "gpu0":
                LOG.info("Waiting for all five LSTM results before GaWF F995/F999")
                while not lstm_ready(original, args.result_parent, args.artifact_parent):
                    time.sleep(30)
            for variant in LANES[args.lane]:
                name = f"gawf_{variant}_seed2"
                with (args.artifact_parent / f"{name}.lock").open("a") as unit_lock:
                    fcntl.flock(unit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    while not gpu_idle(gpu):
                        time.sleep(30)
                    result = args.result_parent / name
                    assert_no_writer(result, process_arguments())
                    original.run_unit("gawf", variant, result, args.artifact_parent / name, False)
            (args.control_dir / f"{args.lane}.done").write_text("All assigned units complete\n")
        except BaseException as exc:
            (args.control_dir / f"{args.lane}.fail").write_text(f"{type(exc).__name__}: {exc}\n")
            raise


if __name__ == "__main__":
    main()
