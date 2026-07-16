#!/usr/bin/env python3
"""Configure the six-model Clutter protocol for chan=1 and seed 42 only.

The six frozen chan=2 best hyperparameters are reused without a chan=1 search. Outputs use an
isolated result namespace so cancelled multi-seed artifacts cannot enter later fg-switch analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.generalization import clutter_best6_multiseed as protocol


def configure_protocol() -> None:
    """Select chan=1, seed 42, and isolated result/monitoring namespaces."""

    protocol.SEEDS = (42,)
    protocol.CHAN_NUM = 1
    protocol.RESULT_ROOT_SUFFIX = "clutter_best6_chan1_seed42_40h_ep150"
    protocol.ARTIFACT_TAG = "clutter_best6_chan1_seed42_ep150"
    protocol.RUN_ID_PREFIX = "amarel-clutter-best6-chan1-seed42-ep150"
    protocol.RUN_DESCRIPTION = (
        "Clutter six frozen chan=2 best hyperparameters reused with chan=1, seed=42 only, "
        "150 full epochs, no early stopping; checkpoints reserved for later fg-switch analysis"
    )


def build_sjc_manifest(run_id: str, remote_root: str, conda_init: str) -> dict[str, Any]:
    """Build the JOBS manifest for seed42 training-only execution on two local GPUs."""

    mutable_names = (
        "SEEDS",
        "CHAN_NUM",
        "RESULT_ROOT_SUFFIX",
        "ARTIFACT_TAG",
        "RUN_ID_PREFIX",
        "RUN_DESCRIPTION",
    )
    original = {name: getattr(protocol, name) for name in mutable_names}
    try:
        configure_protocol()
        manifest = protocol.build_manifest([run_id], remote_root, conda_init)
        manifest["id"] = f"sjc-clutter-best6-chan1-seed42-ep150-{run_id}"
        manifest["description"] = f"{protocol.RUN_DESCRIPTION}; training/checkpoints only"
        manifest["host"] = "sjc-remote"
        manifest["status"] = "running"
        manifest["scheduler"] = {
            "type": "process",
            "job_ids": [],
            "run_ids": [run_id],
            "tmux_session": run_id,
            "process_patterns": ["run_clutter_best6_chan1_seed42_2gpu.sh"],
            "collect_gpu": True,
        }
        manifest["paths"]["log_globs"] = [
            f"experiments/amarel/artifacts/{protocol.ARTIFACT_TAG}/{run_id}.log"
        ]
        manifest["notes"] = [
            note
            for note in manifest["notes"]
            if not note.startswith("Submitted as 1 independent Slurm array")
        ]
        manifest["notes"].extend(
            [
                "Two persistent workers split the six units across sjc-remote GPUs 0 and 1.",
                "A unit done marker is written immediately after its training output validates.",
                "No balanced test or fg-switch analysis is run by this launcher.",
            ]
        )
        return manifest
    finally:
        for name, value in original.items():
            setattr(protocol, name, value)


def _emit_sjc_manifest() -> None:
    """Parse and emit the sjc-remote seed42 manifest-only command."""

    parser = argparse.ArgumentParser(description=build_sjc_manifest.__doc__)
    parser.add_argument("emit-sjc-manifest", nargs="?")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--conda-init", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(
        build_sjc_manifest(args.run_id, args.remote_root, args.conda_init),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main() -> None:
    """Configure and execute the shared protocol command-line interface."""

    if len(sys.argv) > 1 and sys.argv[1] == "emit-sjc-manifest":
        _emit_sjc_manifest()
        return
    configure_protocol()
    protocol.main()


if __name__ == "__main__":
    main()
