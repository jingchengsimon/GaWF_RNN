#!/usr/bin/env python3
"""Configure the six-model Clutter multi-seed protocol for current-frame-only input.

This wrapper reuses the six frozen chan=2 best hyperparameters while changing only the input
channel count and output namespaces. It exposes the same task mapping, validation, status, and
manifest commands as ``clutter_best6_multiseed.py``.
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

ANALYSIS_RESULT_ROOT = (
    "results/anal_index/G_behaviour/export_fg_switch_offset_acc/data/"
    "clutter_best6_jointswitch_balanced_chan1"
)


def configure_protocol() -> None:
    """Select the isolated chan=1 result, artifact, and monitoring namespaces."""

    protocol.CHAN_NUM = 1
    protocol.RESULT_ROOT_SUFFIX = "clutter_best6_multiseed_40h_chan1_ep150"
    protocol.ARTIFACT_TAG = "clutter_best6_chan1_10seed_ep150"
    protocol.RUN_ID_PREFIX = "amarel-clutter-best6-chan1-10seed-ep150"
    protocol.RUN_DESCRIPTION = (
        "Clutter six frozen chan=2 best hyperparameters reused with chan=1, seeds 1-10, "
        "150 full epochs, no early stopping"
    )


def build_sjc_manifest(run_id: str, remote_root: str, conda_init: str) -> dict[str, Any]:
    """Build a JOBS.md registry manifest for dual-GPU training plus balanced testing."""

    mutable_names = (
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
        manifest["id"] = f"sjc-clutter-best6-chan1-10seed-ep150-{run_id}"
        manifest["description"] = (
            f"{protocol.RUN_DESCRIPTION}; each valid checkpoint is then evaluated on the "
            "strict jointswitch-balanced test set"
        )
        manifest["host"] = "sjc-remote"
        manifest["scheduler"] = {
            "type": "process",
            "job_ids": [],
            "run_ids": [run_id],
            "tmux_session": run_id,
            "process_patterns": ["run_clutter_best6_chan1_2gpu.sh"],
            "collect_gpu": True,
        }
        manifest["paths"]["log_globs"] = [
            f"experiments/amarel/artifacts/{protocol.ARTIFACT_TAG}/{run_id}.log"
        ]
        manifest["paths"]["result_paths"].append(ANALYSIS_RESULT_ROOT)
        manifest["notes"] = [
            note
            for note in manifest["notes"]
            if not note.startswith("Submitted as ten independent Slurm arrays")
        ]
        for config, unit in zip(protocol.all_task_configs(), manifest["tracking"]["units"]):
            unit["analysis_result_dir"] = f"{ANALYSIS_RESULT_ROOT}/{config.unit_id}"
            unit["required_analysis_globs"] = [
                "fg_switch_offset_acc_*.npz",
                "fg_switch_offset_meta_*.json",
            ]
        manifest["notes"].extend(
            [
                "Two persistent workers split the 60 units across sjc-remote GPUs 0 and 1.",
                (
                    "A unit done marker is written only after strict balanced fg-switch "
                    "analysis succeeds."
                ),
                "Balanced test suffix: 40h-float32-jointswitch-balanced.",
            ]
        )
        return manifest
    finally:
        for name, value in original.items():
            setattr(protocol, name, value)


def _emit_sjc_manifest() -> None:
    """Parse and emit the sjc-remote manifest-only command."""

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
