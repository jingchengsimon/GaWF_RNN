#!/usr/bin/env python3
"""Register the SentiHood LSTM-Final Amarel job in the persistent dashboard.

Run this locally immediately after a successful Slurm submission. The tracker
counts one valid result only when the LSTM-Final metrics JSON matches the recipe
and its companion pickle/checkpoint files exist on Amarel.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


RESULT_STEM = "lstm_sentihood_h50_emb50_lr0.01_wd0.0_edo0.001_rdo0.001"
RETENTION_POLICY = "human_confirmation_required"


def build_spec(args: argparse.Namespace) -> Dict[str, Any]:
    result_glob = f"results/train_data/{args.result_suffix}/{RESULT_STEM}_metrics.json"
    return {
        "id": f"sentihood-lstm-final-{args.job_id}",
        "description": (
            "SentiHood LSTM-Final reproducibility run: query-pair preprocessing, "
            "balanced None/Positive/Negative sampling, h50/emb50/lr0.01"
        ),
        "machine": "amarel",
        "job_ids": [str(args.job_id)],
        "remote_root": args.remote_root,
        "retention_policy": RETENTION_POLICY,
        "tracker": {
            "type": "metrics_grid",
            "expected_total": 1,
            "result_glob": result_glob,
            "match": {
                "dataset": {"equals": "sentihood"},
                "model_type": {"equals": "lstm"},
                "hidden_size": {"equals": 50},
                "embed_dim": {"equals": 50},
                "lr": {"equals": 0.01},
                "weight_decay": {"equals": 0.0},
                "pooling": {"equals": "last"},
                "balance_train_labels": {"equals": True},
            },
            "uniqueness_fields": [
                "model_type",
                "hidden_size",
                "embed_dim",
                "lr",
                "weight_decay",
                "balance_train_labels",
            ],
            "companion_files": ["{stem}.pkl", "{stem}_model.pth"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job_id", required=True, help="Slurm job id returned by sbatch.")
    parser.add_argument("--remote_root", default="~/FAW_RNN")
    parser.add_argument("--result_suffix", default="sentihood_lstm_final")
    parser.add_argument(
        "--dashboard_root",
        default="/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes",
        help="Parent directory that contains the central dashboard package.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the JSON task spec without registering it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    spec_json = json.dumps(spec, sort_keys=True)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return

    dashboard_root = Path(args.dashboard_root)
    cmd = [
        sys.executable,
        "-m",
        "dashboard.manage_tasks",
        "register",
        "--spec",
        spec_json,
    ]
    subprocess.run(cmd, cwd=dashboard_root, check=True)


if __name__ == "__main__":
    main()
