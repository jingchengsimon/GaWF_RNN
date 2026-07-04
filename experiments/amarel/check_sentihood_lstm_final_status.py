#!/usr/bin/env python3
"""Check SentiHood LSTM-Final output validity and print benchmark metrics."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def expected_paths(root: str, result_suffix: str) -> Dict[str, str]:
    stem = "lstm_sentihood_h50_emb50_lr0.01_wd0.0_edo0.001_rdo0.001"
    base = os.path.join(root, "results", "train_data", result_suffix)
    return {
        "metrics": os.path.join(base, f"{stem}_metrics.json"),
        "pkl": os.path.join(base, f"{stem}.pkl"),
        "model": os.path.join(base, f"{stem}_model.pth"),
    }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--result_suffix", default="sentihood_lstm_final")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    paths = expected_paths(root, args.result_suffix)
    summary: Dict[str, Any] = {
        "result_suffix": args.result_suffix,
        "paths": paths,
        "metrics_exists": os.path.isfile(paths["metrics"]),
        "pkl_exists": os.path.isfile(paths["pkl"]),
        "model_exists": os.path.isfile(paths["model"]),
        "valid": False,
        "reason": "",
    }
    if not summary["metrics_exists"]:
        summary["reason"] = "missing_metrics"
    else:
        try:
            metrics = load_json(paths["metrics"])
        except (OSError, json.JSONDecodeError) as exc:
            summary["reason"] = f"bad_metrics_json:{exc}"
        else:
            expected = {
                "dataset": "sentihood",
                "model_type": "lstm",
                "hidden_size": 50,
                "embed_dim": 50,
                "pooling": "last",
                "balance_train_labels": True,
            }
            mismatches = {}
            for key, value in expected.items():
                if metrics.get(key) != value:
                    mismatches[key] = metrics.get(key)
            if mismatches:
                summary["reason"] = f"metrics_mismatch:{mismatches}"
            elif not summary["pkl_exists"]:
                summary["reason"] = "missing_pkl"
            elif not summary["model_exists"]:
                summary["reason"] = "missing_model"
            else:
                summary["valid"] = True
                summary["reason"] = "ok"
                for key in (
                    "best_epoch_1based",
                    "best_val_score",
                    "test_aspect_f1_at_best",
                    "test_sentiment_acc_at_best",
                    "test_aspect_auc_at_best",
                    "test_sentiment_auc_at_best",
                    "actual_epochs",
                    "stopped_by_patience",
                ):
                    summary[key] = metrics.get(key)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            if key == "paths":
                continue
            print(f"{key}={value}")
        for name, path in paths.items():
            print(f"{name}_path={path}")
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
