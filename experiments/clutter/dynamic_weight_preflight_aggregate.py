"""Aggregate four two-epoch sanity reports and compare timing with LSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("lstm", "mlstm", "hyperlstm", "brims")


def parse_args() -> argparse.Namespace:
    """Parse preflight aggregation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    """Load the exact four seed-1 reports and add LSTM-relative timing ratios."""
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight summary: {args.output}")
    reports = []
    for model in MODELS:
        path = args.reports_root / f"{model}-seed01.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["model"] != model or int(report["seed"]) != 1:
            raise ValueError(f"Unexpected identity in {path}")
        if report["source_commit"] != args.source_commit:
            raise ValueError(f"Source commit mismatch in {path}")
        if not report["loss_decreased"] or not report["all_finite"]:
            raise ValueError(f"Sanity gate failed in {path}")
        reports.append(report)

    lstm_seconds = float(reports[0]["wall_clock_seconds_per_epoch"])
    for report in reports:
        report["wall_clock_ratio_vs_lstm"] = (
            float(report["wall_clock_seconds_per_epoch"]) / lstm_seconds
        )
    return {
        "status": "passed",
        "source_commit": args.source_commit,
        "seed": 1,
        "epochs": 2,
        "models": reports,
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = aggregate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
