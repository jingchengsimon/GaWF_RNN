"""
Generate *_metrics.json from existing results *.pkl files.

Use when training saved the pkl but failed before writing metrics (e.g. due to
summarize_experiment_metrics error). Reads each pkl, builds the same summary
dict as summarize_experiment_metrics(), and writes it to <stem>_metrics.json.

Usage:
  # Single file
  python pkl_to_metrics_json.py results/train_data/sector_40h_adamw/gru_sector_acc_h105_lr0.0005_wd0.0001_do0.pkl

  # All pkl files in a directory (no _model.pth in name)
  python pkl_to_metrics_json.py results/train_data/sector_40h_adamw/

  # Override hyperparams if filename does not match expected pattern
  python pkl_to_metrics_json.py path/to/results.pkl --dataset_suffix 40h --num_epochs 150 --optimizer adamw
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import pickle
from pathlib import Path

# Allow importing utils from project root
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.train_helpers import summarize_experiment_metrics, save_metrics_summary


# Stem pattern: model_type + mode (sector|coord|allchars) + optional _acc + _hN + _lr... + _wd... + _do... + optional _nofb|_fbN
_STEM_RE = re.compile(
    r"^(\w+)_(sector|coord|allchars)(_acc)?_h(\d+)_lr([\d.]+)_wd([\d.]+)_do([\d.]+)(_nofb|_fb\d+)?$",
    re.IGNORECASE,
)


def parse_stem(stem: str) -> dict | None:
    """Parse results stem into kwargs for summarize_experiment_metrics. Returns None if no match."""
    m = _STEM_RE.match(stem.strip())
    if not m:
        return None
    model_type, dataset_mode, _acc, hidden_size, lr, wd, do = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
    return {
        "model_type": model_type.lower(),
        "dataset_mode": dataset_mode.lower(),
        "hidden_size": int(hidden_size),
        "lr": float(lr),
        "weight_decay": float(wd),
        "dropout": float(do),
    }


def pkl_paths_from_dir(dir_path: str) -> list[str]:
    """Return list of .pkl paths in directory, excluding *_model.pth sidecars (we only have .pkl)."""
    root = Path(dir_path)
    if not root.is_dir():
        return []
    out = []
    for f in root.glob("*.pkl"):
        # skip if it looks like a model-only save
        if "_model.pkl" in f.name:
            continue
        out.append(str(f))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate _metrics.json from training result .pkl files.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "path",
        nargs="+",
        help="Path(s) to .pkl file(s), or a single directory to process all .pkl inside it.",
    )
    ap.add_argument(
        "--dataset_suffix",
        default="",
        help="Dataset suffix for metrics (e.g. 40h). Default: empty.",
    )
    ap.add_argument(
        "--dataset_mode",
        default=None,
        help="Override dataset_mode (sector|coord|allchars). Default: parsed from filename.",
    )
    ap.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (default: use actual_epochs from pkl).",
    )
    ap.add_argument(
        "--optimizer",
        default="adamw",
        help="Optimizer name for metrics. Default: adamw.",
    )
    args = ap.parse_args()

    # Collect all .pkl files
    pkl_files = []
    for p in args.path:
        path = Path(p).resolve()
        if path.is_file():
            if path.suffix.lower() == ".pkl":
                pkl_files.append(str(path))
            else:
                print(f"Skip (not .pkl): {path}", file=sys.stderr)
        elif path.is_dir():
            pkl_files.extend(pkl_paths_from_dir(str(path)))
        else:
            print(f"Not found: {path}", file=sys.stderr)

    if not pkl_files:
        print("No .pkl files to process.", file=sys.stderr)
        sys.exit(1)

    for pkl_path in pkl_files:
        stem = Path(pkl_path).stem
        parsed = parse_stem(stem)
        if parsed is None:
            print(f"Warning: could not parse stem '{stem}', using defaults for missing fields.", file=sys.stderr)
            parsed = {
                "model_type": "unknown",
                "dataset_mode": args.dataset_mode or "sector",
                "hidden_size": 0,
                "lr": 0.0,
                "weight_decay": 0.0,
                "dropout": 0.0,
            }
        if args.dataset_mode is not None:
            parsed["dataset_mode"] = args.dataset_mode

        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        num_epochs = args.num_epochs
        if num_epochs is None:
            num_epochs = int(results.get("actual_epochs", 0))

        metric_summary = summarize_experiment_metrics(
            results,
            model_type=parsed["model_type"],
            dataset_suffix=args.dataset_suffix,
            dataset_mode=parsed["dataset_mode"],
            num_epochs=num_epochs,
            hidden_size=parsed["hidden_size"],
            lr=parsed["lr"],
            weight_decay=parsed["weight_decay"],
            dropout=parsed["dropout"],
            optimizer=args.optimizer,
        )
        out_path = str(Path(pkl_path).with_suffix("")) + "_metrics.json"
        save_metrics_summary(metric_summary, out_path, logger=None)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
