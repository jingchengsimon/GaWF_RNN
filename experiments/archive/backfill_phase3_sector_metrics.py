#!/usr/bin/env python3
"""
Backfill Phase 3 sector metrics into existing *_metrics.json files from saved *.pkl files.

For each Phase 3 result folder, this script computes sector accuracy at the epoch where
validation sector accuracy is maximal:
  - train_acc_sector_at_best_val_sector
  - val_acc_sector_at_best
  - overfit_gap_sector
  - best_epoch_val_acc_sector_1based

Run from the repo root. Use --apply to write JSON updates; without it, this is a dry run.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from glob import glob
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import numpy.core.multiarray as np_multiarray
import numpy.core.numeric as np_numeric

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", np_numeric)
sys.modules.setdefault("numpy._core.multiarray", np_multiarray)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj).__name__}")
    return obj


def _sector_arrays(results: Dict[str, Any]) -> Tuple[np.ndarray | None, np.ndarray | None]:
    train = _as_float_array(results.get("train_acc_pos"))
    if train is None:
        train = _as_float_array(results.get("train_metric_pos"))
    val = _as_float_array(results.get("val_acc_pos"))
    if val is None:
        val = _as_float_array(results.get("val_metric_pos"))
    return train, val


def _compute_sector_metrics(results: Dict[str, Any]) -> Dict[str, Any] | None:
    train, val = _sector_arrays(results)
    if train is None or val is None or len(train) == 0 or len(val) == 0:
        return None
    n = min(len(train), len(val), int(results.get("actual_epochs", len(val)) or len(val)))
    train = train[:n]
    val = val[:n]
    if n == 0 or np.all(np.isnan(val)):
        return None

    best_idx = int(np.nanargmax(val))
    train_acc = float(train[best_idx])
    val_acc = float(val[best_idx])
    return {
        "train_acc_sector_at_best_val_sector": train_acc,
        "val_acc_sector_at_best": val_acc,
        "overfit_gap_sector": train_acc - val_acc,
        "best_epoch_val_acc_sector_1based": best_idx + 1,
    }


def _iter_phase3_pkls(train_data_root: str) -> Iterable[str]:
    patterns = [
        os.path.join(train_data_root, "gen_phase3_*_ep*", "*.pkl"),
        os.path.join(train_data_root, "gen_phase3_short_*_ep*", "*.pkl"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob(pattern)):
            if path.endswith("_metrics.pkl"):
                continue
            if path not in seen:
                seen.add(path)
                yield path


def _metrics_path_for_pkl(pkl_path: str) -> str:
    stem = pkl_path[:-4] if pkl_path.endswith(".pkl") else pkl_path
    return f"{stem}_metrics.json"


def backfill(train_data_root: str, apply: bool) -> Tuple[int, int, int]:
    scanned = 0
    updated = 0
    skipped = 0
    for pkl_path in _iter_phase3_pkls(train_data_root):
        scanned += 1
        metrics_path = _metrics_path_for_pkl(pkl_path)
        if not os.path.isfile(metrics_path):
            print(f"SKIP no metrics json: {metrics_path}")
            skipped += 1
            continue

        results = _load_pickle(pkl_path)
        sector_metrics = _compute_sector_metrics(results)
        if sector_metrics is None:
            print(f"SKIP no sector arrays: {pkl_path}")
            skipped += 1
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        changed = any(metrics.get(k) != v for k, v in sector_metrics.items())
        if not changed:
            continue
        print(f"UPDATE {metrics_path}: {sector_metrics}")
        if apply:
            metrics.update(sector_metrics)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
                f.write("\n")
        updated += 1
    return scanned, updated, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train_data_root",
        default="results/train_data",
        help="Root containing Phase 3 result folders",
    )
    ap.add_argument("--apply", action="store_true", help="Write JSON updates")
    args = ap.parse_args()

    os.chdir(_repo_root())
    scanned, updated, skipped = backfill(args.train_data_root, args.apply)
    mode = "applied" if args.apply else "dry-run"
    print(f"Done ({mode}). scanned={scanned}, updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()
