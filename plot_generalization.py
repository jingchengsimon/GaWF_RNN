#!/usr/bin/env python3
"""
Plot generalization summaries from Phase 3 CSVs under experiments/generalization/artifacts/.

Produces (per run):
  - Overfit gap vs dataset scale
  - Train char acc vs scale
  - Val char acc vs scale

Use --csv_tag for the CSV suffix produced by collect_results.py (e.g. _short_ep50
for phase3_summary_{scale}_short_ep50.csv). Short pipeline default base is _short;
epoch is appended in the bash scripts as _ep${NUM_EPOCHS}.
Full pipeline: default csv_tag "" -> phase3_summary_{scale}.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE_ORDER = ["4h", "10h", "20h", "40h"]
MODELS = ["rnn", "lstm", "gru", "gawf"]
STYLES = {
    "rnn": dict(color="#1f77b4", marker="o"),
    "lstm": dict(color="#ff7f0e", marker="s"),
    "gru": dict(color="#2ca02c", marker="^"),
    "gawf": dict(color="#d62728", marker="D"),
}


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_rows_for_scales(artifact_dir: str, csv_tag: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for scale in SCALE_ORDER:
        name = f"phase3_summary_{scale}{csv_tag}.csv"
        path = os.path.join(artifact_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def build_scale_model_map(
    rows: List[Dict[str, str]], value_key: str
) -> Dict[str, Dict[str, float]]:
    by_scale_model: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        scale = r["scale"]
        model = r["model"]
        by_scale_model[scale][model] = float(r[value_key])
    return by_scale_model


def plot_lines(
    by_scale_model: Dict[str, Dict[str, float]],
    ylabel: str,
    title: str,
    out_path_base: str,
) -> None:
    x = list(range(len(SCALE_ORDER)))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in MODELS:
        ys = [by_scale_model[s].get(m, float("nan")) for s in SCALE_ORDER]
        st = STYLES[m]
        ax.plot(x, ys, label=m.upper(), linewidth=2, markersize=7, **st)

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_ORDER)
    ax.set_xlabel("Dataset scale (train hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    fig.savefig(out_path_base + ".pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_path_base + ".png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_path_base}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact_dir",
        default="",
        help="Directory with phase3_summary_*.csv (default: experiments/generalization/artifacts)",
    )
    ap.add_argument(
        "--csv_tag",
        type=str,
        default="",
        help='CSV suffix e.g. "_short_ep50" -> phase3_summary_4h_short_ep50.csv (default: "")',
    )
    ap.add_argument(
        "--out_dir",
        default="",
        help="Figure output directory (default: results/anal_figs/generalization)",
    )
    args = ap.parse_args()
    root = _repo_root()
    art = args.artifact_dir or os.path.join(
        root, "experiments", "generalization", "artifacts"
    )
    art = os.path.abspath(art)

    tag_suffix = args.csv_tag.lstrip("_") if args.csv_tag else ""
    mid = f"_{tag_suffix}" if tag_suffix else ""

    out_dir = args.out_dir or os.path.join(root, "results", "anal_figs", "generalization")
    out_dir = os.path.abspath(out_dir)

    rows = load_rows_for_scales(art, args.csv_tag)
    if not rows:
        raise SystemExit(f"No rows loaded from {art} (tag={args.csv_tag!r})")

    gap_map = build_scale_model_map(rows, "overfit_gap")
    plot_lines(
        gap_map,
        ylabel="Overfit gap (train acc − val acc @ best val epoch)",
        title="Generalization: overfit gap vs train set size",
        out_path_base=os.path.join(out_dir, f"overfit_gap_vs_scale{mid}"),
    )

    train_map = build_scale_model_map(rows, "train_acc")
    plot_lines(
        train_map,
        ylabel="Train char accuracy (%)",
        title="Train accuracy vs train set size (best-val-epoch where available)",
        out_path_base=os.path.join(out_dir, f"train_acc_vs_scale{mid}"),
    )

    val_map = build_scale_model_map(rows, "val_acc")
    plot_lines(
        val_map,
        ylabel="Validation char accuracy (%)",
        title="Validation accuracy vs train set size (best-val-epoch where available)",
        out_path_base=os.path.join(out_dir, f"val_acc_vs_scale{mid}"),
    )


if __name__ == "__main__":
    main()
