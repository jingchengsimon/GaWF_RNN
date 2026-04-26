#!/usr/bin/env python3
"""
Plot generalization summaries from Phase 3 CSVs under experiments/generalization/artifacts/.

Produces (per run):
  - Overfit gap vs dataset scale (char + sector panels)
  - Train accuracy vs scale (char + sector panels)
  - Val accuracy vs scale (char + sector panels)

Use --csv_tag for the CSV suffix produced by collect_results.py, e.g.
  _short_ep100 -> phase3_summary_{scale}_short_ep100.csv (short pipeline),
  _ep100 -> phase3_summary_{scale}_ep100.csv (full pipeline, profile=full; see phase3_train_scale.sh).
Epoch is _ep${NUM_EPOCHS}; default NUM_EPOCHS=100 unless overridden.

By default only PNG is written; pass --save-pdf to also emit PDF.
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
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        raw = r.get(value_key)
        by_scale_model[scale][model] = float(raw) if raw not in (None, "") else float("nan")
    return by_scale_model


def plot_lines(
    by_scale_model: Dict[str, Dict[str, float]],
    ylabel: str,
    title: str,
    ax,
    *,
    show_legend: bool,
) -> None:
    x = list(range(len(SCALE_ORDER)))
    for m in MODELS:
        ys = [by_scale_model[s].get(m, float("nan")) for s in SCALE_ORDER]
        st = STYLES[m]
        ax.plot(x, ys, label=m.upper(), linewidth=2, markersize=7, **st)

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_ORDER)
    ax.set_xlabel("Dataset scale (train hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)


def plot_metric_pair(
    char_map: Dict[str, Dict[str, float]],
    sector_map: Dict[str, Dict[str, float]],
    char_ylabel: str,
    sector_ylabel: str,
    title: str,
    out_path_base: str,
    *,
    save_pdf: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=False)
    plot_lines(char_map, char_ylabel, "Char", axes[0], show_legend=True)
    plot_lines(sector_map, sector_ylabel, "Sector", axes[1], show_legend=False)
    fig.suptitle(title)

    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    fig.savefig(out_path_base + ".png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    if save_pdf:
        fig.savefig(out_path_base + ".pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    if save_pdf:
        print(f"Saved {out_path_base}.png / .pdf")
    else:
        print(f"Saved {out_path_base}.png")


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
        help='CSV suffix e.g. "_short_ep100" -> phase3_summary_4h_short_ep100.csv (default: "")',
    )
    ap.add_argument(
        "--out_dir",
        default="",
        help="Figure output directory (default: results/anal_figs/generalization)",
    )
    ap.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also write PDF alongside PNG (default: PNG only).",
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

    save_pdf = bool(args.save_pdf)

    gap_char_map = build_scale_model_map(rows, "overfit_gap_char")
    gap_sector_map = build_scale_model_map(rows, "overfit_gap_sector")
    plot_metric_pair(
        gap_char_map,
        gap_sector_map,
        char_ylabel="Overfit gap (@ best val char epoch)",
        sector_ylabel="Overfit gap (@ best val sector epoch)",
        title="Generalization: overfit gap vs train set size",
        out_path_base=os.path.join(out_dir, f"overfit_gap_vs_scale{mid}"),
        save_pdf=save_pdf,
    )

    train_char_map = build_scale_model_map(rows, "train_acc_char")
    train_sector_map = build_scale_model_map(rows, "train_acc_sector")
    plot_metric_pair(
        train_char_map,
        train_sector_map,
        char_ylabel="Train char accuracy (%)",
        sector_ylabel="Train sector accuracy (%)",
        title="Train accuracy vs train set size",
        out_path_base=os.path.join(out_dir, f"train_acc_vs_scale{mid}"),
        save_pdf=save_pdf,
    )

    val_char_map = build_scale_model_map(rows, "val_acc_char")
    val_sector_map = build_scale_model_map(rows, "val_acc_sector")
    plot_metric_pair(
        val_char_map,
        val_sector_map,
        char_ylabel="Validation char accuracy (%)",
        sector_ylabel="Validation sector accuracy (%)",
        title="Validation accuracy vs train set size",
        out_path_base=os.path.join(out_dir, f"val_acc_vs_scale{mid}"),
        save_pdf=save_pdf,
    )


if __name__ == "__main__":
    main()
