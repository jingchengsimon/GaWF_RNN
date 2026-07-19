#!/usr/bin/env python3
"""
TEMPORARY: aggregate *_metrics.json from verify_temp_rnn_hidden_ablation.sh runs
and plot RNN h=256 vs h=275 overfit gap (and train/val acc) vs scale.

Y-axis limits align with Phase3 short plots: from phase3_summary_*{ref_csv_tag}.csv
(same 16 ref points as utils_viz/plot_generalization.py, 5% pad); expand if verify points fall outside.

Does not import training code.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

SCALES: List[Tuple[str, str]] = [
    ("4h", "4h-float32"),
    ("10h", "10h-float32"),
    ("20h", "20h-float32"),
    ("40h", "40h-float32"),
]
H_LIST = (256, 275)
SCALE_ORDER = [s[0] for s in SCALES]
# Match utils_viz/plot_generalization.py STYLES: RNN h275 vs phase3 "rnn"; h256 vs "gawf" (ablation stand-in).
_COLOR_RNN275 = "#1f77b4"
_MARKER_RNN275 = "o"
_COLOR_RNN256 = "#d62728"
_MARKER_RNN256 = "D"


def _repo_root() -> str:
    return PROJECT_ROOT


def _row_from_metrics(m: Dict[str, Any], scale: str) -> Dict[str, Any]:
    train_acc = m.get("train_acc_at_best_val")
    if train_acc is None:
        train_acc = m.get("best_train_acc_char")
    val_acc = m.get("val_acc_at_best")
    if val_acc is None:
        val_acc = m.get("best_val_acc_char")
    og = m.get("overfit_gap")
    if og is None and train_acc is not None and val_acc is not None:
        og = float(train_acc) - float(val_acc)
    hs = m.get("hidden_size", "")
    return {
        "scale": scale,
        "model": f"rnn_h{hs}",
        "overfit_gap": og,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }


def load_rows(root: str, ep: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sk, _ds in SCALES:
        for h in H_LIST:
            d = os.path.join(
                root, "results", "train_data", f"verify_rnn_h{h}_{sk}_ep{ep}"
            )
            if not os.path.isdir(d):
                print(f"WARN: missing dir {d}", file=sys.stderr)
                continue
            paths = glob.glob(os.path.join(d, "*_metrics.json"))
            if not paths:
                print(f"WARN: no *_metrics.json in {d}", file=sys.stderr)
                continue
            with open(paths[0], "r", encoding="utf-8") as f:
                m = json.load(f)
            rows.append(_row_from_metrics(m, sk))
    return rows


def _float_csv_field(row: Dict[str, str], key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return float("nan")
    return float(v)


def load_phase3_ref_rows(artifact_dir: str, ref_tag: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for scale in SCALE_ORDER:
        name = f"phase3_summary_{scale}{ref_tag}.csv"
        path = os.path.join(artifact_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing Phase3 ref for ylim: {path} (run collect_results phase3 or adjust --ref_csv_tag)"
            )
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def ylim_for_metric(
    ref_rows: List[Dict[str, str]],
    verify_rows: List[Dict[str, Any]],
    key: str,
    pad_frac: float = 0.05,
) -> Optional[Tuple[float, float]]:
    ref_vals: List[float] = []
    for r in ref_rows:
        v = _float_csv_field(r, key)
        if v == v:
            ref_vals.append(v)
    if not ref_vals:
        return None
    rlo, rhi = min(ref_vals), max(ref_vals)
    if rlo == rhi:
        rlo -= 0.5
        rhi += 0.5
    span0 = rhi - rlo
    lo, hi = rlo - pad_frac * span0, rhi + pad_frac * span0

    ver: List[float] = []
    for r in verify_rows:
        v = r.get(key)
        if v is not None and v == v:
            ver.append(float(v))
    if ver:
        vlo, vhi = min(ver), max(ver)
        if vlo < lo or vhi > hi:
            lo = min(lo, vlo)
            hi = max(hi, vhi)
            span1 = hi - lo
            lo -= pad_frac * span1
            hi += pad_frac * span1
    return (lo, hi)


def plot_one(
    rows: List[Dict[str, Any]],
    value_key: str,
    ylabel: str,
    title: str,
    out_base: str,
    ylim: Optional[Tuple[float, float]],
) -> None:
    x = list(range(len(SCALE_ORDER)))
    by_h: Dict[int, List[float]] = {256: [], 275: []}
    for sk in SCALE_ORDER:
        for h in H_LIST:
            found = next(
                (r for r in rows if r["scale"] == sk and r["model"] == f"rnn_h{h}"),
                None,
            )
            v = (
                float(found[value_key])
                if found and found.get(value_key) is not None
                else float("nan")
            )
            by_h[h].append(v)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # Same hues/markers as phase3 rnn / gawf (see utils_viz.plot_generalization.STYLES).
    ax.plot(
        x,
        by_h[275],
        color=_COLOR_RNN275,
        marker=_MARKER_RNN275,
        linewidth=2,
        markersize=7,
        label="RNN h=275",
    )
    ax.plot(
        x,
        by_h[256],
        color=_COLOR_RNN256,
        marker=_MARKER_RNN256,
        linewidth=2,
        markersize=7,
        label="RNN h=256",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_ORDER)
    ax.set_xlabel("Dataset scale (train hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_base}.png (ylim={ylim})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=50, help="Epoch suffix in result dir name")
    ap.add_argument(
        "--out_dir",
        default="",
        help="Figure output (default: category-indexed behaviour figures)",
    )
    ap.add_argument(
        "--write_csv",
        default="",
        help="If set, write combined CSV to this path",
    )
    ap.add_argument(
        "--ref_csv_tag",
        type=str,
        default="_short_ep50",
        help="Same as utils_viz/plot_generalization --csv_tag (e.g. _short_ep50)",
    )
    ap.add_argument(
        "--artifact_dir",
        default="",
        help="Directory with phase3_summary_*.csv (default: experiments/generalization/artifacts)",
    )
    ap.add_argument(
        "--no_ylim",
        action="store_true",
        help="Do not set ylim (matplotlib autoscale only)",
    )
    args = ap.parse_args()
    root = _repo_root()
    os.chdir(root)
    out_dir = args.out_dir or str(
        output_dir("G_behaviour", "verify_temp_rnn_hidden_ablation_plot", "figs")
    )
    out_dir = os.path.abspath(out_dir)
    art = args.artifact_dir or os.path.join(
        root, "experiments", "generalization", "artifacts"
    )
    art = os.path.abspath(art)

    rows = load_rows(root, args.epoch)
    if len(rows) < 8:
        print(
            f"ERROR: expected 8 rows, got {len(rows)}. Train runs missing?",
            file=sys.stderr,
        )
        sys.exit(1)

    ylims: Dict[str, Optional[Tuple[float, float]]] = {
        "overfit_gap": None,
        "train_acc": None,
        "val_acc": None,
    }
    if not args.no_ylim:
        try:
            ref_rows = load_phase3_ref_rows(art, args.ref_csv_tag)
        except FileNotFoundError as e:
            print(f"WARN: {e}; using autoscale.", file=sys.stderr)
        else:
            for k in ylims:
                ylims[k] = ylim_for_metric(ref_rows, rows, k)

    if args.write_csv:
        p = args.write_csv
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["scale", "model", "train_acc", "val_acc", "overfit_gap"]
            )
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {p}")

    ep = args.epoch
    mid = f"_verify_rnn_h256_h275_ep{ep}"
    for key, stem, ttitle in (
        ("overfit_gap", "overfit_gap", "overfit gap"),
        ("train_acc", "train_acc", "train"),
        ("val_acc", "val_acc", "val"),
    ):
        plot_one(
            rows,
            key,
            {
                "overfit_gap": "Overfit gap (train acc − val acc @ best val epoch)",
                "train_acc": "Train char accuracy (%)",
                "val_acc": "Validation char accuracy (%)",
            }[key],
            f"TEMP ablation: RNN h=256 vs h=275 ({ttitle}) ep={ep} [ylim vs Phase3{args.ref_csv_tag}]",
            os.path.join(out_dir, f"{stem}{mid}"),
            ylims[key],
        )


if __name__ == "__main__":
    main()
