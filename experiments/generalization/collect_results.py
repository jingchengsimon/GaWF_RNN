#!/usr/bin/env python3
"""
Aggregate metrics from generalization experiment runs (Phase 1–3).

Reads *_metrics.json written by train_model.py (via summarize_experiment_metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob
from typing import Any, Dict, List


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_metrics_files(metrics_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob(os.path.join(metrics_dir, "*_metrics.json")))
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["_path"] = p
        out.append(row)
    return out


def score_row(m: Dict[str, Any]) -> float:
    if "val_acc_at_best" in m and m["val_acc_at_best"] is not None:
        return float(m["val_acc_at_best"])
    return float(m.get("best_val_acc_char") or 0.0)


def phase1_best(metrics_dir: str, scale_key: str) -> Dict[str, Any]:
    rows = load_metrics_files(metrics_dir)
    if not rows:
        raise RuntimeError(f"No *_metrics.json under {metrics_dir}")
    best = max(rows, key=score_row)
    return {
        "scale": scale_key,
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "val_acc_at_best": score_row(best),
        "metrics_path": best["_path"],
    }


def cmd_phase1_aggregate(args: argparse.Namespace) -> None:
    """Write phase1 JSON from four Phase-1 result directories (4h,10h,20h,40h order)."""
    os.chdir(_repo_root())
    scales = ["4h", "10h", "20h", "40h"]
    if len(args.metrics_dirs) != len(scales):
        raise SystemExit(
            f"Expected {len(scales)} --metrics_dirs (one per scale), got {len(args.metrics_dirs)}"
        )
    out: Dict[str, Any] = {}
    for scale, mdir in zip(scales, args.metrics_dirs):
        out[scale] = phase1_best(mdir, scale)
    dst = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {dst}")


def cmd_phase2_summary(args: argparse.Namespace) -> None:
    """
    Summarize Phase2 LR sanity check per scale.
    Expects metrics under results/train_data/gen_phase2_{scale}/ from three model runs.
    """
    os.chdir(_repo_root())
    art = os.path.join(os.path.dirname(__file__), "artifacts")
    with open(os.path.join(art, "phase1_best.json"), "r", encoding="utf-8") as f:
        p1 = json.load(f)
    scale = args.scale
    p1_row = p1[scale]
    gawf_lr = float(p1_row["lr"])
    metrics_dir = args.metrics_dir
    rows = load_metrics_files(metrics_dir)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for m in rows:
        mt = m.get("model_type")
        if mt not in ("rnn", "lstm", "gru"):
            continue
        by_model.setdefault(mt, []).append(m)

    summary_rows: List[Dict[str, Any]] = []
    final_hparams: Dict[str, Any] = {"scale": scale, "gawf": {}, "rnn": {}, "lstm": {}, "gru": {}}

    final_hparams["gawf"] = {
        "lr": p1_row["lr"],
        "weight_decay": p1_row["weight_decay"],
        "hidden_size": 256,
    }

    for mt in ("rnn", "lstm", "gru"):
        ms = by_model.get(mt, [])
        if not ms:
            raise RuntimeError(f"No metrics for model {mt} under {metrics_dir}")
        best_m = max(ms, key=score_row)
        best_lr = float(best_m["lr"])
        best_val = score_row(best_m)
        gawf_lr_rows = [x for x in ms if abs(float(x["lr"]) - gawf_lr) < 1e-12]
        if not gawf_lr_rows:
            raise RuntimeError(f"No run for {mt} with lr={gawf_lr} in {metrics_dir}")
        gawf_lr_m = max(gawf_lr_rows, key=score_row)
        gawf_lr_val = score_row(gawf_lr_m)
        lr_match = abs(best_lr - gawf_lr) < 1e-12
        use_own = (best_val - gawf_lr_val) > 0.02
        chosen_lr = best_lr if use_own else gawf_lr
        hid = {"rnn": 275, "lstm": 80, "gru": 105}[mt]
        final_hparams[mt] = {
            "lr": chosen_lr,
            "weight_decay": float(p1_row["weight_decay"]),
            "hidden_size": hid,
        }
        summary_rows.append(
            {
                "scale": scale,
                "model": mt,
                "best_lr": best_lr,
                "best_val_acc": best_val,
                "gawf_lr_val_acc": gawf_lr_val,
                "lr_match": lr_match,
                "use_own_lr_over_2pct": use_own,
                "final_lr": chosen_lr,
            }
        )

    os.makedirs(art, exist_ok=True)
    csv_path = os.path.join(art, f"phase2_summary_{scale}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {csv_path}")

    # Merge into combined final hparams file
    combined_path = os.path.join(art, "phase2_final_hparams.json")
    combined: Dict[str, Any] = {}
    if os.path.isfile(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            combined = json.load(f)
    combined[scale] = final_hparams
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Updated {combined_path}")


def _metrics_row_to_phase3_csv_row(scale: str, m: Dict[str, Any]) -> Dict[str, Any]:
    mt = m.get("model_type")
    train_acc_char = m.get("train_acc_at_best_val")
    if train_acc_char is None:
        train_acc_char = m.get("best_train_acc_char")
    val_acc_char = m.get("val_acc_at_best")
    if val_acc_char is None:
        val_acc_char = m.get("best_val_acc_char")
    og_char = m.get("overfit_gap")
    if og_char is None and train_acc_char is not None and val_acc_char is not None:
        og_char = float(train_acc_char) - float(val_acc_char)

    train_acc_sector = m.get("train_acc_sector_at_best_val_sector")
    if train_acc_sector is None:
        train_acc_sector = m.get("best_train_acc_pos")
    val_acc_sector = m.get("val_acc_sector_at_best")
    if val_acc_sector is None:
        val_acc_sector = m.get("best_val_acc_pos")
    og_sector = m.get("overfit_gap_sector")
    if og_sector is None and train_acc_sector is not None and val_acc_sector is not None:
        og_sector = float(train_acc_sector) - float(val_acc_sector)
    return {
        "scale": scale,
        "model": mt,
        "lr": m.get("lr"),
        "wd": m.get("weight_decay"),
        "early_stop_epoch": m.get("early_stop_epoch_1based", m.get("actual_epochs")),
        # Backward-compatible aliases for the original char plots.
        "train_acc": train_acc_char,
        "val_acc": val_acc_char,
        "overfit_gap": og_char,
        "train_acc_char": train_acc_char,
        "val_acc_char": val_acc_char,
        "overfit_gap_char": og_char,
        "train_acc_sector": train_acc_sector,
        "val_acc_sector": val_acc_sector,
        "overfit_gap_sector": og_sector,
        "best_epoch_val_acc_sector_1based": m.get("best_epoch_val_acc_sector_1based"),
        "stopped_by_patience": m.get("stopped_by_patience"),
    }


def cmd_phase3_table(args: argparse.Namespace) -> None:
    os.chdir(_repo_root())
    rows: List[Dict[str, Any]] = []
    for mdir in args.metrics_dirs:
        for m in load_metrics_files(mdir):
            mt = m.get("model_type")
            if mt not in ("rnn", "lstm", "gru", "gawf"):
                continue
            rows.append(_metrics_row_to_phase3_csv_row(args.scale, m))
    art = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(art, exist_ok=True)
    tag = getattr(args, "out_tag", "") or ""
    csv_path = os.path.join(art, f"phase3_summary_{args.scale}{tag}.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")


def phase1_best_gawf_only(metrics_dir: str, scale_key: str) -> Dict[str, Any]:
    rows = [m for m in load_metrics_files(metrics_dir) if m.get("model_type") == "gawf"]
    if not rows:
        rows = load_metrics_files(metrics_dir)
    if not rows:
        raise RuntimeError(f"No *_metrics.json under {metrics_dir}")
    best = max(rows, key=score_row)
    return {
        "scale": scale_key,
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "val_acc_at_best": score_row(best),
        "metrics_path": best["_path"],
    }


def cmd_phase1_short(args: argparse.Namespace) -> None:
    """Phase 1 short: three searched scales + 40h preset dir (GAWF lr/wd from existing run)."""
    os.chdir(_repo_root())
    scales_three = ["4h", "10h", "20h"]
    if len(args.metrics_dirs) != 3:
        raise SystemExit(f"Expected 3 metrics_dirs, got {len(args.metrics_dirs)}")
    out: Dict[str, Any] = {}
    for sc, mdir in zip(scales_three, args.metrics_dirs):
        out[sc] = phase1_best(mdir, sc)
    preset = os.path.abspath(args.preset_40h_dir)
    out["40h"] = phase1_best_gawf_only(preset, "40h")
    art = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(art, exist_ok=True)
    dst = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {dst}")


def cmd_emit_hparams_shared(args: argparse.Namespace) -> None:
    """All four models share Phase-1 (per-scale) lr/wd; hidden sizes fixed by architecture."""
    os.chdir(_repo_root())
    with open(os.path.abspath(args.phase1_best), "r", encoding="utf-8") as f:
        p1: Dict[str, Any] = json.load(f)
    hidden = {"gawf": 256, "rnn": 275, "lstm": 80, "gru": 105}
    combined: Dict[str, Any] = {}
    for scale in ("4h", "10h", "20h", "40h"):
        if scale not in p1:
            raise KeyError(f"Missing scale {scale} in {args.phase1_best}")
        row = p1[scale]
        lr = float(row["lr"])
        wd = float(row["weight_decay"])
        combined[scale] = {
            "scale": scale,
            "gawf": {"lr": lr, "weight_decay": wd, "hidden_size": hidden["gawf"]},
            "rnn": {"lr": lr, "weight_decay": wd, "hidden_size": hidden["rnn"]},
            "lstm": {"lr": lr, "weight_decay": wd, "hidden_size": hidden["lstm"]},
            "gru": {"lr": lr, "weight_decay": wd, "hidden_size": hidden["gru"]},
        }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {out_path}")


def cmd_phase3_import_dir(args: argparse.Namespace) -> None:
    """Build phase3_summary CSV from a single folder of *_metrics.json (e.g. sector_40h_adamw)."""
    os.chdir(_repo_root())
    mdir = os.path.abspath(args.metrics_dir)
    rows: List[Dict[str, Any]] = []
    for m in load_metrics_files(mdir):
        mt = m.get("model_type")
        if mt not in ("rnn", "lstm", "gru", "gawf"):
            continue
        rows.append(_metrics_row_to_phase3_csv_row(args.scale, m))
    art = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(art, exist_ok=True)
    tag = getattr(args, "out_tag", "") or ""
    csv_path = os.path.join(art, f"phase3_summary_{args.scale}{tag}.csv")
    if not rows:
        raise RuntimeError(f"No model metrics rows under {mdir}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="Collect generalization experiment metrics")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1", help="Aggregate Phase 1 GAWF grid (four dirs -> phase1 JSON)")
    p1.add_argument(
        "metrics_dirs",
        nargs=4,
        help="Dirs: 4h, 10h, 20h, 40h Phase-1 result folders (e.g. gen_phase1_gawf_* or gen_phase1_short_gawf_*) in that order",
    )
    p1.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: experiments/generalization/artifacts/phase1_best.json)",
    )
    p1.set_defaults(func=cmd_phase1_aggregate)

    p2 = sub.add_parser("phase2", help="Phase 2 LR sanity summary for one scale")
    p2.add_argument("--scale", required=True, choices=["4h", "10h", "20h", "40h"])
    p2.add_argument(
        "--metrics_dir",
        required=True,
        help="e.g. results/train_data/gen_phase2_4h",
    )
    p2.set_defaults(func=cmd_phase2_summary)

    p3 = sub.add_parser("phase3", help="Merge Phase 3 per-model dirs into one CSV")
    p3.add_argument("--scale", required=True, choices=["4h", "10h", "20h", "40h"])
    p3.add_argument(
        "--out_tag",
        type=str,
        default="",
        help="Suffix before .csv (e.g. _short_ep100 -> phase3_summary_4h_short_ep100.csv)",
    )
    p3.add_argument(
        "metrics_dirs",
        nargs="+",
        help="Result dir(s) for that scale: typically one shared dir (gen_phase3_*_epN with all models' *_metrics.json) or legacy four per-model dirs",
    )
    p3.set_defaults(func=cmd_phase3_table)

    p1s = sub.add_parser(
        "phase1_short",
        help="Legacy: 3 Phase-1 dirs + 40h preset dir -> phase1_best_short.json (prefer `phase1` with 4 dirs + --out for new runs)",
    )
    p1s.add_argument(
        "metrics_dirs",
        nargs=3,
        help="gen_phase1_short_gawf_4h, _10h, _20h in order",
    )
    p1s.add_argument(
        "--preset_40h_dir",
        type=str,
        default="results/train_data/sector_40h_adamw",
        help="Folder with prior40h runs (*_metrics.json); lr/wd taken from gawf file",
    )
    p1s.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: experiments/generalization/artifacts/phase1_best_short.json)",
    )
    p1s.set_defaults(func=cmd_phase1_short)

    pe = sub.add_parser(
        "emit_hparams_shared",
        help="phase1_best*.json -> phase2_final_hparams_short.json (same lr/wd for all models per scale)",
    )
    pe.add_argument(
        "--phase1_best",
        type=str,
        default="experiments/generalization/artifacts/phase1_best_short.json",
    )
    pe.add_argument(
        "--out",
        type=str,
        default="experiments/generalization/artifacts/phase2_final_hparams_short.json",
    )
    pe.set_defaults(func=cmd_emit_hparams_shared)

    pi = sub.add_parser(
        "phase3_import",
        help="Single metrics_dir (four models) -> phase3_summary_<scale><tag>.csv",
    )
    pi.add_argument("--scale", required=True, choices=["4h", "10h", "20h", "40h"])
    pi.add_argument("--metrics_dir", required=True)
    pi.add_argument(
        "--out_tag",
        type=str,
        default="_short",
        help="CSV suffix (e.g. _short_ep100); must match phase3_train --out_tag / plot --csv_tag",
    )
    pi.set_defaults(func=cmd_phase3_import_dir)

    art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    args = p.parse_args()
    if args.cmd == "phase1" and not getattr(args, "out", None):
        args.out = os.path.join(art_dir, "phase1_best.json")
    if args.cmd == "phase1_short" and not args.out:
        args.out = os.path.join(art_dir, "phase1_best_short.json")
    args.func(args)


if __name__ == "__main__":
    main()
