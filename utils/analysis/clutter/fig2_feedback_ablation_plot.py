"""Standalone shuffle-only feedback-ablation figure (single poster-style panel).

Shows the digit- and sector-readout test accuracy for three conditions -- GaWF baseline,
Shuffle digit, Shuffle sector -- with no shuffle_all. Bars are grouped by readout (the three
condition bars for each readout sit flush together), each bar carrying the across-seed mean, a
mean ± SEM error bar, and one dot per seed. ``Fig*`` output names are normalized to PDF and
``Supple*`` output names to PNG, matching the curated ``results/save/`` convention.

By default the baseline bar heights come from ``best_acc_test_mean_std.csv`` (the canonical
multiseed test accuracy used by the best-model figure), while the two shuffle bars come from the
per-seed ``ablation_metrics.json`` files. ``--baseline_source ablation`` instead takes all three
conditions from those metrics, for protocols whose recurrent rollout differs from the canonical
test evaluation.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.abspath(
    _anal_os.path.join(_anal_os.path.dirname(__file__), "..", "..", "..")
)
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.multiseed_plotting import add_seed_points

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# (data key, legend label, colour) for the two readouts; order here is left-to-right group
# order (sector group on the left, digit group on the right). Colours align with
# core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
READOUTS = (("sector_acc", "Sector", "#264653"), ("char_acc", "Digit", "#E76F51"))
# Displayed conditions in order, with their pretty x labels. baseline is drawn from the
# canonical test CSV; the two shuffle conditions from the ablation metrics.
CONDITIONS = (
    ("baseline", "GaWF\nbaseline"),
    ("shuffle_digit", "Shuffle\ndigit"),
    ("shuffle_sector", "Shuffle\nsector"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default=str(_ANAL_PROJECT_ROOT + "/results/save_data/fig2/gawf_shuffle_ablation"),
        help="Parent dir with per-seed gawf-seed*/ablation_metrics.json.",
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=str(
            _ANAL_PROJECT_ROOT
            + "/results/save_data/fig1/test_accuracy_summary/best_acc_test_mean_std.csv"
        ),
        help="Per-seed canonical test accuracy CSV (source,model,seed,char_acc,sector_acc).",
    )
    parser.add_argument("--model", type=str, default="gawf")
    parser.add_argument(
        "--shuffle_anova_long_csv",
        type=Path,
        default=None,
        help="Use the reset-excluded Figure 4 shuffle long table for all three bars.",
    )
    parser.add_argument(
        "--baseline_source",
        choices=("canonical_csv", "ablation"),
        default="canonical_csv",
        help="Use the canonical CSV baseline or the baseline condition in the ablation metrics.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--out_name", type=str, default="fig_ablation_shuffle_standalone.png")
    parser.add_argument("--ymin", type=float, default=50.0)
    parser.add_argument("--ymax", type=float, default=95.0)
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each 10-seed bar.",
    )
    parser.add_argument(
        "--yticks",
        nargs="+",
        type=float,
        default=(55.0, 65.0, 75.0, 85.0, 95.0),
        help="Optional explicit y-axis tick positions.",
    )
    return parser.parse_args()


def _baseline_from_csv(path: str, model: str) -> Dict[str, np.ndarray]:
    """Return per-seed baseline arrays for the requested model from the canonical test CSV."""

    char: List[float] = []
    sector: List[float] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model:
                char.append(float(row["char_acc"]))
                sector.append(float(row["sector_acc"]))
    if not char:
        raise ValueError(f"No rows for model={model!r} in {path}")
    return {"char_acc": np.asarray(char), "sector_acc": np.asarray(sector)}


def _conditions_from_ablation(
    ablation_dir: str,
    conditions_to_load: tuple[str, ...],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return selected per-seed condition arrays from ablation metrics."""

    files = sorted(glob.glob(os.path.join(ablation_dir, "gawf-seed*", "ablation_metrics.json")))
    if len(files) != 10:
        raise FileNotFoundError(
            f"Expected ten gawf-seed*/ablation_metrics.json under {ablation_dir}, found {len(files)}."
        )
    collected: Dict[str, Dict[str, List[float]]] = {}
    for path in files:
        conditions = json.load(open(path))["conditions"]
        for cond in conditions_to_load:
            collected.setdefault(cond, {"char_acc": [], "sector_acc": []})
            for key in ("char_acc", "sector_acc"):
                collected[cond][key].append(float(conditions[cond][key]))
    return {c: {k: np.asarray(v) for k, v in d.items()} for c, d in collected.items()}


def _conditions_from_shuffle_anova(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read one reset-excluded accuracy per seed and shuffle condition from Figure 4."""

    collected: Dict[str, Dict[str, List[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["object"] != "hidden_activation":
                continue
            condition = row["condition"]
            collected.setdefault(condition, {"char_acc": [], "sector_acc": []})
            collected[condition]["char_acc"].append(float(row["digit_acc"]))
            collected[condition]["sector_acc"].append(float(row["sector_acc"]))
    if set(collected) != {name for name, _ in CONDITIONS} or any(
        len(values["char_acc"]) != 10 for values in collected.values()
    ):
        raise ValueError(f"Expected ten Figure 4 rows per condition in {path}")
    return {
        condition: {key: np.asarray(values) for key, values in metrics.items()}
        for condition, metrics in collected.items()
    }


def _output_path(save_dir: str, out_name: str) -> Path:
    """Normalize curated Figure/Supplementary filenames to their required format."""

    path = Path(save_dir) / out_name
    if path.stem.startswith("Fig"):
        return path.with_suffix(".pdf")
    if path.stem.startswith("Supple"):
        return path.with_suffix(".png")
    return path


def main() -> None:
    args = parse_args()
    if args.save_dir is None:
        args.save_dir = str(output_dir("G_behaviour", "viz_feedback_ablation", "figs"))
    if args.ymin >= args.ymax:
        raise ValueError(f"ymin must be smaller than ymax, got {args.ymin} >= {args.ymax}.")
    if args.yticks is not None and any(
        tick < args.ymin or tick > args.ymax for tick in args.yticks
    ):
        raise ValueError("yticks must lie within the requested y-axis range.")
    condition_data = (
        _conditions_from_shuffle_anova(args.shuffle_anova_long_csv)
        if args.shuffle_anova_long_csv is not None
        else _conditions_from_ablation(
            args.ablation_dir, ("baseline", "shuffle_digit", "shuffle_sector")
        )
    )
    if args.shuffle_anova_long_csv is not None:
        baseline = condition_data["baseline"]
    elif args.baseline_source == "canonical_csv":
        baseline = _baseline_from_csv(args.baseline_csv, args.model)
    else:
        baseline = condition_data["baseline"]
    data = {
        "baseline": baseline,
        "shuffle_digit": condition_data["shuffle_digit"],
        "shuffle_sector": condition_data["shuffle_sector"],
    }
    n_seeds = baseline["char_acc"].size
    if n_seeds != 10 or any(
        values.size != n_seeds for condition in data.values() for values in condition.values()
    ):
        raise ValueError("Every displayed Fig2 condition/readout must contain exactly ten seeds.")

    conds = [c for c, _ in CONDITIONS]
    labels = {c: lbl for c, lbl in CONDITIONS}
    n = len(conds)
    width = 0.88
    step = 1.0
    group_gap = 1.5  # 1.5 bar widths between the two groups, matching core_objects_2x2
    # Left-to-right group order follows READOUTS (sector group first, then digit).
    group_base = {key: i * (n * step + group_gap) for i, (key, _lbl, _c) in enumerate(READOUTS)}
    rng = np.random.default_rng(0)

    with plt.rc_context(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
        }
    ):
        # Match one row of core_objects_aggregate_2x2 (5.05in wide x 8.2in / 2 rows tall) so the
        # panel height aligns with that figure's first row and the bars share its tall-narrow
        # width-to-height ratio, while staying a single axis.
        fig, axis = plt.subplots(figsize=(5.5, 2.4))
        xticks: List[float] = []
        xticklabels: List[str] = []
        for key, label, color in READOUTS:
            base = group_base[key]
            for j, cond in enumerate(conds):
                xpos = base + j * step
                values = data[cond][key]
                sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                axis.bar(
                    xpos,
                    float(values.mean()),
                    width=width,
                    color=color,
                    edgecolor="none",
                    label=label if j == 0 else None,
                    yerr=sem,
                    capsize=3,
                    error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
                )
                add_seed_points(
                    axis,
                    np.asarray([xpos]),
                    values[:, None],
                    bar_width=width,
                    show=args.show_seed_points,
                    rng=rng,
                )
                xticks.append(xpos)
                xticklabels.append(labels[cond])

        axis.set_ylabel("Test accuracy (%)")
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels, fontsize=9)
        axis.set_xlim(-0.7, max(group_base.values()) + (n - 1) * step + 0.7)
        axis.set_ylim(args.ymin, args.ymax)
        y_ticks = args.yticks
        if y_ticks is None:
            y_ticks = np.arange(args.ymin, args.ymax + 0.1, 5.0)
        axis.set_yticks(y_ticks)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        handles, legend_labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        out_path = _output_path(args.save_dir, args.out_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    print(f"Saved figure: {out_path}  (baseline n={n_seeds} seeds)")


if __name__ == "__main__":
    main()
