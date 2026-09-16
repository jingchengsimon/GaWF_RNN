"""Visualize GaWF feedback-component ablation metrics.

Reads exactly ten ``ablation_metrics.json`` files and produces:
- a grouped bar chart of char/sector accuracy per ablation condition
- switch recovery curves with seed mean ± SEM for char and sector readouts

Outputs (in --save_dir):
- fig_ablation_2x2.png
- fig_ablation_switch_recovery.png
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
import glob
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GaWF feedback ablation summary figures."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing ablation_metrics.json.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional ordered subset of saved ablation conditions to plot.",
    )
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on 10-seed bar charts.",
    )
    return parser.parse_args()


def _load_metrics(data_dir: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*-seed*", "ablation_metrics.json")))
    if not paths:
        path = os.path.join(data_dir, "ablation_metrics.json")
        paths = [path] if os.path.isfile(path) else []
    if len(paths) != 10:
        raise RuntimeError(f"Expected exactly ten ablation metrics files, found {len(paths)}.")
    metrics = [json.load(open(path)) for path in paths]
    invalid = [
        path
        for path, record in zip(paths, metrics)
        if record.get("exclude_window_initial_frame") is not True
    ]
    if invalid:
        raise RuntimeError(
            "Ablation recovery input includes or does not document rollout t=0: "
            + ", ".join(invalid)
        )
    return metrics


def _conditions(metrics: List[Dict[str, Any]], selected: List[str] | None) -> List[str]:
    available = list(
        metrics[0].get("conditions_order", metrics[0]["conditions"].keys())
    )
    conditions = available if selected is None else selected
    for condition in conditions:
        if any(condition not in record["conditions"] for record in metrics):
            raise RuntimeError(f"Condition {condition!r} is not present in every seed.")
    return conditions


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and cross-seed SEM over exactly ten training seeds."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != 10:
        raise ValueError(f"Expected exactly ten seeds, got {values.shape[0]}.")
    mean = values.mean(axis=0)
    return mean, values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _pretty_condition(name: str) -> str:
    return name.replace("_", "\n")


def _offset_label(offset: int) -> str:
    if offset == 1:
        return "switch"
    if offset < 0:
        return f"pre{abs(offset)}"
    return f"post{offset}"


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _save_figure(fig: plt.Figure, out_path: str) -> None:
    """Save matching PNG and PDF outputs for the ablation figure."""

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)


def _plot_bar(
    metrics: List[Dict[str, Any]],
    conds: List[str],
    out_path: str,
    *,
    show_seed_points: bool,
) -> None:
    char = np.asarray(
        [[record["conditions"][c]["char_acc"] for c in conds] for record in metrics],
        dtype=np.float64,
    )
    sector = np.asarray(
        [[record["conditions"][c]["sector_acc"] for c in conds] for record in metrics],
        dtype=np.float64,
    )
    char_mean, char_sem = _mean_sem(char)
    sector_mean, sector_sem = _mean_sem(sector)
    x = np.arange(len(conds), dtype=np.float32)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    # Digit/sector colours aligned with core_objects_aggregate_2x2 (digit #E76F51, sector #264653).
    bars0 = ax.bar(
        x - width / 2, char_mean, width=width, yerr=char_sem, color="#E76F51", label="char"
    )
    bars1 = ax.bar(
        x + width / 2,
        sector_mean,
        width=width,
        yerr=sector_sem,
        color="#264653",
        label="sector",
    )
    rng = np.random.default_rng(0)
    add_seed_points(
        ax, x - width / 2, char, bar_width=width, show=show_seed_points, rng=rng
    )
    add_seed_points(
        ax, x + width / 2, sector, bar_width=width, show=show_seed_points, rng=rng
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_condition(c) for c in conds])
    ax.set_ylim(0.0, 100.0)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Feedback-component ablation")
    _style_axis(ax)

    for bars in (bars0, bars1):
        for bar in bars:
            h = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(99.0, h + 1.5),
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _plot_recovery(metrics: List[Dict[str, Any]], conds: List[str], out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)
    colors = {
        "baseline": "#4C78A8",
        "clear_digit": "#54A24B",
        "clear_sector": "#F58518",
        "clear_all": "#E45756",
        "shuffle_digit": "#72B7B2",
        "shuffle_sector": "#B279A2",
        "shuffle_all": "#E45756",
    }
    first = metrics[0]["conditions"][conds[0]]
    first_offsets = np.asarray(
        first.get("switch_offsets", first["switch_post_offsets"]),
        dtype=np.int64,
    )
    wanted = {-10, -5, 1, 5, 10}
    selected_indices = np.asarray(
        [index for index, offset in enumerate(first_offsets) if int(offset) in wanted],
        dtype=np.int64,
    )
    selected_labels = [_offset_label(int(first_offsets[index])) for index in selected_indices]
    if selected_indices.size == 0:
        raise RuntimeError("No key pre/switch/post offsets found in ablation metrics")
    x = np.arange(first_offsets.size, dtype=np.int64)

    for ax, key, title, chance_level, chance_label in [
        (axes[0], "char", "Character readout", 10.0, "chance = 10%"),
        (axes[1], "sector", "Sector readout", 100.0 / 9.0, "chance = 11.1%"),
    ]:
        for cond in conds:
            rows = [record["conditions"][cond] for record in metrics]
            if "switch_offsets" in rows[0]:
                offset_key = "switch_offsets"
                offsets = np.asarray(rows[0]["switch_offsets"], dtype=np.int64)
                value_key = f"switch_{key}_acc"
            else:
                offset_key = "switch_post_offsets"
                offsets = np.asarray(rows[0]["switch_post_offsets"], dtype=np.int64)
                value_key = f"switch_post_{key}_acc"
            if not np.array_equal(offsets, first_offsets):
                raise RuntimeError(f"Switch offsets differ for ablation condition {cond!r}")
            values = np.asarray([row[value_key] for row in rows], dtype=np.float64)
            if any(
                not np.array_equal(np.asarray(row[offset_key]), first_offsets)
                for row in rows
            ):
                raise RuntimeError(f"Seed offsets differ for ablation condition {cond!r}")
            mean, sem = _mean_sem(values)
            ax.plot(
                x,
                mean,
                marker="o",
                markevery=selected_indices.tolist(),
                linewidth=1.8,
                markersize=4.0,
                label=cond,
                color=colors.get(cond),
            )
            if np.any(sem):
                ax.fill_between(
                    x,
                    mean - sem,
                    mean + sem,
                    color=colors.get(cond),
                    alpha=0.55,
                    linewidth=0,
                )
        ax.axhline(
            chance_level,
            color="0.3",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
        ax.text(
            0.99,
            chance_level + 1.2,
            chance_label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="0.3",
            fontsize=8,
        )
        ax.set_title(title)
        ax.set_xlabel("Frame relative to switch")
        ax.set_ylim(0.0, 100.0)
        ax.set_xticks(selected_indices, selected_labels)
        if "switch" in selected_labels:
            switch_index = selected_indices[selected_labels.index("switch")]
            ax.axvline(switch_index, color="0.35", linewidth=1.0, linestyle="--")
        _style_axis(ax)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
        fontsize=8,
        ncol=len(conds),
    )
    fig.suptitle(
        "Switch-window recovery under feedback ablation (mean ± SEM)",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])
    _save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()
    if args.data_dir is None:
        args.data_dir = str(output_dir("G_behaviour", "feedback_ablation", "data"))
    if args.save_dir is None:
        args.save_dir = str(output_dir("G_behaviour", "viz_feedback_ablation", "figs"))
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    metrics = _load_metrics(data_dir)
    conditions = _conditions(metrics, args.conditions)

    _plot_bar(
        metrics,
        conditions,
        os.path.join(save_dir, "fig_ablation_2x2.png"),
        show_seed_points=args.show_seed_points,
    )
    _plot_recovery(
        metrics,
        conditions,
        os.path.join(save_dir, "fig_ablation_switch_recovery.png"),
    )


if __name__ == "__main__":
    main()
