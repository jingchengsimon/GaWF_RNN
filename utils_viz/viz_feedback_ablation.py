"""Visualize GaWF feedback-component ablation metrics.

Reads ``ablation_metrics.json`` from ``utils_anal/feedback_ablation.py`` and produces:
- a grouped bar chart of char/sector accuracy per ablation condition
- post-fg-switch recovery curves for char and sector readouts

Outputs (in --save_dir):
- fig_ablation_2x2.png
- fig_ablation_switch_recovery.png
"""
from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
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
        default=str(output_dir("G_behaviour", "feedback_ablation", "data")),
        help="Directory containing ablation_metrics.json.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("G_behaviour", "viz_feedback_ablation", "figs")),
        help="Directory for PNG outputs.",
    )
    return parser.parse_args()


def _load_metrics(data_dir: str) -> Dict[str, Any]:
    path = os.path.join(data_dir, "ablation_metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def _conditions(metrics: Dict[str, Any]) -> List[str]:
    return list(metrics.get("conditions_order", metrics["conditions"].keys()))


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


def _plot_bar(metrics: Dict[str, Any], out_path: str) -> None:
    conds = _conditions(metrics)
    char = np.asarray(
        [metrics["conditions"][c]["char_acc"] for c in conds],
        dtype=np.float32,
    )
    sector = np.asarray(
        [metrics["conditions"][c]["sector_acc"] for c in conds],
        dtype=np.float32,
    )
    x = np.arange(len(conds), dtype=np.float32)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bars0 = ax.bar(x - width / 2, char, width=width, color="#4C78A8", label="char")
    bars1 = ax.bar(x + width / 2, sector, width=width, color="#F58518", label="sector")
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


def _plot_recovery(metrics: Dict[str, Any], out_path: str) -> None:
    conds = _conditions(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)
    colors = {
        "baseline": "#4C78A8",
        "clear_digit": "#54A24B",
        "clear_sector": "#F58518",
        "clear_all": "#E45756",
        "shuffle_digit": "#72B7B2",
        "shuffle_sector": "#B279A2",
    }
    first = metrics["conditions"][conds[0]]
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
            row = metrics["conditions"][cond]
            if "switch_offsets" in row:
                offsets = np.asarray(row["switch_offsets"], dtype=np.int64)
                value_key = f"switch_{key}_acc"
            else:
                offsets = np.asarray(row["switch_post_offsets"], dtype=np.int64)
                value_key = f"switch_post_{key}_acc"
            values = np.asarray(row[value_key], dtype=np.float32)
            if not np.array_equal(offsets, first_offsets):
                raise RuntimeError(f"Switch offsets differ for ablation condition {cond!r}")
            ax.plot(
                x,
                values,
                marker="o",
                markevery=selected_indices.tolist(),
                linewidth=1.8,
                markersize=4.0,
                label=cond,
                color=colors.get(cond),
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
    fig.suptitle("Switch-window recovery under feedback ablation", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])
    _save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    metrics = _load_metrics(data_dir)

    _plot_bar(metrics, os.path.join(save_dir, "fig_ablation_2x2.png"))
    _plot_recovery(
        metrics,
        os.path.join(save_dir, "fig_ablation_switch_recovery.png"),
    )


if __name__ == "__main__":
    main()
