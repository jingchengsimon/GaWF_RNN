"""Visualize per-offset switch-window accuracies as bar charts per model.

Reads exported npz from ``utils_anal/export_fg_switch_offset_acc.py``:
- ``fg_switch_offset_acc_*.npz`` / ``bg_switch_offset_acc_*.npz``: 1x2 panels
  (foreground character + sector); bg uses **bg_switch** windows only in the export script.
  Legacy bg npz files without ``sector_acc`` fall back to a single character panel.

Outputs (in ``--save_dir``):
- ``fg_<ckpt_tag>_switch_offset_acc.png`` or ``bg_<ckpt_tag>_switch_offset_acc.png``
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-model fg/bg-switch offset accuracy bar charts."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/fg_switch_offset_acc",
        help="Directory containing fg_switch_offset_acc_* / bg_switch_offset_acc_*.npz files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/fg_switch_offset_acc",
        help="Output directory for figures.",
    )
    return parser.parse_args()


def _collect_npz(data_dir: str) -> List[str]:
    names = sorted(
        n
        for n in os.listdir(data_dir)
        if n.endswith(".npz")
        and (
            n.startswith("fg_switch_offset_acc_") or n.startswith("bg_switch_offset_acc_")
        )
    )
    return [os.path.join(data_dir, n) for n in names]


def _npz_kind_and_tag(npz_path: str) -> Tuple[str, str]:
    base = os.path.basename(npz_path)
    if base.startswith("fg_switch_offset_acc_"):
        return "fg", base[len("fg_switch_offset_acc_") : -len(".npz")]
    if base.startswith("bg_switch_offset_acc_"):
        return "bg", base[len("bg_switch_offset_acc_") : -len(".npz")]
    raise ValueError(base)


def _style_axes_spines(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_one(npz_path: str, save_dir: str) -> None:
    data = np.load(npz_path)
    labels = data["offset_labels"].tolist()
    char_acc = data["char_acc"].astype(np.float32)
    counts = data["frame_counts"].astype(np.int64)

    x = np.arange(len(labels), dtype=np.int64)
    kind, ckpt_tag = _npz_kind_and_tag(npz_path)
    model_name = ckpt_tag.split("_", 1)[0].upper()
    has_sector = "sector_acc" in data.files

    if has_sector:
        sector_acc = data["sector_acc"].astype(np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), sharex=True)

        ax0 = axes[0]
        ax0.bar(x, char_acc, color="#4472C4", alpha=0.90, width=0.75)
        ax0.set_title("Character accuracy (fg digit)", fontsize=12)
        ax0.set_ylabel("Accuracy (%)")
        ax0.set_ylim(0.0, 100.0)
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=30, ha="right")
        ax0.grid(axis="y", alpha=0.3)
        _style_axes_spines(ax0)

        ax1 = axes[1]
        ax1.bar(x, sector_acc, color="#ED7D31", alpha=0.90, width=0.75)
        ax1.set_title("Sector accuracy", fontsize=12)
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0.0, 100.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.grid(axis="y", alpha=0.3)
        _style_axes_spines(ax1)

        sw = "fg" if kind == "fg" else "bg"
        fig.suptitle(
            f"{model_name} test {sw}-switch windows ({ckpt_tag})",
            fontsize=12,
        )
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7.5, 4.2))
        ax0.bar(x, char_acc, color="#4472C4", alpha=0.90, width=0.75)
        ax0.set_title("Character accuracy (legacy bg export)", fontsize=12)
        ax0.set_ylabel("Accuracy (%)")
        ax0.set_ylim(0.0, 100.0)
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=30, ha="right")
        ax0.grid(axis="y", alpha=0.3)
        _style_axes_spines(ax0)
        fig.suptitle(
            f"{model_name} test bg-switch offset (legacy) ({ckpt_tag})",
            fontsize=12,
        )

    count_text = "frames: " + " | ".join(f"{lbl}={int(n)}" for lbl, n in zip(labels, counts))
    fig.text(0.5, 0.01, count_text, ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{kind}_{ckpt_tag}_switch_offset_acc.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    npz_paths = _collect_npz(data_dir)
    if not npz_paths:
        raise RuntimeError(
            f"No fg_switch_offset_acc_* or bg_switch_offset_acc_*.npz files found in {data_dir}"
        )

    for npz_path in npz_paths:
        _plot_one(npz_path, save_dir)


if __name__ == "__main__":
    main()
