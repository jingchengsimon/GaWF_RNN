"""Plot completed Clutter multi-seed training curves with mean ± sample SD.

Reads per-seed training ``.pkl`` histories from model/seed directories and writes a two-panel
character/sector accuracy figure plus a CSV containing the final validation summary.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils_viz.fg_switch_offset_acc import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER


def parse_args() -> argparse.Namespace:
    """Parse input result root and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--save_png", required=True)
    parser.add_argument("--save_summary_csv", required=True)
    return parser.parse_args()


def load_histories(root: str) -> dict[str, list[dict[str, np.ndarray]]]:
    """Load all available model histories, preserving seed-level records."""

    grouped: dict[str, list[dict[str, np.ndarray]]] = {}
    for unit_dir in Path(root).glob("*-seed??"):
        model = unit_dir.name.rsplit("-seed", 1)[0]
        if model not in MODEL_ORDER:
            continue
        pkl_files = sorted(unit_dir.glob("*.pkl"))
        if not pkl_files:
            continue
        with pkl_files[0].open("rb") as handle:
            history = pickle.load(handle)
        grouped.setdefault(model, []).append(history)
    if not grouped:
        raise RuntimeError(f"No completed pkl histories found under {root}")
    return grouped


def stack_histories(histories: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Pad histories with NaN and return shape ``(seed, epoch)``."""

    arrays = [np.asarray(item[key], dtype=np.float64).reshape(-1) for item in histories]
    width = max(array.size for array in arrays)
    stacked = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for index, array in enumerate(arrays):
        stacked[index, : array.size] = array
    return stacked


def main() -> None:
    """Render validation character/sector curves and final validation summary."""

    args = parse_args()
    grouped = load_histories(args.input_root)
    models = [model for model in MODEL_ORDER if model in grouped]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharex=True)
    for axis, key, title in (
        (axes[0], "val_acc_char", "Validation character accuracy"),
        (axes[1], "val_acc_pos", "Validation sector accuracy"),
    ):
        for model in models:
            values = stack_histories(grouped[model], key)
            x = np.arange(1, values.shape[1] + 1)
            mean = np.nanmean(values, axis=0)
            sd = (
                np.nanstd(values, axis=0, ddof=1)
                if values.shape[0] > 1
                else np.zeros_like(mean)
            )
            color = MODEL_COLORS[model]
            axis.plot(x, mean, color=color, linewidth=2.0, label=MODEL_LABELS[model])
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0.0, 105.0)
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right", ncol=2)
    fig.suptitle("Clutter best-6 multi-seed training curves (mean ± SD)")
    fig.tight_layout()
    output = Path(args.save_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    summary = Path(args.save_summary_csv)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "n_seeds",
                "final_val_char_mean",
                "final_val_char_sd",
                "final_val_sector_mean",
                "final_val_sector_sd",
            ]
        )
        for model in models:
            chars = stack_histories(grouped[model], "val_acc_char")
            sectors = stack_histories(grouped[model], "val_acc_pos")
            char_final = chars[:, -1]
            sector_final = sectors[:, -1]
            writer.writerow(
                [
                    model,
                    len(grouped[model]),
                    float(np.nanmean(char_final)),
                    float(np.nanstd(char_final, ddof=1)) if len(grouped[model]) > 1 else 0.0,
                    float(np.nanmean(sector_final)),
                    float(np.nanstd(sector_final, ddof=1)) if len(grouped[model]) > 1 else 0.0,
                ]
            )
    print(f"Saved figure: {output.resolve()}")


if __name__ == "__main__":
    main()
