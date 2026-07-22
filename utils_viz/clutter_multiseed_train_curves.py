"""Plot completed Clutter multi-seed training curves with mean ± sample SD.

Reads per-seed training ``.pkl`` histories from model/seed directories and writes a two-panel
character/sector accuracy figure, optional matching training/validation-loss figures, and a CSV
containing the final validation accuracy summary.
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
from utils_anal.anal_paths import output_dir


def parse_args() -> argparse.Namespace:
    """Parse input result root and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    figure_dir = output_dir("G_behaviour", "clutter_multiseed_train_curves", "figs")
    data_dir = output_dir("G_behaviour", "clutter_multiseed_train_curves", "data")
    parser.add_argument("--save_png", default=str(figure_dir / "train_curves.png"))
    parser.add_argument(
        "--save_loss_png",
        default=None,
        help="Optional path for validation character/sector loss mean ± SD curves.",
    )
    parser.add_argument(
        "--save_train_loss_png",
        default=None,
        help="Optional path for training character/sector loss mean ± SD curves.",
    )
    parser.add_argument(
        "--seed_filter_csv",
        default=None,
        help="Optional model/seed CSV defining the exact comparison cohort.",
    )
    parser.add_argument("--save_summary_csv", default=str(data_dir / "train_summary.csv"))
    return parser.parse_args()


def load_seed_filter(csv_path: str | None) -> set[tuple[str, int]] | None:
    """Load an optional exact model/seed cohort from a CSV containing model and seed columns."""

    if csv_path is None:
        return None
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "model" not in rows[0] or "seed" not in rows[0]:
        raise ValueError(f"Seed filter CSV must contain model and seed columns: {csv_path}")
    return {(row["model"], int(row["seed"])) for row in rows}


def load_histories(
    root: str,
    allowed_units: set[tuple[str, int]] | None = None,
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Load all available model histories, preserving seed-level records."""

    grouped: dict[str, list[dict[str, np.ndarray]]] = {}
    loaded_units: set[tuple[str, int]] = set()
    for unit_dir in Path(root).glob("*-seed??"):
        model, seed_text = unit_dir.name.rsplit("-seed", 1)
        if model not in MODEL_ORDER:
            continue
        unit = (model, int(seed_text))
        if allowed_units is not None and unit not in allowed_units:
            continue
        pkl_files = sorted(unit_dir.glob("*.pkl"))
        if not pkl_files:
            continue
        with pkl_files[0].open("rb") as handle:
            history = pickle.load(handle)
        grouped.setdefault(model, []).append(history)
        loaded_units.add(unit)
    if not grouped:
        raise RuntimeError(f"No completed pkl histories found under {root}")
    if allowed_units is not None and loaded_units != allowed_units:
        missing = sorted(allowed_units - loaded_units)
        raise RuntimeError(f"Missing filtered model/seed histories under {root}: {missing}")
    return grouped


def stack_histories(histories: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Pad histories with NaN and return shape ``(seed, epoch)``."""

    arrays = [np.asarray(item[key], dtype=np.float64).reshape(-1) for item in histories]
    width = max(array.size for array in arrays)
    stacked = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for index, array in enumerate(arrays):
        stacked[index, : array.size] = array
    return stacked


def plot_mean_std_panels(
    grouped: dict[str, list[dict[str, np.ndarray]]],
    models: list[str],
    metric_specs: tuple[tuple[str, str], tuple[str, str]],
    *,
    ylabel: str,
    suptitle: str,
    output_path: str,
    accuracy_limits: bool = False,
    legend_loc: str = "best",
    xticks: tuple[int, ...] | None = None,
    y_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (12.6, 4.9),
) -> None:
    """Plot two validation metrics using the mean and sample SD across completed seeds."""

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for panel_index, (axis, (key, title)) in enumerate(zip(axes, metric_specs)):
        for model in models:
            values = stack_histories(grouped[model], key)
            x = np.arange(1, values.shape[1] + 1)
            mean = np.nanmean(values, axis=0)
            sd = np.nanstd(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
            color = MODEL_COLORS[model]
            axis.plot(x, mean, color=color, linewidth=2.0, label=MODEL_LABELS[model])
            axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        if accuracy_limits:
            axis.set_ylim(0.0, 105.0)
        elif y_limits is not None:
            axis.set_ylim(*y_limits[panel_index])
        else:
            axis.set_ylim(bottom=0.0)
        if xticks is not None:
            axis.set_xlim(float(xticks[0]), float(xticks[-1]))
            axis.set_xticks(xticks)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, loc=legend_loc, ncol=2)
    fig.suptitle(suptitle)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {output.resolve()}")


def main() -> None:
    """Render validation character/sector curves and final validation summary."""

    args = parse_args()
    allowed_units = load_seed_filter(args.seed_filter_csv)
    grouped = load_histories(args.input_root, allowed_units)
    models = [model for model in MODEL_ORDER if model in grouped]
    plot_mean_std_panels(
        grouped,
        models,
        (
            ("val_acc_char", "Validation character accuracy"),
            ("val_acc_pos", "Validation sector accuracy"),
        ),
        ylabel="Accuracy (%)",
        suptitle="Clutter best-6 multi-seed training curves (mean ± SD)",
        output_path=args.save_png,
        accuracy_limits=True,
        legend_loc="lower right",
    )
    if args.save_loss_png:
        plot_mean_std_panels(
            grouped,
            models,
            (
                ("val_loss_char", "Validation character loss"),
                ("val_loss_pos", "Validation sector loss"),
            ),
            ylabel="Cross-entropy loss",
            suptitle="Clutter best-6 multi-seed validation loss (mean ± SD)",
            output_path=args.save_loss_png,
            xticks=(0, 50, 100, 150),
            y_limits=((0.25, 1.25), (0.1, 0.5)),
            figsize=(12.6, 4.2),
        )
    if args.save_train_loss_png:
        plot_mean_std_panels(
            grouped,
            models,
            (
                ("train_loss_char", "Training character loss"),
                ("train_loss_pos", "Training sector loss"),
            ),
            ylabel="Cross-entropy loss",
            suptitle="Clutter best-6 multi-seed training loss (mean ± SD)",
            output_path=args.save_train_loss_png,
            xticks=(0, 50, 100, 150),
        )

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


if __name__ == "__main__":
    main()
