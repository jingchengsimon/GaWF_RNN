"""Summarize and plot the formal Clutter 4h/10h/20h/40h comparison.

Input is the fixed-best6, ten-seed result tree. The script validates all 240 metrics files,
writes seed-level and mean-with-SEM CSV plus a JSON summary below ``results/data/analysis``,
and renders one curated 2-by-3 PDF below ``results/save``. Historical category figures are not
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fg_switch_offset_acc import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
)


CATEGORY = "G_behaviour"
SCRIPT_NAME = "data_scale_comparison"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PDF = PROJECT_ROOT / "results/save/data_scale_performance_2x3_10seed.pdf"
SCALES = ("4h", "10h", "20h", "40h")
MODELS = ("rnn", "lstm", "gru", "gawf", "mamba", "s5")
SEEDS = tuple(range(1, 11))
EVAL_DATASET_SUFFIX = "40h-uint8"
ACTUAL_EPOCHS = 150
FORMAL_HPARAMS = {
    "rnn": (275, 0.001, 0.00001),
    "lstm": (80, 0.001, 0.001),
    "gru": (105, 0.005, 0.001),
    "gawf": (256, 0.005, 0.001),
    "mamba": (170, 0.001, 0.001),
    "s5": (256, 0.001, 0.0),
}
METRIC_FIELDS = (
    "train_acc_char",
    "val_acc_char",
    "overfit_gap_char",
    "train_acc_sector",
    "val_acc_sector",
    "overfit_gap_sector",
    "best_epoch_char",
    "best_epoch_sector",
)
CSV_FIELDS = (
    "seed",
    "scale",
    "model",
    "hidden_size",
    "lr",
    "weight_decay",
    "actual_epochs",
    *METRIC_FIELDS,
    "source_metrics",
)
SUMMARY_FIELDS = (
    "scale",
    "model",
    "n_seeds",
    *(f"{stat}_{field}" for field in METRIC_FIELDS for stat in ("mean", "sem")),
)
Row = dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    """Parse the formal result root and optional canonical output overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Formal clutter_formal_4scale_ep150 directory containing scale/model-seed leaves.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Original result root recorded as provenance when input-root is a local staging copy.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    return parser.parse_args()


def _number(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"Metrics are missing all required fields: {keys}")


def _validate_protocol(
    payload: dict[str, Any], path: Path, scale: str, model: str, seed: int
) -> None:
    width, lr, weight_decay = FORMAL_HPARAMS[model]
    expected = {
        "model_type": model,
        "seed": seed,
        "dataset_suffix": f"{scale}-uint8",
        "eval_dataset_suffix": EVAL_DATASET_SUFFIX,
        "actual_epochs": ACTUAL_EPOCHS,
        "patience": 0,
        "stopped_by_patience": False,
        "hidden_size": width,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Protocol mismatch in {path}: {mismatches}")


def load_runs(input_root: Path, source_root: Path | None = None) -> list[Row]:
    """Load and validate the formal four-scale, six-model, ten-seed metrics."""

    recorded_root = source_root or input_root.resolve()
    rows: list[Row] = []
    for scale in SCALES:
        for model in MODELS:
            for seed in SEEDS:
                leaf = input_root / scale / f"{model}-seed{seed:02d}"
                paths = sorted(leaf.glob("*_metrics.json"))
                if len(paths) != 1:
                    raise FileNotFoundError(
                        f"Expected one metrics JSON in {leaf}, found {len(paths)}"
                    )
                path = paths[0]
                payload = json.loads(path.read_text(encoding="utf-8"))
                _validate_protocol(payload, path, scale, model, seed)
                train_char = _number(payload, "train_acc_at_best_val", "best_train_acc_char")
                val_char = _number(payload, "val_acc_at_best", "best_val_acc_char")
                train_sector = _number(
                    payload,
                    "train_acc_sector_at_best_val_sector",
                    "best_train_acc_pos",
                )
                val_sector = _number(payload, "val_acc_sector_at_best", "best_val_acc_pos")
                rows.append(
                    {
                        "seed": seed,
                        "scale": scale,
                        "model": model,
                        "hidden_size": int(payload["hidden_size"]),
                        "lr": float(payload["lr"]),
                        "weight_decay": float(payload["weight_decay"]),
                        "actual_epochs": int(payload["actual_epochs"]),
                        "train_acc_char": train_char,
                        "val_acc_char": val_char,
                        "overfit_gap_char": float(
                            payload.get("overfit_gap", train_char - val_char)
                        ),
                        "train_acc_sector": train_sector,
                        "val_acc_sector": val_sector,
                        "overfit_gap_sector": float(
                            payload.get("overfit_gap_sector", train_sector - val_sector)
                        ),
                        "best_epoch_char": int(
                            _number(payload, "best_epoch_val_acc_1based", "best_epoch_char")
                        ),
                        "best_epoch_sector": int(
                            _number(
                                payload,
                                "best_epoch_val_acc_sector_1based",
                                "best_epoch_pos",
                            )
                        ),
                        "source_metrics": str(recorded_root / path.relative_to(input_root)),
                    }
                )
    return rows


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("SEM requires at least two independent seeds")
    return fmean(values), stdev(values) / math.sqrt(len(values))


def summarize_runs(rows: list[Row]) -> list[Row]:
    """Return one mean-with-SEM row per scale and model."""

    summary: list[Row] = []
    for scale in SCALES:
        for model in MODELS:
            group = [row for row in rows if row["scale"] == scale and row["model"] == model]
            if {int(row["seed"]) for row in group} != set(SEEDS):
                raise RuntimeError(f"Incomplete seed set for scale={scale}, model={model}")
            result: Row = {"scale": scale, "model": model, "n_seeds": len(group)}
            for field in METRIC_FIELDS:
                mean, sem = _mean_sem([float(row[field]) for row in group])
                result[f"mean_{field}"] = mean
                result[f"sem_{field}"] = sem
            summary.append(result)
    return summary


def _write_csv(path: Path, rows: list[Row], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(
    path: Path, input_root: Path, rows: list[Row], summary: list[Row]
) -> None:
    counts = Counter((str(row["scale"]), str(row["model"])) for row in rows)
    payload = {
        "input_root": str(input_root.resolve()),
        "scales": list(SCALES),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "eval_dataset_suffix": EVAL_DATASET_SUFFIX,
        "actual_epochs": ACTUAL_EPOCHS,
        "expected_runs": len(SCALES) * len(MODELS) * len(SEEDS),
        "observed_runs": len(rows),
        "formal_hparams": {
            model: {"hidden_size": values[0], "lr": values[1], "weight_decay": values[2]}
            for model, values in FORMAL_HPARAMS.items()
        },
        "counts_by_scale_model": {
            scale: {model: counts[(scale, model)] for model in MODELS} for scale in SCALES
        },
        "mean_sem": summary,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_summary_csv(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[Row] = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    key: (value if key in {"scale", "model"} else float(value))
                    for key, value in raw.items()
                }
            )
    return rows


def _panel_limits(rows: list[Row], field: str) -> tuple[float, float]:
    """Return five-point rounded limits containing every mean and SEM error bar."""

    lower = min(float(row[f"mean_{field}"]) - float(row[f"sem_{field}"]) for row in rows)
    upper = max(float(row[f"mean_{field}"]) + float(row[f"sem_{field}"]) for row in rows)
    is_gap = field.startswith("overfit_gap")
    if is_gap:
        lower = min(lower, 0.0)
    padding = max(0.5 if is_gap else 1.0, 0.08 * (upper - lower))
    lower = max(0.0, 5.0 * math.floor((lower - padding) / 5.0))
    upper = 5.0 * math.ceil((upper + padding) / 5.0)
    if not is_gap:
        upper = min(100.0, upper)
    return lower, upper


def _plot_cross_scale(rows: list[Row], output_pdf: Path) -> Path:
    """Render Location/Identity rows and train/validation/gap columns."""

    by_identity = {(str(row["scale"]), str(row["model"])): row for row in rows}
    x = list(range(len(SCALES)))
    panels = (
        ("train_acc_sector", "val_acc_sector", "overfit_gap_sector"),
        ("train_acc_char", "val_acc_char", "overfit_gap_char"),
    )
    column_titles = ("Training accuracy", "Validation accuracy", "Train–validation gap")
    row_labels = ("Location", "Identity")
    panel_limits = {
        field: _panel_limits(rows, field) for fields in panels for field in fields
    }
    with plt.rc_context(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.35), sharex=True)
        for row_index, (fields, row_label) in enumerate(zip(panels, row_labels)):
            for column_index, (axis, field, title) in enumerate(
                zip(axes[row_index], fields, column_titles)
            ):
                for model in MODELS:
                    group = [by_identity[(scale, model)] for scale in SCALES]
                    axis.errorbar(
                        x,
                        [float(row[f"mean_{field}"]) for row in group],
                        yerr=[float(row[f"sem_{field}"]) for row in group],
                        color=MODEL_COLORS[model],
                        marker=MODEL_MARKERS[model],
                        linewidth=1.2,
                        markersize=3.2,
                        capsize=1.8,
                        capthick=0.7,
                        label=MODEL_LABELS[model],
                    )
                if row_index == 0:
                    axis.set_title(title)
                axis.set_xticks(x, SCALES)
                axis.grid(axis="y", linewidth=0.5, alpha=0.22)
                axis.spines[["top", "right"]].set_visible(False)
                axis.set_ylim(*panel_limits[field])
                if column_index < 2:
                    if column_index == 0:
                        axis.set_ylabel(f"{row_label}\nAccuracy (%)")
                else:
                    axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
                    axis.set_ylabel("Gap (pp)")
                if row_index == 1:
                    axis.set_xlabel("Training data (hours)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(MODELS), frameon=False)
        fig.subplots_adjust(
            left=0.105, right=0.995, bottom=0.14, top=0.84, hspace=0.34, wspace=0.43
        )
        label_y = max(axis.get_position().y1 for axis in axes[0]) + 0.035
        for label, axis in zip("ABC", axes[0]):
            position = axis.get_position()
            fig.text(
                position.x0 - 0.012,
                label_y,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return output_pdf


def run(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Write structured summaries, then render all figures from the mean-with-SEM CSV."""

    data_dir = args.data_dir or output_dir(CATEGORY, SCRIPT_NAME, "data")
    source_root = args.source_root or args.input_root.resolve()
    rows = load_runs(args.input_root, source_root)
    summary = summarize_runs(rows)
    all_runs_csv = data_dir / "data_scale_seed_runs_10seed.csv"
    mean_sem_csv = data_dir / "data_scale_mean_sem_10seed.csv"
    summary_json = data_dir / "data_scale_summary_10seed.json"
    _write_csv(all_runs_csv, rows, CSV_FIELDS)
    _write_csv(mean_sem_csv, summary, SUMMARY_FIELDS)
    _write_summary_json(summary_json, source_root, rows, summary)
    saved_summary = _read_summary_csv(mean_sem_csv)
    output_pdf = Path(args.output_pdf)
    figure = _plot_cross_scale(saved_summary, output_pdf)
    return [all_runs_csv, mean_sem_csv, summary_json], [figure]


def main() -> None:
    """Run the formal ten-seed data-scale summary and visualization workflow."""

    data_files, figure_files = run(parse_args())
    print("Data:")
    for path in data_files:
        print(path)
    print("Figures:")
    for path in figure_files:
        print(path)


if __name__ == "__main__":
    main()
