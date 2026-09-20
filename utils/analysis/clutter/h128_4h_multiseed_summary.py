"""Summarize and plot the 4h, width-128 Clutter comparison.

Inputs are exactly one metrics JSON and one training PKL for each of six models and ten seeds.
The script validates the formal protocol, writes seed-level CSV, mean-with-SEM CSV, a JSON
summary, and float32 NPZ curves below ``results/data/analysis/G_behaviour``. It renders a
canonical PNG below ``results/figs/G_behaviour`` and the requested PDF below ``results/save``.
The 2-by-4 layout follows the visual grammar of the retained multiseed Figure 2 without
inventing target-switch or shuffle-ablation measurements absent from this experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.fg_switch_offset_acc import MODEL_COLORS, MODEL_LABELS
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATEGORY = "G_behaviour"
SCRIPT_NAME = "h128_4h_multiseed_summary"
MODEL_ORDER = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
SEEDS = tuple(range(1, 11))
EPOCHS = 150
FORMAL_HPARAMS = {
    "gawf": (128, 0.005, 0.001),
    "rnn": (128, 0.001, 0.00001),
    "lstm": (128, 0.001, 0.001),
    "gru": (128, 0.005, 0.001),
    "mamba": (128, 0.001, 0.001),
    "s5": (128, 0.001, 0.0),
}
BAR_METRICS = (
    "best_val_location",
    "best_val_identity",
    "gap_location",
    "gap_identity",
)
CURVE_KEYS = (
    "val_acc_location",
    "val_acc_identity",
    "val_loss_location",
    "val_loss_identity",
)
PKL_KEYS = {
    "val_acc_location": "val_acc_pos",
    "val_acc_identity": "val_acc_char",
    "val_loss_location": "val_loss_pos",
    "val_loss_identity": "val_loss_char",
}
SEED_CSV_FIELDS = (
    "model",
    "seed",
    "best_val_location",
    "best_val_identity",
    "final_val_location",
    "final_val_identity",
    "gap_location",
    "gap_identity",
    "best_epoch_location",
    "best_epoch_identity",
    "source_metrics",
    "source_pkl",
)
SUMMARY_CSV_FIELDS = (
    "model",
    "n_seeds",
    *(f"{stat}_{metric}" for metric in BAR_METRICS for stat in ("mean", "sem")),
)
Row = dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    """Parse input provenance and output overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/data/clutter/runs/data_scale/clutter_h128_comparison_4h_ep150"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/G/MIMOlab/Codes/aim3_gawf_rnn/results/data/clutter/runs/data_scale/"
            "clutter_h128_comparison_4h_ep150"
        ),
        help="Authoritative remote result root recorded in provenance fields.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=PROJECT_ROOT / "results/save/Fig2_clutter_4h_h128_multiseed_2x4.pdf",
    )
    return parser.parse_args()


def _number(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"Metrics are missing all required fields: {keys}")


def _validate_protocol(payload: dict[str, Any], path: Path, model: str, seed: int) -> None:
    width, learning_rate, weight_decay = FORMAL_HPARAMS[model]
    expected = {
        "model_type": model,
        "seed": seed,
        "dataset_suffix": "4h-uint8",
        "eval_dataset_suffix": "40h-uint8",
        "actual_epochs": EPOCHS,
        "patience": 0,
        "stopped_by_patience": False,
        "hidden_size": width,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "use_acceleration": True,
        "use_mmap": True,
        "input_cast_mode": "device",
        "frame_layout": "compact",
        "shuffle_block_size": -1,
        "checkpoint_interval_epochs": 5,
    }
    if model in {"rnn", "lstm", "gru", "gawf"}:
        expected["num_layers"] = 1
    elif model == "mamba":
        expected["mamba_d_model"] = 128
    else:
        expected.update({"s5_d_model": 128, "s5_state_size": 128, "s5_num_layers": 1})
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Protocol mismatch in {path}: {mismatches}")


def _load_curve(payload: dict[str, Any], key: str, path: Path) -> np.ndarray:
    array = np.asarray(payload[key], dtype=np.float64)
    if array.shape != (EPOCHS,):
        raise ValueError(f"Expected {key} shape {(EPOCHS,)} in {path}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {key}: {path}")
    return array


def load_runs(
    input_root: Path, source_root: Path
) -> tuple[list[Row], dict[str, np.ndarray], list[Path]]:
    """Load and validate all 60 seed bundles.

    Returns seed rows, float64 arrays shaped ``(models, seeds, epochs)`` for each curve, and
    the exact 120 source files used to compute the result.
    """

    rows: list[Row] = []
    curves = {
        key: np.empty((len(MODEL_ORDER), len(SEEDS), EPOCHS), dtype=np.float64)
        for key in CURVE_KEYS
    }
    inputs: list[Path] = []
    for model_index, model in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            leaf = input_root / f"{model}-seed{seed:02d}"
            metrics_paths = sorted(leaf.glob("*_metrics.json"))
            pkl_paths = sorted(leaf.glob("*.pkl"))
            if len(metrics_paths) != 1 or len(pkl_paths) != 1:
                raise FileNotFoundError(
                    f"Expected one metrics JSON and one PKL in {leaf}; "
                    f"found metrics={len(metrics_paths)} pkl={len(pkl_paths)}"
                )
            metrics_path, pkl_path = metrics_paths[0], pkl_paths[0]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with pkl_path.open("rb") as handle:
                history = pickle.load(handle)
            _validate_protocol(metrics, metrics_path, model, seed)
            for curve_key, pkl_key in PKL_KEYS.items():
                curves[curve_key][model_index, seed_index] = _load_curve(
                    history, pkl_key, pkl_path
                )

            best_location = _number(
                metrics, "val_acc_sector_at_best", "best_val_acc_pos"
            )
            best_identity = _number(metrics, "val_acc_at_best", "best_val_acc_char")
            if not math.isclose(
                best_location,
                float(np.max(curves["val_acc_location"][model_index, seed_index])),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError(f"Best Location accuracy disagrees with history: {leaf}")
            if not math.isclose(
                best_identity,
                float(np.max(curves["val_acc_identity"][model_index, seed_index])),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError(f"Best Identity accuracy disagrees with history: {leaf}")

            relative_metrics = metrics_path.relative_to(input_root)
            relative_pkl = pkl_path.relative_to(input_root)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "best_val_location": best_location,
                    "best_val_identity": best_identity,
                    "final_val_location": float(
                        curves["val_acc_location"][model_index, seed_index, -1]
                    ),
                    "final_val_identity": float(
                        curves["val_acc_identity"][model_index, seed_index, -1]
                    ),
                    "gap_location": _number(metrics, "overfit_gap_sector"),
                    "gap_identity": _number(metrics, "overfit_gap"),
                    "best_epoch_location": int(
                        _number(
                            metrics,
                            "best_epoch_val_acc_sector_1based",
                            "best_epoch_pos",
                        )
                    ),
                    "best_epoch_identity": int(
                        _number(metrics, "best_epoch_val_acc_1based", "best_epoch_char")
                    ),
                    "source_metrics": str(source_root / relative_metrics),
                    "source_pkl": str(source_root / relative_pkl),
                }
            )
            inputs.extend((metrics_path, pkl_path))
    if len(rows) != len(MODEL_ORDER) * len(SEEDS):
        raise RuntimeError(f"Expected 60 seed rows, found {len(rows)}")
    return rows, curves, inputs


def _mean_sem(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("SEM requires at least two independent seeds")
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def summarize_runs(rows: list[Row]) -> list[Row]:
    """Return one seed-level mean-with-SEM row per model."""

    summary: list[Row] = []
    for model in MODEL_ORDER:
        group = [row for row in rows if row["model"] == model]
        if {int(row["seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"Incomplete seed set for {model}")
        record: Row = {"model": model, "n_seeds": len(group)}
        for metric in BAR_METRICS:
            mean, sem = _mean_sem([float(row[metric]) for row in group])
            record[f"mean_{metric}"] = mean
            record[f"sem_{metric}"] = sem
        summary.append(record)
    return summary


def _write_csv(path: Path, rows: list[Row], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.55, zorder=0)
    axis.tick_params(direction="out", length=2.5, width=0.7)


def _plot_seed_bars(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    y_limits: tuple[float, float],
    show_xticks: bool,
) -> None:
    positions = np.arange(len(MODEL_ORDER), dtype=np.float64)
    rng = np.random.default_rng(0)
    for model_index, model in enumerate(MODEL_ORDER):
        mean, sem = _mean_sem(values[model_index])
        axis.bar(
            positions[model_index],
            mean,
            width=0.72,
            yerr=sem,
            color=MODEL_COLORS[model],
            edgecolor="none",
            capsize=2.2,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
            zorder=2,
        )
        add_seed_points(
            axis,
            np.asarray([positions[model_index]]),
            values[model_index, :, None],
            bar_width=0.72,
            rng=rng,
        )
    axis.set_xticks(
        positions,
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=40,
        ha="right",
    )
    if not show_xticks:
        axis.tick_params(axis="x", which="both", labelbottom=False)
    axis.set_ylim(*y_limits)
    _style_axis(axis)


def _plot_curves(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    y_limits: tuple[float, float],
    show_xlabel: bool,
    show_xticks: bool,
) -> None:
    epochs = np.arange(1, EPOCHS + 1, dtype=np.float64)
    for model_index, model in enumerate(MODEL_ORDER):
        mean = values[model_index].mean(axis=0)
        sem = values[model_index].std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        axis.plot(epochs, mean, color=MODEL_COLORS[model], linewidth=1.45, zorder=2)
        axis.fill_between(
            epochs,
            mean - sem,
            mean + sem,
            color=MODEL_COLORS[model],
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    axis.set_xlim(1, EPOCHS)
    axis.set_xticks((1, 50, 100, 150))
    axis.set_ylim(*y_limits)
    if show_xlabel:
        axis.set_xlabel("Epoch")
    if not show_xticks:
        axis.tick_params(axis="x", which="both", labelbottom=False)
    _style_axis(axis)


def _plot_figure(
    curves: dict[str, np.ndarray],
    rows: list[Row],
    figure_png: Path,
    output_pdf: Path,
) -> None:
    bar_values = {
        metric: np.asarray(
            [
                [
                    float(
                        next(
                            row[metric]
                            for row in rows
                            if row["model"] == model and row["seed"] == seed
                        )
                    )
                    for seed in SEEDS
                ]
                for model in MODEL_ORDER
            ],
            dtype=np.float64,
        )
        for metric in BAR_METRICS
    }
    with plt.rc_context(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.55))
        _plot_seed_bars(
            axes[0, 0],
            bar_values["best_val_location"],
            y_limits=(75.0, 90.0),
            show_xticks=False,
        )
        _plot_seed_bars(
            axes[1, 0],
            bar_values["best_val_identity"],
            y_limits=(48.0, 72.0),
            show_xticks=True,
        )
        _plot_curves(
            axes[0, 1],
            curves["val_acc_location"],
            y_limits=(40.0, 90.0),
            show_xlabel=False,
            show_xticks=False,
        )
        _plot_curves(
            axes[1, 1],
            curves["val_acc_identity"],
            y_limits=(0.0, 75.0),
            show_xlabel=True,
            show_xticks=True,
        )
        _plot_curves(
            axes[0, 2],
            curves["val_loss_location"],
            y_limits=(0.3, 1.7),
            show_xlabel=False,
            show_xticks=False,
        )
        _plot_curves(
            axes[1, 2],
            curves["val_loss_identity"],
            y_limits=(1.0, 4.5),
            show_xlabel=True,
            show_xticks=True,
        )
        _plot_seed_bars(
            axes[0, 3],
            bar_values["gap_location"],
            y_limits=(4.0, 12.0),
            show_xticks=False,
        )
        _plot_seed_bars(
            axes[1, 3],
            bar_values["gap_identity"],
            y_limits=(20.0, 45.0),
            show_xticks=True,
        )

        for row_index in range(2):
            axes[row_index, 0].set_ylabel("Accuracy (%)")
            axes[row_index, 1].set_ylabel("Accuracy (%)")
            axes[row_index, 2].set_ylabel("Loss")
            axes[row_index, 3].set_ylabel("Gap (pp)")

        fig.subplots_adjust(
            left=0.095,
            right=0.995,
            bottom=0.235,
            top=0.755,
            hspace=0.43,
            wspace=0.47,
        )
        column_centers = [
            np.mean(
                [
                    axes[row, column].get_position().x0
                    + axes[row, column].get_position().width / 2
                    for row in range(2)
                ]
            )
            for column in range(4)
        ]
        title_y = max(axes[0, column].get_position().y1 for column in range(4)) + 0.035
        for x, title in zip(
            column_centers,
            (
                "Best validation\naccuracy",
                "Validation accuracy\n(mean +/- SEM)",
                "Validation loss\n(mean +/- SEM)",
                "Train-validation\ngap",
            ),
        ):
            fig.text(x, title_y, title, ha="center", va="bottom", fontsize=8)
        for label, axis in zip("ABCD", axes[0]):
            position = axis.get_position()
            fig.text(
                position.x0 - 0.012,
                title_y,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        row_centers = [
            np.mean(
                [
                    axes[row, column].get_position().y0
                    + axes[row, column].get_position().height / 2
                    for column in range(4)
                ]
            )
            for row in range(2)
        ]
        for y, label in zip(row_centers, ("Location", "Identity")):
            fig.text(0.018, y, label, rotation=90, ha="center", va="center", fontsize=8)

        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.2)
            for model in MODEL_ORDER
        ]
        fig.legend(
            handles,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(MODEL_ORDER),
            fontsize=6.5,
            handlelength=1.4,
            columnspacing=1.0,
        )
        figure_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_png, dpi=300)
        fig.savefig(output_pdf)
        plt.close(fig)


def run(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Validate inputs, write structured summaries, and render the 2-by-4 figure."""

    data_dir = args.data_dir or output_dir(CATEGORY, SCRIPT_NAME, "data")
    figure_dir = args.figure_dir or output_dir(CATEGORY, SCRIPT_NAME, "figs")
    rows, curves, input_paths = load_runs(args.input_root, args.source_root)
    summary = summarize_runs(rows)
    seed_csv = data_dir / "h128_4h_seed_runs_10seed.csv"
    summary_csv = data_dir / "h128_4h_mean_sem_10seed.csv"
    curves_npz = data_dir / "h128_4h_curves_10seed.npz"
    summary_json = data_dir / "h128_4h_summary_10seed.json"
    delivery_json = data_dir / "h128_4h_delivery.json"
    figure_png = figure_dir / "clutter_4h_h128_multiseed_2x4.png"

    _write_csv(seed_csv, rows, SEED_CSV_FIELDS)
    _write_csv(summary_csv, summary, SUMMARY_CSV_FIELDS)
    curve_arrays: dict[str, np.ndarray] = {
        "model_order": np.asarray(MODEL_ORDER),
        "seeds": np.asarray(SEEDS, dtype=np.int64),
        "epochs": np.arange(1, EPOCHS + 1, dtype=np.int64),
    }
    for key, values in curves.items():
        curve_arrays[f"{key}_seed"] = values.astype(np.float32)
        curve_arrays[f"{key}_mean"] = values.mean(axis=1).astype(np.float32)
        curve_arrays[f"{key}_sem"] = (
            values.std(axis=1, ddof=1) / np.sqrt(len(SEEDS))
        ).astype(np.float32)
    np.savez_compressed(curves_npz, **curve_arrays)

    input_sha = _aggregate_sha256(input_paths, args.input_root)
    summary_payload = {
        "input_root": str(args.input_root.resolve()),
        "source_root": str(args.source_root),
        "models": list(MODEL_ORDER),
        "seeds": list(SEEDS),
        "n_models": len(MODEL_ORDER),
        "n_seeds_per_model": len(SEEDS),
        "observed_runs": len(rows),
        "input_files": len(input_paths),
        "input_bytes": sum(path.stat().st_size for path in input_paths),
        "input_aggregate_sha256": input_sha,
        "protocol": {
            "train_data": "4h-uint8",
            "validation_data": "40h-uint8",
            "width": 128,
            "num_layers": 1,
            "epochs": EPOCHS,
            "patience": 0,
        },
        "seed_unit": "one independent training seed per point and SEM sample",
        "s5_seed05": {
            "status": "strictly validated after epoch-30 checkpoint recovery",
            "included": True,
        },
        "mean_sem": summary,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    _plot_figure(curves, rows, figure_png, args.output_pdf)
    delivery_payload = {
        "figure_png": str(figure_png.resolve()),
        "figure_png_sha256": _file_sha256(figure_png),
        "output_pdf": str(args.output_pdf.resolve()),
        "output_pdf_sha256": _file_sha256(args.output_pdf),
        "input_aggregate_sha256": input_sha,
        "source_runs": len(rows),
    }
    delivery_json.write_text(json.dumps(delivery_payload, indent=2) + "\n", encoding="utf-8")
    return [seed_csv, summary_csv, curves_npz, summary_json, delivery_json], [
        figure_png,
        args.output_pdf,
    ]


def main() -> None:
    """Run the complete h128 4h multiseed summary workflow."""

    data_files, figure_files = run(parse_args())
    print("Data:")
    for path in data_files:
        print(path)
    print("Figures:")
    for path in figure_files:
        print(path)


if __name__ == "__main__":
    main()
