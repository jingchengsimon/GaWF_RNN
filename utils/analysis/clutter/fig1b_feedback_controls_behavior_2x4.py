"""Render the 2x4 behaviour comparison for the CM-MNIST feedback-control campaign.

The layout follows the retained best-six 2x4 summary: one row per readout (Location = sector,
Identity = character) and four columns with the same roles — test performance, validation dynamics,
target-switch recovery, and the feedback-shuffle ablation. Five series are drawn: the four
parameter-matched feedback controls plus the strict (multiplicative) GaWF reference evaluated under
the same feedback-control protocol. The reference Figure 1 family is never modified.

Inputs
- ``--curves-root``: feedback-control ``shuffle`` export (switch-aligned baseline/shuffle curves).
- ``--units-root``: feedback-control reset-excluded test accuracy export per unit.
- ``--histories-root``: feedback-control per-unit ``*.pkl`` training histories.
- ``--reference-test-csv``: strict GaWF reset-excluded test accuracy (ten seeds).
- ``--reference-histories-root``: strict GaWF validation-loss histories.
- ``--reference-shuffle-root``: strict GaWF shuffle export on the identical feedback-control
  protocol (same split, K=10, pre_K=10, offset 0 excluded).

Outputs
- ``results/data/analysis/G_behaviour/<script>/``: NPZ bundle, CSV summary, ``key_results.json``.
- ``results/figs/G_behaviour/<script>/``: PNG and PDF.
- curated copy under ``results/save/`` unless ``--no_save_copy`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir  # noqa: E402
from utils.analysis.clutter.fig1b_feedback_controls_target_switch import (  # noqa: E402
    CATEGORY,
    CONDITION_ORDER,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    _mean_sem,
    _offset_label,
    build_curves,
    load_units,
)

SCRIPT_NAME = Path(__file__).stem

ROW_KEYS = ("sector", "char")
ROW_LABELS = {"sector": "Location", "char": "Identity"}
REFERENCE_MODEL = "gawf_strict"
LABELS = {**MODEL_LABELS, REFERENCE_MODEL: "GaWF (strict)"}
COLORS = {**MODEL_COLORS, REFERENCE_MODEL: "#333333"}
MARKERS = {**MODEL_MARKERS, REFERENCE_MODEL: "X"}
HISTORY_TRACK = {"sector": "val_loss_pos", "char": "val_loss_char"}
ABLATION_CONDITIONS = ("baseline", "shuffle_digit", "shuffle_sector")
ABLATION_LABELS = ("Baseline", "Shuffle\ndigit", "Shuffle\nsector")
ABLATION_TINTS = ("#264653", "#E76F51", "#F4A261")
DEFAULT_CURVES_ROOT = (
    PROJECT_ROOT / "results" / "save_data" / "fig1" / "fbctrl_feedback_controls" / "shuffle_all"
)
DEFAULT_UNITS_ROOT = (
    PROJECT_ROOT / "results" / "save_data" / "fig1" / "fbctrl_feedback_controls" / "test_all"
)
DEFAULT_HISTORIES_ROOT = (
    PROJECT_ROOT / "results" / "save_data" / "fig1" / "fbctrl_feedback_controls" / "histories"
)
DEFAULT_REFERENCE_TEST_CSV = (
    PROJECT_ROOT
    / "results"
    / "data"
    / "analysis"
    / "fig1_reset_excluded_behavior_6model_10seed_v8"
    / "final"
    / "reset_excluded_test_accuracy_10seed.csv"
)
DEFAULT_REFERENCE_HISTORIES_ROOT = (
    PROJECT_ROOT / "results" / "save_data" / "fig1" / "validation_loss_histories"
)
DEFAULT_REFERENCE_SHUFFLE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "data"
    / "analysis"
    / "supple1_feedback_shuffle_recovery_resetexcluded_10seed_v1"
)


def parse_args() -> argparse.Namespace:
    """Parse the feedback-control export roots, reference roots, and destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves-root", type=Path, default=DEFAULT_CURVES_ROOT)
    parser.add_argument("--units-root", type=Path, default=DEFAULT_UNITS_ROOT)
    parser.add_argument("--histories-root", type=Path, default=DEFAULT_HISTORIES_ROOT)
    parser.add_argument("--reference-test-csv", type=Path, default=DEFAULT_REFERENCE_TEST_CSV)
    parser.add_argument(
        "--reference-histories-root", type=Path, default=DEFAULT_REFERENCE_HISTORIES_ROOT
    )
    parser.add_argument(
        "--reference-shuffle-root", type=Path, default=DEFAULT_REFERENCE_SHUFFLE_ROOT
    )
    parser.add_argument("--without_reference", action="store_true")
    parser.add_argument(
        "--save-pdf",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "save"
        / "cmmnist_feedback_controls_behavior_2x4.pdf",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "save"
        / "cmmnist_feedback_controls_behavior_2x4.png",
    )
    parser.add_argument("--no_save_copy", action="store_true")
    return parser.parse_args()


def _unit_key(directory: Path) -> tuple[str, int]:
    """Split one history directory name into model type and seed."""

    model, _, seed_text = directory.name.rpartition("-seed")
    if model not in MODEL_ORDER or not seed_text.isdigit():
        raise RuntimeError(f"Unrecognized history directory: {directory.name}")
    return model, int(seed_text)


def load_loss_histories(root: Path) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Load the per-epoch validation-loss track of every completed feedback-control unit."""

    if not root.is_dir():
        raise RuntimeError(f"Missing feedback-control history root: {root}")
    histories: dict[str, dict[str, dict[int, np.ndarray]]] = {
        model: {row_key: {} for row_key in ROW_KEYS} for model in MODEL_ORDER
    }
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        model, seed = _unit_key(directory)
        pickles = sorted(directory.glob("*.pkl"))
        if len(pickles) != 1:
            raise RuntimeError(f"Expected exactly one history pickle in {directory}")
        with pickles[0].open("rb") as stream:
            payload = pickle.load(stream)
        for row_key, track in HISTORY_TRACK.items():
            values = payload.get(track)
            if values is None:
                raise RuntimeError(f"{pickles[0]} has no {track} track")
            histories[model][row_key][seed] = np.asarray(values, dtype=np.float64)
    for model in MODEL_ORDER:
        for row_key in ROW_KEYS:
            if not histories[model][row_key]:
                raise RuntimeError(f"No loss histories for {model}/{row_key}")
    return histories


def load_reference_gawf(
    test_csv: Path, histories_root: Path, shuffle_root: Path
) -> dict[str, Any]:
    """Load the retained strict-GaWF reference under the feedback-control protocol."""

    with test_csv.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("model") == "gawf"]
    if len(rows) != 10:
        raise RuntimeError(f"Expected ten strict-GaWF test rows in {test_csv}, got {len(rows)}")
    rows.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    test = {
        row_key: np.asarray([float(row[f"{row_key}_acc"]) for row in rows], dtype=np.float64)
        for row_key in ROW_KEYS
    }

    histories: dict[str, dict[int, np.ndarray]] = {row_key: {} for row_key in ROW_KEYS}
    curves: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {row_key: [] for row_key in ROW_KEYS}
        for condition in CONDITION_ORDER
    }
    offsets: np.ndarray | None = None
    for seed in seeds:
        history_dir = histories_root / f"gawf-seed{seed:02d}"
        pickles = sorted(history_dir.glob("*.pkl"))
        if len(pickles) != 1:
            raise RuntimeError(f"Expected exactly one strict-GaWF history in {history_dir}")
        with pickles[0].open("rb") as stream:
            payload = pickle.load(stream)
        for row_key, track in HISTORY_TRACK.items():
            histories[row_key][seed] = np.asarray(payload[track], dtype=np.float64)

        shuffle_path = shuffle_root / f"gawf-seed{seed:02d}" / "ablation_metrics.json"
        shuffle = json.loads(shuffle_path.read_text(encoding="utf-8"))
        if shuffle.get("exclude_window_initial_frame") is not True:
            raise RuntimeError(f"Strict-GaWF shuffle export includes the reset frame: {shuffle_path}")
        current = np.asarray(shuffle["switch_offsets"], dtype=np.int64)
        if offsets is None:
            offsets = current
        elif not np.array_equal(offsets, current):
            raise RuntimeError(f"Strict-GaWF offset grid differs from previous seeds: {shuffle_path}")
        for condition in CONDITION_ORDER:
            for row_key in ROW_KEYS:
                curves[condition][row_key].append(
                    np.asarray(
                        shuffle["conditions"][condition][f"switch_{row_key}_acc"],
                        dtype=np.float64,
                    )
                )

    recovery = {
        row_key: _mean_sem(curves["baseline"][row_key]) for row_key in ROW_KEYS
    }
    ablation = {
        condition: {
            row_key: np.asarray(
                [float(np.mean(series)) for series in curves[condition][row_key]], dtype=np.float64
            )
            for row_key in ROW_KEYS
        }
        for condition in ABLATION_CONDITIONS
    }
    return {
        "seeds": seeds,
        "test": test,
        "histories": histories,
        "recovery": recovery,
        "ablation": ablation,
    }


def build_panels(
    curves: dict[str, Any],
    grouped: dict[str, Any],
    histories: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one unified panel payload for every drawn series."""

    models = [*MODEL_ORDER] + ([REFERENCE_MODEL] if reference else [])
    panels: dict[str, Any] = {
        "models": models,
        "test": {},
        "histories": {},
        "recovery": {},
        "ablation": {},
        "seeds": {},
    }
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        panels["test"][model] = {
            row_key: np.asarray(
                [entry[f"{row_key}_acc"] for entry in grouped[model]["test"] if entry],
                dtype=np.float64,
            )
            for row_key in ROW_KEYS
        }
        panels["histories"][model] = histories[model]
        panels["recovery"][model] = {
            row_key: (
                np.asarray(payload["baseline"][row_key]["mean"], dtype=np.float64),
                np.asarray(payload["baseline"][row_key]["sem"], dtype=np.float64),
            )
            for row_key in ROW_KEYS
        }
        panels["ablation"][model] = {
            condition: {
                row_key: np.asarray(payload[condition][row_key]["seed_curves"], dtype=np.float64)
                .mean(axis=1)
                for row_key in ROW_KEYS
            }
            for condition in ABLATION_CONDITIONS
        }
        panels["seeds"][model] = payload["n_seeds"]
    if reference is not None:
        panels["test"][REFERENCE_MODEL] = reference["test"]
        panels["histories"][REFERENCE_MODEL] = reference["histories"]
        panels["recovery"][REFERENCE_MODEL] = reference["recovery"]
        panels["ablation"][REFERENCE_MODEL] = reference["ablation"]
        panels["seeds"][REFERENCE_MODEL] = len(reference["seeds"])
    return panels


def _bar_with_seeds(
    axis: plt.Axes,
    models: list[str],
    values: dict[str, np.ndarray],
    *,
    show_xticks: bool,
) -> None:
    """Draw one bar per series with individual seed dots and a SEM error bar."""

    for index, model in enumerate(models):
        samples = np.asarray(values[model], dtype=np.float64)
        mean = samples.mean()
        sem = samples.std(ddof=1) / np.sqrt(samples.size) if samples.size > 1 else 0.0
        axis.bar(
            index,
            mean,
            width=0.66,
            color=COLORS[model],
            alpha=0.9,
            zorder=2,
            hatch="//" if model == REFERENCE_MODEL else None,
            edgecolor="white" if model == REFERENCE_MODEL else None,
        )
        axis.errorbar(index, mean, yerr=sem, color="#333333", capsize=2.6, linewidth=0.9, zorder=4)
        jitter = np.linspace(-0.16, 0.16, samples.size)
        axis.scatter(index + jitter, samples, s=9, color="#555555", alpha=0.75, zorder=3)
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(
        [LABELS[model] for model in models],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    stacked = np.concatenate([np.asarray(values[model], dtype=np.float64) for model in models])
    span = float(stacked.max() - stacked.min())
    axis.set_ylim(float(stacked.min() - 0.45 * span), float(stacked.max() + 0.15 * span))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_histories(
    axis: plt.Axes,
    models: list[str],
    histories: dict[str, Any],
    row_key: str,
    *,
    show_xlabel: bool,
) -> None:
    """Draw mean plus SEM validation-loss curves per series."""

    longest = 0
    for model in models:
        per_seed = histories[model][row_key]
        seeds = sorted(per_seed)
        stacked = np.stack([per_seed[seed] for seed in seeds])
        longest = max(longest, stacked.shape[1])
        epochs = np.arange(1, stacked.shape[1] + 1)
        mean = stacked.mean(axis=0)
        sem = (
            stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            if stacked.shape[0] > 1
            else np.zeros_like(mean)
        )
        axis.plot(
            epochs,
            mean,
            color=COLORS[model],
            linewidth=1.6,
            linestyle="--" if model == REFERENCE_MODEL else "-",
            zorder=3,
        )
        if np.any(sem > 0):
            axis.fill_between(
                epochs, mean - sem, mean + sem, color=COLORS[model], alpha=0.18, zorder=2
            )
    axis.set_xlim(1, longest)
    axis.set_xlabel("Epoch" if show_xlabel else "", fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_recovery(
    axis: plt.Axes,
    offsets: np.ndarray,
    models: list[str],
    recovery: dict[str, Any],
    row_key: str,
    *,
    show_xlabel: bool,
) -> None:
    """Draw the switch-aligned recovery curves of the baseline condition."""

    for model in models:
        mean, sem = recovery[model][row_key]
        axis.plot(
            offsets,
            mean,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=3.0,
            linewidth=1.4,
            linestyle="--" if model == REFERENCE_MODEL else "-",
            zorder=3,
        )
        if np.any(sem > 0):
            axis.fill_between(
                offsets, mean - sem, mean + sem, color=COLORS[model], alpha=0.18, zorder=2
            )
    axis.axvline(0, color="#D55E00", linestyle="--", linewidth=1.1, zorder=1)
    axis.set_xticks([-10, 1, 4, 10])
    axis.set_xticklabels(
        ["pre10", "switch", "post4", "post10"], fontsize=7.5, rotation=20, ha="right"
    )
    axis.set_xlim(-10.5, 10.5)
    if show_xlabel:
        axis.set_xlabel("Frame relative to target switch", fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_ablation(
    axis: plt.Axes,
    models: list[str],
    ablation: dict[str, Any],
    row_key: str,
    *,
    show_xticks: bool,
) -> None:
    """Draw the baseline, shuffled-digit, and shuffled-sector bars per series."""

    width = 0.8 / (len(models) + 2) * 3.0 if len(models) > 4 else 0.26
    overall: list[float] = []
    for model_index, model in enumerate(models):
        for condition_index, condition in enumerate(ABLATION_CONDITIONS):
            samples = np.asarray(ablation[model][condition][row_key], dtype=np.float64)
            mean = samples.mean()
            sem = samples.std(ddof=1) / np.sqrt(samples.size) if samples.size > 1 else 0.0
            overall.append(mean)
            x = model_index + (condition_index - 1) * width
            axis.bar(
                x,
                mean,
                width=width,
                color=ABLATION_TINTS[condition_index],
                alpha=0.92 if condition_index == 0 else 0.75,
                zorder=2,
                label=ABLATION_LABELS[condition_index] if model_index == 0 else None,
            )
            axis.errorbar(x, mean, yerr=sem, color="#333333", capsize=1.8, linewidth=0.8, zorder=4)
    span = max(overall) - min(overall)
    axis.set_ylim(min(overall) - 0.45 * span, max(overall) + 0.15 * span)
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(
        [LABELS[model] for model in models],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    axis.set_xlim(-0.5, len(models) - 0.5)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def render_figure(
    panels: dict[str, Any],
    offsets: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render the feedback-control 2x4 behaviour comparison."""

    models = panels["models"]
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(11.8, 6.2))
        for row, row_key in enumerate(ROW_KEYS):
            _bar_with_seeds(
                axes[row][0],
                models,
                {model: panels["test"][model][row_key] for model in models},
                show_xticks=row == 1,
            )
            _plot_histories(
                axes[row][1],
                models,
                panels["histories"],
                row_key,
                show_xlabel=row == 1,
            )
            _plot_recovery(
                axes[row][2],
                offsets,
                models,
                panels["recovery"],
                row_key,
                show_xlabel=row == 1,
            )
            _plot_ablation(
                axes[row][3],
                models,
                panels["ablation"],
                row_key,
                show_xticks=row == 1,
            )
        titles = (
            "A  Reset-excluded test accuracy",
            "B  Validation loss",
            "C  Target-switch recovery",
            "D  Feedback shuffle ablation",
        )
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, pad=12, fontweight="semibold")

        figure.subplots_adjust(
            left=0.062, right=0.99, bottom=0.215, top=0.845, hspace=0.36, wspace=0.30
        )
        row_centers = [
            float(np.mean([axis.get_position().y0 + axis.get_position().height / 2 for axis in row]))
            for row in axes
        ]
        for row_key, center in zip(ROW_KEYS, row_centers):
            figure.text(0.012, center, ROW_LABELS[row_key], rotation=90, va="center", fontsize=10)
        handles = [
            Line2D(
                [0],
                [0],
                color=COLORS[model],
                linewidth=2.5,
                marker=MARKERS[model],
                markersize=4,
                linestyle="--" if model == REFERENCE_MODEL else "-",
            )
            for model in models
        ]
        figure.legend(
            handles,
            [LABELS[model] for model in models],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(models),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.4,
        )
        ablation_handles = [Line2D([0], [0], color=tint, linewidth=7.0) for tint in ABLATION_TINTS]
        figure.legend(
            ablation_handles,
            [label.replace("\n", " ") for label in ABLATION_LABELS],
            loc="lower right",
            bbox_to_anchor=(0.995, 0.078),
            ncol=3,
            frameon=False,
            fontsize=7.5,
            handlelength=1.4,
        )
        counts = ", ".join(f"{LABELS[model]} n={panels['seeds'][model]}" for model in models)
        figure.text(
            0.5,
            0.008,
            "CM-MNIST feedback controls, 150 epochs, seeds 1-10; A: reset-excluded test accuracy; B:"
            " validation loss on the 40h split;\nC: joint-switch-balanced 10-digit test, reset at"
            " every foreground switch, offset 0 excluded; D: sequence-512 reset-excluded shuffle"
            f" conditions ({counts}).\nGaWF (strict) is the retained multiplicative-feedback model on"
            " the identical evaluation protocol; it is not a result of this campaign's Amarel array.",
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        plt.close(figure)


def write_structured_outputs(
    panels: dict[str, Any], offsets: np.ndarray, data_dir: Path
) -> dict[str, Path]:
    """Write the aggregated NPZ bundle and CSV summary."""

    data_dir.mkdir(parents=True, exist_ok=True)
    models = panels["models"]
    arrays: dict[str, np.ndarray] = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "offset_labels": np.asarray([_offset_label(int(o)) for o in offsets]),
        "model_order": np.asarray(models),
    }
    rows: list[dict[str, Any]] = []
    for row_key in ROW_KEYS:
        for model in models:
            arrays[f"{row_key}__{model}__n_seeds"] = np.asarray(
                [panels["seeds"][model]], dtype=np.int64
            )
            arrays[f"{row_key}__{model}__test_acc"] = np.asarray(
                panels["test"][model][row_key], dtype=np.float32
            )
            seeds = sorted(panels["histories"][model][row_key])
            arrays[f"{row_key}__{model}__val_loss_seeds"] = np.asarray(seeds, dtype=np.int64)
            arrays[f"{row_key}__{model}__val_loss"] = np.stack(
                [panels["histories"][model][row_key][seed] for seed in seeds]
            ).astype(np.float32)
            mean, sem = panels["recovery"][model][row_key]
            arrays[f"{row_key}__{model}__recovery_mean"] = np.asarray(mean, dtype=np.float32)
            arrays[f"{row_key}__{model}__recovery_sem"] = np.asarray(sem, dtype=np.float32)
            for condition in ABLATION_CONDITIONS:
                arrays[f"{row_key}__{model}__{condition}__acc"] = np.asarray(
                    panels["ablation"][model][condition][row_key], dtype=np.float32
                )
            test_values = np.asarray(panels["test"][model][row_key], dtype=np.float64)
            row: dict[str, Any] = {
                "readout": row_key,
                "model": model,
                "n_seeds": panels["seeds"][model],
                "test_acc_mean": float(test_values.mean()),
                "test_acc_sem": float(test_values.std(ddof=1) / np.sqrt(test_values.size))
                if test_values.size > 1
                else 0.0,
                "val_loss_final_mean": float(
                    np.mean(
                        [panels["histories"][model][row_key][seed][-1] for seed in seeds]
                    )
                ),
                "post_switch_acc_mean": float(
                    np.asarray(mean, dtype=np.float64)[np.asarray(offsets) >= 1].mean()
                ),
            }
            for condition in ABLATION_CONDITIONS:
                values = np.asarray(panels["ablation"][model][condition][row_key], dtype=np.float64)
                row[f"{condition}_acc_mean"] = float(values.mean())
                row[f"{condition}_acc_sem"] = (
                    float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                )
            rows.append(row)
    npz_path = data_dir / "feedback_controls_behavior_2x4.npz"
    np.savez_compressed(npz_path, **arrays)
    csv_path = data_dir / "feedback_controls_behavior_2x4.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    key_results = {
        f"{row['readout']}.{row['model']}.{key}": float(value)
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    }
    key_path = data_dir / "key_results.json"
    key_path.write_text(json.dumps(key_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"npz": npz_path, "csv": csv_path, "key": key_path}


def main() -> None:
    """Load the completed exports, aggregate, and render the 2x4 figure."""

    args = parse_args()
    grouped = load_units(args.units_root, args.curves_root)
    curves = build_curves(grouped)
    histories = load_loss_histories(args.histories_root)
    reference = None
    if not args.without_reference:
        reference = load_reference_gawf(
            args.reference_test_csv,
            args.reference_histories_root,
            args.reference_shuffle_root,
        )
    panels = build_panels(curves, grouped, histories, reference)
    data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
    fig_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    written = write_structured_outputs(panels, np.asarray(curves["offsets"]), data_dir)
    png_path = fig_dir / "cmmnist_feedback_controls_behavior_2x4.png"
    pdf_path = fig_dir / "cmmnist_feedback_controls_behavior_2x4.pdf"
    render_figure(panels, np.asarray(curves["offsets"]), png_path, pdf_path)
    if not args.no_save_copy:
        render_figure(panels, np.asarray(curves["offsets"]), args.save_png, args.save_pdf)
    for row_key in ROW_KEYS:
        print(f"[{ROW_LABELS[row_key]}]")
        for model in panels["models"]:
            values = np.asarray(panels["test"][model][row_key], dtype=np.float64)
            mean, _ = panels["recovery"][model][row_key]
            post = np.asarray(curves["offsets"]) >= 1
            baseline = np.asarray(panels["ablation"][model]["baseline"][row_key], dtype=np.float64)
            shuffled = np.asarray(panels["ablation"][model]["shuffle_all"][row_key], dtype=np.float64) if "shuffle_all" in panels["ablation"][model] else None
            extra = "" if shuffled is None else f" shuffle-all delta {baseline.mean() - shuffled.mean():+.2f}"
            print(
                f"  {LABELS[model]:>14} n={panels['seeds'][model]:>2}"
                f" test {values.mean():.2f}"
                f" post-switch {np.asarray(mean)[post].mean():.2f}{extra}"
            )
    for name, path in written.items():
        print(f"{name}: {path}")
    print(f"figure: {png_path}")
    print(f"figure: {pdf_path}")
    if not args.no_save_copy:
        print(f"curated: {args.save_png}")
        print(f"curated: {args.save_pdf}")


if __name__ == "__main__":
    main()
