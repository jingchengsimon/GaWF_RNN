"""Render the 2x4 behaviour comparison for the CM-MNIST feedback-control campaign.

The layout follows the retained best-six 2x4 summary: one row per readout (Location = sector,
Identity = character) and four columns with the same roles — test performance, validation dynamics,
target-switch recovery, and the feedback-shuffle ablation. Ten series are drawn: the four
parameter-matched feedback controls from this campaign plus the six retained best-six Clutter models
(GaWF, RNN, LSTM, GRU, Mamba, S5) re-evaluated on the identical feedback-control protocol. Each
control keeps the palette of the baseline it ablates, so a colour marks one architecture family and
the retained models are additionally drawn dashed with open markers and hatched bars.

Column coverage differs by construction: the five open-loop baselines were never run with the
feedback-shuffle ablation, so column D carries only the four controls plus strict GaWF and marks the
remaining slots as not run. The reference Figure 1 family is never modified.

Inputs
- ``--curves-root``: feedback-control ``shuffle`` export (switch-aligned baseline/shuffle curves).
- ``--units-root``: feedback-control reset-excluded test accuracy export per unit.
- ``--histories-root``: feedback-control per-unit ``*.pkl`` training histories.
- ``--reference-test-csv``: reset-excluded test accuracy of the six retained Clutter models.
- ``--reference-histories-root``: retained-model validation-loss histories.
- ``--reference-shuffle-root``: retained-model shuffle export where it exists (strict GaWF only);
  conditions use the same split, K=10, pre_K=10, offset 0 excluded.
- ``--baseline-recovery-root``: retained-model reset-excluded target-switch recovery export.

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
from collections.abc import Sequence
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
from utils.analysis.clutter.clutter_multiseed_summary import (  # noqa: E402
    load_recovery_curves,
)
from utils.analysis.clutter.fg_switch_offset_acc import (  # noqa: E402
    MODEL_COLORS as BASELINE_COLORS,
    MODEL_LABELS as BASELINE_LABELS,
    MODEL_MARKERS as BASELINE_MARKERS,
    MODEL_ORDER as BASELINE_ORDER,
)

SCRIPT_NAME = Path(__file__).stem

ROW_KEYS = ("sector", "char")
ROW_LABELS = {"sector": "Location", "char": "Identity"}
BASELINE_MODELS: tuple[str, ...] = tuple(BASELINE_ORDER)
REFERENCE_MODEL = "gawf"
LABELS = {**MODEL_LABELS, **BASELINE_LABELS, REFERENCE_MODEL: "GaWF (strict)"}
COLORS = {**MODEL_COLORS, **BASELINE_COLORS}
MARKERS = {**MODEL_MARKERS, **BASELINE_MARKERS}
LINESTYLE = {
    model: ("--" if model in BASELINE_MODELS else "-")
    for model in (*MODEL_ORDER, *BASELINE_MODELS)
}
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
DEFAULT_BASELINE_RECOVERY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "data"
    / "analysis"
    / "fig1_target_switch_recovery_resetexcluded_6model_10seed_v4"
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
    parser.add_argument(
        "--baseline-recovery-root",
        type=Path,
        default=DEFAULT_BASELINE_RECOVERY_ROOT,
        help="Reset-excluded retained-model target-switch recovery export (one npz per seed).",
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


def _load_test_rows(
    test_csv: Path, model: str, expected_seeds: int = 10
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Read one retained model's reset-excluded test rows."""

    with test_csv.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("model") == model]
    if len(rows) != expected_seeds:
        raise RuntimeError(
            f"Expected {expected_seeds} {model} test rows in {test_csv}, got {len(rows)}"
        )
    rows.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    test = {
        row_key: np.asarray([float(row[f"{row_key}_acc"]) for row in rows], dtype=np.float64)
        for row_key in ROW_KEYS
    }
    return seeds, test


def _load_reference_histories(
    histories_root: Path, model: str, seeds: Sequence[int]
) -> dict[str, dict[int, np.ndarray]]:
    """Load one retained model's per-epoch validation-loss tracks for the given seeds."""

    histories: dict[str, dict[int, np.ndarray]] = {row_key: {} for row_key in ROW_KEYS}
    for seed in seeds:
        history_dir = histories_root / f"{model}-seed{seed:02d}"
        pickles = sorted(history_dir.glob("*.pkl"))
        if len(pickles) != 1:
            raise RuntimeError(f"Expected exactly one {model} history in {history_dir}")
        with pickles[0].open("rb") as stream:
            payload = pickle.load(stream)
        for row_key, track in HISTORY_TRACK.items():
            values = payload.get(track)
            if values is None:
                raise RuntimeError(f"{pickles[0]} has no {track} track")
            histories[row_key][seed] = np.asarray(values, dtype=np.float64)
    return histories


def _load_shuffle_curves(
    shuffle_root: Path, model: str, seeds: Sequence[int]
) -> tuple[np.ndarray, dict[str, dict[str, list[np.ndarray]]]] | None:
    """Load one model's switch-aligned shuffle export, or ``None`` when it was never run."""

    paths = [shuffle_root / f"{model}-seed{seed:02d}" / "ablation_metrics.json" for seed in seeds]
    if not any(path.is_file() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"Incomplete shuffle export for {model} under {shuffle_root}")

    offsets: np.ndarray | None = None
    curves: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {row_key: [] for row_key in ROW_KEYS} for condition in CONDITION_ORDER
    }
    for path in paths:
        shuffle = json.loads(path.read_text(encoding="utf-8"))
        if shuffle.get("exclude_window_initial_frame") is not True:
            raise RuntimeError(f"{model} shuffle export includes the reset frame: {path}")
        current = np.asarray(shuffle["switch_offsets"], dtype=np.int64)
        if offsets is None:
            offsets = current
        elif not np.array_equal(offsets, current):
            raise RuntimeError(f"{model} offset grid differs from previous seeds: {path}")
        for condition in CONDITION_ORDER:
            for row_key in ROW_KEYS:
                curves[condition][row_key].append(
                    np.asarray(
                        shuffle["conditions"][condition][f"switch_{row_key}_acc"],
                        dtype=np.float64,
                    )
                )
    if offsets is None:
        raise RuntimeError(f"No usable offset grid for {model} under {shuffle_root}")
    return offsets, curves


def load_baseline_models(
    test_csv: Path,
    histories_root: Path,
    shuffle_root: Path,
    recovery_root: Path | None = None,
    models: Sequence[str] = BASELINE_MODELS,
    expected_offsets: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Load every retained reference model on the feedback-control evaluation protocol.

    Recovery curves come from the reset-excluded six-model export when it covers the model, and fall
    back to the shuffle export's baseline condition otherwise. The feedback-shuffle ablation is only
    loaded for models that actually have that export; ``ablation`` is ``None`` for the open-loop
    baselines that never ran it.
    """

    recovery_offsets: np.ndarray | None = None
    recovery_curves: dict[str, dict[str, np.ndarray]] = {}
    if recovery_root is not None and recovery_root.is_dir():
        recovery_offsets, recovery_curves = load_recovery_curves(recovery_root)

    payload: dict[str, dict[str, Any]] = {}
    for model in models:
        seeds, test = _load_test_rows(test_csv, model)
        histories = _load_reference_histories(histories_root, model, seeds)
        shuffle = None if shuffle_root is None else _load_shuffle_curves(shuffle_root, model, seeds)

        # Prefer the model's own shuffle export so a series already drawn from it keeps its exact
        # numbers; the canonical six-model recovery export only fills in models that lack one.
        if shuffle is not None:
            grid = shuffle[0]
            recovery = {row_key: _mean_sem(shuffle[1]["baseline"][row_key]) for row_key in ROW_KEYS}
        elif model in recovery_curves:
            if recovery_offsets is None:
                raise RuntimeError(f"Missing recovery offset grid for {model}")
            grid = recovery_offsets
            recovery = {
                row_key: _mean_sem(list(np.asarray(recovery_curves[model][row_key])))
                for row_key in ROW_KEYS
            }
        else:
            raise RuntimeError(
                f"No recovery export for {model}: pass --baseline-recovery-root or a shuffle export"
            )
        if expected_offsets is not None and not np.array_equal(
            np.asarray(grid, dtype=np.int64), np.asarray(expected_offsets, dtype=np.int64)
        ):
            raise RuntimeError(
                f"{model} recovery offsets do not match the feedback-control offset grid"
            )

        ablation = None
        if shuffle is not None:
            ablation = {
                condition: {
                    row_key: np.asarray(shuffle[1][condition][row_key], dtype=np.float64).mean(
                        axis=1
                    )
                    for row_key in ROW_KEYS
                }
                for condition in ABLATION_CONDITIONS
            }
        payload[model] = {
            "seeds": seeds,
            "test": test,
            "histories": histories,
            "recovery": recovery,
            "ablation": ablation,
        }
    return payload


def load_reference_gawf(
    test_csv: Path, histories_root: Path, shuffle_root: Path
) -> dict[str, Any]:
    """Load the retained strict-GaWF reference under the feedback-control protocol."""

    return load_baseline_models(
        test_csv,
        histories_root,
        shuffle_root,
        recovery_root=None,
        models=(REFERENCE_MODEL,),
    )[REFERENCE_MODEL]


def build_panels(
    curves: dict[str, Any],
    grouped: dict[str, Any],
    histories: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one unified panel payload for every drawn series."""

    if reference is None:
        baseline_models: dict[str, dict[str, Any]] = {}
    elif "test" in reference:
        baseline_models = {REFERENCE_MODEL: reference}
    else:
        baseline_models = {model: payload for model, payload in reference.items()}
    models = [*MODEL_ORDER, *baseline_models]
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
    for model, payload in baseline_models.items():
        panels["test"][model] = payload["test"]
        panels["histories"][model] = payload["histories"]
        panels["recovery"][model] = payload["recovery"]
        panels["seeds"][model] = len(payload["seeds"])
        if payload.get("ablation") is not None:
            panels["ablation"][model] = payload["ablation"]
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
        is_baseline = model in BASELINE_MODELS
        axis.bar(
            index,
            mean,
            width=0.66,
            color=COLORS[model],
            alpha=0.9,
            zorder=2,
            hatch="//" if is_baseline else None,
            edgecolor="white" if is_baseline else None,
        )
        axis.errorbar(index, mean, yerr=sem, color="#333333", capsize=2.6, linewidth=0.9, zorder=4)
        jitter = np.linspace(-0.16, 0.16, samples.size)
        axis.scatter(index + jitter, samples, s=9, color="#555555", alpha=0.75, zorder=3)
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(
        [LABELS[model] for model in models],
        rotation=32 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=6.8 if show_xticks else 8,
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
            linewidth=1.2 if model in BASELINE_MODELS else 1.7,
            linestyle=LINESTYLE[model],
            zorder=3,
        )
        if np.any(sem > 0):
            axis.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                color=COLORS[model],
                alpha=0.12 if model in BASELINE_MODELS else 0.20,
                zorder=2,
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
            markersize=2.8 if model in BASELINE_MODELS else 3.4,
            linewidth=1.1 if model in BASELINE_MODELS else 1.5,
            linestyle=LINESTYLE[model],
            markerfacecolor="white" if model in BASELINE_MODELS else COLORS[model],
            markeredgewidth=0.9,
            zorder=3,
        )
        if np.any(sem > 0):
            axis.fill_between(
                offsets,
                mean - sem,
                mean + sem,
                color=COLORS[model],
                alpha=0.10 if model in BASELINE_MODELS else 0.18,
                zorder=2,
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
    """Draw the baseline, shuffled-digit, and shuffled-sector bars per series.

    Models without a feedback-shuffle export keep their slot and are labelled as not run so the
    column stays aligned with the other three panels.
    """

    width = 0.8 / (len(models) + 2) * 3.0 if len(models) > 4 else 0.26
    overall: list[float] = []
    unrun: list[int] = []
    for model_index, model in enumerate(models):
        if model not in ablation:
            unrun.append(model_index)
            continue
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
    bottom = min(overall) - 0.45 * span
    axis.set_ylim(bottom, max(overall) + 0.15 * span)
    for model_index in unrun:
        axis.text(
            model_index,
            bottom + 0.06 * span,
            "not\nrun",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color="#8A8A8A",
            zorder=5,
        )
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(
        [LABELS[model] for model in models],
        rotation=32 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=6.8 if show_xticks else 8,
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
        figure, axes = plt.subplots(2, 4, figsize=(15.0, 7.0))
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
            left=0.055, right=0.992, bottom=0.215, top=0.795, hspace=0.40, wspace=0.33
        )
        row_centers = [
            float(np.mean([axis.get_position().y0 + axis.get_position().height / 2 for axis in row]))
            for row in axes
        ]
        for row_key, center in zip(ROW_KEYS, row_centers):
            figure.text(0.012, center, ROW_LABELS[row_key], rotation=90, va="center", fontsize=10)
        control_models = [model for model in models if model not in BASELINE_MODELS]
        baseline_models = [model for model in models if model in BASELINE_MODELS]
        for group, anchor in ((control_models, 0.997), (baseline_models, 0.958)):
            if not group:
                continue
            group_handles = [
                Line2D(
                    [0],
                    [0],
                    color=COLORS[model],
                    linewidth=1.8 if model in BASELINE_MODELS else 2.5,
                    marker=MARKERS[model],
                    markersize=4,
                    linestyle=LINESTYLE[model],
                    markerfacecolor="white" if model in BASELINE_MODELS else COLORS[model],
                )
                for model in group
            ]
            figure.legend(
                group_handles,
                [LABELS[model] for model in group],
                loc="upper center",
                bbox_to_anchor=(0.5, anchor),
                ncol=len(group),
                frameon=False,
                handlelength=2.0,
                columnspacing=1.4,
                fontsize=8,
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
        seed_counts = sorted({panels["seeds"][model] for model in models})
        seed_text = " and ".join(str(count) for count in seed_counts)
        missing = [model for model in models if model not in panels["ablation"]]
        caption_lines = [
            "CM-MNIST behaviour comparison, ten models, 150 epochs,"
            f" {seed_text} seeds per model. A: reset-excluded test accuracy;"
            " B: validation loss on the 40h split.",
            "C: joint-switch-balanced 10-digit test, reset at every foreground switch, offset 0"
            " excluded; D: sequence-512 reset-excluded shuffle conditions.",
            "Column C sources: the four controls and strict GaWF come from this campaign's"
            " shuffle export; RNN, LSTM, GRU, Mamba and S5 come from their canonical reset-excluded"
            " recovery export, whose switch-window definition differs slightly.",
            "Solid with filled markers: the four feedback-control models of this campaign. Dashed"
            " with open markers and hatched bars: the six retained best-six Clutter models,"
            " including strict GaWF, which is not a result of this campaign's Amarel array.",
        ]
        if missing:
            caption_lines.append(
                "Column D covers only"
                f" {', '.join(LABELS[model] for model in models if model in panels['ablation'])};"
                f" {', '.join(LABELS[model] for model in missing)} were never run with the"
                " feedback-shuffle protocol and are marked not run."
            )
        figure.text(
            0.5,
            0.004,
            "\n".join(caption_lines),
            ha="center",
            va="bottom",
            fontsize=6.4,
            linespacing=1.5,
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
            ablation_available = model in panels["ablation"]
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
            arrays[f"{row_key}__{model}__ablation_available"] = np.asarray(
                [int(ablation_available)], dtype=np.int64
            )
            for condition in ABLATION_CONDITIONS:
                arrays[f"{row_key}__{model}__{condition}__acc"] = (
                    np.asarray(panels["ablation"][model][condition][row_key], dtype=np.float32)
                    if ablation_available
                    else np.full((1,), np.nan, dtype=np.float32)
                )
            test_values = np.asarray(panels["test"][model][row_key], dtype=np.float64)
            row: dict[str, Any] = {
                "readout": row_key,
                "model": model,
                "n_seeds": panels["seeds"][model],
                "ablation_available": int(ablation_available),
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
                if ablation_available:
                    values = np.asarray(
                        panels["ablation"][model][condition][row_key], dtype=np.float64
                    )
                    row[f"{condition}_acc_mean"] = float(values.mean())
                    row[f"{condition}_acc_sem"] = (
                        float(values.std(ddof=1) / np.sqrt(values.size))
                        if values.size > 1
                        else 0.0
                    )
                else:
                    row[f"{condition}_acc_mean"] = float("nan")
                    row[f"{condition}_acc_sem"] = float("nan")
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
        if isinstance(value, (int, float)) and not (isinstance(value, float) and np.isnan(value))
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
    baseline_models: dict[str, dict[str, Any]] = {}
    if not args.without_reference:
        baseline_models = load_baseline_models(
            args.reference_test_csv,
            args.reference_histories_root,
            args.reference_shuffle_root,
            recovery_root=args.baseline_recovery_root,
            expected_offsets=np.asarray(curves["offsets"], dtype=np.int64),
        )
    panels = build_panels(curves, grouped, histories, baseline_models or None)
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
            extra = " shuffle-ablation not run"
            if model in panels["ablation"]:
                baseline = np.asarray(
                    panels["ablation"][model]["baseline"][row_key], dtype=np.float64
                )
                shuffled = np.asarray(
                    panels["ablation"][model]["shuffle_digit"][row_key], dtype=np.float64
                )
                extra = f" shuffle-digit delta {baseline.mean() - shuffled.mean():+.2f}"
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
