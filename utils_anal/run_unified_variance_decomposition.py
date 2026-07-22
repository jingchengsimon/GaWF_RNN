"""Run the unified balanced variance decomposition from saved trial representations.

The input JSON points to saved mmap-friendly arrays and, optionally, a GaWF trajectory NPZ.
No model, dataset, checkpoint, or plotting-time inference is loaded.  Outputs are one tidy CSV,
one per-object NPZ containing every repeated per-unit fraction, one four-cell figure per object,
one seven-object summary figure, one compact four-object aggregate figure, a consistency report,
and a provenance manifest.

Minimal input manifest::

    {
      "trajectory_npz": "/path/gawf_gate_trajectory.npz",
      "objects": {
        "encoder_activation": {"path": "/path/encoder_activation.npy"},
        "input_gate": {"path": "/path/input_gate.npy"},
        "hidden_state": {"path": "/path/hidden_state.npy"},
        "recurrent_gate": {"path": "/path/recurrent_gate.npy"}
      }
    }

The trajectory supplies only labels, the 19-element feedback vector, and static input/recurrent
weights. Gates are always read from saved arrays and are never regenerated from ``U/V``.
Effective weights may also be supplied as saved arrays; otherwise they are evaluated blockwise as
saved gate times static weight. Array files may keep multiple trial or representation axes as long
as their total size is exactly ``n_trials * n_units`` in C order. Large synapse arrays must be
uncompressed ``.npy`` files so they can be memory-mapped.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_anal.anal_paths import output_dir
from utils_anal.variance_decomposition import (
    CM_FACTORS,
    TRIAL_FACTORS,
    RepeatedDecomposition,
    StreamingMoments,
    balanced_subsample_indices,
    decompose_repeated_blocks,
    unbalanced_condition_mean_bridge,
)
from utils.publication_paths import publication_figures_dir


CATEGORY = "D_variance_decomposition"
SCRIPT_NAME = "unified"
OBJECT_UNITS = {
    "encoder_activation": 1152,
    "input_gate": 256 * 1152,
    "effective_input_weight": 256 * 1152,
    "hidden_state": 256,
    "recurrent_gate": 256 * 256,
    "effective_recurrent_weight": 256 * 256,
    "feedback_vector": 19,
}
OBJECT_ORDER = tuple(OBJECT_UNITS)
SYNAPSE_OBJECTS = {
    "input_gate",
    "effective_input_weight",
    "recurrent_gate",
    "effective_recurrent_weight",
}
PUBLISHED = {
    ("input_gate", "aggregate_cm"): (81.38, 11.53, 7.09),
    ("recurrent_gate", "aggregate_cm"): (31.05, 61.59, 7.36),
    ("input_gate", "aggregate_trial"): (36.77, 5.21, 3.20, 54.82),
    ("recurrent_gate", "aggregate_trial"): (12.90, 25.59, 3.06, 58.46),
    ("encoder_activation", "aggregate_cm"): (55.91, 4.54, 39.55),
}
HISTORICAL_HIDDEN_CM = {
    "sector": 0.30080200072478297,
    "digit": 0.5229833089026543,
    "interaction": 0.17621469037256268,
}


class BlockSource(Protocol):
    """Read selected trials and a contiguous flattened unit slice."""

    num_trials: int
    num_units: int

    def read(self, trial_indices: np.ndarray, unit_slice: slice) -> np.ndarray:
        """Return a two-dimensional trial-by-unit block."""


class ArraySource:
    """Memory-mapped source for an existing saved representation."""

    def __init__(self, path: Path, num_units: int, num_trials: int) -> None:
        if path.suffix != ".npy":
            raise ValueError(f"Saved representation must be mmap .npy, got {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim < 2 or array.size != num_trials * num_units:
            raise ValueError(
                f"{path} must contain {num_trials} trials x {num_units} units, "
                f"observed shape {array.shape}"
            )
        self.array = array.reshape(num_trials, num_units)
        self.num_trials, self.num_units = self.array.shape

    def read(self, trial_indices: np.ndarray, unit_slice: slice) -> np.ndarray:
        return np.asarray(self.array[trial_indices, unit_slice])


class WeightedSource:
    """Multiply one saved gate block by matching saved static weights."""

    def __init__(self, gate_source: BlockSource, weight: np.ndarray) -> None:
        flattened = np.asarray(weight, dtype=np.float32).reshape(-1)
        if flattened.size != gate_source.num_units:
            raise ValueError(
                f"Static weight has {flattened.size} entries; expected {gate_source.num_units}"
            )
        self.gate_source = gate_source
        self.weight = flattened
        self.num_trials = gate_source.num_trials
        self.num_units = gate_source.num_units

    def read(self, trial_indices: np.ndarray, unit_slice: slice) -> np.ndarray:
        return self.gate_source.read(trial_indices, unit_slice) * self.weight[unit_slice][None, :]


class FeedbackSource:
    """Read the saved 19-element feedback/readout vector."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.feedback = np.asarray(arrays["feedback"], dtype=np.float32).reshape(-1, 19)
        self.num_trials, self.num_units = self.feedback.shape

    def read(self, trial_indices: np.ndarray, unit_slice: slice) -> np.ndarray:
        return self.feedback[trial_indices, unit_slice]


def parse_args() -> argparse.Namespace:
    """Parse the saved-data runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--repeats", type=int, choices=(20,), default=20)
    parser.add_argument("--unit_block_size", type=int, default=4096)
    parser.add_argument("--trial_batch_size", type=int, default=32)
    parser.add_argument("--memory_budget_gib", type=float, default=2.0)
    parser.add_argument(
        "--publication_fig_dir",
        type=Path,
        default=None,
        help=(
            "Official PDF destination. Defaults to AIM3_PUBLICATION_FIGURES_DIR or the local "
            "6-Writing/Aim3/Figures sibling tree when available."
        ),
    )
    return parser.parse_args()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"feedback", "labels", "weight_ih", "weight_hh"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Trajectory {path} is missing arrays: {missing}")
        return {key: np.asarray(archive[key]) for key in required}


def _load_inputs(path: Path) -> tuple[np.ndarray, dict[str, BlockSource], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    objects = payload.get("objects", {})
    trajectory = None
    if "trajectory_npz" in payload:
        trajectory = _load_trajectory(_resolve(base, payload["trajectory_npz"]))
        labels = np.asarray(trajectory["labels"], dtype=np.int64).reshape(-1, 2)
    elif "labels" in payload:
        labels_path = _resolve(base, payload["labels"])
        labels = np.asarray(np.load(labels_path, mmap_mode="r"), dtype=np.int64).reshape(-1, 2)
    else:
        raise ValueError("input manifest requires trajectory_npz or labels")

    sources: dict[str, BlockSource] = {}
    for object_name in ("encoder_activation", "input_gate", "hidden_state", "recurrent_gate"):
        spec = objects.get(object_name)
        if not spec or "path" not in spec:
            raise ValueError(f"input manifest is missing objects.{object_name}.path")
        sources[object_name] = ArraySource(
            _resolve(base, spec["path"]), OBJECT_UNITS[object_name], labels.shape[0]
        )
    for effective_name, gate_name, weight_name in (
        ("effective_input_weight", "input_gate", "weight_ih"),
        ("effective_recurrent_weight", "recurrent_gate", "weight_hh"),
    ):
        spec = objects.get(effective_name)
        if spec and "path" in spec:
            sources[effective_name] = ArraySource(
                _resolve(base, spec["path"]), OBJECT_UNITS[effective_name], labels.shape[0]
            )
            continue
        if trajectory is not None:
            weight = trajectory[weight_name]
        else:
            weight_spec = payload.get("weights", {}).get(weight_name)
            if not weight_spec:
                raise ValueError(
                    f"input manifest requires objects.{effective_name}.path or "
                    f"weights.{weight_name}"
                )
            weight = np.load(_resolve(base, weight_spec), mmap_mode="r", allow_pickle=False)
        sources[effective_name] = WeightedSource(sources[gate_name], weight)

    feedback_spec = objects.get("feedback_vector")
    if feedback_spec and "path" in feedback_spec:
        sources["feedback_vector"] = ArraySource(
            _resolve(base, feedback_spec["path"]), OBJECT_UNITS["feedback_vector"], labels.shape[0]
        )
    elif trajectory is not None:
        sources["feedback_vector"] = FeedbackSource(trajectory)
    else:
        raise ValueError("input manifest requires objects.feedback_vector.path or trajectory_npz")
    for object_name, source in sources.items():
        if source.num_trials != labels.shape[0]:
            raise ValueError(
                f"{object_name} has {source.num_trials} trials but labels have {labels.shape[0]}"
            )
        if source.num_units != OBJECT_UNITS[object_name]:
            raise ValueError(
                f"{object_name} has {source.num_units} units, expected {OBJECT_UNITS[object_name]}"
            )
    return labels, sources, payload


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _per_unit_draw_mean(values: np.ndarray) -> np.ndarray:
    """Return one finite eta-squared value per unit after averaging repeated draws."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"per-unit values must be draws x units, observed shape {values.shape}")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.nansum(values, axis=0, dtype=np.float64)
    means = np.divide(
        sums,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    distribution = means[np.isfinite(means)]
    if distribution.size == 0:
        raise ValueError("per-unit values contain no finite draw-averaged units")
    return distribution


def _annotate_aggregate_means(
    axis: plt.Axes,
    bars: Any,
    means: np.ndarray,
    highs: np.ndarray,
) -> None:
    """Label aggregate bars with their repeated-draw mean percentages."""

    for bar, mean, high in zip(bars, means, highs):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(float(high) + 0.025, 1.055),
            f"{100.0 * float(mean):.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _save_object_npz(
    data_dir: Path,
    object_name: str,
    result: RepeatedDecomposition,
) -> Path:
    arrays: dict[str, np.ndarray] = {}
    for factor, values in result.aggregate_cm.items():
        arrays[f"aggregate_cm_{factor}"] = values.astype(np.float32, copy=False)
    for factor, values in result.aggregate_trial.items():
        arrays[f"aggregate_trial_{factor}"] = values.astype(np.float32, copy=False)
    for factor, values in result.per_unit_cm.items():
        arrays[f"per_unit_cm_{factor}"] = values.astype(np.float32, copy=False)
    for factor, values in result.per_unit_trial.items():
        arrays[f"per_unit_trial_{factor}"] = values.astype(np.float32, copy=False)
    for factor, values in result.unweighted_per_unit_mean_cm.items():
        arrays[f"unweighted_per_unit_mean_cm_{factor}"] = values.astype(np.float32, copy=False)
    for factor, values in result.unweighted_per_unit_mean_trial.items():
        arrays[f"unweighted_per_unit_mean_trial_{factor}"] = values.astype(np.float32, copy=False)
    for name, values in result.consistency.items():
        arrays[f"consistency_{name}"] = values.astype(np.float64, copy=False)
    destination = data_dir / f"{object_name}_per_unit_distributions.npz"
    np.savez_compressed(destination, **arrays)
    return destination


def _tidy_rows(
    object_name: str,
    result: RepeatedDecomposition,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    variants = (
        ("condition_mean_aggregate", result.aggregate_cm, CM_FACTORS),
        ("trial_level_aggregate", result.aggregate_trial, TRIAL_FACTORS),
        ("condition_mean_per_unit", result.unweighted_per_unit_mean_cm, CM_FACTORS),
        ("trial_level_per_unit", result.unweighted_per_unit_mean_trial, TRIAL_FACTORS),
    )
    for variant, values_by_factor, factors in variants:
        for factor in factors:
            mean, low, high = _mean_ci(values_by_factor[factor])
            rows.append(
                {
                    "object": object_name,
                    "variant": variant,
                    "factor": factor,
                    "value": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def _summary_only(result: RepeatedDecomposition) -> RepeatedDecomposition:
    """Drop large per-unit distributions after they have been saved and plotted."""

    return RepeatedDecomposition(
        aggregate_cm=result.aggregate_cm,
        aggregate_trial=result.aggregate_trial,
        per_unit_cm={},
        per_unit_trial={},
        unweighted_per_unit_mean_cm=result.unweighted_per_unit_mean_cm,
        unweighted_per_unit_mean_trial=result.unweighted_per_unit_mean_trial,
        consistency=result.consistency,
    )


def _plot_object(
    figure_dir: Path,
    object_name: str,
    result: RepeatedDecomposition,
) -> Path:
    colors = {
        "sector": "#4477AA",
        "digit": "#EE6677",
        "interaction": "#228833",
        "residual": "#BBBBBB",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    aggregate_cells = (
        (axes[0, 0], "Condition-mean aggregate", result.aggregate_cm, CM_FACTORS),
        (axes[0, 1], "Trial-level aggregate", result.aggregate_trial, TRIAL_FACTORS),
    )
    for axis, title, values_by_factor, factors in aggregate_cells:
        means, lows, highs = zip(*(_mean_ci(values_by_factor[factor]) for factor in factors))
        means_array = np.asarray(means)
        highs_array = np.asarray(highs)
        x = np.arange(len(factors))
        errors = np.asarray([means_array - lows, highs_array - means_array])
        bars = axis.bar(
            x,
            means_array,
            color=[colors[factor] for factor in factors],
            yerr=errors,
            capsize=3,
        )
        _annotate_aggregate_means(axis, bars, means_array, highs_array)
        axis.set_xticks(x, factors, rotation=20)
        axis.set_ylim(0.0, 1.08)
        axis.set_title(title)
        axis.set_ylabel(r"variance fraction ($\eta^2$)")

    per_unit_cells = (
        (axes[1, 0], "Condition-mean per-unit distribution", result.per_unit_cm, CM_FACTORS),
        (axes[1, 1], "Trial-level per-unit distribution", result.per_unit_trial, TRIAL_FACTORS),
    )
    for axis, title, values_by_factor, factors in per_unit_cells:
        distributions = [_per_unit_draw_mean(values_by_factor[factor]) for factor in factors]
        x = np.arange(len(factors))
        violins = axis.violinplot(
            distributions,
            positions=x,
            widths=0.75,
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )
        for body, factor in zip(violins["bodies"], factors):
            body.set_facecolor(colors[factor])
            body.set_edgecolor("black")
            body.set_alpha(0.8)
        violins["cmeans"].set_color("black")
        violins["cmeans"].set_linewidth(1.2)
        axis.set_xticks(x, factors, rotation=20)
        axis.set_ylim(0.0, 1.08)
        axis.set_title(title)
        axis.set_ylabel(r"variance fraction ($\eta^2$)")
    unit_note = (
        "Unit axis indexes SYNAPSES, not neurons; per-synapse eta^2 answers a different "
        "question from per-neuron eta^2."
        if object_name in SYNAPSE_OBJECTS
        else f"Unit axis indexes {OBJECT_UNITS[object_name]} representation units."
    )
    fig.suptitle(f"{object_name}\n{unit_note}", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Violins show the distribution across units after averaging each unit across "
        "the fixed-seed balanced draws; black lines mark distribution means.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    destination = figure_dir / f"{object_name}_four_cells.png"
    fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return destination


def _plot_summary(
    figure_dir: Path, results: dict[str, RepeatedDecomposition], *, repeats: int
) -> Path:
    fig, axis = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(OBJECT_ORDER))
    bottom = np.zeros(len(OBJECT_ORDER), dtype=np.float64)
    colors = {"sector": "#4477AA", "digit": "#EE6677", "interaction": "#228833"}
    for factor in CM_FACTORS:
        values = np.asarray([results[name].aggregate_cm[factor].mean() for name in OBJECT_ORDER])
        bars = axis.bar(x, values, bottom=bottom, label=factor, color=colors[factor])
        for bar, value, base in zip(bars, values, bottom):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                base + value / 2.0,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
        bottom += values
    axis.set_xticks(x, [name.replace("_", "\n") for name in OBJECT_ORDER], fontsize=8)
    axis.set_ylabel(r"condition-mean aggregate variance fraction ($\eta^2$)")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, ncol=3)
    axis.set_title(f"Unified balanced GaWF variance decomposition ({repeats} fixed-seed draws)")
    fig.text(
        0.5,
        0.01,
        "Input/recurrent gate and effective-weight unit axes index SYNAPSES, not neurons; "
        "per-synapse eta^2 asks a different question from per-neuron eta^2.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    destination = figure_dir / "all_objects_condition_mean_aggregate.png"
    fig.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return destination


def _plot_compact_aggregate(
    figure_dir: Path,
    results: dict[str, RepeatedDecomposition],
    publication_fig_dir: Path | None = None,
) -> Path:
    """Plot the poster-style condition-mean aggregate summary for four core objects."""

    object_rows = (
        (("input_gate", "Input gate"), ("recurrent_gate", "Recurrent gate")),
        (
            ("encoder_activation", "Encoder\nactivation"),
            ("hidden_state", "Hidden\nactivation"),
        ),
    )
    factors = CM_FACTORS
    # This high-contrast navy/coral/mustard palette is distinct from the model and gate palettes
    # used by the adjacent 2-by-3 and 1-by-3 publication panels.
    colors = {"sector": "#264653", "digit": "#E76F51", "interaction": "#E9C46A"}
    # The row-wide axis contains two logical quadrants.  This data width gives each bar the same
    # physical width-to-height ratio as a bar in the GRU afferent-gate panel of the 1-by-3 figure.
    bar_width = 0.095
    # Three bars occupy each group; leave exactly 1.5 bar widths between adjacent group edges.
    category_centers = np.arange(2, dtype=np.float64) * (3.0 + 1.5) * bar_width
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(5.05, 8.2), sharey=True)
        for axis, objects in zip(axes, object_rows):
            for factor_index, factor in enumerate(factors):
                statistics = [
                    _mean_ci(results[object_name].aggregate_cm[factor])
                    for object_name, _ in objects
                ]
                means = 100.0 * np.asarray([item[0] for item in statistics])
                lows = 100.0 * np.asarray([item[1] for item in statistics])
                highs = 100.0 * np.asarray([item[2] for item in statistics])
                positions = category_centers + (factor_index - 1) * bar_width
                axis.bar(
                    positions,
                    means,
                    width=bar_width,
                    color=colors[factor],
                    yerr=np.asarray([means - lows, highs - means]),
                    capsize=2.5,
                    label=factor.title(),
                    error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#333333"},
                )
            axis.set_xticks(category_centers, [label for _, label in objects])
            axis.set_ylim(0.0, 105.0)
            axis.set_yticks(np.arange(0.0, 100.1, 20.0))
            axis.set_axisbelow(True)
            axis.grid(axis="y", linewidth=0.7, alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        fig.subplots_adjust(left=0.24, right=0.865, bottom=0.08, top=0.90, hspace=0.22)
        row_center = np.mean(
            [axis.get_position().y0 + axis.get_position().height / 2 for axis in axes]
        )
        fig.text(
            0.108,
            row_center,
            "Explained variance (%)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=16,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            # Offset the legend's first patch so its center sits on the shared y-axis.
            bbox_to_anchor=(0.205, 0.937),
            ncol=3,
            frameon=False,
            handlelength=1.15,
            handleheight=0.85,
            handletextpad=0.22,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
        destination = figure_dir / "core_objects_aggregate_2x2.png"
        fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.04)
        if publication_fig_dir is not None:
            fig.savefig(
                publication_fig_dir / "core_objects_aggregate_2x2.pdf",
                bbox_inches="tight",
                pad_inches=0.04,
            )
        plt.close(fig)
    return destination


def _hidden_unbalanced_bridge(
    source: BlockSource,
    labels: np.ndarray,
    *,
    batch_size: int,
    memory_budget_bytes: int,
) -> dict[str, float]:
    accumulator = StreamingMoments(source.num_units, memory_budget_bytes=memory_budget_bytes)
    all_trials = np.arange(labels.shape[0], dtype=np.int64)
    for start in range(0, labels.shape[0], batch_size):
        indices = all_trials[start : start + batch_size]
        accumulator.update(source.read(indices, slice(0, source.num_units)), labels[indices])
    return unbalanced_condition_mean_bridge(
        accumulator.total_sum, accumulator.cell_sum, accumulator.cell_count
    )


def _consistency_report(
    results: dict[str, RepeatedDecomposition],
    balance: Any,
    hidden_bridge: dict[str, float],
) -> str:
    lines = [
        "UNIFIED VARIANCE DECOMPOSITION CONSISTENCY CHECKS",
        f"balanced n per cell: {balance.n_per_cell}",
        f"trials retained per draw: {balance.trials_retained_per_draw}",
        f"trials discarded per draw: {balance.trials_discarded_per_draw}",
        f"repeats: {balance.repeats}; seed: {balance.seed}",
        "",
    ]
    for object_name in OBJECT_ORDER:
        result = results[object_name]
        lines.append(f"[{object_name}]")
        for name, values in result.consistency.items():
            lines.append(f"{name}: max={float(np.max(np.abs(values))):.12g}")
        for factor in CM_FACTORS:
            aggregate = float(result.aggregate_cm[factor].mean())
            unweighted = float(result.unweighted_per_unit_mean_cm[factor].mean())
            lines.append(
                f"cm {factor}: aggregate={aggregate:.8f}; "
                f"unweighted_per_unit_mean={unweighted:.8f}; gap={unweighted - aggregate:+.8f}"
            )
        lines.append("")

    lines += ["REGRESSION AGAINST PUBLISHED NUMBERS"]
    for (object_name, variant), old_values in PUBLISHED.items():
        result = results[object_name]
        values_by_factor = getattr(result, variant)
        factors = CM_FACTORS if variant == "aggregate_cm" else TRIAL_FACTORS
        for factor, old in zip(factors, old_values):
            mean, low, high = _mean_ci(values_by_factor[factor] * 100.0)
            outside = old < low or old > high
            explanation = (
                "UNEXPLAINED outside subsample interval; run fails until protocol difference is "
                "resolved"
                if outside
                else "within subsample interval"
            )
            lines.append(
                f"{object_name} {variant} {factor}: new={mean:.4f}% "
                f"CI=[{low:.4f}, {high:.4f}] old={old:.4f}% "
                f"diff={mean - old:+.4f} pp; {explanation}"
            )

    lines += ["", "HIDDEN STATE BALANCING BRIDGE"]
    hidden = results["hidden_state"]
    for factor in CM_FACTORS:
        balanced = float(hidden.aggregate_cm[factor].mean())
        bridge = hidden_bridge[factor]
        old = HISTORICAL_HIDDEN_CM[factor]
        lines.append(
            f"{factor}: existing={old:.6f}; unbalanced_bridge={bridge:.6f}; "
            f"balanced={balanced:.6f}; balancing_shift={balanced - bridge:+.6f}; "
            f"remaining_bridge_gap={bridge - old:+.6f}"
        )
    lines.append(f"unbalanced bridge fraction sum={hidden_bridge['sum']:.8f}")
    return "\n".join(lines) + "\n"


def _save_trace_arrays(
    data_dir: Path,
    results: dict[str, RepeatedDecomposition],
    balance: Any,
    hidden_bridge: dict[str, float],
) -> int:
    """Save every balance, bridge, and regression number printed in the text report."""

    np.savez_compressed(
        data_dir / "balance_and_hidden_bridge.npz",
        n_per_cell=np.asarray(balance.n_per_cell, dtype=np.int64),
        total_trials_available=np.asarray(balance.total_trials_available, dtype=np.int64),
        trials_retained_per_draw=np.asarray(balance.trials_retained_per_draw, dtype=np.int64),
        trials_discarded_per_draw=np.asarray(balance.trials_discarded_per_draw, dtype=np.int64),
        repeats=np.asarray(balance.repeats, dtype=np.int64),
        seed=np.asarray(balance.seed, dtype=np.int64),
        hidden_unbalanced_sector=np.asarray(hidden_bridge["sector"], dtype=np.float64),
        hidden_unbalanced_digit=np.asarray(hidden_bridge["digit"], dtype=np.float64),
        hidden_unbalanced_interaction=np.asarray(hidden_bridge["interaction"], dtype=np.float64),
        hidden_unbalanced_sum=np.asarray(hidden_bridge["sum"], dtype=np.float64),
        hidden_existing_sector=np.asarray(HISTORICAL_HIDDEN_CM["sector"], dtype=np.float64),
        hidden_existing_digit=np.asarray(HISTORICAL_HIDDEN_CM["digit"], dtype=np.float64),
        hidden_existing_interaction=np.asarray(
            HISTORICAL_HIDDEN_CM["interaction"], dtype=np.float64
        ),
    )
    regression_arrays: dict[str, np.ndarray] = {}
    unexplained = 0
    for (object_name, variant), old_values in PUBLISHED.items():
        result = results[object_name]
        factors = CM_FACTORS if variant == "aggregate_cm" else TRIAL_FACTORS
        values_by_factor = getattr(result, variant)
        for factor, old in zip(factors, old_values):
            values = values_by_factor[factor] * 100.0
            mean, low, high = _mean_ci(values)
            stem = f"{object_name}_{variant}_{factor}"
            regression_arrays[f"{stem}_draws"] = values.astype(np.float64)
            regression_arrays[f"{stem}_old"] = np.asarray(old, dtype=np.float64)
            regression_arrays[f"{stem}_mean"] = np.asarray(mean, dtype=np.float64)
            regression_arrays[f"{stem}_ci_low"] = np.asarray(low, dtype=np.float64)
            regression_arrays[f"{stem}_ci_high"] = np.asarray(high, dtype=np.float64)
            regression_arrays[f"{stem}_difference"] = np.asarray(mean - old, dtype=np.float64)
            if old < low or old > high:
                unexplained += 1
    np.savez_compressed(data_dir / "published_regression_trace.npz", **regression_arrays)
    return unexplained


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_index(index_root: Path) -> None:
    index_path = index_root / "INDEX.md"
    if index_path.is_file():
        return
    lines = [
        "# Analysis output index",
        "",
        "Analysis figures and data are split into parallel `results/anal_figs/` and "
        "`results/anal_data/` trees. Figures are flat within each category; data and its "
        "provenance `manifest.json` remain grouped by script in the latter, "
        "while figures are kept at category level.",
        "",
        "## Categories",
        "",
        "- `A_raw_gate`: raw gates without condition labels",
        "- `B_gate_by_context`: raw gates partitioned by condition",
        "- `C_delta_gate`: per-synapse grand-mean-subtracted gates",
        "- `D_variance_decomposition`: variance apportioned to factors",
        "- `E_relevance_alignment`: activation-derived relevance/alignment",
        "- `F_timing`: switch-frame or event-latency analyses",
        "- `G_behaviour`: task performance without gate internals",
        "- `H_controls`: confound, convergence, and invariance controls",
        "",
        "## Unified decomposition",
        "",
        "`results/anal_figs/D_variance_decomposition/` and the parallel unified data directory "
        "contain all seven representations, four decomposition cells, repeated-draw intervals, "
        "per-unit distributions, and consistency checks. Gate unit axes index synapses, not "
        "neurons.",
        "",
    ]
    index_root.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all seven objects through the same balanced decomposition module."""

    args = parse_args()
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        raise ValueError("memory_budget_gib must be a finite positive value")
    memory_budget_bytes = int(args.memory_budget_gib * 1024**3)
    publication_dir = publication_figures_dir(args.publication_fig_dir, create=True)
    labels, sources, input_payload = _load_inputs(args.input_manifest.resolve())
    draws, balance = balanced_subsample_indices(
        labels,
        repeats=args.repeats,
        seed=args.seed,
        memory_budget_bytes=memory_budget_bytes,
    )
    print(
        f"Balanced design: n={balance.n_per_cell}, retained={balance.trials_retained_per_draw}, "
        f"discarded={balance.trials_discarded_per_draw}, repeats={balance.repeats}, "
        f"seed={balance.seed}",
        flush=True,
    )
    data_dir = output_dir(CATEGORY, SCRIPT_NAME, "data")
    figure_dir = output_dir(CATEGORY, SCRIPT_NAME, "figs")
    results: dict[str, RepeatedDecomposition] = {}
    tidy_rows: list[dict[str, str | float]] = []
    for object_name in OBJECT_ORDER:
        source = sources[object_name]
        print(f"Decomposing {object_name} ({source.num_units} units)...", flush=True)
        result = decompose_repeated_blocks(
            source.read,
            labels,
            draws,
            num_units=source.num_units,
            unit_block_size=args.unit_block_size,
            trial_batch_size=args.trial_batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )
        _save_object_npz(data_dir, object_name, result)
        _plot_object(figure_dir, object_name, result)
        tidy_rows.extend(_tidy_rows(object_name, result))
        results[object_name] = _summary_only(result)
        del result
    _plot_summary(figure_dir, results, repeats=balance.repeats)
    _plot_compact_aggregate(figure_dir, results, publication_dir)

    csv_path = data_dir / "unified_variance_decomposition.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["object", "variant", "factor", "value", "ci_low", "ci_high"],
        )
        writer.writeheader()
        writer.writerows(tidy_rows)

    hidden_bridge = _hidden_unbalanced_bridge(
        sources["hidden_state"],
        labels,
        batch_size=args.trial_batch_size,
        memory_budget_bytes=memory_budget_bytes,
    )
    report = _consistency_report(results, balance, hidden_bridge)
    print(report, end="")
    (data_dir / "consistency_checks.txt").write_text(report, encoding="utf-8")
    unexplained = _save_trace_arrays(data_dir, results, balance, hidden_bridge)

    key_results = {
        f"{object_name}.{variant}.{factor}": float(values.mean())
        for object_name, result in results.items()
        for variant, values_by_factor in (
            ("aggregate_cm", result.aggregate_cm),
            ("aggregate_trial", result.aggregate_trial),
        )
        for factor, values in values_by_factor.items()
    }
    data_files = sorted(
        item.relative_to(data_dir).as_posix()
        for item in data_dir.rglob("*")
        if item.is_file()
    )
    expected_figure_names = {
        *(f"{object_name}_four_cells.png" for object_name in OBJECT_ORDER),
        "all_objects_condition_mean_aggregate.png",
        "core_objects_aggregate_2x2.png",
    }
    figure_files = sorted(
        item.name
        for item in figure_dir.iterdir()
        if item.is_file() and item.name in expected_figure_names
    )
    files = [f"data/{path}" for path in data_files]
    files.extend(f"figs/{path}" for path in figure_files)
    manifest = {
        "script_path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "git_commit": _git_commit(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "category": CATEGORY,
        "data_root": str(data_dir.relative_to(PROJECT_ROOT)),
        "figure_root": str(figure_dir.relative_to(PROJECT_ROOT)),
        "data_files": data_files,
        "figure_files": figure_files,
        "files_written": files,
        "key_numerical_results": key_results,
        "balance": balance.__dict__,
        "input_manifest": str(args.input_manifest.resolve()),
        "input": input_payload,
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_index(PROJECT_ROOT / "results" / "anal_index")
    print(f"Saved unified data to {data_dir}; figures to {figure_dir}")
    if unexplained:
        raise RuntimeError(
            f"{unexplained} published regression values fall outside their repeated-draw "
            "intervals and remain unexplained; see consistency_checks.txt"
        )


if __name__ == "__main__":
    main()
