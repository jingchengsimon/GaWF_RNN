"""Plot partial no-tanh training trajectories against matched historical baselines.

Inputs are frozen per-seed metric prefixes from the two active DSW campaigns and complete
150-epoch pickle histories from the matched Amarel GaWF/RNN baselines. Outputs are a 2-by-2
Digit/Sector by Train/Validation figure, long-form epoch data, coverage, and provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.analysis.anal_paths import output_dir


CATEGORY = "G_behaviour"
SCRIPT_NAME = "partial_nonlinearity_training_curves_20260922"
COMPLETE_NAME = "nonlinearity_training_curves_complete_20260923"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEEDS = tuple(range(1, 11))
EPOCHS = 150
METRICS = (
    "train_acc_char",
    "val_acc_char",
    "train_acc_sector",
    "val_acc_sector",
)
PARTIAL_KEYS = {
    "train_acc_char": "train_acc_char",
    "val_acc_char": "val_acc_char",
    "train_acc_sector": "train_metric_pos",
    "val_acc_sector": "val_metric_pos",
}
BASELINE_KEYS = {
    "train_acc_char": "train_acc_char",
    "val_acc_char": "val_acc_char",
    "train_acc_sector": "train_acc_pos",
    "val_acc_sector": "val_acc_pos",
}
EXPECTED_PARTIAL_SEEDS = {
    "gawf_legacy_notanh": set(SEEDS),
    "rnn_inloop_notanh": {1, 2, 5, 6, 7, 8, 9, 10},
}
MODEL_ORDER = ("gawf", "gawf_legacy_notanh", "rnn", "rnn_inloop_notanh")
MODEL_LABELS = {
    "gawf": "GaWF baseline",
    "gawf_legacy_notanh": "GaWF no-tanh",
    "rnn": "RNN baseline",
    "rnn_inloop_notanh": "RNN in-loop no-tanh",
}
MODEL_COLORS = {
    "gawf": "#0072B2",
    "gawf_legacy_notanh": "#E69F00",
    "rnn": "#009E73",
    "rnn_inloop_notanh": "#CC79A7",
}


@dataclass(frozen=True)
class Run:
    """One seed's observed training trajectories and provenance."""

    model: str
    seed: int
    completed_epochs: int
    metrics: dict[str, np.ndarray]
    source: str


def parse_args() -> argparse.Namespace:
    """Parse the frozen input root and optional output overrides."""

    default_input = (
        PROJECT_ROOT
        / "results/data/analysis/G_behaviour"
        / SCRIPT_NAME
        / "inputs"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Require complete 150-epoch histories for all four models and all ten seeds.",
    )
    return parser.parse_args()


def _validated_array(value: Any, expected: int, source: Path, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,):
        raise ValueError(f"Expected {key} shape {(expected,)} in {source}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {key} from {source}")
    return array


def load_partial(input_root: Path, model: str) -> list[Run]:
    """Load one frozen partial campaign and validate its seed coverage."""

    model_root = input_root / model
    metadata_path = model_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_label") != model:
        raise ValueError(f"Model identity mismatch in {metadata_path}")
    records = {int(item["seed"]): item for item in metadata["records"]}
    if set(records) != EXPECTED_PARTIAL_SEEDS[model]:
        raise ValueError(
            f"Unexpected seed set for {model}: {sorted(records)}; "
            f"expected {sorted(EXPECTED_PARTIAL_SEEDS[model])}"
        )

    runs: list[Run] = []
    for seed, record in sorted(records.items()):
        path = model_root / f"seed{seed:02d}.npz"
        with np.load(path) as payload:
            completed = int(payload["completed_epochs"])
            if completed != int(record["completed_epochs"]) or not 1 <= completed <= EPOCHS:
                raise ValueError(f"Invalid completed epoch count in {path}: {completed}")
            metrics = {
                metric: _validated_array(payload[key], completed, path, key)
                for metric, key in PARTIAL_KEYS.items()
            }
        runs.append(
            Run(
                model=model,
                seed=seed,
                completed_epochs=completed,
                metrics=metrics,
                source=str(record["checkpoint"]),
            )
        )
    return runs


def load_baseline(input_root: Path, model: str) -> list[Run]:
    """Load a complete 10-seed historical baseline."""

    runs: list[Run] = []
    for seed in SEEDS:
        path = input_root / model / f"seed{seed:02d}.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or int(payload.get("actual_epochs", -1)) != EPOCHS:
            raise ValueError(f"Expected a complete {EPOCHS}-epoch history in {path}")
        metrics = {
            metric: _validated_array(payload[key], EPOCHS, path, key)
            for metric, key in BASELINE_KEYS.items()
        }
        runs.append(
            Run(
                model=model,
                seed=seed,
                completed_epochs=EPOCHS,
                metrics=metrics,
                source=str(payload.get("checkpoint_path", path.resolve())),
            )
        )
    return runs


def _mean_sem(runs: list[Run], metric: str, epochs: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([run.metrics[metric][:epochs] for run in runs])
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    return mean, sem


def write_epoch_csv(path: Path, runs: dict[str, list[Run]]) -> None:
    """Write every observed seed/epoch metric value in long form."""

    fields = ("model", "seed", "epoch", "metric", "value", "source")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODEL_ORDER:
            for run in runs[model]:
                for metric in METRICS:
                    for epoch, value in enumerate(run.metrics[metric], start=1):
                        writer.writerow(
                            {
                                "model": model,
                                "seed": run.seed,
                                "epoch": epoch,
                                "metric": metric,
                                "value": float(value),
                                "source": run.source,
                            }
                        )


def write_coverage_csv(path: Path, runs: dict[str, list[Run]]) -> None:
    """Write the exact seed coverage used by the plot."""

    fields = ("model", "seed", "completed_epochs", "included_in_group_mean", "source")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODEL_ORDER:
            common = min(run.completed_epochs for run in runs[model])
            for run in runs[model]:
                writer.writerow(
                    {
                        "model": model,
                        "seed": run.seed,
                        "completed_epochs": run.completed_epochs,
                        "included_in_group_mean": f"epochs_1_to_{common}",
                        "source": run.source,
                    }
                )


def build_summary(runs: dict[str, list[Run]]) -> dict[str, Any]:
    """Build coverage and common-endpoint numerical summaries."""

    complete = all(
        len(model_runs) == len(SEEDS)
        and all(run.completed_epochs == EPOCHS for run in model_runs)
        for model_runs in runs.values()
    )
    summary: dict[str, Any] = {
        "snapshot": "2026-09-23 EDT" if complete else "2026-09-22 EDT",
        "complete_10seed_150epoch": complete,
        "models": {},
    }
    for model in MODEL_ORDER:
        model_runs = runs[model]
        common = min(run.completed_epochs for run in model_runs)
        endpoints: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            mean, sem = _mean_sem(model_runs, metric, common)
            endpoints[metric] = {
                "epoch": common,
                "mean": float(mean[-1]),
                "sem": float(sem[-1]),
            }
        summary["models"][model] = {
            "label": MODEL_LABELS[model],
            "seeds": [run.seed for run in model_runs],
            "n_seeds": len(model_runs),
            "completed_epochs_by_seed": {
                str(run.seed): run.completed_epochs for run in model_runs
            },
            "common_prefix_epochs": common,
            "common_prefix_endpoint": endpoints,
        }
    comparisons: dict[str, Any] = {}
    for baseline, variant in (
        ("gawf", "gawf_legacy_notanh"),
        ("rnn", "rnn_inloop_notanh"),
    ):
        baseline_by_seed = {run.seed: run for run in runs[baseline]}
        variant_runs = runs[variant]
        common = min(run.completed_epochs for run in variant_runs)
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            variant_values = np.asarray(
                [run.metrics[metric][common - 1] for run in variant_runs]
            )
            baseline_values = np.asarray(
                [baseline_by_seed[run.seed].metrics[metric][common - 1] for run in variant_runs]
            )
            differences = variant_values - baseline_values
            metric_results[metric] = {
                "variant_mean": float(variant_values.mean()),
                "baseline_mean_on_matched_seeds": float(baseline_values.mean()),
                "paired_difference_mean": float(differences.mean()),
                "paired_difference_sem": float(
                    differences.std(ddof=1) / math.sqrt(differences.size)
                ),
            }
        comparisons[f"{variant}_minus_{baseline}"] = {
            "epoch": common,
            "seeds": [run.seed for run in variant_runs],
            "metrics": metric_results,
        }
    summary["matched_common_epoch_comparisons"] = comparisons
    if complete:
        summary["notes"] = [
            "All four conditions use 10 seeds and all 150 completed epochs.",
            "Bands are seed mean plus or minus SEM; thin solid lines are individual no-tanh seeds.",
        ]
    else:
        summary["notes"] = [
            "Baseline bands use all 10 seeds for all 150 epochs.",
            "Partial-condition thick lines and bands stop at the common completed prefix.",
            "Partial-condition thin lines show each seed beyond the common prefix when available.",
            "RNN in-loop no-tanh seeds 3 and 4 had not started at snapshot time.",
        ]
    return summary


def plot_curves(path_png: Path, path_pdf: Path, runs: dict[str, list[Run]]) -> None:
    """Render Digit/Sector and Train/Validation trajectories."""

    complete = all(
        len(model_runs) == len(SEEDS)
        and all(run.completed_epochs == EPOCHS for run in model_runs)
        for model_runs in runs.values()
    )
    panels = (
        ("train_acc_char", "Digit — train", (15, 100)),
        ("val_acc_char", "Digit — validation", (15, 100)),
        ("train_acc_sector", "Sector — train", (55, 100)),
        ("val_acc_sector", "Sector — validation", (55, 100)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True)
    handles: dict[str, Any] = {}
    for axis, (metric, title, limits) in zip(axes.flat, panels):
        for model in MODEL_ORDER:
            model_runs = runs[model]
            color = MODEL_COLORS[model]
            is_partial = model.endswith("notanh")
            common = min(run.completed_epochs for run in model_runs)
            if is_partial:
                for run in model_runs:
                    epoch = np.arange(1, run.completed_epochs + 1)
                    axis.plot(epoch, run.metrics[metric], color=color, alpha=0.13, lw=0.75)
            mean, sem = _mean_sem(model_runs, metric, common)
            epoch = np.arange(1, common + 1)
            label = MODEL_LABELS[model]
            if is_partial and not complete:
                label += f" (n={len(model_runs)}, mean through ep {common})"
            line = axis.plot(
                epoch,
                mean,
                color=color,
                lw=2.2,
                ls="-" if is_partial else "--",
                label=label,
            )[0]
            axis.fill_between(epoch, mean - sem, mean + sem, color=color, alpha=0.14)
            handles[model] = line
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlim(1, EPOCHS)
        axis.set_ylim(*limits)
        axis.set_ylabel("Accuracy (%)")
        axis.grid(alpha=0.22, linewidth=0.6)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    labels = [handles[model].get_label() for model in MODEL_ORDER]
    fig.legend(
        [handles[model] for model in MODEL_ORDER],
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        (
            "Matched GaWF/RNN training trajectories — complete 10-seed comparison"
            if complete
            else "Matched GaWF/RNN training trajectories — partial snapshot"
        ),
        fontsize=14,
        weight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Bands: mean ± SEM across 10 seeds. Thin no-tanh lines: individual seeds."
            if complete
            else "Bands: mean ± SEM. Thin no-tanh lines: individual completed seed prefixes. "
            "RNN no-tanh seeds 3–4 not started."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Validate the snapshot, write structured data, and render the figure."""

    args = parse_args()
    output_name = COMPLETE_NAME if args.complete else SCRIPT_NAME
    data_dir = args.data_dir or output_dir(CATEGORY, SCRIPT_NAME, "data")
    figure_dir = args.figure_dir or output_dir(CATEGORY, SCRIPT_NAME, "figs")
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_suffix = "_complete_20260923" if args.complete else ""

    runs = {
        "gawf": load_baseline(args.input_root, "gawf"),
        "gawf_legacy_notanh": (
            load_baseline(args.input_root, "gawf_legacy_notanh")
            if args.complete
            else load_partial(args.input_root, "gawf_legacy_notanh")
        ),
        "rnn": load_baseline(args.input_root, "rnn"),
        "rnn_inloop_notanh": (
            load_baseline(args.input_root, "rnn_inloop_notanh")
            if args.complete
            else load_partial(args.input_root, "rnn_inloop_notanh")
        ),
    }
    write_epoch_csv(data_dir / f"epoch_curves{data_suffix}.csv", runs)
    write_coverage_csv(data_dir / f"coverage{data_suffix}.csv", runs)
    summary = build_summary(runs)
    (data_dir / f"summary{data_suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_curves(
        figure_dir / f"{output_name}_2x2.png",
        figure_dir / f"{output_name}_2x2.pdf",
        runs,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
