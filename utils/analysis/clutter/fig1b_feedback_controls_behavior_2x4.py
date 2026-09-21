"""Render the 2x4 behaviour comparison for the CM-MNIST feedback-control campaign.

The layout follows the retained best-six 2x4 summary: one row per readout (Location = sector,
Identity = character) and four columns with the same roles — test performance, validation
dynamics, target-switch recovery, and the feedback-shuffle ablation. Only the four
parameter-matched feedback controls appear, so the reference Figure 1 family is not modified.

Inputs
- ``--curves-root``: ``feedback_controls_shuffle_resetexcluded_10seed_v1`` (switch-aligned
  baseline/shuffle curves per unit).
- ``--units-root``: ``feedback_controls_reset_excluded_test_10seed_v1`` (per-unit test accuracy).
- ``--histories-root``: per-unit ``*.pkl`` training histories with the per-epoch loss tracks.

Outputs
- ``results/data/analysis/G_behaviour/<script>/``: NPZ bundle, per-unit and per-model CSVs,
  ``key_results.json``.
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
HISTORY_TRACK = {
    "sector": {"loss": "val_loss_pos"},
    "char": {"loss": "val_loss_char"},
}
ABLATION_CONDITIONS = ("baseline", "shuffle_digit", "shuffle_sector")
ABLATION_LABELS = ("Baseline", "Shuffle\ndigit", "Shuffle\nsector")
ABLATION_TINTS = ("#264653", "#E76F51", "#F4A261")
DEFAULT_CURVES_ROOT = (
    PROJECT_ROOT
    / "results"
    / "save_data"
    / "fig1"
    / "fbctrl_feedback_controls"
    / "shuffle_all"
)
DEFAULT_UNITS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "save_data"
    / "fig1"
    / "fbctrl_feedback_controls"
    / "test_all"
)
DEFAULT_HISTORIES_ROOT = (
    PROJECT_ROOT
    / "results"
    / "save_data"
    / "fig1"
    / "fbctrl_feedback_controls"
    / "histories"
)


def parse_args() -> argparse.Namespace:
    """Parse the feedback-control export roots and output destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves-root", type=Path, default=DEFAULT_CURVES_ROOT)
    parser.add_argument("--units-root", type=Path, default=DEFAULT_UNITS_ROOT)
    parser.add_argument("--histories-root", type=Path, default=DEFAULT_HISTORIES_ROOT)
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


def load_loss_histories(root: Path) -> dict[str, dict[int, np.ndarray]]:
    """Load the per-epoch validation-loss track of every completed unit."""

    if not root.is_dir():
        raise RuntimeError(f"Missing feedback-control history root: {root}")
    histories: dict[str, dict[str, dict[int, np.ndarray]]] = {
        model: {"sector": {}, "char": {}} for model in MODEL_ORDER
    }
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        model, seed = _unit_key(directory)
        pickles = sorted(directory.glob("*.pkl"))
        if len(pickles) != 1:
            raise RuntimeError(f"Expected exactly one history pickle in {directory}")
        with pickles[0].open("rb") as stream:
            payload = pickle.load(stream)
        for row_key, track in HISTORY_TRACK.items():
            values = payload.get(track["loss"])
            if values is None:
                raise RuntimeError(f"{pickles[0]} has no {track['loss']} track")
            histories[model][row_key][seed] = np.asarray(values, dtype=np.float64)
    for model in MODEL_ORDER:
        for row_key in ROW_KEYS:
            if not histories[model][row_key]:
                raise RuntimeError(f"No loss histories for {model}/{row_key}")
    return histories


def _bar_with_seeds(
    axis: plt.Axes,
    models: tuple[str, ...],
    values: dict[str, np.ndarray],
    *,
    percent: bool,
    color_lookup: dict[str, str],
    ylabel: str | None = None,
    show_xticks: bool = False,
) -> None:
    """Draw one bar per model with individual seed dots and a SEM error bar."""

    positions = np.arange(len(models), dtype=np.float64)
    for index, model in enumerate(models):
        samples = np.asarray(values[model], dtype=np.float64)
        mean = samples.mean()
        sem = samples.std(ddof=1) / np.sqrt(samples.size) if samples.size > 1 else 0.0
        axis.bar(
            index,
            mean * (100.0 if percent else 1.0),
            width=0.66,
            color=color_lookup[model],
            alpha=0.9,
            zorder=2,
        )
        axis.errorbar(
            index,
            mean * (100.0 if percent else 1.0),
            yerr=sem * (100.0 if percent else 1.0),
            color="#333333",
            capsize=2.6,
            linewidth=0.9,
            zorder=4,
        )
        jitter = np.linspace(-0.16, 0.16, samples.size)
        axis.scatter(
            index + jitter,
            samples * (100.0 if percent else 1.0),
            s=9,
            color="#555555",
            alpha=0.75,
            zorder=3,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in models],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    samples_all = np.concatenate([values[model] for model in models]) * (100.0 if percent else 1.0)
    span = float(samples_all.max() - samples_all.min())
    axis.set_ylim(
        float(samples_all.min() - 0.45 * span),
        float(samples_all.max() + 0.15 * span),
    )
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_histories(
    axis: plt.Axes,
    histories: dict[str, np.ndarray],
    *,
    percent: bool,
    show_xlabel: bool,
    ylabel: str | None = None,
) -> None:
    """Draw mean plus SEM validation-loss curves per model."""

    for model in MODEL_ORDER:
        seeds = sorted(histories[model])
        stacked = np.stack([histories[model][seed] for seed in seeds])
        epochs = np.arange(1, stacked.shape[1] + 1)
        mean = stacked.mean(axis=0) * (100.0 if percent else 1.0)
        sem = (
            stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            if stacked.shape[0] > 1
            else np.zeros_like(mean)
        )
        axis.plot(epochs, mean, color=MODEL_COLORS[model], linewidth=1.6, zorder=3)
        if np.any(sem > 0):
            axis.fill_between(
                epochs, mean - sem, mean + sem, color=MODEL_COLORS[model], alpha=0.20, zorder=2
            )
    axis.set_xlim(1, max(len(values) for model in MODEL_ORDER for values in histories[model].values()))
    axis.set_xlabel("Epoch" if show_xlabel else "", fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_recovery(
    axis: plt.Axes,
    curves: dict[str, Any],
    row_key: str,
    *,
    show_xlabel: bool,
    ylabel: str | None = None,
) -> None:
    """Draw the switch-aligned recovery curves of the baseline condition."""

    offsets = np.asarray(curves["offsets"], dtype=np.float64)
    for model in MODEL_ORDER:
        payload = curves["models"][model]["baseline"][row_key]
        axis.plot(
            offsets,
            np.asarray(payload["mean"], dtype=np.float64),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=3.0,
            linewidth=1.4,
            zorder=3,
        )
        sem = np.asarray(payload["sem"], dtype=np.float64)
        if np.any(sem > 0):
            axis.fill_between(
                offsets,
                payload["mean"] - sem,
                payload["mean"] + sem,
                color=MODEL_COLORS[model],
                alpha=0.18,
                zorder=2,
            )
    axis.axvline(0, color="#D55E00", linestyle="--", linewidth=1.1, zorder=1)
    ticks = [-10, 1, 4, 10]
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        ["pre10", "switch", "post4", "post10"],
        fontsize=7.5,
        rotation=20,
        ha="right",
    )
    axis.set_xlim(-10.5, 10.5)
    if show_xlabel:
        axis.set_xlabel("Frame relative to target switch", fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _plot_ablation(
    axis: plt.Axes,
    curves: dict[str, Any],
    row_key: str,
    *,
    show_xticks: bool,
    ylabel: str | None = None,
) -> None:
    """Draw the baseline, shuffled-digit, and shuffled-sector bars per model."""

    width = 0.26
    for model_index, model in enumerate(MODEL_ORDER):
        for condition_index, condition in enumerate(ABLATION_CONDITIONS):
            payload = curves["models"][model][condition][row_key]
            seed_overall = np.asarray(payload["seed_curves"], dtype=np.float64).mean(axis=1)
            mean = seed_overall.mean()
            sem = (
                seed_overall.std(ddof=1) / np.sqrt(seed_overall.size)
                if seed_overall.size > 1
                else 0.0
            )
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
            axis.errorbar(
                x,
                mean,
                yerr=sem,
                color="#333333",
                capsize=2.0,
                linewidth=0.8,
                zorder=4,
            )
    overall = [
        float(
            np.asarray(curves["models"][model][condition][row_key]["seed_curves"], dtype=np.float64)
            .mean(axis=1)
            .mean()
        )
        for model in MODEL_ORDER
        for condition in ABLATION_CONDITIONS
    ]
    span = max(overall) - min(overall)
    axis.set_ylim(min(overall) - 0.45 * span, max(overall) + 0.15 * span)
    positions = np.arange(len(MODEL_ORDER), dtype=np.float64)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=20 if show_xticks else 0,
        ha="right" if show_xticks else "center",
        fontsize=7.5 if show_xticks else 8,
    )
    if not show_xticks:
        axis.set_xticklabels([])
    axis.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def render_figure(
    curves: dict[str, Any],
    histories: dict[str, Any],
    grouped: dict[str, Any],
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Render the feedback-control 2x4 behaviour comparison."""

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(11.8, 6.0))
        for row, row_key in enumerate(ROW_KEYS):
            test_values = {
                model: np.asarray(
                    [
                        entry[f"{row_key}_acc"]
                        for entry in grouped[model]["test"]
                        if entry
                    ],
                    dtype=np.float64,
                )
                for model in MODEL_ORDER
            }
            _bar_with_seeds(
                axes[row][0],
                MODEL_ORDER,
                test_values,
                percent=False,
                color_lookup=MODEL_COLORS,
                show_xticks=row == 1,
            )
            _plot_histories(
                axes[row][1],
                {model: histories[model][row_key] for model in MODEL_ORDER},
                percent=False,
                show_xlabel=row == 1,
            )
            _plot_recovery(
                axes[row][2],
                curves,
                row_key,
                show_xlabel=row == 1,
            )
            _plot_ablation(
                axes[row][3],
                curves,
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
            left=0.062, right=0.99, bottom=0.20, top=0.83, hspace=0.36, wspace=0.30
        )
        row_centers = [
            float(np.mean([axis.get_position().y0 + axis.get_position().height / 2 for axis in row]))
            for row in axes
        ]
        for row_key, center in zip(ROW_KEYS, row_centers):
            figure.text(
                0.012,
                center,
                ROW_LABELS[row_key],
                rotation=90,
                va="center",
                fontsize=10,
            )
        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=2.5, marker=MODEL_MARKERS[model],
                   markersize=4)
            for model in MODEL_ORDER
        ]
        figure.legend(
            handles,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(MODEL_ORDER),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.6,
        )
        ablation_handles = [
            Line2D([0], [0], color=tint, linewidth=7.0) for tint in ABLATION_TINTS
        ]
        figure.legend(
            ablation_handles,
            [label.replace("\n", " ") for label in ABLATION_LABELS],
            loc="lower right",
            bbox_to_anchor=(0.995, 0.072),
            ncol=3,
            frameon=False,
            fontsize=7.5,
            handlelength=1.4,
        )
        counts = ", ".join(
            f"{MODEL_LABELS[model]} n={curves['models'][model]['n_seeds']}" for model in MODEL_ORDER
        )
        figure.text(
            0.5,
            0.008,
            "CM-MNIST feedback controls, 150 epochs, seeds 1-10; A: reset-excluded test accuracy; B:"
            " validation loss on the 40h split;\nC: joint-switch-balanced test, reset at every"
            " foreground switch, offset 0 excluded; D: sequence-512 reset-excluded shuffle"
            f" conditions ({counts}).",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=300)
        figure.savefig(output_pdf)
        plt.close(figure)


def write_structured_outputs(
    curves: dict[str, Any],
    histories: dict[str, Any],
    grouped: dict[str, Any],
    data_dir: Path,
) -> dict[str, Path]:
    """Write the aggregated NPZ bundle and CSV summaries."""

    data_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "offsets": np.asarray(curves["offsets"], dtype=np.int64),
        "offset_labels": np.asarray([_offset_label(int(o)) for o in curves["offsets"]]),
    }
    rows: list[dict[str, Any]] = []
    for row_key in ROW_KEYS:
        for model in MODEL_ORDER:
            payload = curves["models"][model]
            arrays[f"{row_key}__{model}__n_seeds"] = np.asarray(
                [payload["n_seeds"]], dtype=np.int64
            )
            for condition in CONDITION_ORDER:
                arrays[f"{row_key}__{model}__{condition}__mean"] = np.asarray(
                    payload[condition][row_key]["mean"], dtype=np.float32
                )
                arrays[f"{row_key}__{model}__{condition}__sem"] = np.asarray(
                    payload[condition][row_key]["sem"], dtype=np.float32
                )
            seeds = sorted(histories[model][row_key])
            arrays[f"{row_key}__{model}__val_loss_seeds"] = np.asarray(seeds, dtype=np.int64)
            arrays[f"{row_key}__{model}__val_loss"] = np.stack(
                [histories[model][row_key][seed] for seed in seeds]
            ).astype(np.float32)
            test_values = np.asarray(
                [entry[f"{row_key}_acc"] for entry in grouped[model]["test"] if entry],
                dtype=np.float64,
            )
            row: dict[str, Any] = {
                "readout": row_key,
                "model": model,
                "n_seeds": payload["n_seeds"],
                "test_acc_mean": float(test_values.mean()),
                "test_acc_sem": float(test_values.std(ddof=1) / np.sqrt(test_values.size))
                if test_values.size > 1
                else 0.0,
                "val_loss_final_mean": float(
                    np.mean([histories[model][row_key][seed][-1] for seed in seeds])
                ),
            }
            for condition in CONDITION_ORDER:
                seed_overall = np.asarray(
                    payload[condition][row_key]["seed_curves"], dtype=np.float64
                ).mean(axis=1)
                row[f"{condition}_acc_mean"] = float(seed_overall.mean())
                row[f"{condition}_acc_sem"] = (
                    float(seed_overall.std(ddof=1) / np.sqrt(seed_overall.size))
                    if seed_overall.size > 1
                    else 0.0
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
    """Load the completed feedback-control exports, aggregate, and render the 2x4 figure."""

    args = parse_args()
    grouped = load_units(args.units_root, args.curves_root)
    curves = build_curves(grouped)
    histories = load_loss_histories(args.histories_root)
    data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
    fig_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    written = write_structured_outputs(curves, histories, grouped, data_dir)
    png_path = fig_dir / "cmmnist_feedback_controls_behavior_2x4.png"
    pdf_path = fig_dir / "cmmnist_feedback_controls_behavior_2x4.pdf"
    render_figure(curves, histories, grouped, png_path, pdf_path)
    if not args.no_save_copy:
        render_figure(curves, histories, grouped, args.save_png, args.save_pdf)
    for row_key in ROW_KEYS:
        print(f"[{ROW_LABELS[row_key]}]")
        for model in MODEL_ORDER:
            test_values = np.asarray(
                [entry[f"{row_key}_acc"] for entry in grouped[model]["test"] if entry],
                dtype=np.float64,
            )
            baseline = curves["models"][model]["baseline"][row_key]
            post = np.asarray(curves["offsets"]) >= 1
            post_mean = np.asarray(baseline["seed_curves"], dtype=np.float64)[:, post].mean()
            print(
                f"  {MODEL_LABELS[model]:>14} n={curves['models'][model]['n_seeds']:>2}"
                f" test {test_values.mean():.2f}"
                f" post-switch {post_mean:.2f}"
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
