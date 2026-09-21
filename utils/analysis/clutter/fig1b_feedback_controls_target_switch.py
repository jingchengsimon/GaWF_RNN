"""Compare completed CM-MNIST feedback-control behaviour around the target switch.

The feedback-control campaign trains four parameter-matched recurrent families on the same 40h
CM-MNIST sector task: GaWF-additive feedback, RNN-FB, GRU-FB and LSTM-FB. Each unit evaluates the
joint-switch-balanced test split with a reset at every foreground switch and tabulates
char/sector accuracy at frame offsets ``-10..-1`` and ``+1..+10`` around that switch (offset 0 is
excluded, i.e. the reset-excluded protocol). This module turns those per-unit JSON exports into one
switch-aligned behaviour comparison and renders a new figure in the visual language of the Figure 1
target-switch timeline: pre-switch span, post-switch span, and one dashed target-switch marker.

Figure 1 and its files are never touched; the curated figure is written under a new name.

Inputs
- ``--units-root``: ``feedback_controls_reset_excluded_test_10seed_v1`` (one ``<model>-seedNN`` dir)
  holding the reset-excluded test accuracy export.
- ``--curves-root``: ``feedback_controls_shuffle_resetexcluded_10seed_v1`` holding
  ``ablation_metrics.json`` with the switch-aligned baseline/shuffle curves.

Outputs
- ``results/data/analysis/G_behaviour/<script>/``: ``switch_offset_curves.npz``,
  ``unit_summary.csv``, ``model_summary.csv`` (plus the automatic ``manifest.json``).
- ``results/figs/G_behaviour/<script>/``: PNG and PDF.
- curated copies under ``results/save/`` unless ``--no_save_copy`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
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

# Must equal this module's stem so ``output_dir`` records the run and writes manifest.json.
SCRIPT_NAME = "fig1b_feedback_controls_target_switch"
CATEGORY = "G_behaviour"

MODEL_ORDER = ("gawf_additive", "rnn_fb", "gru_fb", "lstm_fb")
MODEL_LABELS = {
    "gawf_additive": "GaWF-additive",
    "rnn_fb": "RNN-FB",
    "gru_fb": "GRU-FB",
    "lstm_fb": "LSTM-FB",
}
MODEL_COLORS = {
    "gawf_additive": "#4C78A8",
    "rnn_fb": "#F58518",
    "gru_fb": "#E45756",
    "lstm_fb": "#54A24B",
}
MODEL_MARKERS = {
    "gawf_additive": "o",
    "rnn_fb": "s",
    "gru_fb": "D",
    "lstm_fb": "^",
}
CONDITION_ORDER = ("baseline", "shuffle_digit", "shuffle_sector", "shuffle_all")
CONDITION_LABELS = {
    "baseline": "baseline",
    "shuffle_digit": "shuffle digit",
    "shuffle_sector": "shuffle sector",
    "shuffle_all": "shuffle all",
}
METRIC_LABELS = {"char": "Character accuracy (%)", "sector": "Sector accuracy (%)"}
TARGET_COLOR = "#D55E00"
DEFAULT_UNITS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "save_data"
    / "fig1"
    / "fbctrl_feedback_controls"
    / "test_all"
)
DEFAULT_CURVES_ROOT = (
    PROJECT_ROOT
    / "results"
    / "save_data"
    / "fig1"
    / "fbctrl_feedback_controls"
    / "shuffle_all"
)


def parse_args() -> argparse.Namespace:
    """Parse the feedback-control export roots, output destinations, and figure options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-root", type=Path, default=DEFAULT_UNITS_ROOT)
    parser.add_argument("--curves-root", type=Path, default=DEFAULT_CURVES_ROOT)
    parser.add_argument(
        "--save-pdf",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "save"
        / "Fig1b_cmmnist_target_switch_timeline_fbctrl_10seed.pdf",
    )
    parser.add_argument(
        "--save-png",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "save"
        / "Fig1b_cmmnist_target_switch_timeline_fbctrl_10seed.png",
    )
    parser.add_argument("--no_save_copy", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_ORDER,
        default=("baseline", "shuffle_all"),
        help="Conditions drawn as figure rows; the model summary always covers all conditions.",
    )
    return parser.parse_args()


def _unit_dirs(root: Path) -> list[Path]:
    """Return sorted ``<model>-seedNN`` unit directories."""

    if not root.is_dir():
        raise RuntimeError(f"Missing feedback-control export root: {root}")
    units = sorted(path for path in root.iterdir() if path.is_dir() and "-seed" in path.name)
    if not units:
        raise RuntimeError(f"No <model>-seedNN unit directories under {root}")
    return units


def _parse_unit(directory: Path) -> tuple[str, int]:
    """Split one unit directory name into model type and seed."""

    model, _, seed_text = directory.name.rpartition("-seed")
    if model not in MODEL_ORDER or not seed_text.isdigit():
        raise RuntimeError(f"Unrecognized feedback-control unit directory: {directory.name}")
    return model, int(seed_text)


def _load_curve_payload(directory: Path) -> dict[str, Any]:
    """Read and validate one unit's switch-aligned ablation export."""

    path = directory / "ablation_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("exclude_window_initial_frame") is not True:
        raise RuntimeError(f"Switch export includes the reset frame or is undocumented: {path}")
    conditions = payload.get("conditions", {})
    missing = [name for name in CONDITION_ORDER if name not in conditions]
    if missing:
        raise RuntimeError(f"Missing conditions {missing} in {path}")
    return payload


def _load_test_export(directory: Path) -> dict[str, Any] | None:
    """Read the reset-excluded test accuracy export of one completed unit, if present."""

    path = directory / "reset_excluded_test_accuracy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_units(units_root: Path, curves_root: Path) -> dict[str, Any]:
    """Group per-unit switch curves and test accuracies by model type."""

    grouped: dict[str, dict[str, list[Any]]] = {
        model: {"seeds": [], "offsets": None, "curves": {}, "test": [], "meta": []}
        for model in MODEL_ORDER
    }
    for directory in _unit_dirs(curves_root):
        model, seed = _parse_unit(directory)
        payload = _load_curve_payload(directory)
        offsets = np.asarray(payload["switch_offsets"], dtype=np.int64)
        entry = grouped[model]
        if entry["offsets"] is None:
            entry["offsets"] = offsets
        elif not np.array_equal(entry["offsets"], offsets):
            raise RuntimeError(f"Offset grid differs from previous units: {directory}")
        entry["seeds"].append(seed)
        entry["meta"].append(
            {
                "unit": directory.name,
                "checkpoint": payload.get("ckpt"),
                "sequence_length": payload.get("sequence_length"),
                "K": payload.get("K"),
                "pre_K": payload.get("pre_K"),
            }
        )
        for condition in CONDITION_ORDER:
            entry["curves"].setdefault(condition, {"char": [], "sector": []})
            entry["curves"][condition]["char"].append(
                np.asarray(payload["conditions"][condition]["switch_char_acc"], dtype=np.float64)
            )
            entry["curves"][condition]["sector"].append(
                np.asarray(payload["conditions"][condition]["switch_sector_acc"], dtype=np.float64)
            )
        unit_dir = units_root / directory.name
        entry["test"].append(_load_test_export(unit_dir) if unit_dir.is_dir() else None)

    for model, entry in grouped.items():
        if not entry["seeds"]:
            raise RuntimeError(f"No completed units found for {model}")
        order = np.argsort(entry["seeds"])
        entry["seeds"] = [entry["seeds"][index] for index in order]
        entry["test"] = [entry["test"][index] for index in order]
        entry["meta"] = [entry["meta"][index] for index in order]
        for condition, metrics in entry["curves"].items():
            for metric, values in metrics.items():
                metrics[metric] = [values[index] for index in order]
    return grouped


def _mean_sem(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the offset-wise seed mean and standard error of the mean."""

    stacked = np.stack([np.asarray(value, dtype=np.float64) for value in values])
    mean = stacked.mean(axis=0)
    sem = (
        stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def build_curves(grouped: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-seed curves into model means and SEMs for every condition and metric."""

    curves: dict[str, Any] = {"offsets": None, "models": {}}
    for model in MODEL_ORDER:
        entry = grouped[model]
        if curves["offsets"] is None:
            curves["offsets"] = np.asarray(entry["offsets"], dtype=np.int64)
        model_payload = {"n_seeds": len(entry["seeds"]), "seeds": list(entry["seeds"])}
        for condition, metrics in entry["curves"].items():
            condition_payload = {}
            for metric, values in metrics.items():
                mean, sem = _mean_sem(values)
                condition_payload[metric] = {
                    "mean": mean.astype(np.float32),
                    "sem": sem.astype(np.float32),
                    "seed_curves": np.stack(values).astype(np.float32),
                }
            model_payload[condition] = condition_payload
        curves["models"][model] = model_payload
    return curves


def _offset_label(offset: int) -> str:
    """Return the Figure 1 style frame label for one signed offset."""

    return f"pre{abs(offset)}" if offset < 0 else f"post{offset}"


def write_structured_outputs(
    curves: dict[str, Any], grouped: dict[str, Any], data_dir: Path
) -> dict[str, Path]:
    """Write the aggregated curves and per-unit/per-model CSV summaries."""

    data_dir.mkdir(parents=True, exist_ok=True)
    offsets = curves["offsets"]
    arrays: dict[str, np.ndarray] = {"offsets": offsets}
    arrays["offset_labels"] = np.asarray([_offset_label(int(o)) for o in offsets])
    for model, payload in curves["models"].items():
        arrays[f"{model}__n_seeds"] = np.asarray([payload["n_seeds"]], dtype=np.int64)
        for condition in CONDITION_ORDER:
            for metric in ("char", "sector"):
                block = payload[condition][metric]
                arrays[f"{model}__{condition}__{metric}__mean"] = block["mean"]
                arrays[f"{model}__{condition}__{metric}__sem"] = block["sem"]
                arrays[f"{model}__{condition}__{metric}__seed_curves"] = block["seed_curves"]
    curves_path = data_dir / "switch_offset_curves.npz"
    np.savez_compressed(curves_path, **arrays)

    unit_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        entry = grouped[model]
        post = offsets >= 1
        for index, seed in enumerate(payload["seeds"]):
            test = entry["test"][index] or {}
            row: dict[str, Any] = {
                "model": model,
                "seed": seed,
                "test_char_acc": test.get("char_acc"),
                "test_sector_acc": test.get("sector_acc"),
                "test_n_windows": test.get("n_windows"),
            }
            for condition in CONDITION_ORDER:
                for metric in ("char", "sector"):
                    series = payload[condition][metric]["seed_curves"][index]
                    row[f"{condition}_{metric}_overall"] = float(np.mean(series))
                    row[f"{condition}_{metric}_post_switch"] = float(np.mean(series[post]))
            unit_rows.append(row)
    unit_path = data_dir / "unit_summary.csv"
    with unit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unit_rows)

    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        row = {"model": model, "n_seeds": payload["n_seeds"]}
        for condition in CONDITION_ORDER:
            for metric in ("char", "sector"):
                mean, sem = _mean_sem(
                    list(payload[condition][metric]["seed_curves"].astype(np.float64))
                )
                row[f"{condition}_{metric}_mean"] = float(np.mean(mean))
                row[f"{condition}_{metric}_sem"] = float(np.std(
                    [np.mean(curve) for curve in payload[condition][metric]["seed_curves"]],
                    ddof=1,
                ) / np.sqrt(payload["n_seeds"])) if payload["n_seeds"] > 1 else 0.0
                row[f"{condition}_{metric}_sem_offsetwise"] = float(np.mean(sem))
        for metric in ("char", "sector"):
            row[f"shuffle_all_minus_baseline_{metric}"] = (
                row[f"shuffle_all_{metric}_mean"] - row[f"baseline_{metric}_mean"]
            )
        test_values = [entry for entry in grouped[model]["test"] if entry]
        for metric, key in (("char", "char_acc"), ("sector", "sector_acc")):
            values = np.asarray([entry[key] for entry in test_values], dtype=np.float64)
            row[f"reset_excluded_test_{metric}_mean"] = float(values.mean()) if values.size else None
            row[f"reset_excluded_test_{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )
        model_rows.append(row)
    model_path = data_dir / "model_summary.csv"
    with model_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_rows)
    key_results: dict[str, float] = {}
    for row in model_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                key_results[f"{row['model']}.{key}"] = float(value)
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        post = curves["offsets"] >= 1
        for metric in ("char", "sector"):
            post_mean = payload["baseline"][metric]["seed_curves"][:, post].mean()
            key_results[f"{model}.baseline_{metric}_post_switch_mean"] = float(post_mean)
    key_path = data_dir / "key_results.json"
    key_path.write_text(json.dumps(key_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"curves": curves_path, "units": unit_path, "models": model_path, "key": key_path}


def _style_axis(axis: plt.Axes, offsets: np.ndarray, metric: str) -> None:
    """Apply the shared timeline styling to one behaviour axis."""

    axis.axvspan(-10.5, -0.5, color="#F2F2F2", zorder=0)
    axis.axvspan(0.5, 10.5, color="#FDF0E6", zorder=0)
    axis.axvline(0, color=TARGET_COLOR, linestyle="--", linewidth=1.2, zorder=1)
    ticks = [-10, -5, 0, 5, 10]
    axis.set_xticks(ticks)
    labels = ["pre10", "pre5", "switch", "post5", "post10"]
    axis.set_xticklabels(labels, fontsize=8)
    for tick_label in axis.get_xticklabels():
        if tick_label.get_text() == "switch":
            tick_label.set_color(TARGET_COLOR)
            tick_label.set_fontweight("bold")
    axis.set_xlim(-10.5, 10.5)
    axis.set_xlabel("Frame offset from the target switch", fontsize=9)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=9)
    axis.tick_params(axis="y", labelsize=8)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.text(
        0.25,
        0.995,
        r"Clutter $k$",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    axis.text(
        0.75,
        0.995,
        r"Clutter $k+1$",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )


def plot_timeline(
    curves: dict[str, Any], conditions: tuple[str, ...], pdf_path: Path, png_path: Path
) -> None:
    """Render the switch-aligned feedback-control behaviour comparison."""

    offsets = np.asarray(curves["offsets"], dtype=np.float64)
    rows = len(conditions)
    fig, axes = plt.subplots(rows, 2, figsize=(8.2, 2.55 * rows + 0.9), squeeze=False)
    for row, condition in enumerate(conditions):
        for column, metric in enumerate(("char", "sector")):
            axis = axes[row][column]
            for model in MODEL_ORDER:
                payload = curves["models"][model][condition][metric]
                mean = np.asarray(payload["mean"], dtype=np.float64)
                sem = np.asarray(payload["sem"], dtype=np.float64)
                axis.plot(
                    offsets,
                    mean,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    markersize=3.2,
                    linewidth=1.4,
                    label=MODEL_LABELS[model],
                    zorder=3,
                )
                if np.any(sem > 0):
                    axis.fill_between(
                        offsets,
                        mean - sem,
                        mean + sem,
                        color=MODEL_COLORS[model],
                        alpha=0.18,
                        zorder=2,
                    )
            axis.set_title(METRIC_LABELS[metric], fontsize=9.5, pad=14)
            axis.margins(y=0.10)
            _style_axis(axis, offsets, metric)
            if column == 0:
                axis.text(
                    -0.185,
                    0.5,
                    CONDITION_LABELS[condition],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#333333",
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=4,
            linewidth=1.4,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    counts = ", ".join(
        f"{MODEL_LABELS[model]} n={curves['models'][model]['n_seeds']}" for model in MODEL_ORDER
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "CM-MNIST target-switch behaviour of the feedback controls (reset-excluded)",
        fontsize=10,
        y=0.99,
    )
    fig.text(
        0.5,
        0.082,
        "Joint-switch-balanced 40h test split, sequence length 512, reset at every foreground switch,"
        " offset 0 excluded;\nmean $\\pm$ SEM across available seeds"
        f" ({counts}).",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.115, right=0.965, top=0.90, bottom=0.20, wspace=0.22, hspace=0.42)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _print_summary(
    curves: dict[str, Any], grouped: dict[str, Any], conditions: tuple[str, ...]
) -> None:
    """Print the compact numeric summary used for reporting."""

    print(f"conditions={list(conditions)} offsets={list(curves['offsets'])}")
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        tests = [entry for entry in grouped[model]["test"] if entry]
        char = np.asarray([entry["char_acc"] for entry in tests], dtype=np.float64)
        sector = np.asarray([entry["sector_acc"] for entry in tests], dtype=np.float64)
        base = payload["baseline"]
        post = curves["offsets"] >= 1
        post_char = base["char"]["seed_curves"][:, post].mean(axis=1)
        post_sector = base["sector"]["seed_curves"][:, post].mean(axis=1)

        def mean_sem(values: np.ndarray) -> str:
            if values.size == 0:
                return "n/a"
            sem = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
            return f"{values.mean():.2f} ± {sem:.2f}"

        print(
            f"{model:>14} n={payload['n_seeds']:>2} |"
            f" test char {mean_sem(char)} | test sector {mean_sem(sector)} |"
            f" post-switch char {mean_sem(post_char)} | post-switch sector {mean_sem(post_sector)}"
        )
    print("shuffle effect (baseline minus shuffle_all, mean over offsets and seeds):")
    for model in MODEL_ORDER:
        payload = curves["models"][model]
        parts = []
        for metric in ("char", "sector"):
            baseline = payload["baseline"][metric]["seed_curves"].mean(axis=1)
            shuffled = payload["shuffle_all"][metric]["seed_curves"].mean(axis=1)
            delta = baseline - shuffled
            sem = delta.std(ddof=1) / np.sqrt(delta.size) if delta.size > 1 else 0.0
            parts.append(f"{metric} {delta.mean():+.2f} ± {sem:.2f}")
        print(f"{model:>14} n={payload['n_seeds']:>2} | " + " | ".join(parts))


def main() -> None:
    """Load the completed feedback-control exports, aggregate, and render the comparison."""

    args = parse_args()
    grouped = load_units(args.units_root, args.curves_root)
    curves = build_curves(grouped)
    conditions = tuple(args.conditions)
    data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
    fig_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    written = write_structured_outputs(curves, grouped, data_dir)
    tag = "-".join(conditions)
    pdf_path = fig_dir / f"fbctrl_target_switch_timeline_{tag}.pdf"
    png_path = fig_dir / f"fbctrl_target_switch_timeline_{tag}.png"
    plot_timeline(curves, conditions, pdf_path, png_path)
    if not args.no_save_copy:
        plot_timeline(curves, conditions, args.save_pdf, args.save_png)
    _print_summary(curves, grouped, conditions)
    print(f"data={data_dir}")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(f"figure: {pdf_path}")
    print(f"figure: {png_path}")
    if not args.no_save_copy:
        print(f"curated: {args.save_pdf}")
        print(f"curated: {args.save_png}")


if __name__ == "__main__":
    main()
