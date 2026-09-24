"""Compare original and no-tanh recurrent models with four wrapped baselines.

Inputs are the frozen six-model reset-excluded test CSV, the two no-tanh per-seed test JSON
collections, and ten 150-epoch training histories per model. Outputs are structured seed-level
test data, epoch-wise mean/SEM curves, endpoint summaries, and a no-shuffle 2-by-4 PNG/PDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils.analysis.anal_paths import output_dir
from utils.analysis.clutter.clutter_multiseed_summary import (
    MODEL_COLORS,
    _style_axis,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATEGORY = "G_behaviour"
SCRIPT_NAME = "best8_nonlinearity_training_test"
MODEL_ORDER = (
    "gawf",
    "gawf_legacy_notanh",
    "rnn",
    "rnn_inloop_notanh",
    "lstm",
    "gru",
    "mamba",
    "s5",
)
SIX_MODEL_ORDER = (
    "gawf_legacy_notanh",
    "rnn_inloop_notanh",
    "lstm",
    "gru",
    "mamba",
    "s5",
)
LABELS = {
    "gawf": "GaWF",
    "gawf_legacy_notanh": "GaWF no-tanh",
    "rnn": "RNN",
    "rnn_inloop_notanh": "RNN in-loop no-tanh",
    "lstm": "LSTM",
    "gru": "GRU",
    "mamba": "Mamba",
    "s5": "S5",
}
FAMILY = {
    "gawf_legacy_notanh": "gawf",
    "rnn_inloop_notanh": "rnn",
    **{model: model for model in ("gawf", "rnn", "lstm", "gru", "mamba", "s5")},
}
READOUTS = {"sector": "pos", "digit": "char"}
CURVE_METRICS = ("train_acc", "val_acc", "val_loss")


def parse_args() -> argparse.Namespace:
    """Parse frozen input roots and the non-overwriting publication-copy destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-test-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final"
        / "reset_excluded_test_accuracy_10seed.csv",
    )
    parser.add_argument(
        "--new-test-root",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour/best8_nonlinearity_behavior_20260923/inputs/test",
    )
    parser.add_argument(
        "--baseline-history-root",
        type=Path,
        default=PROJECT_ROOT / "results/save_data/fig1/validation_loss_histories",
    )
    parser.add_argument(
        "--new-history-root",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour"
        / "partial_nonlinearity_training_curves_20260922/inputs_complete_20260923",
    )
    parser.add_argument(
        "--curated-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "results/save/Figures",
    )
    parser.add_argument(
        "--stem",
        default="best8_multiseed_noshuffle_training_test_2x4_seq32",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse validated inputs and render a new figure stem without rewriting tables.",
    )
    return parser.parse_args()


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return seed mean and SEM along axis zero."""

    array = np.asarray(values, dtype=np.float64)
    return array.mean(axis=0), array.std(axis=0, ddof=1) / np.sqrt(array.shape[0])


def _load_test_metrics(
    baseline_csv: Path, new_root: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Load exactly ten reset-excluded test seeds for all eight models."""

    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: {"digit": {}, "sector": {}}
    )
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"].lower()
            if model not in MODEL_ORDER:
                continue
            seed = int(row["seed"])
            grouped[model]["digit"][seed] = float(row["char_acc"])
            grouped[model]["sector"][seed] = float(row["sector_acc"])

    for model in ("gawf_legacy_notanh", "rnn_inloop_notanh"):
        paths = sorted(new_root.glob(f"{model}/{model}-seed*/reset_excluded_test_accuracy.json"))
        if len(paths) != 10:
            raise RuntimeError(f"Expected ten reset-excluded JSONs for {model}, found {len(paths)}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "model": model,
                "sequence_length": 32,
                "excluded_timestep": 0,
                "n_frames": 55769,
            }
            mismatches = {
                key: (payload.get(key), value)
                for key, value in expected.items()
                if payload.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Protocol mismatch in {path}: {mismatches}")
            seed = int(payload["seed"])
            grouped[model]["digit"][seed] = float(payload["char_acc"])
            grouped[model]["sector"][seed] = float(payload["sector_acc"])

    output: dict[str, dict[str, np.ndarray]] = {}
    expected_seeds = set(range(1, 11))
    for model in MODEL_ORDER:
        if set(grouped[model]["digit"]) != expected_seeds:
            raise RuntimeError(f"Incomplete digit test seeds for {model}")
        if set(grouped[model]["sector"]) != expected_seeds:
            raise RuntimeError(f"Incomplete sector test seeds for {model}")
        output[model] = {
            readout: np.asarray([grouped[model][readout][seed] for seed in range(1, 11)])
            for readout in READOUTS
        }
    return output


def _history_paths(model: str, baseline_root: Path, new_root: Path) -> list[Path]:
    """Return the ten history pickle paths for one model."""

    if model in {"gawf_legacy_notanh", "rnn_inloop_notanh"}:
        return sorted((new_root / model).glob("seed*.pkl"))
    return sorted(baseline_root.glob(f"{model}-seed*/*.pkl"))


def _load_histories(
    baseline_root: Path, new_root: Path
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load 10-by-150 training, validation, and validation-loss arrays for all models."""

    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model in MODEL_ORDER:
        paths = _history_paths(model, baseline_root, new_root)
        if len(paths) != 10:
            raise RuntimeError(f"Expected ten histories for {model}, found {len(paths)}")
        model_data: dict[str, dict[str, list[np.ndarray]]] = {
            readout: {metric: [] for metric in CURVE_METRICS} for readout in READOUTS
        }
        for path in paths:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if int(payload.get("actual_epochs", -1)) != 150:
                raise RuntimeError(f"Expected actual_epochs=150 in {path}")
            for readout, suffix in READOUTS.items():
                for metric in CURVE_METRICS:
                    key = f"{metric}_{suffix}"
                    values = np.asarray(payload[key], dtype=np.float64)
                    if values.shape != (150,) or not np.isfinite(values).all():
                        raise RuntimeError(f"Invalid {key} in {path}: {values.shape}")
                    model_data[readout][metric].append(values)
        output[model] = {
            readout: {
                metric: np.stack(arrays, axis=0)
                for metric, arrays in per_readout.items()
            }
            for readout, per_readout in model_data.items()
        }
    return output


def _write_structured_outputs(
    data_dir: Path,
    tests: dict[str, dict[str, np.ndarray]],
    histories: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, object]:
    """Write seed-level test, epoch-curve, endpoint, and ranking files."""

    destinations = (
        data_dir / "test_accuracy_8model_10seed.csv",
        data_dir / "epoch_curves_8model_10seed.csv",
        data_dir / "endpoint_summary_8model_10seed.csv",
        data_dir / "summary.json",
    )
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite structured outputs: {existing}")

    with destinations[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "seed", "digit_acc", "sector_acc"))
        for model in MODEL_ORDER:
            for seed in range(10):
                writer.writerow(
                    (model, seed + 1, tests[model]["digit"][seed], tests[model]["sector"][seed])
                )

    with destinations[1].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "readout", "metric", "epoch", "mean", "sem"))
        for model in MODEL_ORDER:
            for readout in READOUTS:
                for metric in CURVE_METRICS:
                    mean, sem = _mean_sem(histories[model][readout][metric])
                    for epoch, (center, error) in enumerate(zip(mean, sem), start=1):
                        writer.writerow((model, readout, metric, epoch, center, error))

    endpoint_rows: list[dict[str, object]] = []
    best_by_metric: dict[str, dict[str, object]] = {}
    for readout in READOUTS:
        for metric in ("test_acc", *CURVE_METRICS):
            values_by_model: dict[str, np.ndarray] = {}
            for model in MODEL_ORDER:
                values_by_model[model] = (
                    tests[model][readout]
                    if metric == "test_acc"
                    else histories[model][readout][metric][:, -1]
                )
                mean, sem = _mean_sem(values_by_model[model])
                endpoint_rows.append(
                    {
                        "model": model,
                        "readout": readout,
                        "metric": metric,
                        "epoch": "" if metric == "test_acc" else 150,
                        "mean": float(mean),
                        "sem": float(sem),
                    }
                )
            reverse = metric != "val_loss"
            six_ranking = sorted(
                SIX_MODEL_ORDER,
                key=lambda model: float(values_by_model[model].mean()),
                reverse=reverse,
            )
            eight_ranking = sorted(
                MODEL_ORDER,
                key=lambda model: float(values_by_model[model].mean()),
                reverse=reverse,
            )
            best_by_metric[f"{readout}.{metric}"] = {
                "six_model_best": six_ranking[0],
                "eight_model_best": eight_ranking[0],
                "six_model_order": six_ranking,
                "eight_model_order": eight_ranking,
            }
    with destinations[2].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    summary: dict[str, object] = {
        "models": list(MODEL_ORDER),
        "primary_six_models": list(SIX_MODEL_ORDER),
        "n_seeds": 10,
        "epochs": 150,
        "test_protocol": {
            "reset_excluded": True,
            "sequence_length": 32,
            "n_frames_per_seed": 55769,
        },
        "best_by_endpoint_mean": best_by_metric,
        "scope_note": (
            "Target-switch recovery is not included because matched no-tanh recovery exports "
            "were unavailable; the old seq512 shuffle panel is intentionally not reused."
        ),
    }
    destinations[3].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _plot_test_axis(
    axis: plt.Axes,
    tests: dict[str, dict[str, np.ndarray]],
    readout: str,
    show_labels: bool,
) -> None:
    """Plot reset-excluded test bars with seed points."""

    positions = np.arange(len(MODEL_ORDER), dtype=np.float64)
    rng = np.random.default_rng(0)
    for index, model in enumerate(MODEL_ORDER):
        values = tests[model][readout]
        mean, sem = _mean_sem(values)
        axis.bar(
            positions[index],
            float(mean),
            yerr=float(sem),
            width=0.72,
            color=MODEL_COLORS[FAMILY[model]],
            edgecolor="black" if model.endswith("notanh") else "none",
            linewidth=0.5,
            hatch="///" if model.endswith("notanh") else None,
            capsize=2.0,
            error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
        )
        add_seed_points(
            axis,
            np.asarray([positions[index]]),
            values[:, None],
            bar_width=0.72,
            show=True,
            rng=rng,
        )
    axis.set_xticks(positions, [LABELS[model] for model in MODEL_ORDER], rotation=42, ha="right")
    axis.tick_params(axis="x", labelsize=6.2)
    if not show_labels:
        axis.tick_params(axis="x", labelbottom=False)
    if readout == "sector":
        axis.set_ylim(88.0, 94.0)
        axis.set_yticks((88, 90, 92, 94))
    else:
        axis.set_ylim(73.0, 88.0)
        axis.set_yticks((74, 78, 82, 86))
    _style_axis(axis)


def _plot_curve_axis(
    axis: plt.Axes,
    histories: dict[str, dict[str, dict[str, np.ndarray]]],
    readout: str,
    metric: str,
    show_xlabel: bool,
) -> None:
    """Plot one 150-epoch metric as seed mean plus SEM."""

    epochs = np.arange(1, 151)
    for model in MODEL_ORDER:
        mean, sem = _mean_sem(histories[model][readout][metric])
        color = MODEL_COLORS[FAMILY[model]]
        style = "--" if model.endswith("notanh") else "-"
        width = 2.0 if model.endswith("notanh") else 1.5
        axis.plot(epochs, mean, color=color, linestyle=style, linewidth=width, zorder=2)
        axis.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0)
    axis.set_xlim(0, 150)
    axis.set_xticks((0, 50, 100, 150))
    if not show_xlabel:
        axis.tick_params(axis="x", labelbottom=False)
    else:
        axis.set_xlabel("Epoch")
    if metric in {"train_acc", "val_acc"}:
        if readout == "sector":
            axis.set_ylim(70, 100)
            axis.set_yticks((70, 80, 90, 100))
        else:
            axis.set_ylim(20, 100)
            axis.set_yticks((20, 40, 60, 80, 100))
    elif readout == "sector":
        axis.set_ylim(0.15, 0.75)
        axis.set_yticks((0.2, 0.4, 0.6))
    else:
        axis.set_ylim(0.3, 2.2)
        axis.set_yticks((0.3, 0.9, 1.5, 2.1))
    _style_axis(axis)


def _render(
    tests: dict[str, dict[str, np.ndarray]],
    histories: dict[str, dict[str, dict[str, np.ndarray]]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render the no-shuffle 8-model comparison."""

    with plt.rc_context(
        {"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    ):
        figure, axes = plt.subplots(
            2, 4, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]}
        )
        for row, readout in enumerate(("sector", "digit")):
            _plot_test_axis(axes[row, 0], tests, readout, show_labels=row == 1)
            _plot_curve_axis(
                axes[row, 1], histories, readout, "train_acc", show_xlabel=row == 1
            )
            _plot_curve_axis(axes[row, 2], histories, readout, "val_acc", show_xlabel=row == 1)
            _plot_curve_axis(axes[row, 3], histories, readout, "val_loss", show_xlabel=row == 1)
        figure.subplots_adjust(
            left=0.06, right=0.98, bottom=0.27, top=0.70, hspace=0.32, wspace=0.34
        )
        legend_handles: list[Line2D | Patch] = []
        for model in MODEL_ORDER:
            color = MODEL_COLORS[FAMILY[model]]
            if model.endswith("notanh"):
                handle = Line2D(
                    [0], [0], color=color, linestyle="--", linewidth=2.0, label=LABELS[model]
                )
            else:
                handle = Line2D([0], [0], color=color, linewidth=2.0, label=LABELS[model])
            legend_handles.append(handle)
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
        column_titles = (
            "Reset-excluded test accuracy",
            "Training accuracy",
            "Validation accuracy",
            "Validation loss",
        )
        for column, title in enumerate(column_titles):
            position = axes[0, column].get_position()
            figure.text(
                position.x0 + position.width / 2,
                position.y1 + 0.025,
                title,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for row, label in enumerate(("Location", "Identity")):
            position = axes[row, 0].get_position()
            figure.text(
                0.018,
                position.y0 + position.height / 2,
                label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
            )
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_path, dpi=300)
        figure.savefig(pdf_path)
        plt.close(figure)


def main() -> None:
    """Validate inputs, write structured summaries, render, and copy the curated figure."""

    args = parse_args()
    if args.render_only:
        data_dir = PROJECT_ROOT / "results/data/analysis" / CATEGORY / SCRIPT_NAME
        figure_dir = PROJECT_ROOT / "results/figs" / CATEGORY
    else:
        data_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "data"))
        figure_dir = Path(output_dir(CATEGORY, SCRIPT_NAME, "figs"))
    tests = _load_test_metrics(args.baseline_test_csv, args.new_test_root)
    histories = _load_histories(args.baseline_history_root, args.new_history_root)
    summary_path = data_dir / "summary.json"
    if args.render_only:
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = _write_structured_outputs(data_dir, tests, histories)
    png_path = figure_dir / f"{args.stem}.png"
    pdf_path = figure_dir / f"{args.stem}.pdf"
    curated_pdf = args.curated_figure_dir / f"{args.stem}.pdf"
    for path in (png_path, pdf_path, curated_pdf):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    _render(tests, histories, png_path, pdf_path)
    args.curated_figure_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, curated_pdf)
    print(json.dumps(summary["best_by_endpoint_mean"], indent=2))
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {curated_pdf}")


if __name__ == "__main__":
    main()
