"""Render the available-seed 14-series nonlinearity comparison.

The test-accuracy panels compare the explicitly selected new, tanh-only, and old GaWF/RNN
semantics plus the original and no-wrap versions of LSTM, GRU, Mamba, and S5. Newly completed
reset-excluded JSONs override the retained five-seed ablation table. The three training-curve
columns remain the validated eight-model, ten-seed curves because matched histories are not yet
complete for every added variant.

Outputs are a seed-level CSV, a coverage JSON, and a non-overwriting 2-by-4 PNG/PDF.  Models in
the same recurrent family always share one colour; bar hatch distinguishes semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.analysis.clutter.clutter_multiseed_summary import MODEL_COLORS, _style_axis
from utils.analysis.clutter.multiseed_plotting import add_seed_points


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ORDER = (
    "gawf_nowrap",
    "gawf_legacy_notanh",
    "gawf",
    "rnn_nowrap",
    "rnn_inloop_notanh",
    "rnn",
    "lstm_nowrap",
    "lstm",
    "gru_nowrap",
    "gru",
    "mamba_nowrap",
    "mamba",
    "s5_nowrap",
    "s5",
)
CURVE_MODELS = (
    "gawf_legacy_notanh",
    "gawf",
    "rnn_inloop_notanh",
    "rnn",
    "lstm",
    "gru",
    "mamba",
    "s5",
)
FAMILY = {
    model: next(
        family for family in ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
        if model.startswith(family)
    )
    for model in MODEL_ORDER
}
LABELS = {
    "gawf_legacy_notanh": "GaWF",
    "gawf_nowrap": "GaWF-tanh",
    "gawf": "Old GaWF",
    "rnn_inloop_notanh": "RNN",
    "rnn_nowrap": "RNN-tanh",
    "rnn": "Old RNN",
    "lstm": "LSTM\nwrap",
    "lstm_nowrap": "LSTM\nno-wrap",
    "gru": "GRU\nwrap",
    "gru_nowrap": "GRU\nno-wrap",
    "mamba": "Mamba\nwrap",
    "mamba_nowrap": "Mamba\nno-wrap",
    "s5": "S5\nwrap",
    "s5_nowrap": "S5\nno-wrap",
}
CURVE_LABELS = {
    "gawf_legacy_notanh": "GaWF",
    "gawf": "Old GaWF",
    "rnn_inloop_notanh": "RNN",
    "rnn": "Old RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "mamba": "Mamba",
    "s5": "S5",
}


def parse_args() -> argparse.Namespace:
    """Parse retained tables, optional completed JSON roots, and output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best8-test-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour/best8_nonlinearity_training_test"
        / "test_accuracy_8model_10seed.csv",
    )
    parser.add_argument(
        "--ablation-test-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour/ablation_nowrap_behavior"
        / "ablation_14model_test_accuracy.csv",
    )
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour/best8_nonlinearity_training_test"
        / "epoch_curves_8model_10seed.csv",
    )
    parser.add_argument(
        "--completed-json-root",
        type=Path,
        action="append",
        default=[],
        help="Root containing model-seedNN/reset_excluded_test_accuracy.json leaves.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results/data/analysis/G_behaviour/best14_nonlinearity_training_test_available",
    )
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "results/figs/G_behaviour",
    )
    parser.add_argument(
        "--curated-figure-dir",
        type=Path,
        default=PROJECT_ROOT / "results/save/Figures_notanh",
    )
    parser.add_argument(
        "--stem",
        default="best14_multiseed_noshuffle_2x4_seq32_available",
    )
    return parser.parse_args()


def _load_tests(args: argparse.Namespace) -> dict[str, dict[str, dict[int, float]]]:
    """Load seed metrics, preferring completed JSONs over retained CSV rows."""

    values: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: {"digit": {}, "sector": {}}
    )
    for path in (args.ablation_test_csv, args.best8_test_csv):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                model = row["model"]
                if model not in MODEL_ORDER:
                    continue
                seed = int(row["seed"])
                values[model]["digit"][seed] = float(
                    row.get("digit_acc", row.get("char_acc", "nan"))
                )
                values[model]["sector"][seed] = float(row["sector_acc"])
    for root in args.completed_json_root:
        for path in sorted(root.glob("*/reset_excluded_test_accuracy.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = str(payload["model"])
            if model not in MODEL_ORDER:
                continue
            expected = {"sequence_length": 32, "excluded_timestep": 0, "n_frames": 55769}
            mismatch = {
                key: (payload.get(key), wanted)
                for key, wanted in expected.items()
                if payload.get(key) != wanted
            }
            if mismatch:
                raise RuntimeError(f"Protocol mismatch in {path}: {mismatch}")
            seed = int(payload["seed"])
            values[model]["digit"][seed] = float(payload["char_acc"])
            values[model]["sector"][seed] = float(payload["sector_acc"])
    for model in MODEL_ORDER:
        digit_seeds = set(values[model]["digit"])
        sector_seeds = set(values[model]["sector"])
        if not digit_seeds or digit_seeds != sector_seeds:
            raise RuntimeError(f"Invalid seed coverage for {model}: {digit_seeds}, {sector_seeds}")
        if not digit_seeds.issubset(set(range(1, 11))):
            raise RuntimeError(f"Unexpected seeds for {model}: {sorted(digit_seeds)}")
    return values


def _load_curves(path: Path) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Load the retained eight-model mean/SEM curves."""

    grouped: dict[tuple[str, str, str], list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["readout"], row["metric"])
            grouped[key].append((int(row["epoch"]), float(row["mean"]), float(row["sem"])))
    output = {}
    for model in CURVE_MODELS:
        for readout in ("sector", "digit"):
            for metric in ("train_acc", "val_acc", "val_loss"):
                rows = sorted(grouped[(model, readout, metric)])
                if [row[0] for row in rows] != list(range(1, 151)):
                    raise RuntimeError(f"Incomplete curve: {model} {readout} {metric}")
                output[(model, readout, metric)] = (
                    np.asarray([row[1] for row in rows]),
                    np.asarray([row[2] for row in rows]),
                )
    return output


def _write_outputs(
    args: argparse.Namespace,
    tests: dict[str, dict[str, dict[int, float]]],
) -> dict[str, int]:
    """Write the exact available seed rows and coverage record."""

    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_data_dir / "test_accuracy_14series_available_seeds.csv"
    coverage_path = args.output_data_dir / "coverage_14series_available_seeds.json"
    for path in (csv_path, coverage_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    coverage: dict[str, int] = {}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "family", "seed", "digit_acc", "sector_acc"))
        for model in MODEL_ORDER:
            seeds = sorted(tests[model]["digit"])
            coverage[model] = len(seeds)
            for seed in seeds:
                writer.writerow(
                    (
                        model,
                        FAMILY[model],
                        seed,
                        tests[model]["digit"][seed],
                        tests[model]["sector"][seed],
                    )
                )
    coverage_path.write_text(
        json.dumps(
            {
                "models": list(MODEL_ORDER),
                "coverage": coverage,
                "test_protocol": {
                    "reset_excluded": True,
                    "sequence_length": 32,
                    "n_frames_per_seed": 55769,
                },
                "curve_scope": "validated best8 ten-seed curves only",
                "inputs": {
                    "best8_test_csv": str(args.best8_test_csv),
                    "ablation_test_csv": str(args.ablation_test_csv),
                    "curve_csv": str(args.curve_csv),
                    "completed_json_roots": [
                        str(path) for path in args.completed_json_root
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return coverage


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Return scalar mean and SEM for at least two seeds."""

    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _plot_test(
    axis: plt.Axes,
    tests: dict[str, dict[str, dict[int, float]]],
    coverage: dict[str, int],
    readout: str,
    show_labels: bool,
) -> None:
    """Plot all fourteen series with every currently completed seed."""

    bar_width = 0.56
    family_gap = 0.42
    positions = []
    cursor = 0.0
    previous_family = None
    for model in MODEL_ORDER:
        family = FAMILY[model]
        if previous_family is not None and family != previous_family:
            cursor += family_gap
        positions.append(cursor)
        cursor += bar_width
        previous_family = family
    positions = np.asarray(positions, dtype=np.float64)
    rng = np.random.default_rng(0)
    for index, model in enumerate(MODEL_ORDER):
        values = np.asarray(list(tests[model][readout].values()), dtype=np.float64)
        mean, sem = _mean_sem(values)
        axis.bar(
            positions[index],
            mean,
            yerr=sem,
            width=bar_width,
            color=MODEL_COLORS[FAMILY[model]],
            edgecolor="none",
            linewidth=0.0,
            capsize=1.6,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
        )
        add_seed_points(
            axis,
            np.asarray([positions[index]]),
            values[:, None],
            bar_width=bar_width,
            show=True,
            rng=rng,
        )
    labels = [f"{LABELS[model]}\nn={coverage[model]}" for model in MODEL_ORDER]
    axis.set_xticks(positions, labels, rotation=48, ha="right")
    axis.tick_params(axis="x", labelsize=4.7, pad=1)
    if not show_labels:
        axis.tick_params(axis="x", labelbottom=False)
    if readout == "sector":
        axis.set_ylim(86.0, 94.0)
        axis.set_yticks((88, 90, 92, 94))
    else:
        axis.set_ylim(68.0, 87.0)
        axis.set_yticks((74, 78, 82, 86))
    _style_axis(axis)


def _plot_curve(
    axis: plt.Axes,
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    readout: str,
    metric: str,
    show_xlabel: bool,
) -> None:
    """Plot the retained eight-model ten-seed curves."""

    epochs = np.arange(1, 151)
    for model in CURVE_MODELS:
        mean, sem = curves[(model, readout, metric)]
        is_relu = model in {"gawf_legacy_notanh", "rnn_inloop_notanh"}
        color = MODEL_COLORS[FAMILY[model]]
        axis.plot(
            epochs,
            mean,
            color=color,
            linestyle="--" if is_relu else "-",
            linewidth=2.0 if is_relu else 1.5,
        )
        axis.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0)
    axis.set_xlim(0, 150)
    axis.set_xticks((0, 50, 100, 150))
    if show_xlabel:
        axis.set_xlabel("Epoch")
    else:
        axis.tick_params(axis="x", labelbottom=False)
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
    tests: dict[str, dict[str, dict[int, float]]],
    coverage: dict[str, int],
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Render fourteen test series plus the retained eight-model training curves."""

    with plt.rc_context(
        {"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    ):
        figure, axes = plt.subplots(
            2, 4, figsize=(12.2, 4.45), gridspec_kw={"width_ratios": [2.15, 1.0, 1.0, 1.0]}
        )
        for row, readout in enumerate(("sector", "digit")):
            _plot_test(axes[row, 0], tests, coverage, readout, show_labels=row == 1)
            _plot_curve(axes[row, 1], curves, readout, "train_acc", show_xlabel=row == 1)
            _plot_curve(axes[row, 2], curves, readout, "val_acc", show_xlabel=row == 1)
            _plot_curve(axes[row, 3], curves, readout, "val_loss", show_xlabel=row == 1)
        figure.subplots_adjust(
            left=0.05, right=0.985, bottom=0.34, top=0.71, hspace=0.32, wspace=0.32
        )
        handles = []
        for model in CURVE_MODELS:
            is_relu = model in {"gawf_legacy_notanh", "rnn_inloop_notanh"}
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=MODEL_COLORS[FAMILY[model]],
                    linestyle="--" if is_relu else "-",
                    linewidth=2.0,
                    label=CURVE_LABELS[model],
                )
            )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.61, 0.995),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
        titles = (
            "Reset-excluded test accuracy (14 series)",
            "Training accuracy (best8)",
            "Validation accuracy (best8)",
            "Validation loss (best8)",
        )
        for column, title in enumerate(titles):
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
                0.014,
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
    """Validate inputs, preserve structured data, and render the new comparison."""

    args = parse_args()
    tests = _load_tests(args)
    curves = _load_curves(args.curve_csv)
    coverage = _write_outputs(args, tests)
    png_path = args.output_figure_dir / f"{args.stem}.png"
    pdf_path = args.output_figure_dir / f"{args.stem}.pdf"
    curated_path = args.curated_figure_dir / f"{args.stem}.pdf"
    for path in (png_path, pdf_path, curated_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    _render(tests, coverage, curves, png_path, pdf_path)
    args.curated_figure_dir.mkdir(parents=True, exist_ok=True)
    curated_path.write_bytes(pdf_path.read_bytes())
    print(json.dumps(coverage, indent=2, sort_keys=True))
    print(f"wrote {pdf_path}")
    print(f"wrote {curated_path}")


if __name__ == "__main__":
    main()
