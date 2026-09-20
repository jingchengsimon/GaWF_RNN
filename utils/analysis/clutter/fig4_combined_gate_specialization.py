"""Render the ICLR Figure 4 gate-specialization summary from structured ten-seed data.

The top row shows GaWF activations followed by the synapse-level gates they modulate. The bottom row
compares destination-unit projections of GaWF gates with LSTM and GRU unit gates. Inputs are the
compact reset-excluded NPZ/JSON summaries; outputs are one development PNG and one 5.5-inch-wide
vector PDF for the ICLR manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from utils.analysis.clutter.fig5_unit_gate_context_plot import (  # noqa: E402
    GATE_COLORS,
    GATE_LABELS,
    MODEL_ORDER,
    MODEL_TITLES,
)
from utils.analysis.clutter.multiseed_plotting import add_seed_points  # noqa: E402
from utils.analysis.variance_decomposition import CM_FACTORS, RepeatedDecomposition  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOP_COLORS = {"sector": "#264653", "digit": "#E76F51", "interaction": "#E9C46A"}
TOP_PANELS = (
    (("encoder_activation", "Encoder\nactivation"), ("hidden_state", "Hidden\nactivation")),
    (("input_gate", "Input gate"), ("recurrent_gate", "Recurrent gate")),
)
GATE_OBJECTS = ("input_gate", "recurrent_gate")
ACTIVATION_OBJECTS = ("encoder_activation", "hidden_state")


def _seed_files(root: Path, prefix: str, filename: str) -> list[Path]:
    """Return the frozen ten-seed files for one Figure 4 representation family."""

    paths = [root / f"{prefix}{seed:02d}" / filename for seed in range(1, 11)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Missing frozen Figure 4 inputs: " + ", ".join(missing))
    return paths


def _npz_mean(path: Path, key: str) -> float:
    """Return one repeated-draw mean from a compact frozen archive."""

    with np.load(path, allow_pickle=False) as arrays:
        return float(np.asarray(arrays[key], dtype=np.float64).mean())


def _repeated(values: list[float]) -> np.ndarray:
    """Return exactly ten seed means."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,):
        raise RuntimeError(f"Expected ten seed means, got {array.shape}.")
    return array


def _result(condition: dict[str, np.ndarray], residual: np.ndarray) -> RepeatedDecomposition:
    """Build the plotting-only decomposition record."""

    return RepeatedDecomposition(
        aggregate_cm=condition,
        aggregate_trial={"residual": residual},
        per_unit_cm={},
        per_unit_trial={},
        unweighted_per_unit_mean_cm={},
        unweighted_per_unit_mean_trial={},
        consistency={},
    )


def _load_core_results(
    gate_root: Path, activation_root: Path, expected_seeds: int
) -> dict[str, RepeatedDecomposition]:
    """Load frozen Figure 4 summaries without importing the GPU collector."""

    if expected_seeds != 10:
        raise ValueError("The formal core-object figure requires exactly ten training seeds.")
    gate_files = _seed_files(gate_root, "seed", "gate_synapse_anova.npz")
    activation_files = _seed_files(activation_root, "gawf-seed", "activation_anova.npz")
    results: dict[str, RepeatedDecomposition] = {}
    for object_name in GATE_OBJECTS:
        condition = {
            factor: _repeated(
                [_npz_mean(path, f"{object_name}_{factor}") / 100.0 for path in gate_files]
            )
            for factor in CM_FACTORS
        }
        residual = _repeated(
            [_npz_mean(path, f"{object_name}_residual_frac") for path in gate_files]
        )
        results[object_name] = _result(condition, residual)
    for object_name, source_name in zip(
        ACTIVATION_OBJECTS, ("input_activation", "hidden_activation")
    ):
        condition = {
            factor: _repeated(
                [_npz_mean(path, f"{source_name}_{factor}") for path in activation_files]
            )
            for factor in CM_FACTORS
        }
        residual = _repeated(
            [_npz_mean(path, f"{source_name}_residual") for path in activation_files]
        )
        results[object_name] = _result(condition, residual)
    return results


def parse_args() -> argparse.Namespace:
    """Parse structured-data inputs and exact figure destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top_row_only", action="store_true",
                        help="Show only core results, with activations before gates.")
    parser.add_argument("--pdf_only", action="store_true", help="Write only the requested PDF.")
    parser.add_argument(
        "--gate_data_root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/data/analysis/fig4_gate_synapse_anova_resetexcluded_10seed"
        ),
    )
    parser.add_argument(
        "--activation_data_root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/data/analysis/fig4_activation_anova_6model_10seed_residual_resetexcluded"
        ),
    )
    parser.add_argument(
        "--unit_gate_report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/data/analysis/fig5_unit_gate_context_reset_excluded_10seed"
            / "unit_gate_context_variance_multiseed.json"
        ),
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/figs/D_variance_decomposition"
            / "Fig4_gate_task_variable_specialization_2x3_10seed.png"
        ),
    )
    parser.add_argument(
        "--output_pdf",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results/save/iclr_figs"
            / "Fig4_gate_task_variable_specialization_2x3_10seed.pdf"
        ),
    )
    parser.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _style_axis(axis: plt.Axes, *, show_ylabels: bool) -> None:
    """Apply the shared zero-to-100 borderless style."""

    axis.set_ylim(0.0, 105.0)
    axis.set_yticks((0.0, 50.0, 100.0))
    axis.tick_params(axis="y", labelleft=show_ylabels)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_core_axis(
    axis: plt.Axes,
    results: dict[str, RepeatedDecomposition],
    objects: tuple[tuple[str, str], ...],
    *,
    title: str,
    show_ylabels: bool,
    show_seed_points: bool,
) -> None:
    """Draw one top-row GaWF core-object panel."""

    centers = np.arange(len(objects), dtype=np.float64)
    width = 0.22
    for factor_index, factor in enumerate(CM_FACTORS):
        values = np.column_stack(
            [100.0 * np.asarray(results[name].aggregate_cm[factor]) for name, _ in objects]
        )
        positions = centers + (factor_index - 1) * width
        means = values.mean(axis=0)
        sems = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        axis.bar(
            positions,
            means,
            width,
            color=TOP_COLORS[factor],
            edgecolor="none",
            yerr=sems,
            capsize=1.8,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        )
        add_seed_points(
            axis,
            positions,
            values,
            bar_width=width,
            show=show_seed_points,
            rng=np.random.default_rng(factor_index),
        )
    axis.set_xticks(centers, [label for _, label in objects])
    axis.set_xlim(-0.48, len(objects) - 0.52)
    axis.set_title(title, pad=5)
    _style_axis(axis, show_ylabels=show_ylabels)


def _plot_unit_gate_axis(
    axis: plt.Axes,
    report: dict,
    model_type: str,
    *,
    show_ylabels: bool,
    show_seed_points: bool,
) -> None:
    """Draw one bottom-row destination-unit or conventional unit-gate panel."""

    factors = tuple(CM_FACTORS)
    x = np.arange(len(factors), dtype=np.float64)
    gate_names = tuple(GATE_LABELS[model_type])
    width = 0.78 / len(gate_names)
    for gate_index, gate_name in enumerate(gate_names):
        fractions = report["models"][model_type]["gates"][gate_name][
            "equal_cell_condition_mean"
        ]["fractions"]
        seed_values = 100.0 * np.asarray([fractions[factor] for factor in factors]).T
        if seed_values.shape != (10, len(factors)):
            raise ValueError(
                f"Expected ten {model_type}/{gate_name} seeds, got {seed_values.shape}."
            )
        offset = (gate_index - (len(gate_names) - 1) / 2) * width
        positions = x + offset
        axis.bar(
            positions,
            seed_values.mean(axis=0),
            width,
            color=GATE_COLORS[(model_type, gate_name)],
            edgecolor="none",
            yerr=seed_values.std(axis=0, ddof=1) / np.sqrt(seed_values.shape[0]),
            capsize=1.8,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
            label=GATE_LABELS[model_type][gate_name],
        )
        add_seed_points(
            axis,
            positions,
            seed_values,
            bar_width=width,
            show=show_seed_points,
            rng=np.random.default_rng(gate_index),
        )
    axis.set_xticks(x, ("Sec", "Dig", "Int"))
    axis.set_xlim(-0.52, 2.52)
    axis.set_title(MODEL_TITLES[model_type], pad=3)
    _style_axis(axis, show_ylabels=show_ylabels)
    axis.set_ylim(0.0, 125.0)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(gate_names),
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.38,
        borderaxespad=0.0,
        fontsize=6.6,
    )


def plot_combined(
    core_results: dict[str, RepeatedDecomposition],
    unit_gate_report: dict,
    output_png: Path | None,
    output_pdf: Path,
    *,
    show_seed_points: bool,
    top_row_only: bool = False,
) -> None:
    """Render the centered two-panel/top and three-panel/bottom ICLR figure."""

    missing = [model for model in MODEL_ORDER if model not in unit_gate_report.get("models", {})]
    if missing and not top_row_only:
        raise KeyError(f"Unit-gate report is missing models: {missing}")
    with plt.rc_context(
        {
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
        }
    ):
        fig = plt.figure(figsize=(5.5, 2.35 if top_row_only else 2.75))
        grid = fig.add_gridspec(
            1 if top_row_only else 2,
            6,
            left=0.105,
            right=0.99,
            bottom=0.20 if top_row_only else 0.14,
            top=0.80 if top_row_only else 0.84,
            height_ratios=(1.0,) if top_row_only else (0.92, 1.18),
            hspace=0.76,
            wspace=0.72,
        )
        top_axes = (fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]))
        bottom_axes = () if top_row_only else (
            fig.add_subplot(grid[1, :2]),
            fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[1, 4:]),
        )
        panels = TOP_PANELS
        titles = ("GaWF activations", "GaWF synapse gates")
        for index, (axis, objects, title) in enumerate(zip(top_axes, panels, titles)):
            _plot_core_axis(
                axis,
                core_results,
                objects,
                title=title,
                show_ylabels=index == 0,
                show_seed_points=show_seed_points,
            )
        for index, (axis, model_type) in enumerate(zip(bottom_axes, MODEL_ORDER)):
            _plot_unit_gate_axis(
                axis,
                unit_gate_report,
                model_type,
                show_ylabels=index == 0,
                show_seed_points=show_seed_points,
            )
        label_axes = top_axes if top_row_only else (*top_axes, *bottom_axes)
        for label, axis in zip("ABCDE", label_axes):
            position = axis.get_position()
            fig.text(
                position.x0 - 0.025,
                position.y1 + 0.015,
                label,
                ha="right",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        fig.legend(
            handles=[
                Patch(facecolor=TOP_COLORS[factor], edgecolor="none") for factor in CM_FACTORS
            ],
            labels=("Sector", "Digit", "Interaction"),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=3,
            frameon=False,
            handlelength=0.9,
            columnspacing=1.0,
        )
        fig.supylabel("Explained variance (%)", x=0.012, fontsize=8.8)
        if output_png is not None:
            output_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_png, dpi=300)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf)
        plt.close(fig)


def main() -> None:
    """Load the two retained result sets and render the combined ICLR figure."""

    args = parse_args()
    core_results = _load_core_results(args.gate_data_root, args.activation_data_root, 10)
    report = {} if args.top_row_only else json.loads(
        args.unit_gate_report.read_text(encoding="utf-8")
    )
    plot_combined(
        core_results,
        report,
        None if args.pdf_only else args.output_png,
        args.output_pdf,
        show_seed_points=args.show_seed_points,
        top_row_only=args.top_row_only,
    )
    if not args.pdf_only:
        print(f"Saved {args.output_png}")
    print(f"Saved {args.output_pdf}")


if __name__ == "__main__":
    main()
