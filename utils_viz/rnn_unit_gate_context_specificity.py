"""Plot LSTM and GRU unit-level gate context variance decompositions.

Input: ``unit_gate_context_variance.json`` from the matching analysis module.
Output: one Figure-03-style PNG for LSTM and one for GRU.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GATE_LABELS = {
    "lstm": {"input": "Input", "forget": "Forget", "output": "Output"},
    "gru": {"reset": "Reset", "update": "Update"},
}


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=(
            "./results/anal_data/rnn_unit_gate_context_specificity/"
            "unit_gate_context_variance.json"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default="./results/anal_figs/rnn_unit_gate_context_specificity",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _annotate_bars(ax: plt.Axes) -> None:
    """Add compact percentage labels above visible bars."""

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.1f}" if bar.get_height() >= 0.08 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, padding=2, fontsize=7, rotation=90)


def plot_model(report: dict, model_type: str, fig_dir: str, dpi: int) -> str:
    """Write one two-panel Figure-03-style decomposition for a recurrent model."""

    model_report = report["models"][model_type]
    gate_names = tuple(GATE_LABELS[model_type])
    factors = ("sector", "digit", "interaction")
    trial_factors = factors + ("residual",)
    width = 0.8 / len(gate_names)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    colors = plt.get_cmap("tab10").colors
    for gate_index, gate_name in enumerate(gate_names):
        gate = model_report["gates"][gate_name]
        condition = gate["equal_cell_condition_mean"]["fractions"]
        trial = gate["equal_cell_trial_total"]["percent"]
        offset = (gate_index - (len(gate_names) - 1) / 2) * width
        axes[0].bar(
            np.arange(len(factors)) + offset,
            [100.0 * condition[factor] for factor in factors],
            width,
            color=colors[gate_index],
            label=GATE_LABELS[model_type][gate_name],
        )
        axes[1].bar(
            np.arange(len(trial_factors)) + offset,
            [trial[factor] for factor in trial_factors],
            width,
            color=colors[gate_index],
            label=GATE_LABELS[model_type][gate_name],
        )
    axes[0].set_xticks(np.arange(len(factors)), [name.title() for name in factors])
    axes[0].set_ylabel("Condition-mean variance (%)")
    axes[0].set_title("Hidden-state-compatible marginalization")
    axes[1].set_xticks(
        np.arange(len(trial_factors)), [name.title() for name in trial_factors]
    )
    axes[1].set_ylabel("Trial-total variance (%)")
    axes[1].set_title("Balanced trial-level ANOVA")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.legend(title="Unit-level gate", frameon=False)
        _annotate_bars(ax)
        ax.margins(y=0.15)
    fig.suptitle(f"{model_type.upper()} context variance decomposition (unit-level gates)")
    fig.tight_layout()
    path = os.path.join(
        fig_dir, f"03_{model_type}_unit_gate_variance_decomposition.png"
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def main() -> None:
    """Load the analysis report and plot both requested models."""

    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    with open(args.report, "r", encoding="utf-8") as file_obj:
        report = json.load(file_obj)
    for model_type in ("lstm", "gru"):
        plot_model(report, model_type, args.fig_dir, args.dpi)


if __name__ == "__main__":
    main()
