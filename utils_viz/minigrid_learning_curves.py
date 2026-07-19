"""Plot MiniGrid RedBlue learning curves for LSTM and GaWF runs.

This script reads the JSONL metric histories stored below ``--data_dir/lstm`` and
``--data_dir/gawf``. It plots success rate and rolling episodic return against
environment steps, including faint raw traces and causal rolling means.

Outputs (in --output_dir):
- minigrid_redblue_learning_curves.png  (2-panel figure), PNG — learning curves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir


MODEL_STYLES = {
    "lstm": {"label": "LSTM", "color": "#9467bd"},
    "gawf": {"label": "GaWF", "color": "#d62728"},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing lstm/ and gawf/ metrics_history.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(output_dir("G_behaviour", "minigrid_learning_curves", "figs")),
        help="Directory for the output PNG.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=100,
        help="Causal rolling-mean window in logged points (default: 100).",
    )
    return parser.parse_args()


def load_history(path: Path) -> dict[str, np.ndarray]:
    """Load the fields needed for plotting from one JSONL history."""
    records: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

    if not records:
        raise RuntimeError(f"No metric records found in {path}")

    required = ("global_step", "success_rate", "episodic_return_100")
    missing = [key for key in required if key not in records[0]]
    if missing:
        raise KeyError(f"Missing required metrics in {path}: {', '.join(missing)}")

    return {
        key: np.asarray([record[key] for record in records], dtype=np.float64) for key in required
    }


def causal_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a causal rolling mean with a shorter window at the beginning."""
    if window < 1:
        raise ValueError("--smooth_window must be at least 1")
    cumulative = np.cumsum(np.insert(values, 0, 0.0), dtype=np.float64)
    indices = np.arange(values.size)
    starts = np.maximum(0, indices - window + 1)
    counts = indices - starts + 1
    return (cumulative[indices + 1] - cumulative[starts]) / counts


def plot_learning_curves(
    histories: dict[str, dict[str, np.ndarray]], output_path: Path, smooth_window: int
) -> None:
    """Create and save the two-panel MiniGrid learning-curve figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharex=True)
    panels = (
        ("success_rate", "Success rate", True),
        ("episodic_return_100", "Episodic return (last 100)", False),
    )

    for ax, (metric, ylabel, is_percent) in zip(axes, panels):
        for model_name, style in MODEL_STYLES.items():
            history = histories[model_name]
            x_millions = history["global_step"] / 1_000_000.0
            raw = history[metric]
            smooth = causal_rolling_mean(raw, smooth_window)

            ax.plot(x_millions, raw, color=style["color"], alpha=0.10, linewidth=0.7)
            ax.plot(
                x_millions,
                smooth,
                color=style["color"],
                linewidth=2.2,
                label=style["label"],
            )
            ax.scatter(x_millions[-1], smooth[-1], s=30, color=style["color"], zorder=4)

        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        if is_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            observed_max = max(float(histories[name][metric].max()) for name in histories)
            ax.set_ylim(0.0, min(1.0, max(0.30, observed_max * 1.08)))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
            ax.set_ylim(bottom=0.0)

    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    fig.suptitle(
        f"MiniGrid RedBlueDoors-8x8 learning curves (rolling mean: {smooth_window} logs)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    """Load both model histories and write the learning-curve figure."""
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    histories = {
        model_name: load_history(data_dir / model_name / "metrics_history.jsonl")
        for model_name in MODEL_STYLES
    }
    output_path = output_dir / "minigrid_redblue_learning_curves.png"
    plot_learning_curves(histories, output_path, args.smooth_window)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
