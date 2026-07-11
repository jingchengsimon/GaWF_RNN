"""Overlay Atari DQN learning curves for the 1-frame Pong sweep.

Reads ``results/train_data/<suffix>/metrics_history.jsonl`` (written by
``train_atari_dqn.py``) and overlays ``episodic_return_100`` vs ``global_step``
for the seven model variants (cnn/rnn/gru/lstm/gawf/s5/mamba), one coloured line
per model. Style mirrors ``utils_viz/model_train_compare_result.py`` (matplotlib
Agg, fixed per-model colours, legend, output under ``results/train_figs``).

The suffix convention matches the launch scripts:
  plain 1-frame Pong      -> ``<prefix>_<model>``          (--setting plain)
  1-frame flickering Pong -> ``<prefix>_flicker_<model>``  (--setting flicker)

Examples:
  python -m utils_viz.atari_learning_curves --setting both
  python -m utils_viz.atari_learning_curves --setting flicker --smooth 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PREFIX = "atari_dqn_pong1f"
DEFAULT_MODELS = ("cnn", "rnn", "gru", "lstm", "gawf", "s5", "mamba")

# Fixed per-model colours so a model reads the same across every figure.
MODEL_COLORS = {
    "cnn": "#7f7f7f",   # grey: the memoryless control
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "gawf": "#d62728",  # red: model of interest
    "s5": "#17becf",
    "mamba": "#ff7f0e",
}

SETTING_TITLES = {
    "plain": "1-frame Pong (MDP control)",
    "flicker": "1-frame Flickering Pong (POMDP, p=0.5)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Atari DQN learning curves across models."
    )
    parser.add_argument(
        "--setting",
        choices=("plain", "flicker", "both"),
        default="both",
        help="Which Pong setting(s) to plot. 'both' draws a two-panel figure.",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Result-suffix prefix.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model tokens to overlay.",
    )
    parser.add_argument(
        "--metric",
        default="episodic_return_100",
        help="JSONL field to plot on the y-axis.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Rolling-mean window (in logged points) for the curves; 1 disables it.",
    )
    parser.add_argument("--data_root", default="results/train_data")
    parser.add_argument("--output_dir", default="results/train_figs/atari_pong_1frame")
    parser.add_argument("--output", default=None, help="Explicit output png path.")
    return parser.parse_args()


def _suffix_for(prefix: str, setting: str, model: str) -> str:
    if setting == "flicker":
        return f"{prefix}_flicker_{model}"
    return f"{prefix}_{model}"


def _load_curve(
    jsonl_path: Path, metric: str
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not jsonl_path.is_file():
        return None
    steps, values = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get(metric)
            s = rec.get("global_step")
            if v is None or s is None:
                continue
            v = float(v)
            if np.isnan(v):  # early logs before any episode completes
                continue
            steps.append(int(s))
            values.append(v)
    if not steps:
        return None
    return np.asarray(steps), np.asarray(values)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average with a shrinking window at the start.

    Each point averages up to ``window`` preceding points (min_periods=1). This
    is causal (no future leakage) and, unlike a zero-padded convolution, does not
    bias the final points toward zero.
    """
    if window <= 1 or values.size < 2:
        return values
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(values.size)
    lo = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)


def _plot_setting(ax, args, setting: str) -> int:
    plotted = 0
    for model in args.models:
        suffix = _suffix_for(args.prefix, setting, model)
        path = Path(args.data_root) / suffix / "metrics_history.jsonl"
        curve = _load_curve(path, args.metric)
        if curve is None:
            print(f"[skip] no data: {path}")
            continue
        steps, values = curve
        color = MODEL_COLORS.get(model, None)
        ax.plot(
            steps,
            _smooth(values, args.smooth),
            label=model,
            color=color,
            linewidth=1.8,
        )
        plotted += 1
    ax.set_title(SETTING_TITLES.get(setting, setting))
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.metric)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(title="model", fontsize=9)
    return plotted


def main() -> None:
    args = parse_args()
    settings = ("plain", "flicker") if args.setting == "both" else (args.setting,)

    fig, axes = plt.subplots(
        1, len(settings), figsize=(7.5 * len(settings), 5.0), sharey=True, squeeze=False
    )
    total = 0
    for ax, setting in zip(axes[0], settings):
        total += _plot_setting(ax, args, setting)
    if total == 0:
        raise SystemExit(
            "No curves found. Check --prefix/--data_root and that runs have logged "
            "metrics_history.jsonl."
        )

    fig.suptitle("Atari DRQN family — learning curves", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = args.setting
        out_path = os.path.join(args.output_dir, f"atari_pong_1frame_{tag}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  ({total} curves)")


if __name__ == "__main__":
    main()
