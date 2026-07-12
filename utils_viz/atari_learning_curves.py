"""Overlay Atari DQN learning curves for the 1-frame Pong sweep (multi-seed).

Reads ``results/train_data/<suffix>/metrics_history.jsonl`` (written by
``train_atari_dqn.py``) and overlays ``episodic_return_100`` vs ``global_step``
for the seven model variants (cnn/rnn/gru/lstm/gawf/s5/mamba), one coloured line
per model. When several seeds are present the seeds are aggregated into a mean
line with a shaded +/- std band. Style mirrors
``utils_viz/model_train_compare_result.py`` (matplotlib Agg, fixed per-model
colours, legend, output under ``results/train_figs``).

Suffix convention matches the launch scripts (seed suffix optional):
  plain 1-frame Pong      -> ``<prefix>_<model>_seed<N>``          (--setting plain)
  1-frame flickering Pong -> ``<prefix>_flicker_<model>_seed<N>``  (--setting flicker)

Examples:
  python -m utils_viz.atari_learning_curves --setting both
  python -m utils_viz.atari_learning_curves --setting flicker --smooth 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
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

N_GRID = 300  # resampling points for cross-seed aggregation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Atari DQN learning curves across models (multi-seed)."
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
        help="Trailing rolling-mean window (in logged points) per seed; 1 disables it.",
    )
    parser.add_argument(
        "--band",
        choices=("std", "sem", "none"),
        default="std",
        help="Shaded band across seeds: std, standard error, or none.",
    )
    parser.add_argument("--data_root", default="results/train_data")
    parser.add_argument("--output_dir", default="results/train_figs/atari_pong_1frame")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Plot only this single seed (no band). Default: aggregate all seeds.",
    )
    parser.add_argument("--output", default=None, help="Explicit output png path.")
    return parser.parse_args()


def _base_suffix(prefix: str, setting: str, model: str) -> str:
    if setting == "flicker":
        return f"{prefix}_flicker_{model}"
    return f"{prefix}_{model}"


def _discover_run_dirs(data_root: str, base: str, seed: Optional[int] = None) -> list[Path]:
    """Run dirs for a model+setting. If seed is given, restrict to that seed."""
    root = Path(data_root)
    if seed is not None:
        one = root / f"{base}_seed{seed}"
        return [one] if one.is_dir() else []
    seed_dirs = sorted(
        (Path(p) for p in glob.glob(str(root / f"{base}_seed*")) if os.path.isdir(p)),
        key=lambda p: int(re.search(r"_seed(\d+)$", p.name).group(1))
        if re.search(r"_seed(\d+)$", p.name)
        else 0,
    )
    if seed_dirs:
        return seed_dirs
    seedless = root / base
    return [seedless] if seedless.is_dir() else []


def _load_curve(jsonl_path: Path, metric: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
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
            v, s = rec.get(metric), rec.get("global_step")
            if v is None or s is None:
                continue
            v = float(v)
            if np.isnan(v):  # early logs before any episode completes
                continue
            steps.append(int(s))
            values.append(v)
    if not steps:
        return None
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(values)[order]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average (causal; no future leakage, no edge spike)."""
    if window <= 1 or values.size < 2:
        return values
    csum = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(values.size)
    lo = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)


def _aggregate_seeds(
    curves: list[tuple[np.ndarray, np.ndarray]], smooth: int
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Resample each seed onto a common grid over the overlapping step range."""
    curves = [c for c in curves if c is not None and c[0].size >= 2]
    if not curves:
        return None
    # Drop clearly-incomplete seeds (e.g. an in-progress re-run) whose max step
    # is far short of the others, so they don't truncate the overlap range.
    if len(curves) > 1:
        cutoff = 0.9 * float(max(c[0][-1] for c in curves))
        kept = [c for c in curves if c[0][-1] >= cutoff]
        if kept:
            curves = kept
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    if hi <= lo:  # no overlap (e.g. one seed barely started); fall back to longest
        steps, values = max(curves, key=lambda c: c[0][-1])
        sm = _smooth(values, smooth)
        return steps, sm, np.zeros_like(sm), 1
    grid = np.linspace(lo, hi, N_GRID)
    stacked = np.vstack([
        np.interp(grid, steps, _smooth(values, smooth)) for steps, values in curves
    ])
    return grid, stacked.mean(0), stacked.std(0), len(curves)


def _plot_setting(ax, args, setting: str) -> int:
    plotted = 0
    for model in args.models:
        base = _base_suffix(args.prefix, setting, model)
        run_dirs = _discover_run_dirs(args.data_root, base, args.seed)
        curves = [_load_curve(d / "metrics_history.jsonl", args.metric) for d in run_dirs]
        agg = _aggregate_seeds(curves, args.smooth)
        if agg is None:
            print(f"[skip] no data: {args.data_root}/{base}[_seed*]")
            continue
        grid, mean, std, n_seeds = agg
        color = MODEL_COLORS.get(model)
        label = model if args.seed is not None else f"{model} (n={n_seeds})"
        ax.plot(grid, mean, label=label, color=color, linewidth=1.8)
        if args.band != "none" and n_seeds > 1:
            band = std / np.sqrt(n_seeds) if args.band == "sem" else std
            ax.fill_between(grid, mean - band, mean + band, color=color, alpha=0.18, linewidth=0)
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

    if args.seed is not None:
        title_tag = f" (seed {args.seed})"
    else:
        title_tag = "" if args.band == "none" else f" ({args.band} band across seeds)"
    fig.suptitle(f"Atari DRQN family — learning curves{title_tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
        out_path = os.path.join(
            args.output_dir, f"atari_pong_1frame_{args.setting}{seed_tag}.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  ({total} model curves)")


if __name__ == "__main__":
    main()
