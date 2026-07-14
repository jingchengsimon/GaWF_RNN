"""Plot Atari DQN learning curves for sweeps or one multi-task run.

Reads ``results/train_data/<suffix>/metrics_history.jsonl`` (written by
``train_atari_dqn.py``) and overlays ``episodic_return_100`` vs ``global_step``
for the seven model variants (ann/rnn/gru/lstm/gawf/s5/mamba), one coloured line
per model. When several seeds are present the seeds are aggregated into a mean
line with a shaded +/- std band. Style mirrors
``utils_viz/model_train_compare_result.py`` (matplotlib Agg, fixed per-model
colours, legend, output under ``results/train_figs``).

Suffix convention states both environment advance and observation history:
  plain fs4/stack1 Pong      -> ``<prefix>_<model>_seed<N>``          (--setting plain)
  flickering fs4/stack1 Pong -> ``<prefix>_flicker_<model>_seed<N>``  (--setting flicker)

Examples:
  python -m utils_viz.atari_learning_curves --setting both
  python -m utils_viz.atari_learning_curves --setting flicker --smooth 20
  python -m utils_viz.atari_learning_curves --run_dir results/train_data/<suffix>
  python -m utils_viz.atari_learning_curves --run_dir results/train_data/<suffix> \
      --include_combined
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

DEFAULT_PREFIX = "atari_dqn_pong_fs4_stack1"
DEFAULT_MODELS = ("ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba")

# Fixed per-model colours so a model reads the same across every figure.
MODEL_COLORS = {
    "ann": "#7f7f7f",   # grey: the memoryless control
    "cnn": "#7f7f7f",   # historical result alias
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "gawf": "#d62728",  # red: model of interest
    "s5": "#17becf",
    "mamba": "#ff7f0e",
}

TASK_COLORS = {
    "ALE/Breakout-v5": "#1f77b4",
    "ALE/Pong-v5": "#ff7f0e",
}

SETTING_TITLES = {
    "plain": "Pong (frame skip 4, stack 1)",
    "flicker": "Flickering Pong (frame skip 4, stack 1, p=0.5)",
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
    parser.add_argument("--output_dir", default="results/train_figs/atari_pong_fs4_stack1")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Plot only this single seed (no band). Default: aggregate all seeds.",
    )
    parser.add_argument("--output", default=None, help="Explicit output png path.")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Plot one run directly, including per-environment multi-task returns.",
    )
    parser.add_argument(
        "--include_combined",
        action="store_true",
        help="Also plot the rolling return pooled across tasks in --run_dir mode.",
    )
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


def _metric_value(record: dict, metric: str):
    """Resolve a dot-separated metric path from one JSONL record."""
    value = record
    for key in metric.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


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
            v, s = _metric_value(rec, metric), rec.get("global_step")
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


def _discover_env_ids(jsonl_path: Path) -> list[str]:
    """Return sorted task names found in a multi-task metrics history."""
    if not jsonl_path.is_file():
        return []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                per_env = json.loads(line).get("per_env", {})
            except json.JSONDecodeError:
                continue
            if isinstance(per_env, dict) and per_env:
                return sorted(str(env_id) for env_id in per_env)
    return []


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


def _env_label(env_id: str) -> str:
    """Convert an ALE environment id to a compact plot label."""
    return env_id.removeprefix("ALE/").removesuffix("-v5")


def _direct_run_title(run_dir: Path, env_ids: list[str]) -> str:
    """Build a protocol-aware title from one run's saved metadata."""
    metadata_path = run_dir / "metrics.json"
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = {}

    tasks = " + ".join(_env_label(env_id) for env_id in env_ids) or "Atari multi-task"
    model = str(metadata.get("model_type", "DQN")).upper()
    frame_skip = metadata.get("frame_skip")
    frame_stack = metadata.get("frame_stack")
    sampling = str(metadata.get("replay_sampling", "global_uniform")).replace("_", "-")
    protocol = ""
    if frame_skip is not None and frame_stack is not None:
        protocol = f" (skip {frame_skip}, stack {frame_stack})"
    return f"{tasks} {model}{protocol} — {sampling} replay"


def _plot_direct_run(args: argparse.Namespace) -> str:
    """Plot per-task returns for one training directory."""
    run_dir = Path(args.run_dir)
    history_path = run_dir / "metrics_history.jsonl"
    if not history_path.is_file():
        raise SystemExit(f"Missing {history_path}")

    env_ids = _discover_env_ids(history_path)
    series = []
    if args.include_combined:
        series.append(("episodic_return_100", "Combined", "#4c4c4c"))
    series.extend(
        (
            f"per_env.{env_id}.episodic_return_100",
            _env_label(env_id),
            TASK_COLORS.get(env_id),
        )
        for env_id in env_ids
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = 0
    for metric, label, color in series:
        curve = _load_curve(history_path, metric)
        if curve is None:
            continue
        steps, values = curve
        ax.plot(steps, _smooth(values, args.smooth), label=label, color=color, linewidth=1.8)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise SystemExit(f"No valid curves found in {history_path}")

    ax.set_title(_direct_run_title(run_dir, env_ids))
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (last 100 episodes)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="task")
    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "atari_multitask_learning_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out_path}  ({plotted} curves)")
    return out_path


def main() -> None:
    args = parse_args()
    if args.run_dir:
        _plot_direct_run(args)
        return
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
            args.output_dir, f"atari_pong_fs4_stack1_{args.setting}{seed_tag}.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out_path}  ({total} model curves)")


if __name__ == "__main__":
    main()
