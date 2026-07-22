"""Plot combined-model strict Pong curves for each seed and across five seeds.

The curated input root contains one directory per protocol/setting/model group,
with one child directory per seed. Every output figure uses a one-row,
two-column layout: plain Pong on the left and flickering Pong on the right.
Per-seed figures overlay all available models. The mean/std figure includes a
model in a panel only when all declared seeds are complete for that setting.
An optional longer plain-Pong campaign adds a second, full-width row whose first
half aligns horizontally with the 1M plain-Pong panel for prefix comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils_anal.anal_paths import output_dir


DEFAULT_MODELS = ("ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba")
DEFAULT_SEEDS = (42, 1, 2, 3, 4)
MODEL_COLORS = {
    "ann": "#7f7f7f",
    "rnn": "#1f77b4",
    "gru": "#2ca02c",
    "lstm": "#9467bd",
    "gawf": "#d62728",
    "s5": "#17becf",
    "mamba": "#ff7f0e",
}
N_GRID = 300


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="results/train_data/rl/atari/pong_6action",
        help="Curated Pong data root.",
    )
    parser.add_argument(
        "--output_root",
        default=str(output_dir("G_behaviour", "atari_seed_curves", "figs")),
        help="Curated Pong figure root.",
    )
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--expected_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--plain_compare_steps",
        type=int,
        default=None,
        help="Optional longer plain-Pong budget, such as 2000000, plotted in row two.",
    )
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite return values and global steps from one JSONL history."""
    steps: list[int] = []
    returns: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            step = record.get("global_step")
            value = record.get("episodic_return_100")
            if step is None or value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            steps.append(int(step))
            returns.append(value)
    if not steps:
        raise RuntimeError(f"No finite episodic returns in {path}")
    order = np.argsort(steps)
    return (
        np.asarray(steps, dtype=np.int64)[order],
        np.asarray(returns, dtype=np.float64)[order],
    )


def smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a causal rolling mean with short leading windows."""
    if window < 1:
        raise ValueError("--smooth must be at least 1")
    if window == 1 or values.size < 2:
        return values
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(values.size)
    starts = np.maximum(0, indices - window + 1)
    return (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)


def validate_seed(
    seed_dir: Path,
    *,
    model: str,
    frame_skip: int,
    frame_stack: int,
    num_layers: int,
    expected_steps: int,
    expected_flicker_prob: float | None = None,
) -> None:
    """Reject incomplete or protocol-mismatched runs before plotting."""
    metrics_path = seed_dir / "metrics.json"
    history_path = seed_dir / "metrics_history.jsonl"
    if not metrics_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(f"Missing metrics/history under {seed_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": model,
        "frame_skip": frame_skip,
        "frame_stack": frame_stack,
        "num_layers": num_layers,
        "global_step": expected_steps,
        "action_space_mode": "minimal",
        "num_actions": 6,
    }
    if expected_flicker_prob is not None:
        expected["flicker_prob"] = expected_flicker_prob
    mismatches = {
        key: (metrics.get(key), value)
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Protocol mismatch in {metrics_path}: {mismatches}")


def collect_curves(
    args: argparse.Namespace,
) -> dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    """Load every completed strict run into setting/model/seed dictionaries."""
    data_root = Path(args.data_root)
    curves: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = {
        "plain": {},
        "flicker": {},
    }
    for setting in curves:
        for model in args.models:
            group_name = (
                f"fs{args.frame_skip}_stack{args.frame_stack}_l{args.num_layers}_"
                f"{setting}_{model}"
            )
            seed_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for seed in args.seeds:
                seed_dir = data_root / group_name / f"seed{seed}"
                if not seed_dir.is_dir():
                    continue
                validate_seed(
                    seed_dir,
                    model=model,
                    frame_skip=args.frame_skip,
                    frame_stack=args.frame_stack,
                    num_layers=args.num_layers,
                    expected_steps=args.expected_steps,
                    expected_flicker_prob=0.0 if setting == "plain" else 0.5,
                )
                seed_curves[seed] = load_curve(seed_dir / "metrics_history.jsonl")
            curves[setting][model] = seed_curves
    return curves


def format_step_tag(steps: int) -> str:
    """Return the compact step tag used by curated comparison directories."""
    if steps <= 0:
        raise ValueError("step count must be positive")
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def collect_plain_compare_curves(
    args: argparse.Namespace,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Load completed runs from an optional longer plain-Pong campaign."""
    curves = {model: {} for model in args.models}
    if args.plain_compare_steps is None:
        return curves
    data_root = Path(args.data_root)
    step_tag = format_step_tag(args.plain_compare_steps)
    for model in args.models:
        group_name = (
            f"fs{args.frame_skip}_stack{args.frame_stack}_l{args.num_layers}_"
            f"plain{step_tag}_{model}"
        )
        for seed in args.seeds:
            seed_dir = data_root / group_name / f"seed{seed}"
            if not seed_dir.is_dir():
                continue
            validate_seed(
                seed_dir,
                model=model,
                frame_skip=args.frame_skip,
                frame_stack=args.frame_stack,
                num_layers=args.num_layers,
                expected_steps=args.plain_compare_steps,
                expected_flicker_prob=0.0,
            )
            curves[model][seed] = load_curve(seed_dir / "metrics_history.jsonl")
    return curves


def style_axis(
    ax: plt.Axes,
    *,
    setting: str,
    frame_skip: int,
    frame_stack: int,
    title_prefix: str | None = None,
) -> None:
    """Apply shared protocol-aware labels to one panel."""
    title = title_prefix or ("Pong" if setting == "plain" else "Flickering Pong (p=0.5)")
    ax.set_title(f"{title} (skip {frame_skip}, stack {frame_stack})")
    ax.set_xlabel("environment steps (×10⁶)")
    ax.set_ylabel("episodic return (last 100)")
    ax.grid(True, alpha=0.3)


def add_shared_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    """Add one de-duplicated model legend below both panels."""
    handles: dict[str, object] = {}
    for ax in axes:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            title="model",
            loc="lower center",
            ncol=min(7, len(handles)),
            frameon=False,
        )


def save_seed_figure(
    curves: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]],
    output_path: Path,
    *,
    seed: int,
    args: argparse.Namespace,
    plain_compare_curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Overlay all available models for one seed in a two-panel figure."""
    has_compare = any(seed in plain_compare_curves[model] for model in args.models)
    if has_compare:
        fig = plt.figure(figsize=(14.6, 8.4))
        grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.06)
        top_plain = fig.add_subplot(grid[0, 0])
        top_flicker = fig.add_subplot(grid[0, 1], sharey=top_plain)
        compare_ax = fig.add_subplot(grid[1, :], sharey=top_plain)
        axes = np.asarray([top_plain, top_flicker])
        legend_axes = np.asarray([top_plain, top_flicker, compare_ax])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.0), sharey=True)
        compare_ax = None
        legend_axes = axes
    plotted = 0
    for ax, setting in zip(axes, ("plain", "flicker")):
        for model in args.models:
            curve = curves[setting][model].get(seed)
            if curve is None:
                continue
            steps, values = curve
            ax.plot(
                steps / 1_000_000.0,
                smooth_curve(values, args.smooth),
                color=MODEL_COLORS.get(model),
                linewidth=1.9,
                label=model,
            )
            plotted += 1
        style_axis(
            ax,
            setting=setting,
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
        )
    if compare_ax is not None:
        compare_model_count = 0
        for model in args.models:
            curve = plain_compare_curves[model].get(seed)
            if curve is None:
                continue
            steps, values = curve
            compare_ax.plot(
                steps / 1_000_000.0,
                smooth_curve(values, args.smooth),
                color=MODEL_COLORS.get(model),
                linewidth=1.9,
                label=model,
            )
            plotted += 1
            compare_model_count += 1
        style_axis(
            compare_ax,
            setting="plain",
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
            title_prefix=(
                f"Pong · {format_step_tag(args.plain_compare_steps)} run · "
                f"{compare_model_count}/{len(args.models)} models"
            ),
        )
        compare_ax.axvline(1.0, color="#555555", linewidth=1.0, alpha=0.7)
    if plotted == 0:
        plt.close(fig)
        return
    comparison_title = " · 1M vs 2M" if has_compare else ""
    fig.suptitle(
        f"Strict 6-action Pong · L{args.num_layers} · seed {seed}{comparison_title}",
        fontsize=13,
    )
    add_shared_legend(fig, legend_axes)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def aggregate_complete_seeds(
    seed_curves: dict[int, tuple[np.ndarray, np.ndarray]],
    expected_seeds: list[int],
    smooth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return an aligned mean/std curve only when every expected seed exists."""
    if set(seed_curves) != set(expected_seeds):
        return None
    ordered = [seed_curves[seed] for seed in expected_seeds]
    lower = max(int(steps[0]) for steps, _ in ordered)
    upper = min(int(steps[-1]) for steps, _ in ordered)
    if upper <= lower:
        raise RuntimeError("Completed seed curves have no overlapping step range")
    grid = np.linspace(lower, upper, N_GRID)
    aligned = np.vstack(
        [np.interp(grid, steps, smooth_curve(values, smooth)) for steps, values in ordered]
    )
    return grid, aligned.mean(axis=0), aligned.std(axis=0)


def save_mean_std_figure(
    curves: dict[str, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]],
    output_path: Path,
    *,
    args: argparse.Namespace,
    plain_compare_curves: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> tuple[int, int]:
    """Overlay complete-model five-seed means and std bands in two panels."""
    has_compare = any(plain_compare_curves[model] for model in args.models)
    if has_compare:
        fig = plt.figure(figsize=(14.6, 8.4))
        grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.06)
        top_plain = fig.add_subplot(grid[0, 0])
        top_flicker = fig.add_subplot(grid[0, 1], sharey=top_plain)
        compare_ax = fig.add_subplot(grid[1, :], sharey=top_plain)
        axes = np.asarray([top_plain, top_flicker])
        legend_axes = np.asarray([top_plain, top_flicker, compare_ax])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.0), sharey=True)
        compare_ax = None
        legend_axes = axes
    completed_groups = 0
    comparison_groups = 0
    for ax, setting in zip(axes, ("plain", "flicker")):
        for model in args.models:
            aggregate = aggregate_complete_seeds(
                curves[setting][model],
                args.seeds,
                args.smooth,
            )
            if aggregate is None:
                continue
            steps, mean, std = aggregate
            x = steps / 1_000_000.0
            color = MODEL_COLORS.get(model)
            ax.plot(x, mean, color=color, linewidth=2.0, label=model)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            completed_groups += 1
        style_axis(
            ax,
            setting=setting,
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
        )
    if compare_ax is not None:
        for model in args.models:
            aggregate = aggregate_complete_seeds(
                plain_compare_curves[model],
                args.seeds,
                args.smooth,
            )
            if aggregate is None:
                continue
            steps, mean, std = aggregate
            x = steps / 1_000_000.0
            color = MODEL_COLORS.get(model)
            compare_ax.plot(x, mean, color=color, linewidth=2.0, label=model)
            compare_ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            comparison_groups += 1
        style_axis(
            compare_ax,
            setting="plain",
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
            title_prefix=(
                f"Pong · {format_step_tag(args.plain_compare_steps)} runs · "
                f"{comparison_groups}/{len(args.models)} models"
            ),
        )
        compare_ax.axvline(1.0, color="#555555", linewidth=1.0, alpha=0.7)
    if completed_groups == 0:
        plt.close(fig)
        return 0, 0
    fig.suptitle(
        f"Strict 6-action Pong · L{args.num_layers} · " f"{len(args.seeds)}-seed mean ± std",
        fontsize=13,
    )
    add_shared_legend(fig, legend_axes)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return completed_groups, comparison_groups


def main() -> None:
    """Write combined per-seed figures and one combined mean/std figure."""
    args = parse_args()
    curves = collect_curves(args)
    plain_compare_curves = collect_plain_compare_curves(args)
    output_dir = Path(args.output_root) / (
        f"fs{args.frame_skip}_stack{args.frame_stack}_l{args.num_layers}_" f"{len(args.seeds)}seed"
    )
    for seed in args.seeds:
        save_seed_figure(
            curves,
            output_dir / f"seed{seed}.png",
            seed=seed,
            args=args,
            plain_compare_curves=plain_compare_curves,
        )
    completed_groups, comparison_groups = save_mean_std_figure(
        curves,
        output_dir / "mean_std.png",
        args=args,
        plain_compare_curves=plain_compare_curves,
    )
    partial_groups = sum(
        1
        for setting in ("plain", "flicker")
        for model in args.models
        if curves[setting][model] and set(curves[setting][model]) != set(args.seeds)
    )
    print(
        f"wrote combined figures under {output_dir}: "
        f"{completed_groups} complete model-setting groups, "
        f"{partial_groups} partial groups, "
        f"{comparison_groups} complete longer-run model groups"
    )


if __name__ == "__main__":
    main()
