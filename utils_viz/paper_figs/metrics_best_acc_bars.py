"""
Horizontal bar chart of best accuracies across models.

Reads metrics JSON files. Y-axis = models (top to bottom), X-axis = accuracy.
Four bars per model (train pos, val pos, train char, val char). No bottom/right spines.

Usage:
  python -m utils_viz.viz_metrics_best_acc_bars
  python -m utils_viz.viz_metrics_best_acc_bars --metrics_dir results/train_data/sector_40h_adamw
  python -m utils_viz.viz_metrics_best_acc_bars --metrics path1.json path2.json ... --out fig.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Keys and display labels: train pos, val pos, train char, val char (same order for all models)
BAR_KEYS = [
    "best_train_acc_pos",
    "best_val_acc_pos",
    "best_train_acc_char",
    "best_val_acc_char",
]
BAR_LABELS = ["Train pos", "Val pos", "Train char", "Val char"]

# Colors for the four bars (consistent across models)
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]  # green, blue, red, purple

# Default metrics files (relative to repo root)
DEFAULT_METRICS_PATHS = [
    "results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_metrics.json",
    "results/train_data/sector_40h_adamw/lstm_sector_acc_h80_lr0.0005_wd0.0001_do0_metrics.json",
    "results/train_data/sector_40h_adamw/gru_sector_acc_h105_lr0.0005_wd0.0001_do0_metrics.json",
    "results/train_data/sector_40h_adamw/rnn_sector_acc_h275_lr0.0005_wd0.0001_do0_metrics.json",
]


def load_metrics(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_display_name(metrics: dict) -> str:
    m = (metrics.get("model_type") or "unknown").strip().upper()
    if m == "GAWF":
        return "GaWF"
    return m


def plot_best_acc_bars(
    metrics_paths: list[str | Path],
    out_path: str | Path,
    x_min: float = 80.0,
    x_max: float = 100.0,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> None:
    """
    Horizontal bar chart: y = models (top to bottom), x = accuracy.
    Four bars per model (train pos, val pos, train char, val char). No bottom/right spines.
    """
    metrics_list = [load_metrics(p) for p in metrics_paths]
    model_names = [model_display_name(m) for m in metrics_list]
    n_models = len(model_names)
    n_bars = len(BAR_KEYS)

    # data[i, j] = accuracy for model i, metric j
    data = np.zeros((n_models, n_bars))
    for i, m in enumerate(metrics_list):
        for j, key in enumerate(BAR_KEYS):
            v = m.get(key)
            if v is not None:
                data[i, j] = float(v)
            else:
                data[i, j] = np.nan

    # Y positions: first model at top (highest y). Each model gets 4 bars with small vertical offset.
    bar_height = 0.2
    y_centers = np.arange(n_models)[::-1].astype(float)  # [3, 2, 1, 0] for 4 models
    y_offsets = np.array([0.3, 0.1, -0.1, -0.3])  # train pos top, val char bottom within group

    fig, ax = plt.subplots(figsize=figsize)
    for j in range(n_bars):
        y_pos = y_centers + y_offsets[j]
        # barh: bar from left=x_min to value; width = value - x_min (clip to avoid negative)
        left = x_min
        widths = np.where(np.isnan(data[:, j]), 0.0, np.maximum(0.0, data[:, j] - left))
        ax.barh(
            y_pos,
            widths,
            height=bar_height,
            left=left,
            label=BAR_LABELS[j],
            color=COLORS[j],
            edgecolor="none",
        )

    ax.set_xlabel("Best accuracy (%)")
    ax.set_ylabel("Model")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, n_models - 0.4)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(model_names)
    ax.legend(loc="lower left", bbox_to_anchor=(0.8, 0.0), ncol=1, frameon=False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("top")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Bar chart of best accuracies from metrics JSON.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Paths to metrics JSON files (order = x-axis order).",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=None,
        help="Directory containing metrics JSON; use with --metrics_glob to pick files.",
    )
    parser.add_argument(
        "--metrics_glob",
        type=str,
        default="*_metrics.json",
        help="Glob for metrics files under --metrics_dir (default: *_metrics.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output figure path (default: <metrics_dir>/best_acc_bars.png or cwd).",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=80.0,
        help="X-axis (accuracy) minimum (default: 80).",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=100.0,
        help="X-axis (accuracy) maximum (default: 100).",
    )
    args = parser.parse_args()

    if args.metrics is not None:
        paths = [Path(p).resolve() for p in args.metrics]
    elif args.metrics_dir is not None:
        d = Path(args.metrics_dir).resolve()
        if not d.is_absolute():
            d = repo_root / d
        paths = sorted(d.glob(args.metrics_glob))
        paths = [p for p in paths if p.is_file()]
    else:
        paths = [repo_root / p for p in DEFAULT_METRICS_PATHS]
        paths = [p for p in paths if p.exists()]

    if not paths:
        raise SystemExit("No metrics files found.")

    if args.out is not None:
        out = Path(args.out)
    else:
        if args.metrics_dir is not None:
            out = Path(args.metrics_dir).resolve()
            if not out.is_absolute():
                out = repo_root / out
        else:
            out = repo_root / "results" / "models" / "sector_40h_adamw"
        out = out / "best_acc_bars.png"
    out = Path(out).resolve()
    if not out.is_absolute():
        out = repo_root / out

    plot_best_acc_bars(
        metrics_paths=paths,
        out_path=out,
        x_min=args.xmin,
        x_max=args.xmax,
    )
    print(f"Saved: {out}")

    # Print all 16 accuracies: 4 models × train/val × pos/char
    metrics_list = [load_metrics(p) for p in paths]
    model_names = [model_display_name(m) for m in metrics_list]
    header = f"{'Model':<8}  {'Train pos':>10}  {'Val pos':>10}  {'Train char':>10}  {'Val char':>10}"
    print()
    print(header)
    print("-" * len(header))
    for name, m in zip(model_names, metrics_list):
        vals = [m.get(k) for k in BAR_KEYS]
        row = f"{name:<8}  " + "  ".join(
            f"{v*100:>10.2f}" if v is not None and v <= 1.0 else
            (f"{v:>10.2f}" if v is not None else f"{'N/A':>10}")
            for v in vals
        )
        print(row)


if __name__ == "__main__":
    main()
