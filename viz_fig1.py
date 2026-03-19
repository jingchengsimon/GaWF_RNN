"""
Figure 1: 1x3 visualization (reproducible, not PNG stitching).

Left:   GaWF character accuracy curve (train/val) from a training results .pkl
Middle: GaWF sector/position accuracy curve (train/val) from the same .pkl
Right:  Best accuracy bar chart across models from *_metrics.json files

Notes:
- The *_metrics.json files only contain best/final summaries, not per-epoch curves.
- Per-epoch curves (train_acc_char/val_acc_char/train_acc_pos/val_acc_pos) are stored
  in the training results .pkl (often gitignored under results/).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAR_KEYS = [
    "best_train_acc_pos",
    "best_val_acc_pos",
    "best_train_acc_char",
    "best_val_acc_char",
]
BAR_LABELS = ["Train pos", "Val pos", "Train char", "Val char"]
BAR_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]  # green, blue, red, purple


def _patch_numpy_pickle_compat() -> None:
    """Compatibility for pickles generated under older numpy versions."""
    import numpy.core.numeric as _num

    try:
        import numpy._core.numeric  # noqa: F401
    except Exception:
        import sys

        sys.modules["numpy._core.numeric"] = _num
        sys.modules["numpy._core"] = np.core


def load_results_pkl(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            "Results .pkl not found. This file is typically gitignored.\n"
            f"Path: {path}\n"
            "Tip: place the original training results .pkl under results/ or pass --gawf_pkl."
        )
    _patch_numpy_pickle_compat()
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_display_name(metrics: dict) -> str:
    m = (metrics.get("model_type") or "unknown").strip().upper()
    if m == "GAWF":
        return "GaWF"
    return m


def plot_gawf_accuracy_axes(
    ax_char,
    ax_pos,
    results: dict,
    epoch_start: int = 0,
    epoch_end: int | None = None,
    lw: float = 2.0,
) -> None:
    actual_epochs = int(results.get("actual_epochs") or len(results["train_acc_char"]))
    plot_end = actual_epochs if epoch_end is None else min(int(epoch_end), actual_epochs)
    plot_start = max(0, min(int(epoch_start), plot_end))
    epochs = np.arange(plot_start, plot_end) + 1  # 1-based display

    ax_char.plot(
        epochs,
        np.asarray(results["train_acc_char"])[plot_start:plot_end],
        label="train accuracy",
        linewidth=lw,
    )
    ax_char.plot(
        epochs,
        np.asarray(results["val_acc_char"])[plot_start:plot_end],
        label="validation accuracy",
        linewidth=lw,
    )
    ax_char.set_xlabel("Epoch")
    ax_char.set_ylabel("Accuracy (%)")
    ax_char.set_title("GaWF model character accuracy")
    ax_char.set_ylim(20, 100)
    ax_char.spines["top"].set_visible(False)
    ax_char.spines["right"].set_visible(False)
    ax_char.legend(fontsize=9, loc="lower right", frameon=False)

    ax_pos.plot(
        epochs,
        np.asarray(results["train_acc_pos"])[plot_start:plot_end],
        label="train accuracy",
        linewidth=lw,
    )
    ax_pos.plot(
        epochs,
        np.asarray(results["val_acc_pos"])[plot_start:plot_end],
        label="validation accuracy",
        linewidth=lw,
    )
    ax_pos.set_xlabel("Epoch")
    ax_pos.set_ylabel("Accuracy (%)")
    ax_pos.set_title("GaWF model position accuracy")
    ax_pos.set_ylim(75, 100)
    ax_pos.spines["top"].set_visible(False)
    ax_pos.spines["right"].set_visible(False)
    ax_pos.legend(fontsize=9, loc="lower right", frameon=False)


def plot_best_acc_bars_ax(
    ax,
    metrics_paths: list[Path],
    x_min: float = 80.0,
    x_max: float = 100.0,
) -> None:
    metrics_list = [load_metrics_json(p) for p in metrics_paths]
    model_names = [model_display_name(m) for m in metrics_list]
    n_models = len(model_names)
    n_bars = len(BAR_KEYS)

    data = np.zeros((n_models, n_bars), dtype=float)
    for i, m in enumerate(metrics_list):
        for j, key in enumerate(BAR_KEYS):
            v = m.get(key)
            data[i, j] = float(v) if v is not None else np.nan

    bar_height = 0.2
    y_centers = np.arange(n_models)[::-1].astype(float)  # first model at top
    y_offsets = np.array([0.3, 0.1, -0.1, -0.3])

    for j in range(n_bars):
        y_pos = y_centers + y_offsets[j]
        left = x_min
        widths = np.where(np.isnan(data[:, j]), 0.0, np.maximum(0.0, data[:, j] - left))
        ax.barh(
            y_pos,
            widths,
            height=bar_height,
            left=left,
            label=BAR_LABELS[j],
            color=BAR_COLORS[j],
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


def make_fig1(
    gawf_pkl: Path,
    metrics_paths: list[Path],
    out: Path,
    dpi: int = 200,
    figsize=(13.5, 3.8),
    epoch_start: int = 0,
    epoch_end: int | None = None,
    bar_xmin: float = 80.0,
    bar_xmax: float = 100.0,
    suptitle: str = "Model performance on the CT-MNIST task",
    suptitle_y: float = 1.05,
    suptitle_fs: float = 14.0,
) -> None:
    results = load_results_pkl(gawf_pkl)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_gawf_accuracy_axes(
        axes[0],
        axes[1],
        results=results,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    plot_best_acc_bars_ax(axes[2], metrics_paths=metrics_paths, x_min=bar_xmin, x_max=bar_xmax)

    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, lab in zip(axes, panel_labels, strict=True):
        ax.text(
            -0.13,
            1.09,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )

    if suptitle:
        fig.suptitle(suptitle, y=suptitle_y, fontsize=suptitle_fs, fontweight="bold")

    fig.tight_layout(w_pad=1.0)

    # Manually adjust the 3rd subplot xlabel Y-position.
    # This uses axes coordinates: y=0 is bottom of the axes area.
    # Tune xlabel_y until it visually aligns with the "Epoch" labels.
    xlabel_y = -0.1
    axes[2].xaxis.set_label_coords(0.5, xlabel_y)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    default_gawf_pkl = repo_root / "results" / "models" / "sector_40h_adamw" / (
        "gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50.pkl"
    )
    default_metrics = [
        repo_root
        / "results"
        / "models"
        / "sector_40h_adamw"
        / "gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_metrics.json",
        repo_root
        / "results"
        / "models"
        / "sector_40h_adamw"
        / "lstm_sector_acc_h80_lr0.0005_wd0.0001_do0_metrics.json",
        repo_root
        / "results"
        / "models"
        / "sector_40h_adamw"
        / "gru_sector_acc_h105_lr0.0005_wd0.0001_do0_metrics.json",
        repo_root
        / "results"
        / "models"
        / "sector_40h_adamw"
        / "rnn_sector_acc_h275_lr0.0005_wd0.0001_do0_metrics.json",
    ]
    default_out = repo_root / "results" / "visualization" / "sector_40h_adamw" / "fig1.png"

    p = argparse.ArgumentParser(
        description="Compose Fig1 (1x3) by plotting from .pkl and *_metrics.json."
    )
    p.add_argument("--gawf_pkl", type=str, default=str(default_gawf_pkl))
    p.add_argument(
        "--metrics",
        nargs="+",
        default=[str(x) for x in default_metrics],
        help="Paths to *_metrics.json files (order = top-to-bottom model order in bar chart).",
    )
    p.add_argument("--out", type=str, default=str(default_out))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--fig_w", type=float, default=13.5)
    p.add_argument("--fig_h", type=float, default=3.8)
    p.add_argument("--epoch_start", type=int, default=0)
    p.add_argument("--epoch_end", type=int, default=None)
    p.add_argument("--bar_xmin", type=float, default=80.0)
    p.add_argument("--bar_xmax", type=float, default=100.0)
    p.add_argument(
        "--suptitle",
        type=str,
        default="Model performance on the CT-MNIST task",
        help="Figure-level title (set empty string to disable).",
    )
    p.add_argument("--suptitle_y", type=float, default=0.95)
    p.add_argument("--suptitle_fs", type=float, default=14.0)
    args = p.parse_args()

    make_fig1(
        gawf_pkl=Path(args.gawf_pkl).expanduser(),
        metrics_paths=[Path(m).expanduser() for m in args.metrics],
        out=Path(args.out).expanduser(),
        dpi=args.dpi,
        figsize=(args.fig_w, args.fig_h),
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        bar_xmin=args.bar_xmin,
        bar_xmax=args.bar_xmax,
        suptitle=args.suptitle,
        suptitle_y=args.suptitle_y,
        suptitle_fs=args.suptitle_fs,
    )
    print(f"Saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

