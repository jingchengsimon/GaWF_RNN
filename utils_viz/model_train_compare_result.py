"""
Overlay training curves of multiple models in one figure.

This script compares model curves from result pickle files under:
results/train_data/<result_suffix>/

Default behavior:
- result_suffix: sector_40h_adamw_0409
- models: gawf rnn
- curves: both train and valid

Special model token handling:
- "gawf" selects gawf pkl without fb suffix.
- "gawf_fb50" selects gawf pkl with "_fb50" suffix.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fix numpy version compatibility for pickle loading.
import numpy.core.numeric as _num

try:
    import numpy._core.numeric  # type: ignore # noqa: F401
except ImportError:
    import sys

    sys.modules["numpy._core.numeric"] = _num
    sys.modules["numpy._core"] = np.core

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils_anal.anal_paths import output_dir


DEFAULT_RESULT_SUFFIX = "sector_40h_adamw_0409"
DEFAULT_MODELS = ["gawf", "rnn"]

CURVE_CHOICES = ("train", "valid", "both")

# First model keeps legacy/default matplotlib colors.
MODEL_COLOR_PAIRS: Sequence[Tuple[str, str]] = (
    ("#1f77b4", "#ff7f0e"),  # first model: train/valid
    ("#2ca02c", "#d62728"),
    ("#9467bd", "#8c564b"),
    ("#17becf", "#bcbd22"),
    ("#e377c2", "#7f7f7f"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay train/valid curves from multiple model result pkl files."
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default=DEFAULT_RESULT_SUFFIX,
        help="Subdirectory under results/train_data.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Model tokens to compare, e.g. gawf rnn or gawf_fb50 rnn. "
            "Supports more than two models."
        ),
    )
    parser.add_argument(
        "--curves",
        type=str,
        choices=CURVE_CHOICES,
        default="both",
        help="Which curves to draw: train, valid, or both.",
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start epoch (0-based, inclusive).",
    )
    parser.add_argument(
        "--epoch_end",
        type=int,
        default=None,
        help="End epoch (0-based, exclusive). None means until the last actual epoch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output png path. If None, generated from result_suffix and models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir("G_behaviour", "model_train_compare_result", "figs")),
        help="Output directory.",
    )
    return parser.parse_args()


def _collect_pkl_files(result_dir: Path) -> List[str]:
    return sorted(
        n
        for n in os.listdir(result_dir)
        if n.endswith(".pkl") and not n.startswith(".") and not n.endswith("~")
    )


def _resolve_model_token(model_token: str, pkl_files: Sequence[str]) -> Tuple[str, List[str]]:
    token = model_token.strip().lower()
    if not token:
        raise RuntimeError("Empty model token in --models.")

    if token == "gawf":
        candidates = [n for n in pkl_files if n.startswith("gawf_") and "_fb" not in n]
        if not candidates:
            raise RuntimeError("Cannot find gawf pkl without fb suffix.")
        candidates = sorted(candidates)
        return candidates[0], candidates

    if token.startswith("gawf_fb"):
        fb_suffix = token.split("_", 1)[1]
        marker = f"_{fb_suffix}"
        candidates = [n for n in pkl_files if n.startswith("gawf_") and marker in n]
        if not candidates:
            raise RuntimeError(f"Cannot find gawf pkl with suffix '{fb_suffix}'.")
        candidates = sorted(candidates)
        return candidates[0], candidates

    # Generic token: prefix match "<token>_".
    candidates = [n for n in pkl_files if n.startswith(f"{token}_")]
    if not candidates:
        raise RuntimeError(f"Cannot find pkl for model token '{model_token}'.")
    candidates = sorted(candidates)
    return candidates[0], candidates


def _actual_epochs(results: Dict[str, np.ndarray]) -> int:
    if "actual_epochs" in results:
        return int(results["actual_epochs"])
    arr = np.asarray(results.get("train_acc_char", []))
    if arr.size == 0:
        return 0
    non_zero_mask = arr > 1.0
    if np.any(non_zero_mask):
        return int(np.where(non_zero_mask)[0][-1] + 1)
    return int(arr.shape[0])


def _slice_meta(
    total_epochs: int, epoch_start: int, epoch_end: Optional[int]
) -> Tuple[int, int, np.ndarray]:
    plot_end = total_epochs if epoch_end is None else min(epoch_end, total_epochs)
    plot_start = max(0, min(epoch_start, plot_end))
    x = np.arange(plot_start, plot_end, dtype=np.int64) + 1
    return plot_start, plot_end, x


def _curve_label(model_token: str, curve_kind: str) -> str:
    return f"{model_token} {curve_kind}"


def _plot_curves(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    model_token: str,
    train_color: str,
    valid_color: str,
    curve_mode: str,
) -> None:
    if x.size == 0 or y.size == 0:
        return
    if curve_mode in ("train", "both"):
        ax.plot(
            x,
            y[0],
            linewidth=1.9,
            color=train_color,
            label=_curve_label(model_token, "train"),
        )
    if curve_mode in ("valid", "both"):
        ax.plot(
            x,
            y[1],
            linewidth=1.9,
            color=valid_color,
            label=_curve_label(model_token, "valid"),
        )


def _safe_get_slice(
    results: Dict[str, np.ndarray], key: str, start: int, end: int
) -> Optional[np.ndarray]:
    if key not in results:
        return None
    arr = np.asarray(results[key])
    if arr.ndim == 0:
        return None
    return arr[start:end]


def _setup_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)


def _save_path(args: argparse.Namespace) -> Path:
    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("results") / "train_figs" / args.result_suffix
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    tag = "_vs_".join(args.models)
    return out_dir / f"compare_{tag}_{args.curves}.png"


def main() -> None:
    args = parse_args()
    result_dir = Path("results") / "train_data" / args.result_suffix
    if not result_dir.exists():
        raise RuntimeError(f"Result directory does not exist: {result_dir}")

    pkl_files = _collect_pkl_files(result_dir)
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found in {result_dir}")

    model_entries: List[Tuple[str, str, Dict[str, np.ndarray], np.ndarray]] = []
    for idx, model_token in enumerate(args.models):
        pkl_name, candidates = _resolve_model_token(model_token, pkl_files)
        pkl_path = result_dir / pkl_name
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        n_epoch = _actual_epochs(results)
        start, end, x = _slice_meta(n_epoch, args.epoch_start, args.epoch_end)
        if x.size == 0:
            raise RuntimeError(
                f"Empty epoch slice for model '{model_token}' with range "
                f"[{args.epoch_start}, {args.epoch_end})."
            )

        print(
            f"[{idx}] {model_token}: {pkl_name} | actual_epochs={n_epoch} | "
            f"plot_range=[{start}, {end})"
        )
        if len(candidates) > 1:
            print(
                f"  [说明] 模型 '{model_token}' 存在多匹配 ({len(candidates)} 个)，"
                f"目前使用: {pkl_name}"
            )
        model_entries.append((model_token, pkl_name, results, x))

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.5))
    ax_char_acc, ax_sector_acc = axes[0]
    ax_char_loss, ax_sector_loss = axes[1]
    ax_glob_char, ax_glob_sector = axes[2]

    _setup_axis(ax_char_acc, "Character accuracy", "Accuracy (%)", ylim=(-5.0, 105.0))
    _setup_axis(ax_sector_acc, "Sector accuracy", "Accuracy (%)", ylim=(40.0, 105.0))
    _setup_axis(ax_char_loss, "Character loss", "Loss")
    _setup_axis(ax_sector_loss, "Sector position loss", "Loss (CE)")
    _setup_axis(
        ax_glob_char, "Character accuracy (global frame)", "Accuracy (%)", ylim=(-5.0, 105.0)
    )
    _setup_axis(
        ax_glob_sector, "Sector accuracy (global frame)", "Accuracy (%)", ylim=(40.0, 105.0)
    )

    has_any = {
        "char_loss": False,
        "sector_loss": False,
        "glob_char": False,
        "glob_sector": False,
    }

    for idx, (model_token, pkl_name, results, x) in enumerate(model_entries):
        color_pair = MODEL_COLOR_PAIRS[idx % len(MODEL_COLOR_PAIRS)]
        train_color, valid_color = color_pair
        start = int(x[0] - 1)
        end = int(x[-1])

        # Row 1
        train_char = _safe_get_slice(results, "train_acc_char", start, end)
        valid_char = _safe_get_slice(results, "val_acc_char", start, end)
        if train_char is not None and valid_char is not None:
            _plot_curves(
                ax_char_acc,
                x,
                np.vstack([train_char, valid_char]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        train_pos = _safe_get_slice(results, "train_acc_pos", start, end)
        valid_pos = _safe_get_slice(results, "val_acc_pos", start, end)
        if train_pos is not None and valid_pos is not None:
            _plot_curves(
                ax_sector_acc,
                x,
                np.vstack([train_pos, valid_pos]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        # Row 2
        train_loss_char = _safe_get_slice(results, "train_loss_char", start, end)
        valid_loss_char = _safe_get_slice(results, "val_loss_char", start, end)
        if train_loss_char is not None and valid_loss_char is not None:
            has_any["char_loss"] = True
            _plot_curves(
                ax_char_loss,
                x,
                np.vstack([train_loss_char, valid_loss_char]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        train_loss_pos = _safe_get_slice(results, "train_loss_pos", start, end)
        valid_loss_pos = _safe_get_slice(results, "val_loss_pos", start, end)
        if train_loss_pos is not None and valid_loss_pos is not None:
            has_any["sector_loss"] = True
            _plot_curves(
                ax_sector_loss,
                x,
                np.vstack([train_loss_pos, valid_loss_pos]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        # Row 3 (global acc only; pre5/post5 intentionally not plotted)
        glob_train_char = _safe_get_slice(results, "glob_train_acc_char", start, end)
        glob_valid_char = _safe_get_slice(results, "glob_val_acc_char", start, end)
        if glob_train_char is not None and glob_valid_char is not None:
            has_any["glob_char"] = True
            _plot_curves(
                ax_glob_char,
                x,
                np.vstack([glob_train_char, glob_valid_char]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        glob_train_pos = _safe_get_slice(results, "glob_train_acc_pos", start, end)
        glob_valid_pos = _safe_get_slice(results, "glob_val_acc_pos", start, end)
        if glob_train_pos is not None and glob_valid_pos is not None:
            has_any["glob_sector"] = True
            _plot_curves(
                ax_glob_sector,
                x,
                np.vstack([glob_train_pos, glob_valid_pos]),
                model_token,
                train_color,
                valid_color,
                args.curves,
            )

        print(f"  -> loaded {pkl_name}")

    if not has_any["char_loss"]:
        ax_char_loss.text(0.5, 0.5, "Character loss not saved", ha="center", va="center")
    if not has_any["sector_loss"]:
        ax_sector_loss.text(0.5, 0.5, "Sector position loss not saved", ha="center", va="center")
    if not has_any["glob_char"]:
        ax_glob_char.text(0.5, 0.5, "Global character acc not saved", ha="center", va="center")
    if not has_any["glob_sector"]:
        ax_glob_sector.text(0.5, 0.5, "Global sector acc not saved", ha="center", va="center")

    for ax in axes.reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        " | ".join(
            [
                f"result={args.result_suffix}",
                f"models={','.join(args.models)}",
                f"curves={args.curves}",
            ]
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])

    out_path = _save_path(args)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
