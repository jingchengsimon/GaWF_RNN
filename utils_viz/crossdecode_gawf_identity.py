"""Visualize GaWF identity cross-decoding and alignment matrices.

Reads outputs from ``utils_anal/crossdecode_gawf_identity.py`` and renders:
- cross-domain confusion matrices for activation-to-modulation and modulation-to-activation
- the 10x10 cosine alignment matrix between digit V patterns and activation class weights

Outputs (in --save_dir):
- fig_crossdecode_confusion_A2M.png  — row-normalized confusion heatmap
- fig_crossdecode_confusion_M2A.png  — row-normalized confusion heatmap
- fig_align_matrix.png              — cosine alignment heatmap
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GaWF identity cross-decoding confusion and alignment matrices."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/crossdecode_gawf_identity",
        help="Directory containing analysis outputs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/crossdecode_gawf_identity",
        help="Directory for PNG figures.",
    )
    return parser.parse_args()


def _load_results(data_dir: str) -> Dict[str, float]:
    path = os.path.join(data_dir, "results.json")
    with open(path, "r") as f:
        return json.load(f)


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True).astype(np.float64)
    denom = np.maximum(denom, 1.0)
    return (cm.astype(np.float64) / denom).astype(np.float32)


def _annotate_matrix(ax, values: np.ndarray, fmt: str, threshold: float) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(float(values[i, j])) >= threshold else "black"
            ax.text(
                j,
                i,
                format(float(values[i, j]), fmt),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )


def _plot_confusion(
    cm_path: str,
    out_path: str,
    *,
    title: str,
    accuracy: float,
    p_value: float,
    ceiling: float,
) -> None:
    cm = np.load(cm_path).astype(np.int64)
    if cm.shape != (10, 10):
        raise ValueError(f"Expected 10x10 confusion matrix at {cm_path}, got {cm.shape}")
    row_norm = _row_normalize(cm)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(row_norm, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(
        f"{title}\nacc={accuracy:.3f}, p={p_value:.4g}, activation ceiling={ceiling:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("Predicted foreground digit")
    ax.set_ylabel("True foreground digit")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    _annotate_matrix(ax, row_norm * 100.0, ".0f", threshold=50.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _plot_alignment(align_path: str, out_path: str, results: Dict[str, float]) -> None:
    align = np.load(align_path).astype(np.float32)
    if align.shape != (10, 10):
        raise ValueError(f"Expected 10x10 alignment matrix at {align_path}, got {align.shape}")
    vmax = float(np.max(np.abs(align)))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(6.1, 5.3))
    im = ax.imshow(
        align,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(
        "Digit V pattern vs activation classifier weight\n"
        f"diag-offdiag={results['align_diag_minus_offdiag']:.3f}, "
        f"p={results['align_perm_p']:.4g}",
        fontsize=11,
    )
    ax.set_xlabel("Activation classifier digit weight")
    ax.set_ylabel("Modulation V digit row")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    _annotate_matrix(ax, align, ".2f", threshold=0.5 * vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    results = _load_results(data_dir)

    _plot_confusion(
        os.path.join(data_dir, "confusion_A2M.npy"),
        os.path.join(save_dir, "fig_crossdecode_confusion_A2M.png"),
        title="Activation-trained classifier on GaWF modulation",
        accuracy=float(results["transfer_acc_A2M"]),
        p_value=float(results["perm_p_A2M"]),
        ceiling=float(results["ceiling_acc"]),
    )
    _plot_confusion(
        os.path.join(data_dir, "confusion_M2A.npy"),
        os.path.join(save_dir, "fig_crossdecode_confusion_M2A.png"),
        title="Modulation-trained classifier on CNN activation",
        accuracy=float(results["transfer_acc_M2A"]),
        p_value=float(results["perm_p_M2A"]),
        ceiling=float(results["ceiling_acc"]),
    )
    _plot_alignment(
        os.path.join(data_dir, "align_matrix.npy"),
        os.path.join(save_dir, "fig_align_matrix.png"),
        results,
    )


if __name__ == "__main__":
    main()
