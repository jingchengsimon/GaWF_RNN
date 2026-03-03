"""
Visualize GaWF gate matrices (gate_ih / gate_hh) as heatmaps.

This script expects the gate dictionary produced by export_gawf_gates.py,
and generates one heatmap per matrix:
- gate_ih: y = hidden units, x = input features
- gate_hh: y = hidden units, x = hidden units

Example commands:

  # 基本可视化（PNG）
  python viz_gawf_gates.py --in ./gawf_gates.pt --outdir ./gawf_gate_figs

  # 使用 z-score 归一化并按 1%% 上分位裁剪
  python viz_gawf_gates.py --in ./gawf_gates.pt --outdir ./gawf_gate_figs \\
      --normalize zscore --clip 0.01
"""

import argparse
import os
from typing import Optional

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GaWF gate matrices (gate_ih / gate_hh) as heatmaps."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default='./gawf_gates.pt',
        type=str,
        help="Path to gate dictionary file (torch.save from export_gawf_gates.py).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./gawf_gate_figs",
        help="Output directory for figures (default: ./gawf_gate_figs).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help=(
            "Optional clipping for color scale. "
            "If 0 < clip < 0.5, interpreted as upper tail percentile (e.g. 0.01 -> clip top 1%%). "
            "If clip >= 0.5, interpreted as absolute symmetric bound for normalized values."
        ),
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "zscore", "minmax"],
        help="Normalization mode before plotting: none / zscore / minmax (default: none).",
    )
    return parser.parse_args()


def normalize_matrix(mat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mat
    flat = mat.astype(np.float64).reshape(-1)
    if mode == "zscore":
        mean = float(flat.mean())
        std = float(flat.std())
        if std == 0.0:
            return mat * 0.0
        return (mat - mean) / std
    if mode == "minmax":
        vmin = float(flat.min())
        vmax = float(flat.max())
        if vmax <= vmin:
            return mat * 0.0
        return (mat - vmin) / (vmax - vmin)
    raise ValueError(f"Unknown normalize mode: {mode}")


def clip_matrix(mat: np.ndarray, clip: Optional[float], normalized: bool) -> np.ndarray:
    if clip is None:
        return mat
    flat = mat.reshape(-1)
    if 0.0 < clip < 0.5:
        # Percentile-based clipping: remove extreme upper (and lower if normalized) values.
        if normalized:
            abs_flat = np.abs(flat)
            bound = float(np.quantile(abs_flat, 1.0 - clip))
            if bound <= 0:
                return mat
            return np.clip(mat, -bound, bound)
        else:
            upper = float(np.quantile(flat, 1.0 - clip))
            lower = float(np.quantile(flat, clip))
            return np.clip(mat, lower, upper)
    else:
        # Absolute symmetric bound (mainly for zscore).
        if clip <= 0:
            return mat
        bound = float(clip)
        if normalized:
            return np.clip(mat, -bound, bound)
        return np.clip(mat, mat.min(), bound)


def plot_heatmap(
    mat: np.ndarray,
    title: str,
    out_path: str,
    vlim: float,
    cmap: str = "RdBu_r",
) -> None:
    """Gate values are centered; colorbar is symmetric [-vlim, vlim], blue (negative) to red (positive)."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        mat, aspect="auto", origin="lower", interpolation="nearest",
        vmin=-vlim, vmax=vlim, cmap=cmap,
    )
    plt.colorbar(im)
    plt.xlabel("Input index")
    plt.ylabel("Hidden index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure: {out_path}")


def main() -> None:
    args = parse_args()

    gate_dict = torch.load(args.in_path, map_location="cpu")

    gate_ih = gate_dict.get("gate_ih", None)
    gate_hh = gate_dict.get("gate_hh", None)
    if gate_ih is None or gate_hh is None:
        raise KeyError("Input file must contain 'gate_ih' and 'gate_hh' tensors.")

    if isinstance(gate_ih, torch.Tensor):
        gate_ih_np = gate_ih.detach().cpu().numpy()
    else:
        gate_ih_np = np.asarray(gate_ih)

    if isinstance(gate_hh, torch.Tensor):
        gate_hh_np = gate_hh.detach().cpu().numpy()
    else:
        gate_hh_np = np.asarray(gate_hh)

    split = gate_dict.get("split", "unknown")
    sample_index = gate_dict.get("sample_index", -1)

    os.makedirs(args.outdir, exist_ok=True)

    # Center gate at 0: (gate - 0.5) -> range [-0.5, 0.5]; then optional normalize/clip.
    gate_ih_centered = gate_ih_np.astype(np.float64) - 0.5
    gate_hh_centered = gate_hh_np.astype(np.float64) - 0.5
    is_normalized = args.normalize != "none"
    gate_ih_proc = clip_matrix(
        normalize_matrix(gate_ih_centered, args.normalize), args.clip, normalized=is_normalized
    )
    gate_hh_proc = clip_matrix(
        normalize_matrix(gate_hh_centered, args.normalize), args.clip, normalized=is_normalized
    )

    # Use a single symmetric range for both matrices based on their joint max magnitude.
    max_abs = float(
        max(
            np.abs(gate_ih_proc).max(initial=0.0),
            np.abs(gate_hh_proc).max(initial=0.0),
        )
    )
    if max_abs == 0.0:
        max_abs = 1e-8

    ih_shape = gate_ih_np.shape
    hh_shape = gate_hh_np.shape

    suffix = args.format.lower()
    ih_path = os.path.join(args.outdir, f"gate_ih.{suffix}")
    hh_path = os.path.join(args.outdir, f"gate_hh.{suffix}")

    ih_title = (
        f"GaWF gate_ih (hidden x input) shape={ih_shape}, "
        f"split={split}, sample_index={sample_index}"
    )
    hh_title = (
        f"GaWF gate_hh (hidden x hidden) shape={hh_shape}, "
        f"split={split}, sample_index={sample_index}"
    )

    # Colorbar: blue (negative) to red (positive), symmetric range [-max_abs, max_abs]
    plot_heatmap(gate_ih_proc, ih_title, ih_path, vlim=max_abs, cmap="RdBu_r")
    plot_heatmap(gate_hh_proc, hh_title, hh_path, vlim=max_abs, cmap="RdBu_r")


if __name__ == "__main__":
    main()

