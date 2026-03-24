"""
Visualize the GaWF hidden-to-hidden weight matrix (W_hh) reordered by panel-4
unit ordering (digit groups 0–9 + untuned tail).

Two-panel figure (1 row × 2 cols):
  Left  — raw W_hh values
  Right — z-scored W_hh (default: global; row-wise or col-wise via --z_mode)

Axes convention:
  x-axis = target unit   (W_hh[target, source] in PyTorch convention)
  y-axis = source unit
  top-left = position (0, 0) = first unit in panel-4 order
  Tick labels = npz row indices.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GaWF W_hh connection matrix (panel-4 unit order)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/whh_data",
        help="Directory containing weight_hh.npy and sorted_npz_order.npy.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/whh",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--z_mode",
        type=str,
        default="global",
        choices=["global", "row-wise", "col-wise"],
        help="Normalization mode for the z-scored panel (default: global).",
    )
    parser.add_argument(
        "--unit_tick_step",
        type=int,
        default=0,
        help="Tick step for both axes; 0 = auto (max(1, H//16), capped at 32).",
    )
    parser.add_argument(
        "--vmax_raw",
        type=float,
        default=None,
        help="Symmetric color limit for raw panel (default: abs-max of matrix).",
    )
    parser.add_argument(
        "--vmax_z",
        type=float,
        default=3.0,
        help="Symmetric color limit for z-scored panel (default: 3.0).",
    )
    parser.add_argument(
        "--tuned_only",
        action="store_true",
        help=(
            "If set, only show connections between tuned units "
            "(digit groups 0–9 in panel-4; excludes the untuned tail). "
            "Requires n_tuned.npy in --input_dir."
        ),
    )
    return parser.parse_args()


def load_data(input_dir: str, tuned_only: bool):
    whh_path = os.path.join(input_dir, "weight_hh.npy")
    ord_path = os.path.join(input_dir, "sorted_npz_order.npy")
    for p in (whh_path, ord_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run anal_utils/analyze_gawf_connection_matrix.py first."
            )
    W_hh = np.load(whh_path).astype(np.float32)
    sorted_npz_order = np.load(ord_path).astype(np.int64)
    if W_hh.ndim != 2 or W_hh.shape[0] != W_hh.shape[1]:
        raise ValueError(f"Expected square W_hh, got {W_hh.shape}")
    if sorted_npz_order.size != W_hh.shape[0]:
        raise ValueError(
            f"sorted_npz_order size {sorted_npz_order.size} != W_hh dim {W_hh.shape[0]}"
        )
    if tuned_only:
        ntuned_path = os.path.join(input_dir, "n_tuned.npy")
        if not os.path.isfile(ntuned_path):
            raise FileNotFoundError(
                f"Required file not found: {ntuned_path}\n"
                "Re-run anal_utils/analyze_gawf_connection_matrix.py to generate it."
            )
        n_tuned = int(np.load(ntuned_path))
        sorted_npz_order = sorted_npz_order[:n_tuned]
        print(f"tuned_only: using first {n_tuned} units (digit groups 0–9).")
    return W_hh, sorted_npz_order


def reorder_matrix(W_hh: np.ndarray, sorted_npz_order: np.ndarray) -> np.ndarray:
    """
    Reorder W_hh into (source, target) orientation with panel-4 unit ordering.

    PyTorch convention: W_hh[target, source].
    Display convention: row = source, col = target → W_hh.T reordered by sorted_npz_order.
    """
    idx = sorted_npz_order
    return W_hh.T[idx, :][:, idx].astype(np.float32)


def compute_zscore(matrix: np.ndarray, mode: str) -> np.ndarray:
    H = matrix.shape[0]
    if mode == "global":
        mu = float(matrix.mean())
        sigma = float(matrix.std())
        sigma = sigma if sigma > 1e-8 else 1e-8
        return ((matrix - mu) / sigma).astype(np.float32)
    if mode == "row-wise":
        z = np.zeros_like(matrix, dtype=np.float32)
        for i in range(H):
            row = matrix[i]
            mu, sigma = float(row.mean()), float(row.std())
            sigma = sigma if sigma > 1e-8 else 1e-8
            z[i] = (row - mu) / sigma
        return z
    if mode == "col-wise":
        z = np.zeros_like(matrix, dtype=np.float32)
        for j in range(H):
            col = matrix[:, j]
            mu, sigma = float(col.mean()), float(col.std())
            sigma = sigma if sigma > 1e-8 else 1e-8
            z[:, j] = (col - mu) / sigma
        return z
    raise ValueError(f"Unknown z_mode: {mode}")


def plot_connection_matrix(
    W_sorted: np.ndarray,
    W_z: np.ndarray,
    sorted_npz_order: np.ndarray,
    out_path: str,
    z_mode: str,
    unit_tick_step: int,
    vmax_raw: float | None,
    vmax_z: float,
) -> None:
    H = W_sorted.shape[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if unit_tick_step <= 0:
        unit_tick_step = max(1, min(32, H // 16))
    ticks = list(range(0, H, unit_tick_step))
    if H - 1 not in ticks:
        ticks.append(H - 1)
    tick_labels = [str(int(sorted_npz_order[i])) for i in ticks]

    if vmax_raw is None:
        vmax_raw = float(np.abs(W_sorted).max())
    vmax_raw = max(vmax_raw, 1e-8)

    z_mode_label = {
        "global": "Global",
        "row-wise": "Row-wise",
        "col-wise": "Col-wise",
    }[z_mode]

    fig_side = max(8.0, min(14.0, 8.0 * (H / 256.0)))
    fig, axes = plt.subplots(1, 2, figsize=(fig_side * 2 + 1.5, fig_side))
    ax0, ax1 = axes

    _cbar_kw = {"pad": 0.02, "fraction": 0.046}
    _imshow_kw = dict(
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        cmap="RdBu_r",
    )

    im0 = ax0.imshow(W_sorted, **_imshow_kw, vmin=-vmax_raw, vmax=vmax_raw)
    ax0.set_title("W_hh — Raw weights\n(source × target, panel-4 order)")
    ax0.set_xlabel("Target unit (npz row index)")
    ax0.set_ylabel("Source unit (npz row index)")
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax0.set_yticks(ticks)
    ax0.set_yticklabels(tick_labels)
    fig.colorbar(im0, ax=ax0, **_cbar_kw)

    im1 = ax1.imshow(W_z, **_imshow_kw, vmin=-vmax_z, vmax=vmax_z)
    ax1.set_title(f"W_hh — {z_mode_label} Z-score\n(source × target, panel-4 order)")
    ax1.set_xlabel("Target unit (npz row index)")
    ax1.set_ylabel("Source unit (npz row index)")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_labels)
    fig.colorbar(im1, ax=ax1, **_cbar_kw)

    fig.suptitle(
        "GaWF Hidden-to-Hidden Weight Matrix (W_hh)\n"
        "Units ordered by panel-4 (FDR + effect filtered, digit groups 0–9 + untuned tail)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    suffix = args.z_mode.replace("-", "")
    if args.tuned_only:
        suffix += "_tuned"
    out_path = os.path.join(save_dir, f"gawf_connection_matrix_{suffix}.png")

    W_hh, sorted_npz_order = load_data(input_dir, args.tuned_only)
    W_sorted = reorder_matrix(W_hh, sorted_npz_order)
    W_z = compute_zscore(W_sorted, args.z_mode)

    plot_connection_matrix(
        W_sorted=W_sorted,
        W_z=W_z,
        sorted_npz_order=sorted_npz_order,
        out_path=out_path,
        z_mode=args.z_mode,
        unit_tick_step=args.unit_tick_step,
        vmax_raw=args.vmax_raw,
        vmax_z=args.vmax_z,
    )


if __name__ == "__main__":
    main()
