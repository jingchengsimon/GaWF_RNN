"""
Visualize avg gate_ih (Panel 2, sigmoid) or avg outer_ih (Panel 1, U*fb*V)
as a 3×3 sector summary RF gallery.

Data files produced by export_gate_avg.py (sector mode, --agg space):
  Panel 2: avg_gate_ih_s{k}_space.npy   — sigmoid gate,      shape (36, H), values [0, 1]
  Panel 1: avg_outer_ih_s{k}_space.npy  — rank-1 outer prod, shape (36, H), signed values

Unit-selection logic mirrors utils_viz/V_basis.py sector_summary:
  Panel 2 (sigmoid, all-positive): rank by mean desc → pick top-1.
  Panel 1 (signed):  rank by abs_mean desc; among those, prefer units whose
    peak spatial block is positive; fall back to global top-1 by abs_mean.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3×3 sector summary gallery from export_gate_avg.py sector outputs.\n"
            "  --panel 2 (default): avg_gate_ih   (sigmoid, [0,1] colorbar)\n"
            "  --panel 1          : avg_outer_ih  (U*fb*V, symmetric colorbar)"
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/gate_avg/sector",
        help="Directory containing avg_gate_ih_s{k}_space.npy / avg_outer_ih_s{k}_space.npy.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/gate_avg",
        help="Root directory to save the output figure.",
    )
    parser.add_argument(
        "--panel",
        type=int,
        default=2,
        choices=[1, 2],
        help=(
            "Which panel to visualize: "
            "2 = avg gate_ih (sigmoid, default); "
            "1 = avg outer_ih (rank-1 U*fb*V, no sigmoid)."
        ),
    )
    parser.add_argument(
        "--rf_h",
        type=int,
        default=6,
        help="Spatial height of the RF grid (default: 6).",
    )
    parser.add_argument(
        "--rf_w",
        type=int,
        default=6,
        help="Spatial width of the RF grid (default: 6).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Override colorbar upper bound.  "
            "Panel 2: sets [0, vmax]; Panel 1: sets [-vmax, vmax].  "
            "If not given, computed from data."
        ),
    )
    return parser.parse_args()


def _pick_top_unit_sigmoid(rf_all: np.ndarray) -> int:
    """Panel 2: all values in [0,1].  Pick unit with highest peak, break ties by mean."""
    H = rf_all.shape[0]
    mean_act = rf_all.mean(axis=(1, 2))     # (H,)
    flat = rf_all.reshape(H, -1)
    peak_val = flat.max(axis=1)             # (H,)
    order = np.lexsort((-mean_act, -peak_val))
    return int(order[0])


def _pick_top_unit_signed(rf_all: np.ndarray) -> int:
    """
    Panel 1: signed values.  Mirrors V_basis.py sector_summary:
      1. Rank by abs_mean desc.
      2. Among those, prefer units whose peak block has a positive signed value.
      3. Among positives: sort by peak_abs desc, abs_mean as secondary.
      4. Fallback to global top-1 by abs_mean.
    """
    H = rf_all.shape[0]
    flat = rf_all.reshape(H, -1)            # (H, 36)
    abs_mean = np.mean(np.abs(rf_all), axis=(1, 2))  # (H,)
    order = np.argsort(-abs_mean)           # desc

    abs_flat = np.abs(flat)
    peak_idx = np.argmax(abs_flat, axis=1)  # (H,)
    peak_signed = flat[np.arange(H), peak_idx]       # (H,)
    peak_abs = np.abs(peak_signed)                    # (H,)

    pos_units = [int(u) for u in order if float(peak_signed[u]) > 0.0]
    if pos_units:
        pu = np.asarray(pos_units, dtype=np.int64)
        sort_idx = np.lexsort((-abs_mean[pu], -peak_abs[pu]))
        return int(pu[sort_idx[0]])
    return int(order[0])


def main() -> None:
    args = parse_args()

    panel = int(args.panel)
    rf_h, rf_w = int(args.rf_h), int(args.rf_w)
    n_spatial = rf_h * rf_w  # 36

    in_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    summary_dir = os.path.join(save_dir, "sector_summary")
    os.makedirs(summary_dir, exist_ok=True)

    file_prefix = "avg_gate_ih" if panel == 2 else "avg_outer_ih"
    pick_fn = _pick_top_unit_sigmoid if panel == 2 else _pick_top_unit_signed

    selected = []
    for sector_id in range(9):
        fname = f"{file_prefix}_s{sector_id}_space.npy"
        fpath = os.path.join(in_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Missing file for sector {sector_id}: {fpath}\n"
                f"Run export_gate_avg.py with --sector {{k}} --agg space first."
            )
        mat = np.load(fpath)  # (36, H)
        if mat.ndim != 2 or mat.shape[0] != n_spatial:
            raise ValueError(
                f"Expected shape ({n_spatial}, H) in {fname}, got {mat.shape}"
            )
        H = mat.shape[1]
        rf_all = mat.T.reshape(H, rf_h, rf_w)  # (H, 6, 6)

        pick = pick_fn(rf_all)
        selected.append(
            {
                "sector": int(sector_id),
                "unit": int(pick),
                "rf": rf_all[pick],  # (6, 6)
            }
        )
        print(
            f"  sector={sector_id}  top unit={pick}  "
            f"mean={float(rf_all[pick].mean()):.4f}  "
            f"peak={float(rf_all[pick].max()):.4f}"
        )

    # --- colorbar range ---
    if panel == 2:
        if args.vmax is not None:
            vmax = float(args.vmax)
        else:
            vmax = float(np.max([s["rf"].max() for s in selected]))
        if vmax <= 0.0:
            vmax = 1e-8
        vmin = 0.0
        cmap = "viridis"
        cbar_label = "sigmoid gate"
        cbar_ticks = np.arange(0.0, 1.0 + 1e-9, 0.2)
    else:
        if args.vmax is not None:
            m = float(args.vmax)
        else:
            m = float(np.max([np.abs(s["rf"]).max() for s in selected]))
        if m == 0.0:
            m = 1e-8
        vmin, vmax = -m, m
        cmap = "RdBu_r"
        cbar_label = "U·fb·V (outer)"
        cbar_ticks = np.linspace(-m, m, 5)

    # --- plot ---
    title_str = (
        "Avg sigmoid gate on input-to-hidden connections (GaWF RNN) — sector summary"
        if panel == 2 else
        "Avg rank-1 outer U·fb·V on input-to-hidden connections (GaWF RNN) — sector summary"
    )
    out_name = (
        "avg_gate_ih_space_sector_summary_top1.png"
        if panel == 2 else
        "avg_outer_ih_space_sector_summary_top1.png"
    )

    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.5))
    im0 = None
    for s in selected:
        k = int(s["sector"])
        r = 2 - (k // 3)  # bottom-up
        c = k % 3
        ax = axes[r, c]
        im0 = ax.imshow(
            s["rf"],
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sector={k} | unit={s['unit']}")

    fig.suptitle(title_str, y=0.985, size=13, weight="bold")
    fig.subplots_adjust(
        left=0.05, right=0.88, top=0.93, bottom=0.05, wspace=0.15, hspace=0.25,
    )

    if im0 is not None:
        grid_bbox = Bbox.union([ax.get_position() for ax in axes.ravel()])
        cbar_width = 0.03
        pad = 0.03
        x0 = min(float(grid_bbox.x1) + pad, 0.98 - cbar_width)
        cax = fig.add_axes([x0, float(grid_bbox.y0), cbar_width, float(grid_bbox.height)])
        cb = fig.colorbar(im0, cax=cax)
        cb.set_ticks(cbar_ticks)
        cb.set_label(cbar_label, fontsize=9)

    out_path = os.path.join(summary_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
