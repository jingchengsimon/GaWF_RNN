"""Plot sequential-feedback, equal-n sector input-gate spatial means.

Input is ``sector_gate_mean_sequential_equal_n.npz`` from the matching analysis module. Outputs
are point-included and point-excluded sector-only 3x3 grids in both PNG and PDF formats.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir("B_gate_by_context", "sector_sigmoid_gate_sequential", "data")
            / "sector_gate_mean_sequential_equal_n.npz"
        ),
    )
    parser.add_argument(
        "--fig_dir",
        default=str(output_dir("B_gate_by_context", "sector_sigmoid_gate_sequential", "figs")),
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def plot_sector_grid(maps: np.ndarray, point_key: str, fig_dir: str, dpi: int) -> tuple[str, str]:
    """Write one sector-only 3x3 raw gate-mean grid as PNG and PDF."""

    values = np.asarray(maps, dtype=np.float32)
    if values.shape != (9, 6, 6):
        raise ValueError(f"Expected maps with shape (9, 6, 6), got {values.shape}")
    if point_key not in ("point_included", "point_excluded"):
        raise ValueError("point_key must be point_included or point_excluded")
    suffix = "included" if point_key == "point_included" else "excluded"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), constrained_layout=True)
    image = None
    for sector, axis in enumerate(axes.flat):
        image = axis.imshow(
            values[sector],
            cmap="RdBu_r",
            norm=norm,
            interpolation="none",
        )
        axis.set_title(f"Sector {sector}", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    fig.suptitle("Sequential input-gate mean (equal-n sectors)\n" f"0.5 point mass {suffix}")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    stem = f"fig2_sector_gate_mean_sequential_equal_n_{point_key}"
    png_path = os.path.join(fig_dir, f"{stem}.png")
    pdf_path = os.path.join(fig_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def main() -> None:
    """Load both gate-mean definitions and render their sector grids."""

    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as loaded:
        for point_key in ("point_included", "point_excluded"):
            plot_sector_grid(loaded[point_key], point_key, args.fig_dir, args.dpi)


if __name__ == "__main__":
    main()
