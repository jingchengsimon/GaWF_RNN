"""Export sector-feedback V spatial modulation maps.

This script loads a trained single-layer GaWF checkpoint, extracts the input-side
V rows corresponding to sector feedback slots, averages over CNN feature channels,
and writes both the signed 6x6 maps and a 3x3 visualization.

Outputs (in --save_dir / --output_dir):
- sector_v_maps.npy  ((9, 6, 6)), float32 - signed sector V maps
- fig1_sector_v_maps.png - 3x3 signed diverging heatmap grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_anal.anal_helpers import build_model_from_ckpt, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export sector feedback V input-side spatial maps for GaWF."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=(
            "./results/train_data/clutter_best_6model_param_matched_40h/"
            "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
        ),
        help="Path to GaWF checkpoint.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("B_gate_by_context", "1_sector_v_modulation", "data")),
        help="Directory for .npy and metadata outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(output_dir("B_gate_by_context", "1_sector_v_modulation", "figs")),
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix appended to output file stems, e.g. 0317.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def infer_feature_shape(model: torch.nn.Module, device: torch.device) -> Tuple[int, int, int]:
    """Infer encoder output shape (C, H, W) from a dummy 96x96 two-channel frame."""
    dummy = torch.zeros(1, 2, 96, 96, device=device, dtype=torch.float32)
    with torch.no_grad():
        feat = model.encoder(dummy)
    if feat.ndim != 4:
        raise RuntimeError(f"Expected encoder output to be 4D; got {tuple(feat.shape)}.")
    _, channels, height, width = feat.shape
    return int(channels), int(height), int(width)


def compute_sector_v_maps(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """Compute signed sector maps from V rows 10..18 as a float32 (9, H, W) array."""
    if not hasattr(model, "V"):
        raise RuntimeError("Checkpoint model does not expose single-layer GaWF parameter V.")

    V = model.V.detach().to(device="cpu", dtype=torch.float32)
    input_size = int(model.encoder_flatten_size)
    recurrent_size = int(model.rnn.hidden_size)
    channels, height, width = infer_feature_shape(model, device)

    if channels * height * width != input_size:
        raise RuntimeError(
            f"Feature shape {(channels, height, width)} does not match input_size={input_size}."
        )
    if V.ndim != 2 or V.shape[1] != input_size + recurrent_size:
        raise RuntimeError(
            f"Unexpected V shape {tuple(V.shape)}; expected (*, {input_size + recurrent_size})."
        )

    num_classes = int(model.num_classes)
    num_pos = int(model.num_pos)
    if num_classes != 10 or num_pos != 9 or V.shape[0] != num_classes + num_pos:
        raise RuntimeError(
            "This script expects single-layer sector GaWF with "
            f"num_classes=10, num_pos=9, V rows=19; got "
            f"num_classes={num_classes}, num_pos={num_pos}, V.shape={tuple(V.shape)}."
        )

    maps = []
    for sector in range(num_pos):
        row_idx = num_classes + sector
        v_in = V[row_idx, :input_size]
        maps.append(v_in.view(channels, height, width).mean(dim=0).numpy())
    return np.stack(maps, axis=0).astype(np.float32, copy=False)


def plot_sector_maps(sector_maps: np.ndarray, output_path: str) -> None:
    """Save a 3x3 diverging heatmap grid with one panel per sector."""
    vmax = float(np.max(np.abs(sector_maps)))
    if vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), constrained_layout=True)
    image = None
    for sector, ax in enumerate(axes.flat):
        image = ax.imshow(
            sector_maps[sector],
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(f"sector {sector}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="mean V")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    """Run V-map export and visualization."""
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.abspath(args.ckpt)
    model = build_model_from_ckpt(ckpt_path, num_pos=9, device=device)
    sector_maps = compute_sector_v_maps(model, device)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    maps_path = os.path.join(args.save_dir, f"sector_v_maps{suffix}.npy")
    fig_path = os.path.join(args.output_dir, f"fig1_sector_v_maps{suffix}.png")
    meta_path = os.path.join(args.save_dir, f"meta{suffix}.json")

    np.save(maps_path, sector_maps.astype(np.float32, copy=False))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": ckpt_path,
                "shape": list(sector_maps.shape),
                "num_classes": int(model.num_classes),
                "num_pos": int(model.num_pos),
                "input_size": int(model.encoder_flatten_size),
                "hidden_size": int(model.rnn.hidden_size),
            },
            f,
            indent=2,
        )
    plot_sector_maps(sector_maps, fig_path)

    print(f"Saved sector V maps to: {maps_path}")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
