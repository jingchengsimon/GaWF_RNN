"""Export sector-conditioned sigmoid gate spatial maps.

This script computes the full mixed GaWF input-side sigmoid gate for every test
frame, routes each frame to its foreground sector, and averages the resulting
gate maps per sector.

Outputs (in --save_dir / --output_dir):
- sector_gate_mean.npy  ((9, 6, 6)), float32 - mean over hidden units
- sector_gate_max.npy   ((9, 6, 6)), float32 - max over hidden units
- meta.json - gate temperature, checkpoint, and per-sector frame counts
- fig2_sector_gate_mean.png - 3x3 viridis heatmap grid in [0, 1]
- fig2_sector_gate_max.png - 3x3 viridis heatmap grid in [0, 1]
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

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute sector-conditioned full sigmoid gate input-side maps."
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
        default="./results/anal_data/2_sector_sigmoid_gate",
        help="Directory for .npy and metadata outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/anal_figs/2_sector_sigmoid_gate",
        help="Directory for figure outputs.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="Also save gate(s) minus the mean over sectors for mean/max maps.",
    )
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


def _validate_single_layer_gawf(model: torch.nn.Module) -> None:
    """Ensure the checkpoint has the single-layer direct-feedback GaWF layout used here."""
    if not hasattr(model, "U") or not hasattr(model, "V"):
        raise RuntimeError("This script expects a single-layer GaWF model with U and V.")
    if int(model.num_classes) != 10 or int(model.num_pos) != 9:
        raise RuntimeError(
            f"Expected num_classes=10 and num_pos=9; got {model.num_classes}, {model.num_pos}."
        )
    if int(model.V.shape[0]) != int(model.num_classes + model.num_pos):
        raise RuntimeError(
            f"Expected V feedback rows=19; got V.shape={tuple(model.V.shape)}. "
            "Projected-feedback checkpoints are not supported by this measurement."
        )


def compute_sector_gate_maps(
    test_ds,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate full sigmoid gate_ih by foreground sector and return mean/max maps."""
    _validate_single_layer_gawf(model)
    model.eval()

    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.encoder_flatten_size)
    fb_dim = int(model.num_classes + model.num_pos)
    channels, height, width = infer_feature_shape(model, device)
    if channels * height * width != input_size:
        raise RuntimeError(
            f"Feature shape {(channels, height, width)} does not match input_size={input_size}."
        )

    acc = np.zeros((9, hidden_size, input_size), dtype=np.float64)
    counts = np.zeros((9,), dtype=np.int64)
    V_ih = model.V[:, :input_size].unsqueeze(0)
    tau = float(model.gate_tau)

    with torch.no_grad():
        for sidx in range(len(test_ds)):
            sample = test_ds[sidx]
            frames, labels = sample[0], sample[1]
            labels_np = (
                labels.cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else np.asarray(labels)
            )
            if labels_np.ndim < 2 or labels_np.shape[1] <= 1:
                raise ValueError(
                    "Expected labels with sector column at index 1; "
                    f"got {labels_np.shape}."
                )

            frames_t = (
                frames.to(device=device, dtype=torch.float32)
                if isinstance(frames, torch.Tensor)
                else torch.as_tensor(frames, dtype=torch.float32, device=device)
            )
            encoded = model.encoder(frames_t).view(frames_t.shape[0], -1)

            for t in range(encoded.shape[0]):
                sector = int(labels_np[t, 1])
                if sector < 0 or sector >= 9:
                    continue

                x_t = encoded[t : t + 1]
                h0 = torch.zeros(1, hidden_size, device=device, dtype=x_t.dtype)
                fb0 = torch.zeros(1, fb_dim, device=device, dtype=x_t.dtype)
                gated_out = model.middle_gawf(x_t, h0, fb0.clamp(-10, 10).unsqueeze(2))
                char_t, pos_t = model.classifier(gated_out)
                fb1 = torch.cat([char_t, pos_t], dim=-1).clamp(-10, 10).unsqueeze(2)
                trans_ih = torch.matmul(model.U, fb1 * V_ih)
                gate_ih = torch.sigmoid(trans_ih / tau).squeeze(0)

                acc[sector] += gate_ih.cpu().numpy().astype(np.float64, copy=False)
                counts[sector] += 1

            if (sidx + 1) % 200 == 0:
                count_text = ", ".join(str(int(x)) for x in counts.tolist())
                print(f"[{sidx + 1}/{len(test_ds)}] sector frame counts: [{count_text}]")

    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise RuntimeError(f"No frames found for sector(s): {missing}")

    avg_gate = acc / counts[:, None, None]
    avg_gate = avg_gate.reshape(9, hidden_size, channels, height, width).mean(axis=2)
    mean_maps = avg_gate.mean(axis=1).astype(np.float32, copy=False)
    max_maps = avg_gate.max(axis=1).astype(np.float32, copy=False)
    return mean_maps, max_maps, counts


def plot_sector_grid(
    maps: np.ndarray,
    output_path: str,
    *,
    cmap: str,
    title_value: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center_value: float | None = None,
    center_zero: bool = False,
) -> None:
    """Save a 3x3 heatmap grid."""
    if center_value is not None:
        if vmin is None or vmax is None:
            raise ValueError("center_value requires explicit vmin and vmax.")
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
        vmin = None
        vmax = None
    elif center_zero:
        absmax = float(np.max(np.abs(maps)))
        if absmax <= 0:
            absmax = 1.0
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
        vmin = None
        vmax = None
    else:
        norm = None

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8), constrained_layout=True)
    image = None
    for sector, ax in enumerate(axes.flat):
        image = ax.imshow(
            maps[sector],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(f"sector {sector}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label=title_value)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    """Run sector-conditioned gate export and visualization."""
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size: {len(test_ds)}")

    ckpt_path = os.path.abspath(args.ckpt)
    print(f"Building model from: {ckpt_path}")
    model = build_model_from_ckpt(ckpt_path, num_pos=num_pos, device=device)

    mean_maps, max_maps, counts = compute_sector_gate_maps(test_ds, model, device)

    mean_path = os.path.join(args.save_dir, "sector_gate_mean.npy")
    max_path = os.path.join(args.save_dir, "sector_gate_max.npy")
    meta_path = os.path.join(args.save_dir, "meta.json")
    mean_fig = os.path.join(args.output_dir, "fig2_sector_gate_mean.png")
    max_fig = os.path.join(args.output_dir, "fig2_sector_gate_max.png")

    np.save(mean_path, mean_maps.astype(np.float32, copy=False))
    np.save(max_path, max_maps.astype(np.float32, copy=False))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": ckpt_path,
                "tau": float(model.gate_tau),
                "n_frames_by_sector": counts.astype(int).tolist(),
                "hidden_size": int(model.rnn.hidden_size),
                "input_size": int(model.encoder_flatten_size),
            },
            f,
            indent=2,
        )

    plot_sector_grid(
        mean_maps,
        mean_fig,
        cmap="RdBu_r",
        title_value="mean gate",
        vmin=0.0,
        vmax=1.0,
        center_value=0.5,
    )
    plot_sector_grid(
        max_maps,
        max_fig,
        cmap="RdBu_r",
        title_value="max gate",
        vmin=0.0,
        vmax=1.0,
        center_value=0.5,
    )

    print(f"Saved mean maps to: {mean_path}")
    print(f"Saved max maps to: {max_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved mean figure to: {mean_fig}")
    print(f"Saved max figure to: {max_fig}")

    if args.contrast:
        mean_contrast = (mean_maps - mean_maps.mean(axis=0, keepdims=True)).astype(np.float32)
        max_contrast = (max_maps - max_maps.mean(axis=0, keepdims=True)).astype(np.float32)
        mean_contrast_path = os.path.join(args.save_dir, "sector_gate_mean_contrast.npy")
        max_contrast_path = os.path.join(args.save_dir, "sector_gate_max_contrast.npy")
        mean_contrast_fig = os.path.join(args.output_dir, "fig2_sector_gate_mean_contrast.png")
        max_contrast_fig = os.path.join(args.output_dir, "fig2_sector_gate_max_contrast.png")
        np.save(mean_contrast_path, mean_contrast)
        np.save(max_contrast_path, max_contrast)
        plot_sector_grid(
            mean_contrast,
            mean_contrast_fig,
            cmap="RdBu_r",
            title_value="mean gate contrast",
            center_zero=True,
        )
        plot_sector_grid(
            max_contrast,
            max_contrast_fig,
            cmap="RdBu_r",
            title_value="max gate contrast",
            center_zero=True,
        )
        print(f"Saved contrast maps to: {mean_contrast_path}, {max_contrast_path}")
        print(f"Saved contrast figures to: {mean_contrast_fig}, {max_contrast_fig}")


if __name__ == "__main__":
    main()
