"""Compare digit V channel modulation against CNN channel activation.

This script loads digit-feedback input-side rows of V from a trained GaWF
checkpoint, averages each row over the 6x6 spatial grid, and plots the resulting
channel-by-digit matrix beside CNN channel activation statistics.

Outputs (in --save_dir / --output_dir):
- digit_v_mod.npy  ((32, 10)), float32 - signed V modulation by channel and digit
- digit_gate_channel_mean.npy  ((32, 10)), float32 - sigmoid gate mean over hidden units
- digit_gate_channel_max.npy  ((32, 10)), float32 - sigmoid gate max over hidden units
- align_matrix_zscore.npy  ((10, 10)), float32 - z-scored digit cosine alignment
- align_matrix_sigmoid_mean_zscore.npy  ((10, 10)), float32 - mean-gate alignment
- align_matrix_sigmoid_max_zscore.npy  ((10, 10)), float32 - max-gate alignment
- fig3_digit_v_vs_activation.png - raw activation and V heatmaps
- fig3_digit_v_vs_activation_zscore.png - row-wise channel z-scored heatmaps
- fig3_digit_v_vs_activation_align_matrix_zscore.png - 10x10 alignment heatmap
- fig3_digit_sigmoid_<mean|max>_vs_activation*.png - sigmoid gate alignment figures
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
        description="Compare digit V channel modulation with CNN channel activation."
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
        default="./results/anal_data/3_digit_v_vs_activation",
        help="Directory for .npy and metadata outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/anal_figs/3_digit_v_vs_activation",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--cnn_stats",
        type=str,
        default="./results/anal_data/cnn_channel/cnn_channel_activation_stats.npz",
        help="Path to cnn_channel_activation_stats.npz or its containing directory.",
    )
    parser.add_argument(
        "--channel_order_path",
        type=str,
        default="./results/anal_data/cnn_channel/channel_order_by_cosine_similarity.npy",
        help="Optional channel order produced by cnn_channel_stats.py.",
    )
    parser.add_argument(
        "--no_channel_order",
        action="store_true",
        help="Disable automatic channel reordering when --channel_order_path exists.",
    )
    parser.add_argument(
        "--permutes",
        type=int,
        default=10000,
        help="Number of column-permutation samples for the alignment p value.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument(
        "--skip_sigmoid_gate",
        action="store_true",
        help="Only regenerate the original V-vs-activation outputs.",
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


def resolve_cnn_stats_path(path_or_dir: str) -> str:
    """Resolve a stats file path, accepting either the .npz file or its directory."""
    path = os.path.abspath(path_or_dir)
    if os.path.isdir(path):
        path = os.path.join(path, "cnn_channel_activation_stats.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CNN stats not found: {path}. Run, for example:\n"
            "python utils_anal/cnn_channel_stats.py --ckpt <optim_gawf.pth> --device cpu"
        )
    return path


def compute_digit_v_mod(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """Compute signed digit V modulation as a float32 (C, 10) matrix."""
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
    if int(model.num_classes) != 10 or int(model.num_pos) != 9 or V.shape[0] != 19:
        raise RuntimeError(
            "This script expects single-layer sector GaWF with "
            f"num_classes=10, num_pos=9, V rows=19; got "
            f"num_classes={model.num_classes}, num_pos={model.num_pos}, V.shape={tuple(V.shape)}."
        )

    profiles = []
    for digit in range(10):
        v_in = V[digit, :input_size]
        profiles.append(v_in.view(channels, height, width).mean(dim=(1, 2)).numpy())
    return np.stack(profiles, axis=1).astype(np.float32, copy=False)


def compute_digit_sigmoid_gate_channels(
    test_ds,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute digit-conditioned sigmoid gate profiles as (C, 10) mean/max matrices."""
    if not hasattr(model, "U") or not hasattr(model, "V"):
        raise RuntimeError("Checkpoint model does not expose single-layer GaWF U/V parameters.")
    if int(model.num_classes) != 10 or int(model.num_pos) != 9:
        raise RuntimeError(
            f"Expected num_classes=10 and num_pos=9; got {model.num_classes}, {model.num_pos}."
        )
    if int(model.V.shape[0]) != int(model.num_classes + model.num_pos):
        raise RuntimeError(
            f"Expected direct-feedback V rows=19; got V.shape={tuple(model.V.shape)}."
        )

    model.eval()
    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.encoder_flatten_size)
    fb_dim = int(model.num_classes + model.num_pos)
    channels, height, width = infer_feature_shape(model, device)
    if channels * height * width != input_size:
        raise RuntimeError(
            f"Feature shape {(channels, height, width)} does not match input_size={input_size}."
        )

    acc = np.zeros((10, hidden_size, channels), dtype=np.float64)
    counts = np.zeros((10,), dtype=np.int64)
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
            if labels_np.ndim < 2:
                raise ValueError(f"Expected labels with digit column; got {labels_np.shape}.")

            frames_t = (
                frames.to(device=device, dtype=torch.float32)
                if isinstance(frames, torch.Tensor)
                else torch.as_tensor(frames, dtype=torch.float32, device=device)
            )
            encoded = model.encoder(frames_t).view(frames_t.shape[0], -1)

            for t in range(encoded.shape[0]):
                digit = int(labels_np[t, 0])
                if digit < 0 or digit >= 10:
                    continue

                x_t = encoded[t : t + 1]
                h0 = torch.zeros(1, hidden_size, device=device, dtype=x_t.dtype)
                fb0 = torch.zeros(1, fb_dim, device=device, dtype=x_t.dtype)
                gated_out = model.middle_gawf(x_t, h0, fb0.clamp(-10, 10).unsqueeze(2))
                char_t, pos_t = model.classifier(gated_out)
                fb1 = torch.cat([char_t, pos_t], dim=-1).clamp(-10, 10).unsqueeze(2)
                trans_ih = torch.matmul(model.U, fb1 * V_ih)
                gate_ih = torch.sigmoid(trans_ih / tau).squeeze(0)
                gate_channels = gate_ih.view(hidden_size, channels, height, width).mean(
                    dim=(2, 3)
                )

                acc[digit] += gate_channels.cpu().numpy().astype(np.float64, copy=False)
                counts[digit] += 1

            if (sidx + 1) % 200 == 0:
                count_text = ", ".join(str(int(x)) for x in counts.tolist())
                print(f"[{sidx + 1}/{len(test_ds)}] digit frame counts: [{count_text}]")

    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise RuntimeError(f"No frames found for digit(s): {missing}")

    avg_gate = acc / counts[:, None, None]
    mean_profiles = avg_gate.mean(axis=1).T.astype(np.float32, copy=False)
    max_profiles = avg_gate.max(axis=1).T.astype(np.float32, copy=False)
    return mean_profiles, max_profiles, counts


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    """Z-score each channel row over digits."""
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((matrix - mean) / std).astype(np.float32, copy=False)


def cosine_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute row-wise cosine similarities between two 2D arrays."""
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    a_norm = np.maximum(np.linalg.norm(a64, axis=1, keepdims=True), eps)
    b_norm = np.maximum(np.linalg.norm(b64, axis=1, keepdims=True), eps)
    return ((a64 / a_norm) @ (b64 / b_norm).T).astype(np.float32, copy=False)


def diag_minus_offdiag(matrix: np.ndarray) -> float:
    """Return mean diagonal alignment minus mean off-diagonal alignment."""
    diag = np.diag(matrix)
    off_mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(diag.mean() - matrix[off_mask].mean())


def compute_zscore_alignment(activation: np.ndarray, v_mod: np.ndarray) -> np.ndarray:
    """Compute digit-by-digit cosine alignment after row-wise channel z-scoring."""
    activation_z = zscore_rows(activation)
    v_mod_z = zscore_rows(v_mod)
    return cosine_matrix(v_mod_z.T, activation_z.T)


def alignment_permutation_p(
    align: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_perm: int,
) -> Tuple[float, np.ndarray]:
    """Permutation p value from randomly shuffling activation digit columns."""
    if n_perm <= 0:
        return float("nan"), np.empty((0,), dtype=np.float32)

    null = np.empty((n_perm,), dtype=np.float32)
    for idx in range(n_perm):
        perm = rng.permutation(align.shape[1])
        null[idx] = diag_minus_offdiag(align[:, perm])
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_perm + 1))
    return p_value, null


def _imshow_with_colorbar(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    cmap: str,
    signed: bool,
) -> None:
    """Draw one matrix heatmap with independent colorbar."""
    if signed:
        vmax = float(np.max(np.abs(data)))
        if vmax <= 0:
            vmax = 1.0
        image = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
    else:
        image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("digit")
    ax.set_ylabel("channel")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(0, data.shape[0], 4))
    plt.colorbar(image, ax=ax, shrink=0.88)


def plot_comparison(
    activation: np.ndarray,
    v_mod: np.ndarray,
    output_path: str,
    *,
    zscored: bool,
    right_title_base: str = "V modulation",
    right_signed: bool = True,
) -> None:
    """Save side-by-side activation and V heatmaps."""
    if zscored:
        activation_plot = zscore_rows(activation)
        v_plot = zscore_rows(v_mod)
        left_title = "CNN activation z-score"
        right_title = f"{right_title_base} z-score"
        left_signed = True
        right_signed_plot = True
    else:
        activation_plot = activation
        v_plot = v_mod
        left_title = "CNN activation"
        right_title = right_title_base
        left_signed = False
        right_signed_plot = right_signed

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6.2), constrained_layout=True)
    _imshow_with_colorbar(
        axes[0],
        activation_plot,
        title=left_title,
        cmap="viridis" if not zscored else "RdBu_r",
        signed=left_signed,
    )
    _imshow_with_colorbar(
        axes[1],
        v_plot,
        title=right_title,
        cmap="RdBu_r" if right_signed_plot else "viridis",
        signed=right_signed_plot,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_alignment_matrix(
    align: np.ndarray,
    output_path: str,
    *,
    p_value: float,
    left_label: str = "V modulation digit",
    title: str = "CNN activation vs V modulation z-score alignment",
) -> None:
    """Save a crossdecode-style 10x10 cosine alignment heatmap."""
    if align.shape != (10, 10):
        raise ValueError(f"Expected alignment matrix shape (10, 10), got {align.shape}.")

    vmax = float(np.max(np.abs(align)))
    vmax = max(vmax, 1e-6)
    scalar = diag_minus_offdiag(align)

    fig, ax = plt.subplots(figsize=(6.1, 5.3))
    image = ax.imshow(
        align,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(
        f"{title}\ndiag-offdiag={scalar:.3f}, p={p_value:.4g}",
        fontsize=11,
    )
    ax.set_xlabel("CNN activation digit")
    ax.set_ylabel(left_label)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    for row in range(align.shape[0]):
        for col in range(align.shape[1]):
            value = float(align[row, col])
            color = "white" if abs(value) >= 0.5 * vmax else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_sigmoid_alignment_outputs(
    activation: np.ndarray,
    gate_profile: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    *,
    label: str,
) -> dict[str, float]:
    """Save raw/z-scored comparison and digit alignment for one sigmoid profile."""
    align = compute_zscore_alignment(activation, gate_profile)
    align_scalar = diag_minus_offdiag(align)
    align_p, align_null = alignment_permutation_p(align, align_scalar, rng, args.permutes)

    align_path = os.path.join(args.save_dir, f"align_matrix_sigmoid_{label}_zscore.npy")
    null_path = os.path.join(
        args.save_dir,
        f"null_align_diag_minus_offdiag_sigmoid_{label}_zscore.npy",
    )
    raw_fig = os.path.join(
        args.output_dir,
        f"fig3_digit_sigmoid_{label}_vs_activation.png",
    )
    z_fig = os.path.join(
        args.output_dir,
        f"fig3_digit_sigmoid_{label}_vs_activation_zscore.png",
    )
    align_fig = os.path.join(
        args.output_dir,
        f"fig3_digit_sigmoid_{label}_vs_activation_align_matrix_zscore.png",
    )

    np.save(align_path, align.astype(np.float32, copy=False))
    np.save(null_path, align_null.astype(np.float32, copy=False))
    plot_comparison(
        activation,
        gate_profile,
        raw_fig,
        zscored=False,
        right_title_base=f"Sigmoid gate {label}",
        right_signed=False,
    )
    plot_comparison(
        activation,
        gate_profile,
        z_fig,
        zscored=True,
        right_title_base=f"Sigmoid gate {label}",
        right_signed=False,
    )
    plot_alignment_matrix(
        align,
        align_fig,
        p_value=align_p,
        left_label=f"sigmoid gate {label} digit",
        title=f"CNN activation vs sigmoid gate {label} z-score alignment",
    )

    print(f"Saved sigmoid {label} z-score alignment matrix to: {align_path}")
    print(f"Saved sigmoid {label} z-score alignment null to: {null_path}")
    print(f"Saved sigmoid {label} raw comparison figure to: {raw_fig}")
    print(f"Saved sigmoid {label} z-score comparison figure to: {z_fig}")
    print(f"Saved sigmoid {label} z-score alignment figure to: {align_fig}")
    return {
        "align_diag_minus_offdiag": align_scalar,
        "align_perm_p": align_p,
    }


def main() -> None:
    """Run digit V export and side-by-side plotting."""
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.abspath(args.ckpt)
    model = build_model_from_ckpt(ckpt_path, num_pos=9, device=device)
    digit_v_mod_raw = compute_digit_v_mod(model, device)
    digit_v_mod_plot = digit_v_mod_raw

    v_path = os.path.join(args.save_dir, "digit_v_mod.npy")
    np.save(v_path, digit_v_mod_raw.astype(np.float32, copy=False))
    print(f"Saved digit V modulation to: {v_path}")

    stats_path = resolve_cnn_stats_path(args.cnn_stats)
    stats = np.load(stats_path)
    if "mean_activation" not in stats:
        raise KeyError(f"{stats_path} does not contain key 'mean_activation'.")
    mean_activation = stats["mean_activation"].astype(np.float32, copy=False)
    if mean_activation.shape != digit_v_mod_raw.shape:
        raise RuntimeError(
            f"mean_activation shape {mean_activation.shape} does not match "
            f"digit_v_mod shape {digit_v_mod_raw.shape}."
        )

    channel_order = None
    order_path = os.path.abspath(args.channel_order_path)
    if not args.no_channel_order and os.path.isfile(order_path):
        channel_order = np.load(order_path).astype(np.int64)
        if channel_order.shape != (digit_v_mod_raw.shape[0],):
            raise RuntimeError(
                f"Channel order shape {channel_order.shape} incompatible with "
                f"{digit_v_mod_raw.shape}."
            )
        mean_activation = mean_activation[channel_order]
        digit_v_mod_plot = digit_v_mod_raw[channel_order]

    align = compute_zscore_alignment(mean_activation, digit_v_mod_plot)
    align_scalar = diag_minus_offdiag(align)
    rng = np.random.default_rng(args.seed)
    align_p, align_null = alignment_permutation_p(
        align,
        align_scalar,
        rng,
        args.permutes,
    )
    align_path = os.path.join(args.save_dir, "align_matrix_zscore.npy")
    null_path = os.path.join(args.save_dir, "null_align_diag_minus_offdiag_zscore.npy")
    raw_fig = os.path.join(args.output_dir, "fig3_digit_v_vs_activation.png")
    z_fig = os.path.join(args.output_dir, "fig3_digit_v_vs_activation_zscore.png")
    align_fig = os.path.join(
        args.output_dir,
        "fig3_digit_v_vs_activation_align_matrix_zscore.png",
    )
    meta_path = os.path.join(args.save_dir, "meta.json")

    np.save(align_path, align.astype(np.float32, copy=False))
    np.save(null_path, align_null.astype(np.float32, copy=False))
    plot_comparison(mean_activation, digit_v_mod_plot, raw_fig, zscored=False)
    plot_comparison(mean_activation, digit_v_mod_plot, z_fig, zscored=True)
    plot_alignment_matrix(align, align_fig, p_value=align_p)

    sigmoid_meta = None
    if not args.skip_sigmoid_gate:
        print("Building test dataset for digit-conditioned sigmoid gate channels...")
        test_ds, num_pos = build_test_dataset(args)
        if int(num_pos) != 9:
            raise RuntimeError(f"Expected sector-mode num_pos=9; got {num_pos}.")
        gate_mean, gate_max, digit_counts = compute_digit_sigmoid_gate_channels(
            test_ds,
            model,
            device,
        )
        gate_mean_path = os.path.join(args.save_dir, "digit_gate_channel_mean.npy")
        gate_max_path = os.path.join(args.save_dir, "digit_gate_channel_max.npy")
        np.save(gate_mean_path, gate_mean.astype(np.float32, copy=False))
        np.save(gate_max_path, gate_max.astype(np.float32, copy=False))
        print(f"Saved sigmoid mean channel profile to: {gate_mean_path}")
        print(f"Saved sigmoid max channel profile to: {gate_max_path}")

        gate_mean_plot = gate_mean[channel_order] if channel_order is not None else gate_mean
        gate_max_plot = gate_max[channel_order] if channel_order is not None else gate_max
        mean_sigmoid_meta = save_sigmoid_alignment_outputs(
            mean_activation,
            gate_mean_plot,
            args,
            rng,
            label="mean",
        )
        max_sigmoid_meta = save_sigmoid_alignment_outputs(
            mean_activation,
            gate_max_plot,
            args,
            rng,
            label="max",
        )
        sigmoid_meta = {
            "digit_gate_channel_mean": gate_mean_path,
            "digit_gate_channel_max": gate_max_path,
            "n_frames_by_digit": digit_counts.astype(int).tolist(),
            "gate_channel_reduction": "mean over 6x6 input spatial grid",
            "gate_unit_reductions": [
                "mean over 256 hidden units",
                "max over 256 hidden units",
            ],
            "sigmoid_mean": mean_sigmoid_meta,
            "sigmoid_max": max_sigmoid_meta,
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": ckpt_path,
                "cnn_stats": stats_path,
                "channel_order_path": order_path if channel_order is not None else None,
                "zscore_mode": "row_wise_channel_over_digits",
                "align_matrix": "cosine(V_zscore_digit, CNN_activation_zscore_digit)",
                "align_diag_minus_offdiag": align_scalar,
                "align_perm_p": align_p,
                "align_permutes": int(args.permutes),
                "align_permutation": "shuffle CNN activation digit columns",
                "shape": list(digit_v_mod_raw.shape),
                "sigmoid_gate_channel_alignment": sigmoid_meta,
            },
            f,
            indent=2,
        )

    print(f"Saved z-score alignment matrix to: {align_path}")
    print(f"Saved z-score alignment null to: {null_path}")
    print(f"Saved raw comparison figure to: {raw_fig}")
    print(f"Saved z-score comparison figure to: {z_fig}")
    print(f"Saved z-score alignment figure to: {align_fig}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
