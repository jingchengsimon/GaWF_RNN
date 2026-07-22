"""
Analyze CNN channel activations in a trained GaWF model.

This script:
- Loads a trained GaWFRNNConv checkpoint.
- Runs the CNN encoder only on all frames from a dataset split.
- Computes a 32-dimensional activation vector per frame via spatial mean pooling.
- Saves per-frame activations and labels, and digit-conditioned statistics.

Outputs (in --output_dir):
- activation_per_sample.npy  (N, 32), float32
- labels.npy                 (N,),    int64 foreground digit labels (0-9)
- cnn_channel_activation_stats.npz with:
    - mean_activation    (32, 10)
    - std_activation     (32, 10)
    - digit_sample_count (10,)
"""

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root (containing both utils_anal and utils_viz) is on sys.path,
# so that we can import utils_viz reliably no matter where the script is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils.clutter_task_models import GaWFRNNConv
from utils.clutter_train_helpers import set_seed
from utils_anal.anal_helpers import (
    build_model_from_ckpt,
    build_test_dataset,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CNN channel activation analysis for trained GaWF model."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        help="Path to trained GaWFRNNConv checkpoint (e.g. *_model.pth).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("E_relevance_alignment", "cnn_channel_stats", "data")),
        help="Directory to save activation arrays and statistics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for DataLoader (over sequence samples, default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Computation device to use (cpu/cuda, default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help=(
            "Load stimuli with numpy mmap_mode='r' for lower memory usage "
            "(recommended for large datasets, default: True)."
        ),
    )
    # Dataset-related options (mirroring export_gawf_gates.py)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Base directory for stimuli/labels. If empty, uses ENV (AIM3_STIMULI_PATH / "
            "FAW_RNN_DATA_PATH) or <repo>/stimuli (same resolution logic as training)."
        ),
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for stimulus_reg-* files (e.g. '40h'). "
            "Same semantics as train_model.py."
        ),
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=True,
        help="Use sector mode (3x3 sectors) for position labels (default: True).",
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters instead of only foreground (default: False).",
    )
    return parser.parse_args()


def compute_activations(
    model: GaWFRNNConv,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run CNN encoder over all frames in the dataset and collect activations/labels.

    For each batch:
    - Input frames have shape (B, T, C, H, W).
    - Labels have shape (B, T, 2) in sector mode, with labels[..., 0] = fg digit (0-9).
    - We flatten batch and time, run encoder on (B*T, C, H, W), and mean-pool over 6x6
      spatial dimensions to obtain a 32-d channel activation vector per frame.

    Returns
    -------
    activations:
        Array of shape (N, 32), float32, where N is total number of frames.
    labels:
        Array of shape (N,), int64 foreground digit labels.
    """
    all_acts = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            frames, labels = batch[0], batch[1]
            # frames: (B, T, C, H, W)
            # labels: (B, T, 2) -> use labels[..., 0] as fg digit
            frames = frames.to(device=device, dtype=torch.float32)
            digits = labels[..., 0].to(device="cpu", dtype=torch.int64)  # keep labels on CPU

            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)

            feats = model.encoder(frames_flat)  # (B*T, 32, 6, 6)
            # Spatial mean pooling over 6x6 -> (B*T, 32)
            acts = feats.mean(dim=(2, 3))

            all_acts.append(acts.detach().cpu().numpy())
            all_labels.append(digits.reshape(-1).numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"[batch {batch_idx + 1}] collected {acts.shape[0]} activations")

    activation_per_sample = np.concatenate(all_acts, axis=0).astype(np.float32, copy=False)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)
    return activation_per_sample, labels_np


def compute_digit_stats(
    activations: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute digit-conditioned statistics over CNN channel activations.

    For each digit d in {0..9}, and channel c in {0..31}:
        mean_activation[c, d] = mean over samples with label == d
        std_activation[c, d]  = std  over samples with label == d
        digit_sample_count[d] = number of samples with label == d
    """
    if activations.ndim != 2 or activations.shape[1] != 32:
        raise ValueError(
            f"Expected activations of shape (N, 32), got {activations.shape}"
        )
    if labels.ndim != 1 or activations.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} incompatible with activations shape {activations.shape}"
        )

    num_channels = activations.shape[1]
    num_digits = 10

    mean_activation = np.zeros((num_channels, num_digits), dtype=np.float32)
    std_activation = np.zeros((num_channels, num_digits), dtype=np.float32)
    digit_sample_count = np.zeros((num_digits,), dtype=np.int64)

    for d in range(num_digits):
        mask = labels == d
        count = int(mask.sum())
        digit_sample_count[d] = count
        if count == 0:
            continue
        vals = activations[mask]  # (count, 32)
        mean_activation[:, d] = vals.mean(axis=0).astype(np.float32, copy=False)
        std_activation[:, d] = vals.std(axis=0, ddof=0).astype(np.float32, copy=False)

    return mean_activation, std_activation, digit_sample_count


def compute_channel_order_by_cosine(mean_activation: np.ndarray) -> np.ndarray:
    """
    使用均值激活矩阵，根据 cosine similarity 计算 CNN 通道的排序。

    每个通道用其在 10 个 digit 上的 10 维均值向量表示；先在通道维度上取平均，
    得到一个全局参考模式，然后按与该参考模式的 cosine similarity 从大到小排序。

    返回值
    ------
    order : np.ndarray
        形状为 (num_channels,) 的数组，元素是「原始通道索引」在新的排序中的顺序
        （例如 [7, 3, 12, ...]）。
    """
    if mean_activation.ndim != 2 or mean_activation.shape[1] != 10:
        raise ValueError(
            f"Expected mean_activation of shape (C, 10), got {mean_activation.shape}"
        )

    channel_vectors = mean_activation.astype(np.float32, copy=False)

    # 全局参考模式：在通道维度上做平均，得到一个 10 维向量。
    ref = channel_vectors.mean(axis=0)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-8:
        ref_norm = 1e-8

    sims = []
    for c in range(channel_vectors.shape[0]):
        v = channel_vectors[c]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            v_norm = 1e-8
        sims.append(float(np.dot(v, ref) / (v_norm * ref_norm)))
    sims = np.asarray(sims, dtype=np.float32)

    # 按与参考向量的 cosine similarity 从大到小排序。
    order = np.argsort(-sims)
    return order


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device, require_cuda_if_requested=True)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Build dataset (test split only) and DataLoader.
    print("Building test dataset (split=test)...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size (sequence samples): {len(test_ds)}")

    # For mmap-backed arrays, DataLoader must use num_workers=0.
    num_workers = 0 if args.use_mmap else 4
    pin_memory = device.type == "cuda" and not args.use_mmap

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 2) Build model and load checkpoint.
    print(f"Loading model from: {args.ckpt}")
    model = build_model_from_ckpt(
        ckpt_path=args.ckpt,
        num_pos=num_pos,
        device=device,
    )
    print(
        f"Loaded GaWFRNNConv (hidden_size={model.hidden_size}, "
        f"encoder_flatten_size={model.encoder_flatten_size})"
    )

    # 3) Run CNN encoder and collect activations / labels.
    print("Computing CNN channel activations for all frames...")
    activation_per_sample, labels_np = compute_activations(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    N = activation_per_sample.shape[0]
    print(f"Total frames (N): {N}")

    # 4) Save per-sample activations and labels.
    act_path = os.path.join(args.save_dir, "activation_per_sample.npy")
    lbl_path = os.path.join(args.save_dir, "labels.npy")
    np.save(act_path, activation_per_sample.astype(np.float32, copy=False))
    np.save(lbl_path, labels_np.astype(np.int64, copy=False))
    print(f"Saved activations to: {act_path}")
    print(f"Saved labels to: {lbl_path}")

    # 5) Compute digit-conditioned statistics.
    print("Computing digit-conditioned statistics...")
    mean_activation, std_activation, digit_sample_count = compute_digit_stats(
        activations=activation_per_sample,
        labels=labels_np,
    )

    # 6) Save statistics to .npz.
    stats_path = os.path.join(args.save_dir, "cnn_channel_activation_stats.npz")
    np.savez(
        stats_path,
        mean_activation=mean_activation,
        std_activation=std_activation,
        digit_sample_count=digit_sample_count,
    )
    print(f"Saved CNN channel activation statistics to: {stats_path}")

    # 7) Compute channel order based on cosine similarity of mean activations.
    channel_order = compute_channel_order_by_cosine(mean_activation)
    order_path = os.path.join(args.save_dir, "channel_order_by_cosine_similarity.npy")
    np.save(order_path, channel_order.astype(np.int64, copy=False))

    print(
        "Channel order by cosine similarity (indices of original channels):\n"
        f"{channel_order.tolist()}"
    )
    print(f"Saved channel order to: {order_path}")


if __name__ == "__main__":
    main()
