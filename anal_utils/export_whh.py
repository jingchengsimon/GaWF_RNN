"""
Extract the static hidden-to-hidden weight matrix (W_hh) from a trained GaWFRNNConv
model and save it along with the panel-4 unit ordering.

The panel-4 ordering is derived from:
  - unit_order_by_cosine_similarity.npy  (from analyze_gawf_hidden_activation.py)
  - tuned_display_order.npy              (from analyze_gawf_hidden_activation.py)

sorted_npz_order[k] = npz_row index of the k-th unit in panel-4 order (digit groups
0–9 by FDR + effect, then untuned tail).

Outputs (in --save_dir):
  weight_hh.npy         (H, H) float32 — raw rnn.weight_hh_l0
  sorted_npz_order.npy  (H,)   int64   — unit npz row indices in panel-4 order
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract W_hh connection matrix from a trained GaWFRNNConv model."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        help="Path to trained GaWFRNNConv checkpoint (*_model.pth).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/hidden_activation_data",
        help=(
            "Directory containing unit_order_by_cosine_similarity.npy "
            "and tuned_display_order.npy."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/whh_data",
        help="Directory to save extracted matrices.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading checkpoint (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.ckpt}")
    state_dict = torch.load(args.ckpt, map_location=device)

    if "rnn.weight_hh_l0" not in state_dict:
        raise KeyError(
            "Key 'rnn.weight_hh_l0' not found in checkpoint. "
            f"Available keys (first 20): {list(state_dict.keys())[:20]}"
        )

    W_hh = state_dict["rnn.weight_hh_l0"].cpu().float().numpy()
    print(f"W_hh shape: {W_hh.shape}")
    print(
        f"W_hh stats: min={W_hh.min():.4f}  max={W_hh.max():.4f}  "
        f"mean={W_hh.mean():.6f}  std={W_hh.std():.4f}"
    )

    cos_order_path = os.path.join(
        args.data_dir, "unit_order_by_cosine_similarity.npy"
    )
    tuned_order_path = os.path.join(
        args.data_dir, "tuned_display_order.npy"
    )
    for p in (cos_order_path, tuned_order_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run anal_utils/analyze_gawf_hidden_activation.py first."
            )

    # unit_order: display position j → npz_row
    # viz_gawf_hidden_activation.load_unit_order returns cos_order[::-1], mirrored here.
    cos_order = np.load(cos_order_path).astype(np.int64)
    unit_order = cos_order[::-1].copy()  # shape (H,)

    # tuned_display_order: panel-4 sorted position k → display position (0..H-1)
    tuned_display_order = np.load(tuned_order_path).astype(np.int64)  # shape (H,)

    H = W_hh.shape[0]
    if unit_order.size != H or tuned_display_order.size != H:
        raise ValueError(
            f"Shape mismatch: W_hh {W_hh.shape}, "
            f"unit_order {unit_order.shape}, "
            f"tuned_display_order {tuned_display_order.shape}"
        )

    # sorted_npz_order[k] = npz_row index of the k-th unit in panel-4 order
    sorted_npz_order = unit_order[tuned_display_order].astype(np.int64)

    # Determine how many units are tuned (digit groups 0-9) vs untuned tail.
    # tuned units occupy the first n_tuned positions in sorted_npz_order.
    tuning_stats_path = os.path.join(
        args.data_dir, "gawf_hidden_tuning_stats.npz"
    )
    if not os.path.isfile(tuning_stats_path):
        raise FileNotFoundError(
            f"Required file not found: {tuning_stats_path}\n"
            "Run anal_utils/analyze_gawf_hidden_activation.py first."
        )
    tuning_stats = np.load(tuning_stats_path)
    is_tuned = tuning_stats["is_tuned"].astype(bool)
    n_tuned = int(is_tuned.sum())

    whh_path = os.path.join(args.save_dir, "weight_hh.npy")
    ord_path = os.path.join(args.save_dir, "sorted_npz_order.npy")
    ntuned_path = os.path.join(args.save_dir, "n_tuned.npy")
    np.save(whh_path, W_hh.astype(np.float32))
    np.save(ord_path, sorted_npz_order)
    np.save(ntuned_path, np.array(n_tuned, dtype=np.int64))
    print(f"Saved W_hh to:            {whh_path}")
    print(f"Saved sorted_npz_order to: {ord_path}")
    print(f"Saved n_tuned to:          {ntuned_path}")
    print(f"n_tuned = {n_tuned} / {H}")
    print(f"First 5 units in panel-4 order (npz_rows): {sorted_npz_order[:5].tolist()}")


if __name__ == "__main__":
    main()
