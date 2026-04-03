"""
Compute class-conditioned average decomposition for input trans_ih.

Supports one mode at a time:
  --sector S : use frames with sector_label==S, decompose 9 sector components
  --digit  D : use frames with fg_digit==D,     decompose 10 digit components

For each qualifying frame:
  trans_ih = U @ (fb * V_ih)                                 # full trans, all fb dims
  outer_k  = U[:, d_k] (outer) V_ih[d_k, :] * fb[d_k]       # selected component basis

Then aggregate input axis by --agg:
  space   -> (36, H)
  feature -> (32, H)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute all-sector/all-digit decomposition for input trans_ih.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        help="Path to trained GaWFRNNConv checkpoint.",
    )
    parser.add_argument("--sector", type=int, default=None, choices=list(range(9)), help="Sector mode: target sector index (0-8).")
    parser.add_argument("--digit", type=int, default=None, choices=list(range(10)), help="Digit mode: target foreground digit (0-9).")
    parser.add_argument(
        "--agg",
        type=str,
        default="space",
        choices=["space", "feature"],
        help="How to aggregate input axis: space or feature.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/gate_avg_allsector",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--hidden_data_dir",
        type=str,
        default="./results/anal_data/hidden_activation",
        help=(
            "Directory containing gawf_hidden_tuning_stats.npz, "
            "unit_order_by_cosine_similarity.npy, and tuned_display_order.npy."
        ),
    )
    # Dataset args
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _agg_ih(
    mat_HI: np.ndarray,
    agg_mode: str,
    n_feat: int,
    h_sp: int,
    w_sp: int,
) -> np.ndarray:
    """Aggregate (H, input_size) matrix to (input_agg, H)."""
    H = mat_HI.shape[0]
    mat = mat_HI.reshape(H, n_feat, h_sp, w_sp)
    if agg_mode == "space":
        agg = mat.mean(axis=1).reshape(H, h_sp * w_sp)
    else:
        agg = mat.mean(axis=(2, 3))
    return agg.T.astype(np.float32)


def compute_avg_decomp(
    test_ds,
    model,
    device: torch.device,
    mode: str,
    selected_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Returns:
      avg_outer_all:  (n_comp, H, input_size)
      avg_outer_sum:  (H, input_size), sum over selected components
      avg_trans_full: (H, input_size), full trans_ih
      n_frames, n_samples
    """
    H = model.rnn.hidden_size
    fb_dim = model.num_classes + model.num_pos
    nc = model.num_classes
    if mode == "sector":
        label_col = 1
        comp_indices = list(range(nc, nc + model.num_pos))
    else:
        label_col = 0
        comp_indices = list(range(model.num_classes))
    n_comp = len(comp_indices)

    outer_all_acc = None  # (n_comp, H, input_size)
    outer_sum_acc = None  # (H, input_size)
    trans_full_acc = None  # (H, input_size)
    n_frames = 0
    n_samples = 0
    n_total = len(test_ds)

    model.eval()
    with torch.no_grad():
        for sidx in range(n_total):
            frames, labels = test_ds[sidx]

            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
            if labels_np.ndim < 2 or labels_np.shape[0] == 0:
                continue

            target_ids = labels_np[:, label_col].astype(np.int64)
            t_indices = np.where(target_ids == selected_idx)[0]
            if len(t_indices) == 0:
                continue

            n_samples += 1

            frames_t = frames.to(device=device, dtype=torch.float32) if isinstance(frames, torch.Tensor) else torch.as_tensor(frames, dtype=torch.float32, device=device)
            x_encoded_raw = model.encoder(frames_t)
            T_len = frames_t.shape[0]
            x_encoded = x_encoded_raw.view(T_len, -1)
            input_size = x_encoded.size(-1)
            V_ih = model.V[:, :input_size].unsqueeze(0)  # (1, fb_dim, input_size)

            if outer_all_acc is None:
                outer_all_acc = np.zeros((n_comp, H, input_size), dtype=np.float64)
                outer_sum_acc = np.zeros((H, input_size), dtype=np.float64)
                trans_full_acc = np.zeros((H, input_size), dtype=np.float64)

            for t in t_indices:
                x_t = x_encoded[t : t + 1]
                h0 = torch.zeros(1, H, device=device, dtype=x_t.dtype)
                fb0 = torch.zeros(1, fb_dim, device=device, dtype=x_t.dtype)

                fb0_c = fb0.clamp(-10, 10).unsqueeze(2)
                gated_out = model.middle_gawf(x_t, h0, fb0_c)
                char_t, pos_t = model.classifier(gated_out)
                prev_fb = torch.cat([char_t, pos_t], dim=-1)

                fb1 = prev_fb.clamp(-10, 10).unsqueeze(2)
                trans_ih = torch.matmul(model.U, fb1 * V_ih)  # (1, H, input_size)
                trans_full_acc += trans_ih.squeeze(0).cpu().numpy().astype(np.float64)

                outer_sum_t = torch.zeros(H, input_size, device=device, dtype=trans_ih.dtype)
                for ci, d_idx in enumerate(comp_indices):
                    u_d = model.U[:, d_idx]
                    v_d = V_ih[0, d_idx, :]
                    fb_d = fb1[0, d_idx, 0]
                    outer_d = torch.outer(u_d, v_d) * fb_d
                    outer_all_acc[ci] += outer_d.cpu().numpy().astype(np.float64)
                    outer_sum_t += outer_d

                outer_sum_acc += outer_sum_t.cpu().numpy().astype(np.float64)
                n_frames += 1

            if (sidx + 1) % 200 == 0:
                print(
                    f"  [{sidx + 1}/{n_total}] qualifying samples: {n_samples} | "
                    f"frames accumulated: {n_frames}"
                )

    if n_frames == 0:
        raise RuntimeError(f"No frames found for {mode}={selected_idx} in test set.")

    avg_outer_all = (outer_all_acc / n_frames).astype(np.float32)
    avg_outer_sum = (outer_sum_acc / n_frames).astype(np.float32)
    avg_trans_full = (trans_full_acc / n_frames).astype(np.float32)
    return avg_outer_all, avg_outer_sum, avg_trans_full, n_frames, n_samples


def _save_digit_boundaries(hidden_data_dir: str, save_dir: str) -> None:
    hidden_dir = os.path.abspath(hidden_data_dir)
    cos_path = os.path.join(hidden_dir, "unit_order_by_cosine_similarity.npy")
    tdo_path = os.path.join(hidden_dir, "tuned_display_order.npy")
    stats_path = os.path.join(hidden_dir, "gawf_hidden_tuning_stats.npz")
    missing = [p for p in (cos_path, tdo_path, stats_path) if not os.path.isfile(p)]
    if missing:
        print(f"[warn] digit_boundaries not saved - missing files: {missing}")
        return
    cos_order = np.load(cos_path).astype(np.int64)
    unit_order = cos_order[::-1].copy()
    tuned_display_order = np.load(tdo_path).astype(np.int64)
    sorted_npz_order = unit_order[tuned_display_order]
    stats = np.load(stats_path)
    is_tuned = stats["is_tuned"].astype(bool)
    preferred_digit = stats["preferred_digit"].astype(np.int64)
    n_tuned = int(is_tuned.sum())
    H_units = int(sorted_npz_order.size)

    boundaries = [0]
    for d in range(10):
        count_d = sum(1 for k in range(n_tuned) if int(preferred_digit[sorted_npz_order[k]]) == d)
        boundaries.append(boundaries[-1] + count_d)
    boundaries.append(H_units)
    digit_boundaries = np.array(boundaries, dtype=np.int64)
    bounds_path = os.path.join(save_dir, "digit_boundaries.npy")
    np.save(bounds_path, digit_boundaries)
    print(f"Saved digit_boundaries to: {bounds_path}")


def main() -> None:
    args = parse_args()
    if (args.sector is None) == (args.digit is None):
        raise ValueError("Specify exactly one of --sector or --digit.")
    if args.sector is not None:
        mode = "sector"
        selected_idx = int(args.sector)
    else:
        mode = "digit"
        selected_idx = int(args.digit)
    save_dir = os.path.join(args.save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)
    agg = args.agg
    tag = f"{mode}{selected_idx}_{agg}"

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size: {len(test_ds)}")

    print(f"Building model from: {args.ckpt}")
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    print(f"hidden_size={model.hidden_size}  num_pos={model.num_pos}")

    n_feat = model.conv_reduce.out_channels
    h_sp, w_sp = model.pool_reduce.output_size

    print(f"Computing decomposition for selected {mode}={selected_idx}, agg={agg}...")
    avg_outer_all, avg_outer_sum, avg_trans_full, n_frames, n_samples = compute_avg_decomp(
        test_ds, model, device, mode, selected_idx
    )
    print(
        f"Done.  n_samples={n_samples}  n_frames={n_frames}\n"
        f"outer_all stats: min={avg_outer_all.min():.4f}  max={avg_outer_all.max():.4f}  mean={avg_outer_all.mean():.4f}\n"
        f"outer_sum stats: min={avg_outer_sum.min():.4f}  max={avg_outer_sum.max():.4f}  mean={avg_outer_sum.mean():.4f}\n"
        f"trans_full stats: min={avg_trans_full.min():.4f}  max={avg_trans_full.max():.4f}  mean={avg_trans_full.mean():.4f}"
    )

    outer_all_agg = np.stack([_agg_ih(avg_outer_all[i], agg, n_feat, h_sp, w_sp) for i in range(avg_outer_all.shape[0])], axis=0)
    outer_sum_agg = _agg_ih(avg_outer_sum, agg, n_feat, h_sp, w_sp)
    trans_full_agg = _agg_ih(avg_trans_full, agg, n_feat, h_sp, w_sp)

    W_ih_np = model.rnn.weight_ih_l0.detach().cpu().numpy()
    W_ih_agg = _agg_ih(W_ih_np, agg, n_feat, h_sp, w_sp)

    all_path = os.path.join(save_dir, f"avg_outer_ih_allcomp_{tag}.npy")
    sum_path = os.path.join(save_dir, f"avg_outer_ih_sumcomp_{tag}.npy")
    full_path = os.path.join(save_dir, f"avg_trans_ih_full_{tag}.npy")
    wih_path = os.path.join(save_dir, f"weight_ih_{agg}.npy")
    meta_path = os.path.join(save_dir, f"avg_gate_meta_allcomp_{tag}.json")

    np.save(all_path, outer_all_agg)
    np.save(sum_path, outer_sum_agg)
    np.save(full_path, trans_full_agg)
    np.save(wih_path, W_ih_agg)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "mode": mode,
                "selected_idx": selected_idx,
                "n_components": int(avg_outer_all.shape[0]),
                "agg": agg,
                "n_frames": n_frames,
                "n_samples": n_samples,
                "hidden_size": int(avg_trans_full.shape[0]),
                "input_size": int(avg_trans_full.shape[1]),
                "n_feat": int(n_feat),
                "h_sp": int(h_sp),
                "w_sp": int(w_sp),
                "agg_shape": list(trans_full_agg.shape),
                "ckpt": os.path.abspath(args.ckpt),
            },
            f,
            indent=2,
        )

    print(f"Saved avg_outer_ih_allcomp ({outer_all_agg.shape}) to: {all_path}")
    print(f"Saved avg_outer_ih_sumcomp ({outer_sum_agg.shape}) to: {sum_path}")
    print(f"Saved avg_trans_ih_full      ({trans_full_agg.shape}) to: {full_path}")
    print(f"Saved weight_ih              ({W_ih_agg.shape}) to: {wih_path}")
    print(f"Saved metadata to: {meta_path}")
    _save_digit_boundaries(args.hidden_data_dir, save_dir)


if __name__ == "__main__":
    main()

