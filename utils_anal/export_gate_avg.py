"""
Compute average gate matrices over test-set frames for a specified fg digit or sector.

Digit + hidden–hidden  (--digit D  without --agg):
  For each frame where fg_label == D, accumulate:
    gate_hh  = sigmoid(U @ (fb * V_hh) / tau)   shape (H, H)
    outer_hh = U[:,D] (outer) V_hh[D,:] * fb[D]  shape (H, H)  — no sigmoid
  Outputs in --save_dir:
    avg_gate_hh_{D}.npy / avg_outer_hh_{D}.npy / avg_gate_meta_{D}.json

Digit + input–hidden  (--digit D  --agg {space|feature}):
  Same frame filter (fg_label == D); gate_ih / outer_ih use basis index d = D (class slot).
  Aggregate along input axis like sector mode:
    avg_gate_ih_d{D}_{agg}.npy / avg_outer_ih_d{D}_{agg}.npy
    weight_ih_{agg}.npy (shared static ih, one file per agg) / avg_gate_meta_d{D}_{agg}.json

Sector + input–hidden  (--sector S  --agg {space|feature}, required):
  For each frame where sector_label == S, accumulate gate_ih with basis index d = nc + S.
  Outputs:
    avg_gate_ih_s{S}_{agg}.npy / avg_outer_ih_s{S}_{agg}.npy
    weight_ih_{agg}.npy / avg_gate_meta_s{S}_{agg}.json
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
    parser = argparse.ArgumentParser(
        description=(
            "Compute average gate matrices over test frames for a fg digit or sector.\n"
            "Exactly one of --digit or --sector.  --agg is required with --sector; "
            "with --digit, omit --agg for hh or set space|feature for ih."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        help="Path to trained GaWFRNNConv checkpoint.",
    )
    # --- mode selection (exactly one required) ---
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=list(range(10)),
        help="Digit mode: target foreground digit (0-9).",
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        choices=list(range(9)),
        help="Sector mode: target foreground sector index (0-8).",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default=None,
        choices=["space", "feature"],
        help=(
            "Input-axis aggregation for gate_ih / W_ih (required with --sector).\n"
            "With --digit: omit for hh analysis; set to space|feature for ih analysis.\n"
            "  space   → mean over 32 feature channels → (36, H)\n"
            "  feature → mean over 6×6 spatial grid   → (32, H)"
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/gate_avg",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Temperature tau for gate computation (default: use model.gate_tau from middle_gawf).",
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
            "unit_order_by_cosine_similarity.npy, and tuned_display_order.npy. "
            "Used to compute per-digit group boundaries saved alongside avg_gate_hh."
        ),
    )
    # Dataset args — mirrors export_gawf_gates.py
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="")
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_avg_gate_hh(
    test_ds,
    model,
    device: torch.device,
    tau: float,
    fg_digit: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Iterate all test samples. For each frame where fg_label == fg_digit,
    compute gate_hh (element-wise gate matrix, same shape as W_hh) and accumulate.
    Also accumulate the rank-1 outer product U[:, d] * fb[d] * V_hh[d, :] (no sigmoid)
    for the fg_digit basis component d.

    gate_hh[i, j] = sigmoid(trans_hh[i, j] / tau)
    where trans_hh = U @ (fb * V_hh), same formula as middle_gawf.

    Returns:
        avg_gate_hh  : (H, H) float32, values in [0, 1]
        avg_outer_hh : (H, H) float32, rank-1 outer product avg (no sigmoid)
        n_frames     : total qualifying frames accumulated
        n_samples    : samples containing at least one qualifying frame
    """
    H = model.rnn.hidden_size
    fb_dim = model.num_classes + model.num_pos

    gate_hh_acc = np.zeros((H, H), dtype=np.float64)
    outer_acc   = np.zeros((H, H), dtype=np.float64)
    n_frames = 0
    n_samples = 0
    n_total = len(test_ds)

    model.eval()
    with torch.no_grad():
        for sidx in range(n_total):
            frames, labels = test_ds[sidx]

            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.asarray(labels)

            if labels_np.ndim < 2 or labels_np.shape[0] == 0:
                continue

            fg_ids = labels_np[:, 0].astype(np.int64)
            t_indices = np.where(fg_ids == fg_digit)[0]

            if len(t_indices) == 0:
                continue

            n_samples += 1

            if isinstance(frames, torch.Tensor):
                frames_t = frames.to(device=device, dtype=torch.float32)
            else:
                frames_t = torch.as_tensor(frames, dtype=torch.float32, device=device)
            # frames_t: (T, C, H_img, W_img)

            # Encode all frames at once; encoder returns a 4-D volume (T, C, h, w)
            # — must flatten to (T, input_size) to match middle_gawf expectations.
            x_encoded_raw = model.encoder(frames_t)       # (T, C_enc, H_enc, W_enc)
            T_len = frames_t.shape[0]
            x_encoded = x_encoded_raw.view(T_len, -1)     # (T, input_size)
            input_size = x_encoded.size(-1)
            V_hh = model.V[:, input_size:].unsqueeze(0)   # (1, fb_dim, H)

            for t in t_indices:
                x_t = x_encoded[t : t + 1]  # (1, input_size)
                h0 = torch.zeros(1, H, device=device, dtype=x_t.dtype)
                fb0 = torch.zeros(1, fb_dim, device=device, dtype=x_t.dtype)

                # Step 1: initial pass with zero feedback → get logits
                fb0_c = fb0.clamp(-10, 10).unsqueeze(2)  # (1, fb_dim, 1)
                gated_out = model.middle_gawf(x_t, h0, fb0_c)
                char_t, pos_t = model.classifier(gated_out)
                prev_fb = torch.cat([char_t, pos_t], dim=-1)

                # Step 2: recompute gate_hh with updated feedback
                fb1 = prev_fb.clamp(-10, 10).unsqueeze(2)  # (1, fb_dim, 1)
                trans_hh = torch.matmul(model.U, fb1 * V_hh)  # (1, H, H)
                gate_hh = torch.sigmoid(trans_hh / tau)        # (1, H, H)

                gate_hh_acc += gate_hh.squeeze(0).cpu().numpy().astype(np.float64)

                # Rank-1 outer product for the fg_digit basis component (no sigmoid)
                d = fg_digit
                u_d     = model.U[:, d]           # (H,)
                v_hh_d  = V_hh[0, d, :]           # (H,)
                fb_d    = fb1[0, d, 0]             # scalar
                outer_d = torch.outer(u_d, v_hh_d) * fb_d  # (H, H)
                outer_acc += outer_d.cpu().numpy().astype(np.float64)

                n_frames += 1

            if (sidx + 1) % 200 == 0:
                print(
                    f"  [{sidx + 1}/{n_total}] qualifying samples: {n_samples} | "
                    f"frames accumulated: {n_frames}"
                )

    if n_frames == 0:
        raise RuntimeError(
            f"No frames with fg_digit={fg_digit} found in test set."
        )

    avg_gate_hh  = (gate_hh_acc / n_frames).astype(np.float32)
    avg_outer_hh = (outer_acc   / n_frames).astype(np.float32)
    return avg_gate_hh, avg_outer_hh, n_frames, n_samples


def _agg_ih(
    mat_HI: np.ndarray,
    agg_mode: str,
    n_feat: int,
    h_sp: int,
    w_sp: int,
) -> np.ndarray:
    """
    Aggregate a (H, input_size) matrix along the input dimension.

    The encoder output is laid out as (n_feat, h_sp, w_sp) flattened to input_size.

    agg_mode == 'space'  : mean over n_feat channels → (H, h_sp*w_sp) → T → (h_sp*w_sp, H)
    agg_mode == 'feature': mean over h_sp*w_sp spatial → (H, n_feat)  → T → (n_feat, H)

    Returns float32 array of shape (h_sp*w_sp, H) or (n_feat, H).
    """
    H = mat_HI.shape[0]
    mat = mat_HI.reshape(H, n_feat, h_sp, w_sp)
    if agg_mode == "space":
        agg = mat.mean(axis=1).reshape(H, h_sp * w_sp)   # (H, 36)
    else:
        agg = mat.mean(axis=(2, 3))                        # (H, n_feat=32)
    return agg.T.astype(np.float32)  # (36, H) or (32, H)


def compute_avg_gate_ih(
    test_ds,
    model,
    device: torch.device,
    tau: float,
    *,
    label_col: int,
    label_value: int,
    d_fb: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Accumulate gate_ih and rank-1 outer_ih for frames matching labels[:, label_col] == label_value.

    label_col 0 : fg digit (same as compute_avg_gate_hh filter); d_fb is digit index 0..9.
    label_col 1 : sector id; d_fb is nc + sector (class/sector basis slot).

    gate_ih[i, j] = sigmoid((U @ (fb * V_ih))[i,j] / tau), shape (H, input_size).
    outer_ih uses row U[:, d_fb], V_ih[d_fb, :], fb[d_fb].
    """
    H      = model.rnn.hidden_size
    fb_dim = model.num_classes + model.num_pos

    gate_ih_acc  = None   # deferred allocation once input_size is known
    outer_acc    = None
    n_frames  = 0
    n_samples = 0
    n_total   = len(test_ds)

    model.eval()
    with torch.no_grad():
        for sidx in range(n_total):
            frames, labels = test_ds[sidx]

            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.asarray(labels)

            if labels_np.ndim < 2 or labels_np.shape[0] == 0:
                continue

            if labels_np.shape[1] <= label_col:
                raise ValueError(
                    f"Labels need column index {label_col}; got shape {labels_np.shape}."
                )

            lab_ids   = labels_np[:, label_col].astype(np.int64)
            t_indices = np.where(lab_ids == label_value)[0]

            if len(t_indices) == 0:
                continue

            n_samples += 1

            if isinstance(frames, torch.Tensor):
                frames_t = frames.to(device=device, dtype=torch.float32)
            else:
                frames_t = torch.as_tensor(frames, dtype=torch.float32, device=device)

            x_encoded_raw = model.encoder(frames_t)       # (T, C_enc, H_enc, W_enc)
            T_len = frames_t.shape[0]
            x_encoded = x_encoded_raw.view(T_len, -1)     # (T, input_size)
            input_size = x_encoded.size(-1)
            V_ih = model.V[:, :input_size].unsqueeze(0)   # (1, fb_dim, input_size)

            if gate_ih_acc is None:
                gate_ih_acc = np.zeros((H, input_size), dtype=np.float64)
                outer_acc   = np.zeros((H, input_size), dtype=np.float64)

            for t in t_indices:
                x_t = x_encoded[t : t + 1]  # (1, input_size)
                h0  = torch.zeros(1, H, device=device, dtype=x_t.dtype)
                fb0 = torch.zeros(1, fb_dim, device=device, dtype=x_t.dtype)

                fb0_c    = fb0.clamp(-10, 10).unsqueeze(2)  # (1, fb_dim, 1)
                gated_out = model.middle_gawf(x_t, h0, fb0_c)
                char_t, pos_t = model.classifier(gated_out)
                prev_fb  = torch.cat([char_t, pos_t], dim=-1)

                fb1      = prev_fb.clamp(-10, 10).unsqueeze(2)   # (1, fb_dim, 1)
                trans_ih = torch.matmul(model.U, fb1 * V_ih)     # (1, H, input_size)
                gate_ih  = torch.sigmoid(trans_ih / tau)          # (1, H, input_size)

                gate_ih_acc += gate_ih.squeeze(0).cpu().numpy().astype(np.float64)

                u_d     = model.U[:, d_fb]          # (H,)
                v_ih_d  = V_ih[0, d_fb, :]          # (input_size,)
                fb_d    = fb1[0, d_fb, 0]           # scalar
                outer_d = torch.outer(u_d, v_ih_d) * fb_d  # (H, input_size)
                outer_acc += outer_d.cpu().numpy().astype(np.float64)

                n_frames += 1

            if (sidx + 1) % 200 == 0:
                print(
                    f"  [{sidx + 1}/{n_total}] qualifying samples: {n_samples} | "
                    f"frames accumulated: {n_frames}"
                )

    if n_frames == 0:
        what = f"fg_digit={label_value}" if label_col == 0 else f"sector={label_value}"
        raise RuntimeError(f"No frames with {what} found in test set.")

    avg_gate_ih  = (gate_ih_acc  / n_frames).astype(np.float32)
    avg_outer_ih = (outer_acc    / n_frames).astype(np.float32)
    return avg_gate_ih, avg_outer_ih, n_frames, n_samples


def _save_digit_boundaries(hidden_data_dir: str, save_dir: str) -> None:
    """Compute and save digit group boundaries from hidden activation tuning stats."""
    hidden_dir = os.path.abspath(hidden_data_dir)
    cos_path   = os.path.join(hidden_dir, "unit_order_by_cosine_similarity.npy")
    tdo_path   = os.path.join(hidden_dir, "tuned_display_order.npy")
    stats_path = os.path.join(hidden_dir, "gawf_hidden_tuning_stats.npz")
    missing = [p for p in (cos_path, tdo_path, stats_path) if not os.path.isfile(p)]
    if missing:
        print(f"[warn] digit_boundaries not saved — missing files: {missing}")
        return
    cos_order           = np.load(cos_path).astype(np.int64)
    unit_order          = cos_order[::-1].copy()
    tuned_display_order = np.load(tdo_path).astype(np.int64)
    sorted_npz_order    = unit_order[tuned_display_order]
    stats               = np.load(stats_path)
    is_tuned            = stats["is_tuned"].astype(bool)
    preferred_digit     = stats["preferred_digit"].astype(np.int64)
    n_tuned             = int(is_tuned.sum())
    H_units             = int(sorted_npz_order.size)

    boundaries = [0]
    for d in range(10):
        count_d = sum(
            1 for k in range(n_tuned)
            if int(preferred_digit[sorted_npz_order[k]]) == d
        )
        boundaries.append(boundaries[-1] + count_d)
    boundaries.append(H_units)
    digit_boundaries = np.array(boundaries, dtype=np.int64)  # shape (12,)

    bounds_path = os.path.join(save_dir, "digit_boundaries.npy")
    np.save(bounds_path, digit_boundaries)
    print(f"Saved digit_boundaries to: {bounds_path}")
    sizes = [int(digit_boundaries[d+1] - digit_boundaries[d]) for d in range(10)]
    print(f"Digit group sizes (0-9): {sizes}  untuned: {H_units - n_tuned}")


def main() -> None:
    args = parse_args()

    if args.digit is None and args.sector is None:
        raise ValueError("Specify exactly one of --digit or --sector.")
    if args.digit is not None and args.sector is not None:
        raise ValueError("Specify exactly one of --digit or --sector, not both.")
    if args.sector is not None and args.agg is None:
        raise ValueError("Sector mode requires --agg {space|feature}.")

    subdir = "digit" if args.digit is not None else "sector"
    save_dir = os.path.join(args.save_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size: {len(test_ds)}")

    print(f"Building model from: {args.ckpt}")
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    tau = float(model.gate_tau if args.tau is None else args.tau)
    print(f"hidden_size={model.hidden_size}  num_pos={model.num_pos}")

    # ------------------------------------------------------------------ digit mode
    if args.digit is not None:
        fg_digit = int(args.digit)
        n_feat = model.conv_reduce.out_channels
        h_sp, w_sp = model.pool_reduce.output_size

        if args.agg is None:
            print(f"Computing avg gate_hh for fg_digit={fg_digit}, tau={tau}...")
            avg_gate_hh, avg_outer_hh, n_frames, n_samples = compute_avg_gate_hh(
                test_ds, model, device, tau, fg_digit
            )
            print(
                f"Done.  n_samples={n_samples}  n_frames={n_frames}\n"
                f"gate_hh stats:  min={avg_gate_hh.min():.4f}  max={avg_gate_hh.max():.4f}  "
                f"mean={avg_gate_hh.mean():.4f}\n"
                f"outer_hh stats: min={avg_outer_hh.min():.4f}  max={avg_outer_hh.max():.4f}  "
                f"mean={avg_outer_hh.mean():.4f}"
            )
            gate_path  = os.path.join(save_dir, f"avg_gate_hh_{fg_digit}.npy")
            outer_path = os.path.join(save_dir, f"avg_outer_hh_{fg_digit}.npy")
            meta_path  = os.path.join(save_dir, f"avg_gate_meta_{fg_digit}.json")
            np.save(gate_path,  avg_gate_hh)
            np.save(outer_path, avg_outer_hh)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "mode": "digit",
                        "fg_digit": fg_digit,
                        "tau": tau,
                        "n_frames": n_frames,
                        "n_samples": n_samples,
                        "hidden_size": int(avg_gate_hh.shape[0]),
                        "ckpt": os.path.abspath(args.ckpt),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved avg_gate_hh  to: {gate_path}")
            print(f"Saved avg_outer_hh to: {outer_path}")
            print(f"Saved metadata     to: {meta_path}")
            _save_digit_boundaries(args.hidden_data_dir, save_dir)

        else:
            agg = args.agg
            tag = f"d{fg_digit}_{agg}"
            print(
                f"Computing avg gate_ih for fg_digit={fg_digit}, agg={agg}, tau={tau}..."
            )
            avg_gate_ih, avg_outer_ih, n_frames, n_samples = compute_avg_gate_ih(
                test_ds,
                model,
                device,
                tau,
                label_col=0,
                label_value=fg_digit,
                d_fb=fg_digit,
            )
            print(
                f"Done.  n_samples={n_samples}  n_frames={n_frames}\n"
                f"gate_ih stats:  min={avg_gate_ih.min():.4f}  max={avg_gate_ih.max():.4f}  "
                f"mean={avg_gate_ih.mean():.4f}\n"
                f"outer_ih stats: min={avg_outer_ih.min():.4f}  max={avg_outer_ih.max():.4f}  "
                f"mean={avg_outer_ih.mean():.4f}"
            )
            gate_agg  = _agg_ih(avg_gate_ih,  agg, n_feat, h_sp, w_sp)
            outer_agg = _agg_ih(avg_outer_ih, agg, n_feat, h_sp, w_sp)
            W_ih_np   = model.rnn.weight_ih_l0.detach().cpu().numpy()
            W_ih_agg  = _agg_ih(W_ih_np, agg, n_feat, h_sp, w_sp)

            gate_path  = os.path.join(save_dir, f"avg_gate_ih_{tag}.npy")
            outer_path = os.path.join(save_dir, f"avg_outer_ih_{tag}.npy")
            wih_path   = os.path.join(save_dir, f"weight_ih_{agg}.npy")
            meta_path  = os.path.join(save_dir, f"avg_gate_meta_{tag}.json")

            np.save(gate_path,  gate_agg)
            np.save(outer_path, outer_agg)
            np.save(wih_path,   W_ih_agg)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "mode": "digit_ih",
                        "fg_digit": fg_digit,
                        "agg": agg,
                        "tau": tau,
                        "n_frames": n_frames,
                        "n_samples": n_samples,
                        "hidden_size": int(avg_gate_ih.shape[0]),
                        "input_size": int(avg_gate_ih.shape[1]),
                        "n_feat": n_feat,
                        "h_sp": h_sp,
                        "w_sp": w_sp,
                        "agg_shape": list(gate_agg.shape),
                        "ckpt": os.path.abspath(args.ckpt),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved avg_gate_ih  ({gate_agg.shape})  to: {gate_path}")
            print(f"Saved avg_outer_ih ({outer_agg.shape}) to: {outer_path}")
            print(f"Saved weight_ih    ({W_ih_agg.shape})  to: {wih_path}")
            print(f"Saved metadata                         to: {meta_path}")
            _save_digit_boundaries(args.hidden_data_dir, save_dir)

    # ---------------------------------------------------------------- sector mode
    else:
        sector  = int(args.sector)
        agg     = args.agg  # validated non-None above
        tag     = f"s{sector}_{agg}"

        # Encoder spatial/feature dimensions (read from model architecture)
        n_feat = model.conv_reduce.out_channels          # 32
        h_sp, w_sp = model.pool_reduce.output_size       # (6, 6)

        nc = model.num_classes
        print(f"Computing avg gate_ih for sector={sector}, agg={agg}, tau={tau}...")
        avg_gate_ih, avg_outer_ih, n_frames, n_samples = compute_avg_gate_ih(
            test_ds,
            model,
            device,
            tau,
            label_col=1,
            label_value=sector,
            d_fb=nc + sector,
        )
        print(
            f"Done.  n_samples={n_samples}  n_frames={n_frames}\n"
            f"gate_ih stats:  min={avg_gate_ih.min():.4f}  max={avg_gate_ih.max():.4f}  "
            f"mean={avg_gate_ih.mean():.4f}\n"
            f"outer_ih stats: min={avg_outer_ih.min():.4f}  max={avg_outer_ih.max():.4f}  "
            f"mean={avg_outer_ih.mean():.4f}"
        )

        # Aggregate along input dimension
        gate_agg  = _agg_ih(avg_gate_ih,  agg, n_feat, h_sp, w_sp)
        outer_agg = _agg_ih(avg_outer_ih, agg, n_feat, h_sp, w_sp)

        # Aggregate W_ih (static; shape (H, input_size) after transposing PyTorch weight)
        W_ih_np = model.rnn.weight_ih_l0.detach().cpu().numpy()  # (H, input_size)
        W_ih_agg = _agg_ih(W_ih_np, agg, n_feat, h_sp, w_sp)

        gate_path  = os.path.join(save_dir, f"avg_gate_ih_{tag}.npy")
        outer_path = os.path.join(save_dir, f"avg_outer_ih_{tag}.npy")
        wih_path   = os.path.join(save_dir, f"weight_ih_{agg}.npy")
        meta_path  = os.path.join(save_dir, f"avg_gate_meta_{tag}.json")

        np.save(gate_path,  gate_agg)
        np.save(outer_path, outer_agg)
        np.save(wih_path,   W_ih_agg)
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "mode": "sector",
                    "sector": sector,
                    "agg": agg,
                    "tau": tau,
                    "n_frames": n_frames,
                    "n_samples": n_samples,
                    "hidden_size": int(avg_gate_ih.shape[0]),
                    "input_size": int(avg_gate_ih.shape[1]),
                    "n_feat": n_feat,
                    "h_sp": h_sp,
                    "w_sp": w_sp,
                    "agg_shape": list(gate_agg.shape),
                    "ckpt": os.path.abspath(args.ckpt),
                },
                f,
                indent=2,
            )
        print(f"Saved avg_gate_ih  ({gate_agg.shape})  to: {gate_path}")
        print(f"Saved avg_outer_ih ({outer_agg.shape}) to: {outer_path}")
        print(f"Saved weight_ih    ({W_ih_agg.shape})  to: {wih_path}")
        print(f"Saved metadata                         to: {meta_path}")
        _save_digit_boundaries(args.hidden_data_dir, save_dir)


if __name__ == "__main__":
    main()
