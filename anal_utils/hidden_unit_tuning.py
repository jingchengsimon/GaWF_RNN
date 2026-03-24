"""
Analyze hidden-unit activations in a trained GaWF model.

Mirrors analyze_cnn_channel_activation.py but collects the recurrent hidden state
(256-d after LayerNorm + ReLU in the GaWF gated path) for every frame in the
test split, using foreground digit labels per frame.

Outputs (in --save_dir):
- activation_per_sample.npy  (N, hidden_size), float32
- labels.npy                 (N,), int64 foreground digit labels (0-9)
- gawf_hidden_activation_stats.npz with:
    - mean_activation    (hidden_size, 10)
    - std_activation     (hidden_size, 10)
    - digit_sample_count (10,)
- unit_order_by_cosine_similarity.npy  (hidden_size,), int64

Additional tuning / FDR outputs:
- gawf_hidden_tuning_stats.npz
- gawf_hidden_tuning_meta.json (optional metadata)
- tuned_display_order.npy
- tuned_group_summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from scipy.stats import f_oneway
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "analyze_gawf_hidden_activation requires scipy.stats.f_oneway; "
        "install scipy (e.g. pip install scipy)."
    ) from exc

try:
    from statsmodels.stats.multitest import multipletests as _sm_multipletests
except ImportError:
    _sm_multipletests = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from train_rnn_updated import MC_RNN_Dataset
from utils.train_gawf_core import GaWFRNNConv
from utils.train_helpers import (
    create_datasets,
    get_base_path,
    load_raw_data,
    prepare_data_paths,
    set_seed,
)
from viz_utils.model_train_single_result import parse_hparams_from_filename


def benjamini_hochberg_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR adjusted q-values (same as statsmodels fdr_bh).
    Returns q-values in the same order as pvals.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate((ps * (m / ranks))[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=np.float64)
    q[order] = q_sorted
    return q


def apply_fdr_bh(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    FDR correction. Returns (reject, qvals) aligned with pvals.
    Uses statsmodels.multipletests when available; otherwise numpy BH q-values.
    reject[k] = (qvals[k] < alpha) for strict consistency with viz tuning rule
    using q < q_thr; multipletests reject uses q <= alpha — we recompute reject
    from q for a single rule: q < alpha.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    if _sm_multipletests is not None:
        reject, qvals, _, _ = _sm_multipletests(
            p, alpha=alpha, method="fdr_bh", maxiter=1
        )
        qvals = np.asarray(qvals, dtype=np.float64)
        reject = qvals < alpha
        return reject.astype(bool), qvals
    qvals = benjamini_hochberg_qvalues(p)
    reject = qvals < alpha
    return reject.astype(bool), qvals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GaWF hidden-unit activation analysis (test split, per-frame)."
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
        default="./results/anal_data/hidden_activation_data",
        help="Directory to save activation arrays and statistics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for DataLoader (over sequence samples).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Computation device (cpu/cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help="Load stimuli with numpy mmap_mode='r' (default: True).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Base directory for stimuli/labels; empty = ENV or repo default.",
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help="Optional suffix for stimulus_reg-* files (e.g. '40h').",
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=True,
        help="Use sector mode for position labels (default: True).",
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters instead of only foreground (default: False).",
    )
    parser.add_argument(
        "--q_thr",
        type=float,
        default=0.05,
        help="FDR q-value threshold for tuned units (strict: q < q_thr).",
    )
    parser.add_argument(
        "--effect_metric",
        type=str,
        default="gap",
        choices=["gap", "si"],
        help="Effect size used with --effect_thr for tuned_units (gap or si).",
    )
    parser.add_argument(
        "--effect_thr",
        type=float,
        default=None,
        help="Effect threshold; default 0.1 for both gap and si when omitted.",
    )
    parser.add_argument(
        "--fdr_method",
        type=str,
        default="bh",
        choices=["bh"],
        help="Multiple-testing method (Benjamini–Hochberg only for now).",
    )
    parser.add_argument(
        "--anova_min_count_per_digit",
        type=int,
        default=2,
        help="If any digit has fewer samples for ANOVA groups, unit is non-significant.",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_test_dataset(args: argparse.Namespace) -> Tuple[MC_RNN_Dataset, int]:
    base_path = get_base_path(override=args.data_dir or None)
    paths = prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("test",)
    )
    stims_test, lbls_test = load_raw_data(
        None, None, None, None, use_mmap=args.use_mmap, paths_tuple=paths
    )

    test_ds, num_pos = create_datasets(
        None,
        None,
        None,
        None,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    return test_ds, num_pos


def build_model_from_ckpt(
    ckpt_path: str,
    num_pos: int,
    device: torch.device,
) -> GaWFRNNConv:
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    dropout_rate = hparams.get("dropout", 0.3)
    num_classes = 10

    model = GaWFRNNConv(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=5,
        device=str(device),
        dropout_rate=dropout_rate,
        hidden_size=hidden_size,
        max_chars=15,
        predict_all_chars=(num_pos == 0),
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing_keys:", load_result.missing_keys)
    print("[load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    model.to(device)
    model.eval()
    return model


def compute_hidden_activations(
    model: GaWFRNNConv,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the GaWF gated recurrence (same as forward(use_feedback=True)) per frame
    and collect hidden vectors. Each batch starts with h=0 and fb=0 (no carry-over).
    """
    Hdim = model.hidden_size
    all_acts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(data_loader):
            frames = frames.to(device=device, dtype=torch.float32)
            digits = labels[..., 0].to(device="cpu", dtype=torch.int64)

            B, T, c_img, h_pix, w_pix = frames.shape
            enc_in = frames.view(B * T, c_img, h_pix, w_pix)
            enc = model.encoder(enc_in)
            x = enc.view(B, T, -1)

            fb_dim = model.num_classes + model.num_pos
            fb = torch.zeros(B, fb_dim, device=device, dtype=torch.float32)
            h = torch.zeros(B, Hdim, device=device, dtype=torch.float32)

            for t in range(T):
                x_t = x[:, t, :]
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                gated_output = model.middle_gawf(x_t, h, fb_t)
                gated_output = F.dropout(
                    gated_output, p=0.5, training=model.training
                )
                char_t, pos_t = model.classifier(gated_output)
                with torch.no_grad():
                    if pos_t is None:
                        fb = char_t
                    else:
                        fb = torch.cat([char_t, pos_t], dim=-1)
                h = gated_output

                all_acts.append(gated_output.detach().cpu().numpy().astype(np.float32))
                all_labels.append(digits[:, t].numpy())

            if (batch_idx + 1) % 10 == 0:
                n_so_far = sum(a.shape[0] for a in all_acts)
                print(f"[batch {batch_idx + 1}] collected {n_so_far} frame activations")

    activation_per_sample = np.concatenate(all_acts, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64, copy=False)
    return activation_per_sample, labels_np


def compute_digit_stats(
    activations: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations (N, H), got {activations.shape}")
    if labels.ndim != 1 or activations.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels shape {labels.shape} incompatible with activations {activations.shape}"
        )

    Hdim, num_digits = activations.shape[1], 10
    mean_activation = np.zeros((Hdim, num_digits), dtype=np.float32)
    std_activation = np.zeros((Hdim, num_digits), dtype=np.float32)
    digit_sample_count = np.zeros((num_digits,), dtype=np.int64)

    for d in range(num_digits):
        mask = labels == d
        count = int(mask.sum())
        digit_sample_count[d] = count
        if count == 0:
            continue
        vals = activations[mask]
        mean_activation[:, d] = vals.mean(axis=0).astype(np.float32, copy=False)
        std_activation[:, d] = vals.std(axis=0, ddof=0).astype(np.float32, copy=False)

    return mean_activation, std_activation, digit_sample_count


def compute_unit_order_by_cosine(mean_activation: np.ndarray) -> np.ndarray:
    if mean_activation.ndim != 2 or mean_activation.shape[1] != 10:
        raise ValueError(
            f"Expected mean_activation of shape (H, 10), got {mean_activation.shape}"
        )

    U = mean_activation.astype(np.float32, copy=False)
    ref = U.mean(axis=0)
    ref_norm = float(np.linalg.norm(ref)) or 1e-8

    sims = np.empty((U.shape[0],), dtype=np.float32)
    for i in range(U.shape[0]):
        v = U[i]
        v_norm = float(np.linalg.norm(v)) or 1e-8
        sims[i] = float(np.dot(v, ref) / (v_norm * ref_norm))

    return np.argsort(-sims).astype(np.int64, copy=False)


def effect_si_from_means(mean_d: np.ndarray, eps: float = 1e-8) -> float:
    """SI = (max - mean(others)) / (max + mean(others) + eps)."""
    mean_d = np.asarray(mean_d, dtype=np.float64).ravel()
    if mean_d.size == 0:
        return 0.0
    k = int(np.argmax(mean_d))
    max_v = float(mean_d[k])
    others = np.delete(mean_d, k)
    if others.size == 0:
        return 0.0
    mo = float(others.mean())
    return float((max_v - mo) / (max_v + mo + eps))


def effect_gap_from_means(mean_d: np.ndarray) -> float:
    """gap = max - second_max (sorted descending)."""
    mean_d = np.asarray(mean_d, dtype=np.float64).ravel()
    if mean_d.size < 2:
        return 0.0
    s = np.sort(mean_d)
    return float(s[-1] - s[-2])


def digit_groups_valid_for_anova(
    digit_sample_count: np.ndarray,
    min_count: int,
) -> bool:
    return bool(np.all(digit_sample_count >= min_count) and np.all(digit_sample_count > 0))


def compute_unit_tuning_stats(
    activation_per_sample: np.ndarray,
    labels: np.ndarray,
    mean_activation: np.ndarray,
    anova_min_count_per_digit: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per original npz row (unit u): ANOVA p-value, preferred digit, effect_si, effect_gap.
    Invalid ANOVA (insufficient counts per digit): p=1, effects still computed from means.
    """
    if activation_per_sample.shape[0] != labels.shape[0]:
        raise ValueError("activation_per_sample and labels length mismatch")
    H = activation_per_sample.shape[1]
    digit_sample_count = np.array(
        [int(np.sum(labels == d)) for d in range(10)], dtype=np.int64
    )
    groups_ok = digit_groups_valid_for_anova(
        digit_sample_count, anova_min_count_per_digit
    )

    p_value = np.ones((H,), dtype=np.float64)
    preferred_digit = np.argmax(mean_activation, axis=1).astype(np.int64)
    effect_si = np.zeros((H,), dtype=np.float32)
    effect_gap = np.zeros((H,), dtype=np.float32)
    anova_invalid = np.ones((H,), dtype=bool)

    for u in range(H):
        mean_d = mean_activation[u].astype(np.float64, copy=False)
        effect_si[u] = np.float32(effect_si_from_means(mean_d))
        effect_gap[u] = np.float32(effect_gap_from_means(mean_d))

    if not groups_ok:
        return (
            p_value,
            preferred_digit,
            effect_si,
            effect_gap,
            digit_sample_count,
            anova_invalid,
        )

    acts_by_digit: List[np.ndarray] = []
    for d in range(10):
        m = labels == d
        acts_by_digit.append(activation_per_sample[m].astype(np.float64, copy=False))

    for u in range(H):
        group_lists = [acts_by_digit[d][:, u] for d in range(10)]
        if any(len(g) < anova_min_count_per_digit for g in group_lists):
            p_value[u] = 1.0
            continue
        try:
            _stat, p = f_oneway(*group_lists)
        except Exception:
            p_value[u] = 1.0
            continue
        if p is None or not np.isfinite(p):
            p_value[u] = 1.0
            continue
        p_value[u] = float(p)
        anova_invalid[u] = False

    return p_value, preferred_digit, effect_si, effect_gap, digit_sample_count, anova_invalid


def compute_is_tuned(
    q_value: np.ndarray,
    effect_si: np.ndarray,
    effect_gap: np.ndarray,
    q_thr: float,
    effect_metric: str,
    effect_thr: float,
    anova_invalid: np.ndarray,
) -> np.ndarray:
    eff = effect_gap if effect_metric == "gap" else effect_si
    eff = np.asarray(eff, dtype=np.float64)
    tuned = (q_value < q_thr) & (eff >= effect_thr) & (~anova_invalid)
    return tuned.astype(bool, copy=False)


def display_order_from_cosine_file(cos_order: np.ndarray) -> np.ndarray:
    """Match viz_utils.viz_gawf_hidden_activation.load_unit_order: use cos_order[::-1]."""
    cos_order = np.asarray(cos_order, dtype=np.int64).ravel()
    return cos_order[::-1].copy()


def npz_to_display_index(display_order_indices: np.ndarray) -> np.ndarray:
    """display_order_indices[j] = npz row at display column j -> inverse map."""
    H = display_order_indices.size
    inv = np.empty(H, dtype=np.int64)
    inv[display_order_indices] = np.arange(H, dtype=np.int64)
    return inv


def compute_argmax_display_order(
    preferred_digit: np.ndarray,
    mean_activation: np.ndarray,
    display_order_indices: np.ndarray,
) -> np.ndarray:
    """
    Column order grouping ALL units by preferred digit (0..9), within-group sorted
    by row-wise z-score of the preferred digit descending. No FDR / significance filter.
    Uses z-score (not effect_gap) so the sort key matches what the heatmap displays.
    """
    H = int(display_order_indices.shape[0])
    inv_disp = npz_to_display_index(display_order_indices)

    # Row-wise z-score: for each unit, normalise across the 10 digit means
    mu = mean_activation.mean(axis=1, keepdims=True)
    sigma = mean_activation.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    z = ((mean_activation - mu) / sigma).astype(np.float64)  # (H, 10)

    col_perm: List[int] = []
    for d in range(10):
        units_d: List[int] = [u for u in range(H) if int(preferred_digit[u]) == d]
        if not units_d:
            continue
        u_arr = np.asarray(units_d, dtype=np.int64)
        z_pref = z[u_arr, d]  # z-score at preferred digit for each unit
        order_local = np.lexsort((u_arr, -z_pref))
        for idx in order_local:
            u = int(u_arr[idx])
            j = int(inv_disp[u])
            col_perm.append(j)
    return np.asarray(col_perm, dtype=np.int64)


def build_tuned_display_order(
    is_tuned: np.ndarray,
    preferred_digit: np.ndarray,
    effect_si: np.ndarray,
    effect_gap: np.ndarray,
    effect_metric: str,
    display_order_indices: np.ndarray,
) -> np.ndarray:
    """
    Column order for panel 3: values are display indices 0..H-1.
    Groups digit 0..9 (tuned & preferred==d), within-group by effect desc, tie npz row.
    Tail: untuned sorted by display index asc.
    """
    H = int(display_order_indices.shape[0])
    eff = effect_gap if effect_metric == "gap" else effect_si
    eff = np.asarray(eff, dtype=np.float64)
    inv_disp = npz_to_display_index(display_order_indices)

    col_perm: List[int] = []
    used = np.zeros(H, dtype=bool)

    for d in range(10):
        units_d: List[int] = []
        for u in range(H):
            if not is_tuned[u]:
                continue
            if int(preferred_digit[u]) != d:
                continue
            units_d.append(u)
        if not units_d:
            continue
        u_arr = np.asarray(units_d, dtype=np.int64)
        eff_u = eff[u_arr]
        order_local = np.lexsort((u_arr, -eff_u))
        for idx in order_local:
            u = int(u_arr[idx])
            j = int(inv_disp[u])
            col_perm.append(j)
            used[j] = True

    tail: List[int] = []
    for j in range(H):
        if not used[j]:
            tail.append(j)
    col_perm.extend(tail)
    return np.asarray(col_perm, dtype=np.int64)


def write_tuned_group_summary(
    path: str,
    is_tuned: np.ndarray,
    preferred_digit: np.ndarray,
    p_value: np.ndarray,
    q_value: np.ndarray,
    effect_si: np.ndarray,
    effect_gap: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    H = is_tuned.size
    lines: List[str] = [
        "GaWF hidden tuning: FDR + effect filtered grouping summary\n",
        f"q_thr={meta.get('q_thr')} (tuned iff q < q_thr)\n",
        f"effect_metric={meta.get('effect_metric')} effect_thr={meta.get('effect_thr')}\n",
        f"fdr_method={meta.get('fdr_method')}\n",
        f"anova_min_count_per_digit={meta.get('anova_min_count_per_digit')}\n",
        "npz_row_index: row in gawf_hidden_activation_stats.npz / tuning npz (original unit id, used as x-axis label in all panels).\n",
        "\n",
    ]

    for d in range(10):
        units_d = [u for u in range(H) if is_tuned[u] and int(preferred_digit[u]) == d]
        lines.append(f"digit {d} (tuned, preferred=={d}, n={len(units_d)}):\n")
        if units_d:
            for u in units_d:
                lines.append(
                    f"  npz_row={u} "
                    f"p={p_value[u]:.6g} q={q_value[u]:.6g} "
                    f"SI={effect_si[u]:.6g} gap={effect_gap[u]:.6g}\n"
                )
        else:
            lines.append("  (empty)\n")
        lines.append("\n")

    untuned = sorted(u for u in range(H) if not is_tuned[u])
    lines.append(f"untuned tail (n={len(untuned)}):\n")
    for u in untuned:
        lines.append(
            f"  npz_row={u} "
            f"p={p_value[u]:.6g} q={q_value[u]:.6g} "
            f"SI={effect_si[u]:.6g} gap={effect_gap[u]:.6g}\n"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved tuning group summary to: {path}")


def run_tuning_pipeline(
    save_dir: str,
    activation_per_sample: np.ndarray,
    labels: np.ndarray,
    mean_activation: np.ndarray,
    cos_order: np.ndarray,
    q_thr: float,
    effect_metric: str,
    effect_thr: float,
    fdr_method: str,
    anova_min_count_per_digit: int,
) -> None:
    p_value, preferred_digit, effect_si, effect_gap, digit_counts, anova_invalid = (
        compute_unit_tuning_stats(
            activation_per_sample,
            labels,
            mean_activation,
            anova_min_count_per_digit,
        )
    )

    if fdr_method != "bh":
        raise ValueError(f"Unsupported fdr_method: {fdr_method}")

    _, q_value = apply_fdr_bh(p_value, alpha=q_thr)
    q_value = np.asarray(q_value, dtype=np.float64)

    for u in range(p_value.shape[0]):
        if anova_invalid[u]:
            p_value[u] = 1.0
            q_value[u] = 1.0

    is_tuned = compute_is_tuned(
        q_value,
        effect_si,
        effect_gap,
        q_thr,
        effect_metric,
        effect_thr,
        anova_invalid,
    )

    display_order_indices = display_order_from_cosine_file(cos_order)

    argmax_order = compute_argmax_display_order(
        preferred_digit, mean_activation, display_order_indices
    )
    argmax_order_path = os.path.join(save_dir, "argmax_display_order.npy")
    np.save(argmax_order_path, argmax_order.astype(np.int64, copy=False))
    print(f"Saved argmax display order to: {argmax_order_path}")

    tuned_order = build_tuned_display_order(
        is_tuned,
        preferred_digit,
        effect_si,
        effect_gap,
        effect_metric,
        display_order_indices,
    )

    tuning_npz = os.path.join(save_dir, "gawf_hidden_tuning_stats.npz")
    np.savez(
        tuning_npz,
        preferred_digit=preferred_digit.astype(np.int64),
        p_value=p_value.astype(np.float64),
        q_value=q_value.astype(np.float64),
        is_tuned=is_tuned.astype(bool),
        effect_si=effect_si.astype(np.float32),
        effect_gap=effect_gap.astype(np.float32),
        mean_activation=mean_activation.astype(np.float32),
    )
    print(f"Saved tuning statistics to: {tuning_npz}")

    order_path = os.path.join(save_dir, "tuned_display_order.npy")
    np.save(order_path, tuned_order.astype(np.int64, copy=False))
    print(f"Saved tuned display column order to: {order_path}")

    meta: Dict[str, Any] = {
        "q_thr": q_thr,
        "effect_metric": effect_metric,
        "effect_thr": effect_thr,
        "fdr_method": fdr_method,
        "anova_min_count_per_digit": anova_min_count_per_digit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "statsmodels_multipletests": _sm_multipletests is not None,
    }
    thr_path = os.path.join(save_dir, "gawf_hidden_tuning_thresholds.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved thresholds/meta to: {thr_path}")

    summary_path = os.path.join(save_dir, "tuned_group_summary.txt")
    write_tuned_group_summary(
        summary_path,
        is_tuned,
        preferred_digit,
        p_value,
        q_value,
        effect_si,
        effect_gap,
        meta,
    )

    meta_path = os.path.join(save_dir, "gawf_hidden_tuning_meta.json")
    meta_full = dict(meta)
    meta_full["digit_sample_count"] = digit_counts.tolist()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_full, f, indent=2)
    print(f"Saved tuning meta to: {meta_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    effect_thr = args.effect_thr
    if effect_thr is None:
        effect_thr = 0.1

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    print("Building test dataset (split=test)...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size (sequence samples): {len(test_ds)}")

    num_workers = 0 if args.use_mmap else 4
    pin_memory = device.type == "cuda" and not args.use_mmap

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

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

    print("Computing GaWF hidden activations for all frames (eval mode)...")
    activation_per_sample, labels_np = compute_hidden_activations(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    N = activation_per_sample.shape[0]
    print(f"Total frames (N): {N}, hidden dim: {activation_per_sample.shape[1]}")

    act_path = os.path.join(args.save_dir, "activation_per_sample.npy")
    lbl_path = os.path.join(args.save_dir, "labels.npy")
    np.save(act_path, activation_per_sample.astype(np.float32, copy=False))
    np.save(lbl_path, labels_np.astype(np.int64, copy=False))
    print(f"Saved activations to: {act_path}")
    print(f"Saved labels to: {lbl_path}")

    print("Computing digit-conditioned statistics...")
    mean_activation, std_activation, digit_sample_count = compute_digit_stats(
        activations=activation_per_sample,
        labels=labels_np,
    )

    stats_path = os.path.join(args.save_dir, "gawf_hidden_activation_stats.npz")
    np.savez(
        stats_path,
        mean_activation=mean_activation,
        std_activation=std_activation,
        digit_sample_count=digit_sample_count,
    )
    print(f"Saved hidden activation statistics to: {stats_path}")
    print(f"mean_activation shape: {mean_activation.shape} (rows=units, cols=digit)")

    cos_order = compute_unit_order_by_cosine(mean_activation)
    order_path = os.path.join(args.save_dir, "unit_order_by_cosine_similarity.npy")
    np.save(order_path, cos_order.astype(np.int64, copy=False))
    print(f"Saved unit order to: {order_path}")

    print("Computing digit tuning (ANOVA + FDR + effect sizes)...")
    run_tuning_pipeline(
        save_dir=args.save_dir,
        activation_per_sample=activation_per_sample,
        labels=labels_np,
        mean_activation=mean_activation,
        cos_order=cos_order,
        q_thr=args.q_thr,
        effect_metric=args.effect_metric,
        effect_thr=effect_thr,
        fdr_method=args.fdr_method,
        anova_min_count_per_digit=args.anova_min_count_per_digit,
    )


if __name__ == "__main__":
    main()
