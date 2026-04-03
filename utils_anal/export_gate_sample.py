"""
Export GaWF gate matrices (gate_ih / gate_hh) for a single sample.

The computation strictly follows the existing GaWFRNNConv implementation:
- Use the same encoder as in train_model.py to obtain x_t.
- Run a single GaWF step at t=0 with zero feedback to obtain char_t / pos_t.
- Build feedback fb = cat([char_t, pos_t]) (logits, no softmax), clamp/unsqueeze.
- Recompute gate_ih / gate_hh (and optionally gated_weight_ih / gated_weight_hh)
  using the same formulas as middle_gawf but with the updated feedback.

Uses the test dataset only (splits=("test",)); train/valid are not loaded.

Example commands:

  # 导出 gate 矩阵（使用 test 集，默认在 sample 0/1/2 中找相同 fg digit）
  python export_gawf_gates.py \\
      --ckpt /path/to/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_model.pth \\
      --split test \\
      --sample_indices 0 1 2 \\
      --out ./gawf_gates.pt

  # 使用 GPU
  python export_gawf_gates.py \\
      --ckpt /path/to/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_model.pth \\
      --split test \\
      --sample_indices 0 1 2 \\
      --out ./gawf_gates.pt \\
      --device cuda
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from train_model import MC_RNN_Dataset
from utils.train_gawf_core import GaWFRNNConv
from utils.train_helpers import set_seed
from utils_anal.anal_helpers import (
    build_model_from_ckpt,
    build_test_dataset,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GaWF gate matrices (gate_ih / gate_hh) for a single sample."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_fb100_model.pth",
        help="Path to trained GaWFRNNConv checkpoint (e.g. *_model.pth).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
        help="Dataset split to sample from (analysis uses test set only).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to select that contain the target fg digit (default: 3).",
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=5,
        choices=list(range(10)),
        help="Target foreground digit (0-9) that all chosen samples must contain (default: 0).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./gawf_gates.pt",
        help="Output path for gate dictionary (default: ./gawf_gates.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device: cpu / cuda (default: cpu).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Temperature tau used in gate computation (default: use model.gate_tau from middle_gawf).",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Also save gated_weight_ih / gated_weight_hh.",
    )

    # Dataset-related options (mirroring train_model defaults)
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
        help="Use sector mode (3x3 sectors) for position labels (default: True, matches training default).",
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters instead of only foreground (default: False).",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help="Load stimuli with numpy mmap_mode='r' (default: True, matches training default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling order (default: 42).",
    )

    return parser.parse_args()


def pick_sample(test_ds: MC_RNN_Dataset, index: int):
    """Select a single sample from the test dataset by index."""
    if index < 0 or index >= len(test_ds):
        raise IndexError(f"Index {index} out of range for test set (len={len(test_ds)}).")
    frames, labels = test_ds[index]
    return frames, labels, test_ds


def select_nonconsecutive_frames_with_same_digit(
    labels: np.ndarray,
    num_frames: int = 3,
    min_time_gap: int = 2,
) -> Tuple[int, List[int]]:
    """
    从单个样本的 label 序列中，选择若干个前景 digit 相同且时间上不连续的 frame 下标。

    Args:
        labels: shape (T, 2 or 3...)，第 0 列为 fg_char_id（sector/coord 模式均如此）。
        num_frames: 期望选择的 frame 数量。
        min_time_gap: 任意两个被选 frame 在时间轴上的最小间隔（>=2 表示不相邻）。

    Returns:
        (digit, frame_indices)：
            digit: 被选中的前景 digit id。
            frame_indices: 在当前 sample 序列中的下标列表（长度为 num_frames）。
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if labels_np.ndim < 2 or labels_np.shape[0] == 0:
        raise ValueError(f"labels has invalid shape: {labels_np.shape}")

    fg_ids = labels_np[:, 0].astype(np.int64)
    T = fg_ids.shape[0]

    # 按 digit 分组所有时间步的位置
    digit_to_indices: dict[int, List[int]] = {}
    for t in range(T):
        d = int(fg_ids[t])
        digit_to_indices.setdefault(d, []).append(t)

    # 优先选择在当前样本中出现次数较多的 digit
    # 对每个 digit 试图贪心选出满足时间间隔约束的若干帧
    candidate = None
    for d, idx_list in sorted(
        digit_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        if len(idx_list) < num_frames:
            continue
        chosen: List[int] = []
        for idx in idx_list:
            if not chosen or all(abs(idx - c) >= min_time_gap for c in chosen):
                chosen.append(idx)
                if len(chosen) == num_frames:
                    candidate = (d, chosen)
                    break
        if candidate is not None:
            break

    # 若无法满足时间间隔约束，退化为同 digit 的前 num_frames 个位置（仍保证 digit 一致）
    if candidate is None:
        for d, idx_list in digit_to_indices.items():
            if len(idx_list) >= num_frames:
                candidate = (d, idx_list[:num_frames])
                break

    if candidate is None:
        # 极端情况：当前样本中没有任何 digit 至少出现 num_frames 次
        # 退化为按照出现顺序取前 num_frames 个时间步（可能 digit 不同）。
        fallback_indices = list(range(min(num_frames, T)))
        fallback_digit = int(fg_ids[0])
        return fallback_digit, fallback_indices

    return candidate


def compute_gates_for_single_frame(
    model: GaWFRNNConv,
    x_frame: torch.Tensor,
    device: torch.device,
    tau: float,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对单个时间步的输入 frame（形状 (1, 1, C, H, W)）计算 GaWF gate 矩阵。

    Returns:
        gate_ih: (1, hidden_size, input_size)
        gate_hh: (1, hidden_size, hidden_size)
        prev_fb: (1, fb_dim)
    """
    x_input = x_frame.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        batch_size, frame_num, channels, height, width = x_input.size()
        x_flat = x_input.view(batch_size * frame_num, channels, height, width)
        x_encoded = model.encoder(x_flat)
        x_seq = x_encoded.view(batch_size, frame_num, -1)

        # 单步（t=0）
        x_t = x_seq[:, 0, :]  # (B, input_size)
        fb_dim = model.num_classes + model.num_pos
        fb = torch.zeros(batch_size, fb_dim, device=device, dtype=torch.float32)

        hidden_size = model.rnn.hidden_size
        h_prev = torch.zeros(batch_size, hidden_size, device=device, dtype=x_t.dtype)

        # 第一次 middle_gawf：fb=0，仅用于得到 logits -> prev_feedback
        fb_t0 = fb.clamp(-10, 10).unsqueeze(2)
        gated_output0 = model.middle_gawf(x_t, h_prev, fb_t0)
        char_t, pos_t = model.classifier(gated_output0)

        # 和 forward() 一致：反馈直接使用 logits
        prev_fb = torch.cat([char_t, pos_t], dim=-1)

        # 使用更新后的反馈重新计算 gate 矩阵
        fb_t1 = prev_fb.clamp(-10, 10).unsqueeze(2)

        input_size = x_t.size(-1)
        V_ih = model.V[:, :input_size].unsqueeze(0)
        V_hh = model.V[:, input_size:].unsqueeze(0)

        if verbose:
            U, V = model.U, model.V
            print("[U] shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
                tuple(U.shape), U.min().item(), U.max().item(), U.mean().item(), U.std().item()))
            print("[V] shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
                tuple(V.shape), V.min().item(), V.max().item(), V.mean().item(), V.std().item()))
            ft1 = fb_t1.squeeze()
            print("[fb_t1] shape=%s min=%.4f max=%.4f std=%.4f" % (
                tuple(fb_t1.shape), ft1.min().item(), ft1.max().item(), ft1.std().item()))
            fb_V_ih = fb_t1 * V_ih
            print("[fb_t1 * V_ih] mean=%.4f max=%.4f" % (fb_V_ih.mean().item(), fb_V_ih.abs().max().item()))

        trans_ih = torch.matmul(model.U, fb_t1 * V_ih)
        trans_hh = torch.matmul(model.U, fb_t1 * V_hh)

        if verbose:
            print("[trans_ih] mean=%.4f max=%.4f" % (trans_ih.mean().item(), trans_ih.abs().max().item()))

        gate_ih = torch.sigmoid(trans_ih / tau)
        gate_hh = torch.sigmoid(trans_hh / tau)

    return gate_ih, gate_hh, prev_fb


def collect_gate_matrices_for_digits(
    test_ds: MC_RNN_Dataset,
    model: GaWFRNNConv,
    device: torch.device,
    tau: Optional[float] = None,
    num_per_digit: int = 100,
    verbose: bool = False,
) -> Tuple[dict, dict]:
    """
    Collect gate_ih and gate_hh matrices for each digit 0-9 from the test set.
    Does not save to file; returns in-memory lists.

    Returns:
        gate_ih_by_digit: dict[int, List[np.ndarray]]  # digit -> list of (H, W) matrices
        gate_hh_by_digit: dict[int, List[np.ndarray]]  # digit -> list of (H, H) matrices
    """
    from collections import defaultdict

    resolved_tau = float(model.gate_tau if tau is None else tau)

    # 1) Build digit -> [(sample_idx, frame_idx), ...] mapping
    digit_to_pairs: dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for sample_idx in range(len(test_ds)):
        frames, labels = test_ds[sample_idx]
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        if labels_np.ndim < 2:
            continue
        fg_ids = labels_np[:, 0].astype(np.int64)
        for t in range(len(fg_ids)):
            d = int(fg_ids[t])
            if 0 <= d <= 9 and len(digit_to_pairs[d]) < num_per_digit:
                digit_to_pairs[d].append((sample_idx, t))

    gate_ih_by_digit: dict[int, List[np.ndarray]] = {}
    gate_hh_by_digit: dict[int, List[np.ndarray]] = {}

    for digit in range(10):
        pairs = digit_to_pairs.get(digit, [])
        gate_ih_list: List[np.ndarray] = []
        gate_hh_list: List[np.ndarray] = []
        for sample_idx, frame_idx in pairs:
            frames, _ = test_ds[sample_idx]
            if isinstance(frames, torch.Tensor):
                frames_t = frames
            else:
                frames_t = torch.as_tensor(frames, dtype=torch.float32)
            single_frame = frames_t[frame_idx : frame_idx + 1]
            x_frame = single_frame.unsqueeze(0)
            gate_ih, gate_hh, _ = compute_gates_for_single_frame(
                model=model,
                x_frame=x_frame,
                device=device,
                tau=resolved_tau,
                verbose=verbose,
            )
            g_ih = gate_ih.squeeze(0).detach().cpu().numpy()
            g_hh = gate_hh.squeeze(0).detach().cpu().numpy()
            gate_ih_list.append(g_ih)
            gate_hh_list.append(g_hh)
        gate_ih_by_digit[digit] = gate_ih_list
        gate_hh_by_digit[digit] = gate_hh_list
        if verbose:
            print(f"Digit {digit}: collected {len(gate_ih_list)} matrices")

    return gate_ih_by_digit, gate_hh_by_digit


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 1) Build test dataset only (same helpers as training, splits=("test",)).
    test_ds, num_pos = build_test_dataset(args)
    ds = test_ds

    # 2) 根据指定的 fg_digit，在 test 集中寻找包含该 digit 的若干 sample。
    target_fg_digit = int(args.digit)
    max_samples = int(args.num_samples)

    frames_per_sample: List[torch.Tensor] = []
    labels_per_sample_np: List[np.ndarray] = []
    sample_indices: List[int] = []

    for sidx in range(len(test_ds)):
        frames, labels = test_ds[sidx]
        if isinstance(frames, torch.Tensor):
            frames_t = frames
        else:
            frames_t = torch.as_tensor(frames)
        if frames_t.ndim != 4:
            raise ValueError(
                f"Unexpected frames tensor shape for sample {sidx} (expect T,C,H,W): {frames_t.shape}"
            )

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        if labels_np.ndim < 2 or labels_np.shape[0] == 0:
            raise ValueError(f"labels for sample {sidx} has invalid shape: {labels_np.shape}")

        fg_ids = labels_np[:, 0].astype(np.int64)
        digits = set(int(d) for d in fg_ids.tolist())

        if target_fg_digit in digits:
            sample_indices.append(int(sidx))
            frames_per_sample.append(frames_t)
            labels_per_sample_np.append(labels_np)
            print(
                f"[match] sample {sidx}: frames shape={tuple(frames_t.shape)}, "
                f"fg digits={sorted(digits)}"
            )
            if len(sample_indices) >= max_samples:
                break

    if not sample_indices:
        raise RuntimeError(
            f"No samples in test set contain fg_digit={target_fg_digit}."
        )
    if len(sample_indices) < max_samples:
        print(
            f"[warn] Only found {len(sample_indices)} samples containing fg_digit={target_fg_digit}, "
            f"less than requested num_samples={max_samples}."
        )

    print(f"Using sample indices (split=test, fg_digit={target_fg_digit}): {sample_indices}")

    # 2.2) 对每个 sample，选取该目标 digit 的一个时间点 t（若该 sample 中不存在该 digit，则退化为 t=0）。
    selected_frame_indices: List[int] = []
    selected_global_indices: List[int] = []

    if isinstance(ds, MC_RNN_Dataset):
        frame_num = ds.frame_num
        chan_num = ds.chan_num
    else:
        frame_num = frames_per_sample[0].shape[0]
        chan_num = 0

    neighbor_frames_tensor = None
    neighbor_global_indices_tensor = None
    data_all = getattr(ds, "data", None)
    neighbor_global_indices: List[List[int]] = []
    neighbor_frames: List[torch.Tensor] = []

    for sidx, frames_t, labels_np in zip(sample_indices, frames_per_sample, labels_per_sample_np):
        fg_ids = labels_np[:, 0].astype(np.int64)
        t_candidates = np.where(fg_ids == target_fg_digit)[0]
        if t_candidates.size == 0:
            print(
                f"[warn] sample {sidx} has no frame with fg_digit={target_fg_digit}; "
                "fallback to t=0 for this sample."
            )
            t = 0
        else:
            t = int(t_candidates[0])
            print(f"[info] sample {sidx} has frame with fg_digit={target_fg_digit} at t={t_candidates}")

        selected_frame_indices.append(int(t))

        # 参考 MC_RNN_Dataset.__getitem__ 中的定义：
        #   start_idx = (idx * frame_num) + chan_num
        global_start_idx = (int(sidx) * frame_num) + chan_num
        g_idx = global_start_idx + t
        selected_global_indices.append(int(g_idx))

        # 额外保存用于可视化的原始帧：对于该时间点 t 的全局索引 g_idx，取 t-1, t, t+1。
        if data_all is not None:
            total_T = int(data_all.shape[0])
            local_indices = []
            local_frames = []
            for offset in (-1, 0, 1):
                gi = int(np.clip(g_idx + offset, 0, total_T - 1))
                local_indices.append(gi)
                frame_img = data_all[gi]  # (H, W) 或 (C, H, W)
                local_frames.append(torch.as_tensor(frame_img, dtype=torch.float32))
            neighbor_global_indices.append(local_indices)          # (3,)
            neighbor_frames.append(torch.stack(local_frames, 0))   # (3, H, W) or (3, C, H, W)

    print(
        f"Selected per-sample frame indices (t)={selected_frame_indices}, "
        f"global={selected_global_indices}"
    )

    if data_all is not None and neighbor_frames:
        neighbor_frames_tensor = torch.stack(neighbor_frames, 0)              # (N, 3, ...)
        neighbor_global_indices_tensor = torch.as_tensor(
            neighbor_global_indices, dtype=torch.long
        )  # (N, 3)

    # 3) Build and load GaWFRNNConv model from checkpoint.
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    print(
        f"Loaded GaWFRNNConv from '{args.ckpt}' "
        f"(hidden_size={model.hidden_size}, num_pos={model.num_pos})"
    )

    # 4) 对选中的多个样本 / 时间点分别计算 gate 矩阵。
    tau = float(model.gate_tau if args.tau is None else args.tau)

    gate_ih_list: List[torch.Tensor] = []
    gate_hh_list: List[torch.Tensor] = []
    prev_fb_list: List[torch.Tensor] = []

    for frames_t, t in zip(frames_per_sample, selected_frame_indices):
        # frames_t 形状 (T, C, H, W)，取单个时间步 t，shape -> (1, C, H, W)
        single_frame = frames_t[t : t + 1]
        x_frame = single_frame.unsqueeze(0)  # (1, 1, C, H, W)
        gate_ih, gate_hh, prev_fb = compute_gates_for_single_frame(
            model=model,
            x_frame=x_frame,
            device=device,
            tau=tau,
        )

        # 只保存去掉 batch 维度后的 2D gate。
        gate_ih_2d = gate_ih.squeeze(0).detach().cpu()
        gate_hh_2d = gate_hh.squeeze(0).detach().cpu()
        prev_fb_vec = prev_fb.squeeze(0).detach().cpu()

        gate_ih_list.append(gate_ih_2d)
        gate_hh_list.append(gate_hh_2d)
        prev_fb_list.append(prev_fb_vec)

    # 7) 准备保存的张量。
    # 为了兼容旧版可视化脚本，仍然保留单个 gate_ih / gate_hh 字段（取第一个时间点）。
    gate_ih_first = gate_ih_list[0]
    gate_hh_first = gate_hh_list[0]
    prev_fb_first = prev_fb_list[0]

    gate_ih_stack = torch.stack(gate_ih_list, dim=0)  # (N, H, W)
    gate_hh_stack = torch.stack(gate_hh_list, dim=0)  # (N, H, W)
    prev_fb_stack = torch.stack(prev_fb_list, dim=0)  # (N, fb_dim)

    save_dict = {
        # 兼容字段：单个 gate（第一个被选时间点）
        "gate_ih": gate_ih_first,
        "gate_hh": gate_hh_first,
        "prev_fb": prev_fb_first,
        # 扩展字段：多个时间点的 gate
        "gate_ih_all": gate_ih_stack,
        "gate_hh_all": gate_hh_stack,
        "prev_fb_all": prev_fb_stack,
        # 维度信息
        "input_size": int(gate_ih_first.shape[1]),
        "hidden_size": int(gate_ih_first.shape[0]),
        "fb_dim": int(prev_fb_first.numel()),
        "tau": tau,
        # 元信息：整体使用到的 sample indices，以及目标前景 digit
        "sample_indices": torch.as_tensor(sample_indices, dtype=torch.long),
        "target_fg_digit": int(target_fg_digit),
        "split": args.split,
        "ckpt_path": os.path.abspath(args.ckpt),
        # 被选中的样本 / 时间点及其信息
        "selected_fg_digit": int(target_fg_digit),
        "selected_frame_indices": torch.as_tensor(
            selected_frame_indices, dtype=torch.long
        ),
        "selected_global_indices": torch.as_tensor(
            selected_global_indices, dtype=torch.long
        ),
    }

    # 如果有邻近帧信息，则一并保存，供可视化脚本使用。
    if neighbor_frames_tensor is not None and neighbor_global_indices_tensor is not None:
        save_dict["neighbor_frames"] = neighbor_frames_tensor
        save_dict["neighbor_global_indices"] = neighbor_global_indices_tensor

    if args.save_weights:
        # 这里仅对第一个选中 frame 保存对应的 gated weight，
        # 与旧版脚本行为一致（单帧 gate）。
        weight_ih_full = model.rnn.weight_ih_l0.detach().cpu()
        weight_hh_full = model.rnn.weight_hh_l0.detach().cpu()
        save_dict["gated_weight_ih"] = (gate_ih_first * weight_ih_full)
        save_dict["gated_weight_hh"] = (gate_hh_first * weight_hh_full)

    out_dir = os.path.dirname(args.save_dir)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(save_dict, args.save_dir)

    # 8) Required diagnostics / prints.
    print(f"Saved gate dictionary to: {args.save_dir}")
    print(f"gate_ih shape: {tuple(gate_ih_2d.shape)}")
    print(f"gate_hh shape: {tuple(gate_hh_2d.shape)}")

    prev_fb_min = prev_fb_vec.min().item()
    prev_fb_max = prev_fb_vec.max().item()
    print(f"prev_fb min: {prev_fb_min:.6f}, max: {prev_fb_max:.6f}")

    gate_ih_min = gate_ih_2d.min().item()
    gate_ih_max = gate_ih_2d.max().item()
    gate_ih_mean = gate_ih_2d.mean().item()
    print(f"gate_ih min: {gate_ih_min:.6f}, max: {gate_ih_max:.6f}, mean: {gate_ih_mean:.6f}")

    gate_hh_min = gate_hh_2d.min().item()
    gate_hh_max = gate_hh_2d.max().item()
    gate_hh_mean = gate_hh_2d.mean().item()
    print(f"gate_hh min: {gate_hh_min:.6f}, max: {gate_hh_max:.6f}, mean: {gate_hh_mean:.6f}")


if __name__ == "__main__":
    main()

