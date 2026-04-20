"""Export per-offset switch-window accuracies on the test split.

``--switch_target fg`` (default): windows from ``fg_switch``; sector model; per-frame
char + sector accuracy vs ``labels[:,:,0]`` / ``labels[:,:,1]``.

``--switch_target bg``: same model and same **foreground** char/sector metrics as fg mode,
but pre5/post5 windows are aligned to **bg_switch** transition times.

The optional ``_finetune_fcchars_only`` / head-training helpers and CLI flags are retained
for reference but **not invoked** from ``main()`` (legacy bg-multiset head path).

CUDA: if ``--device cuda`` and ``CUDA_VISIBLE_DEVICES`` is unset, selects a GPU via
``pick_cuda_device_index_prefer_no_python()`` (same idea as ``train_model.py``, but
prefers a card with no Python compute process).

Outputs (in ``--save_dir``):
- fg: ``fg_switch_offset_acc_<ckpt_tag>.npz``, ``fg_switch_offset_meta_<ckpt_tag>.json``
- bg: ``bg_switch_offset_acc_<ckpt_tag>.npz``, ``bg_switch_offset_meta_<ckpt_tag>.json``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.train_acceleration import (
    AccelerationConfig,
    run_forward_with_feedback,
    setup_acceleration,
)
from utils.train_predict_all_chars import loss_char_all_chars
from utils.train_helpers import pick_cuda_device_index_prefer_no_python
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device

OFFSET_ORDER: List[int] = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
OFFSET_LABELS: List[str] = ["pre5", "pre4", "pre3", "pre2", "pre1", "post1", "post2", "post3", "post4", "post5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-offset (pre5..post5) test accuracies for one or more checkpoints."
    )
    parser.add_argument(
        "--switch_target",
        type=str,
        choices=("fg", "bg"),
        default="fg",
        help="Use fg_switch or bg_switch for transition timing (dataset head implied).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./results/train_data/sector_40h_adamw_0409",
        help="Directory containing *_model.pth checkpoints.",
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths. If set, --ckpt_dir is ignored.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/fg_switch_offset_acc",
        help="Directory to save exported npz/json files.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--head_epochs",
        type=int,
        default=5,
        help="Unused: reserved for legacy ``_finetune_fcchars_only`` (not called from main).",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-3,
        help="Unused: reserved for legacy ``_finetune_fcchars_only`` (not called from main).",
    )
    parser.add_argument(
        "--head_batch_size",
        type=int,
        default=0,
        help="Unused: reserved for legacy ``_finetune_fcchars_only`` (not called from main).",
    )
    parser.add_argument(
        "--no_accel",
        action="store_true",
        default=False,
        help="Unused: reserved for legacy ``_finetune_fcchars_only`` (not called from main).",
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="40h-float32",
        help="Stimulus/label filename tail after split base (e.g. 40h-float32 -> stimulus_reg-test-40h-float32.npy).",
    )
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _collect_ckpts(args: argparse.Namespace) -> List[str]:
    if args.ckpts:
        paths = [os.path.abspath(p) for p in args.ckpts]
    else:
        ckpt_dir = os.path.abspath(args.ckpt_dir)
        names = sorted(n for n in os.listdir(ckpt_dir) if n.endswith("_model.pth"))
        paths = [os.path.join(ckpt_dir, n) for n in names]
    if not paths:
        raise RuntimeError("No checkpoints found to evaluate.")
    return paths


def _parse_model_key(ckpt_path: str) -> str:
    base = os.path.basename(ckpt_path).lower()
    for key in ("gawf", "rnn", "lstm", "gru"):
        if base.startswith(f"{key}_"):
            return key
    return "unknown"


def _build_offset_targets_from_switch(switch_01: np.ndarray) -> np.ndarray:
    """
    Per-frame offset in {-5..-1, 1..5} with the same rules as fg pre5/post5 windowing.
    """
    switch_01 = np.asarray(switch_01).astype(np.int32, copy=False)
    num_frames = int(switch_01.shape[0])
    switches = np.where(switch_01 != 0)[0].tolist()

    forbidden = np.zeros(num_frames, dtype=bool)
    for i in range(1, len(switches)):
        s_prev, s_curr = switches[i - 1], switches[i]
        if s_curr - s_prev < 10:
            left = s_prev + 1
            right = s_curr
            forbidden[left:right] = True

    post_offset = np.zeros(num_frames, dtype=np.int8)
    pre_offset = np.zeros(num_frames, dtype=np.int8)
    for s in switches:
        for dt in range(0, 5):
            t = s + dt
            if 0 <= t < num_frames:
                post_offset[t] = np.int8(dt + 1)
        for dt in range(1, 6):
            t = s - dt
            if 0 <= t < num_frames:
                pre_offset[t] = np.int8(-dt)

    post_offset[forbidden] = 0
    pre_offset[forbidden] = 0
    pre_offset[post_offset != 0] = 0

    offset_targets = np.zeros(num_frames, dtype=np.int8)
    offset_targets[pre_offset != 0] = pre_offset[pre_offset != 0]
    offset_targets[post_offset != 0] = post_offset[post_offset != 0]
    return offset_targets


def _offset_stats_template_fg() -> Dict[int, Dict[str, int]]:
    return {
        off: {
            "char_correct": 0,
            "char_total": 0,
            "pos_correct": 0,
            "pos_total": 0,
        }
        for off in OFFSET_ORDER
    }


def _tensor_to_cpu_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_ckpt_offset_acc(
    ckpt_path: str,
    model: torch.nn.Module,
    test_ds,
    device: torch.device,
    batch_size: int,
    *,
    switch_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-offset char (fg id) and sector accuracy; windows from ``fg_switch`` or ``bg_switch``.
    """
    if switch_source == "fg":
        switch_arr = getattr(test_ds, "fg_switch", None)
        sw_name = "fg_switch"
    elif switch_source == "bg":
        switch_arr = getattr(test_ds, "bg_switch", None)
        sw_name = "bg_switch"
    else:
        raise ValueError(f"switch_source must be 'fg' or 'bg', got {switch_source!r}")
    if switch_arr is None:
        raise RuntimeError(f"Test dataset does not expose {sw_name}; cannot build offset labels.")

    offset_targets = _build_offset_targets_from_switch(switch_arr)
    stats = _offset_stats_template_fg()

    dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model_key = _parse_model_key(ckpt_path)
    use_feedback = True if model_key == "gawf" else None
    seq_len = int(getattr(test_ds, "frame_num", 32))
    chan_num = int(getattr(test_ds, "chan_num", 2))

    model.eval()
    samples_done = 0
    with torch.no_grad():
        for bidx, batch in enumerate(dl):
            inputs, labels = batch[0], batch[1]
            if inputs.dtype == torch.float64:
                inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            out_char, out_pos = run_forward_with_feedback(
                model,
                inputs,
                use_feedback=use_feedback,
            )

            pred_char_ok = (torch.argmax(out_char, dim=2) == labels[:, :, 0].long())
            pred_pos_ok = (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long())
            pred_char_ok_np = _tensor_to_cpu_np(pred_char_ok)
            pred_pos_ok_np = _tensor_to_cpu_np(pred_pos_ok)

            bs = int(pred_char_ok_np.shape[0])
            global_start = bidx * batch_size * seq_len + chan_num
            global_end = global_start + bs * seq_len
            batch_offsets = offset_targets[global_start:global_end].reshape(bs, seq_len)

            for off in OFFSET_ORDER:
                m = batch_offsets == off
                n = int(m.sum())
                if n == 0:
                    continue
                stats[off]["char_correct"] += int((pred_char_ok_np & m).sum())
                stats[off]["char_total"] += n
                stats[off]["pos_correct"] += int((pred_pos_ok_np & m).sum())
                stats[off]["pos_total"] += n

            samples_done += bs
            if samples_done % 200 < bs or bidx == 0:
                print(f"  [{switch_source} eval] batches={bidx + 1}, sequences≈{samples_done}")

    char_acc = np.zeros(len(OFFSET_ORDER), dtype=np.float32)
    pos_acc = np.zeros(len(OFFSET_ORDER), dtype=np.float32)
    frame_counts = np.zeros(len(OFFSET_ORDER), dtype=np.int64)
    for i, off in enumerate(OFFSET_ORDER):
        c_tot = stats[off]["char_total"]
        p_tot = stats[off]["pos_total"]
        frame_counts[i] = c_tot
        char_acc[i] = (
            100.0 * float(stats[off]["char_correct"]) / float(c_tot) if c_tot > 0 else 0.0
        )
        pos_acc[i] = (
            100.0 * float(stats[off]["pos_correct"]) / float(p_tot) if p_tot > 0 else 0.0
        )
    return char_acc, pos_acc, frame_counts


def evaluate_ckpt_offset_acc_fg(
    ckpt_path: str,
    model: torch.nn.Module,
    test_ds,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for ``evaluate_ckpt_offset_acc(..., switch_source='fg')``."""
    return evaluate_ckpt_offset_acc(
        ckpt_path, model, test_ds, device, batch_size, switch_source="fg"
    )


# Legacy: all-chars ``fcchars`` finetune (not invoked from ``main()``). Restore imports from
# ``utils_anal.anal_helpers`` (``build_train_dataset_allchars``, ``build_rnn_allchars_model_from_sector_ckpt``)
# if this path is re-enabled.
def _finetune_fcchars_only(
    model: torch.nn.Module,
    train_ds,
    device: torch.device,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    max_chars: int,
    seed: int,
    use_accel: bool,
) -> None:
    torch.manual_seed(seed)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        (p for p in model.fcchars.parameters() if p.requires_grad),
        lr=lr,
    )
    if use_accel:
        accel_cfg = AccelerationConfig(use_acceleration=True)
        autocast_fn, scaler, _, _, pin_memory = setup_acceleration(
            accel_cfg, device, logger=None
        )
        amp_on = scaler is not None
        tqdm.write(
            f"  [bg head] acceleration: AMP={'on' if amp_on else 'off'}, "
            f"pin_memory={pin_memory}"
        )
    else:
        autocast_fn = lambda _: nullcontext()
        scaler = None
        pin_memory = False
        tqdm.write("  [bg head] acceleration: disabled (--no_accel)")

    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    n_steps = len(dl)
    non_blocking = bool(pin_memory and device.type == "cuda")
    for ep in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(
            dl,
            desc=f"[bg head] epoch {ep + 1}/{epochs}",
            total=n_steps,
            unit="batch",
            leave=True,
        )
        for batch in pbar:
            inputs, labels = batch[0], batch[1]
            if inputs.dtype == torch.float64:
                inputs = inputs.float()
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            with autocast_fn(device):
                out_char, _ = run_forward_with_feedback(model, inputs, use_feedback=None)
                loss, _ = loss_char_all_chars(
                    out_char, labels, criterion, max_chars, device
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                bad_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.fcchars.parameters()
                )
                if bad_grad:
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    loss_val = float(loss.detach().float().item())
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.fcchars.parameters(), max_norm=1.0
                    )
                    scaler.step(opt)
                    scaler.update()
                    loss_val = float(loss.detach().float().item())
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.fcchars.parameters(), max_norm=1.0)
                opt.step()
                loss_val = float(loss.detach().item())

            epoch_loss += loss_val
            n_batches += 1
            mean_so_far = epoch_loss / n_batches
            pbar.set_postfix(loss=f"{loss_val:.4f}", mean=f"{mean_so_far:.4f}")
        mean_loss = epoch_loss / max(n_batches, 1)
        tqdm.write(f"  [bg head train] epoch {ep + 1}/{epochs} done, mean loss={mean_loss:.4f}")
    model.eval()


def _save_outputs_fg(
    save_dir: str,
    ckpt_path: str,
    char_acc: np.ndarray,
    pos_acc: np.ndarray,
    frame_counts: np.ndarray,
) -> None:
    ckpt_tag = os.path.basename(ckpt_path).replace("_model.pth", "")
    npz_path = os.path.join(save_dir, f"fg_switch_offset_acc_{ckpt_tag}.npz")
    meta_path = os.path.join(save_dir, f"fg_switch_offset_meta_{ckpt_tag}.json")

    np.savez(
        npz_path,
        offset_order=np.asarray(OFFSET_ORDER, dtype=np.int64),
        offset_labels=np.asarray(OFFSET_LABELS, dtype="<U6"),
        char_acc=char_acc.astype(np.float32),
        sector_acc=pos_acc.astype(np.float32),
        frame_counts=frame_counts.astype(np.int64),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "switch_target": "fg",
                "offset_order": OFFSET_ORDER,
                "offset_labels": OFFSET_LABELS,
                "frame_counts": frame_counts.astype(np.int64).tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved npz:  {npz_path}")
    print(f"Saved meta: {meta_path}")


def _save_outputs_bg(
    save_dir: str,
    ckpt_path: str,
    char_acc: np.ndarray,
    pos_acc: np.ndarray,
    frame_counts: np.ndarray,
) -> None:
    ckpt_tag = os.path.basename(ckpt_path).replace("_model.pth", "")
    npz_path = os.path.join(save_dir, f"bg_switch_offset_acc_{ckpt_tag}.npz")
    meta_path = os.path.join(save_dir, f"bg_switch_offset_meta_{ckpt_tag}.json")

    np.savez(
        npz_path,
        offset_order=np.asarray(OFFSET_ORDER, dtype=np.int64),
        offset_labels=np.asarray(OFFSET_LABELS, dtype="<U6"),
        char_acc=char_acc.astype(np.float32),
        sector_acc=pos_acc.astype(np.float32),
        frame_counts=frame_counts.astype(np.int64),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": os.path.abspath(ckpt_path),
                "switch_target": "bg",
                "offset_order": OFFSET_ORDER,
                "offset_labels": OFFSET_LABELS,
                "frame_counts": frame_counts.astype(np.int64).tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved npz:  {npz_path}")
    print(f"Saved meta: {meta_path}")


def main() -> None:
    args = parse_args()
    cuda_visible_preset = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    if args.device == "cuda" and not cuda_visible_preset:
        cuda_index = pick_cuda_device_index_prefer_no_python()
        if cuda_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)
            print(
                f"CUDA_VISIBLE_DEVICES={cuda_index} "
                f"(prefer GPU without Python compute; preset env overrides this)."
            )
        else:
            print(
                "Could not auto-pick CUDA index (nvidia-smi unavailable?); "
                "using default CUDA device order."
            )
    elif args.device == "cuda" and cuda_visible_preset:
        print(
            f"Using preset CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    device = resolve_device(args.device, require_cuda_if_requested=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    print(f"Test dataset size: {len(test_ds)}")

    ckpt_paths = _collect_ckpts(args)
    print(f"Found {len(ckpt_paths)} checkpoints.")

    for ckpt in ckpt_paths:
        print(f"\n[eval] {ckpt}")
        model = build_model_from_ckpt(ckpt, num_pos=num_pos, device=device)
        char_acc, pos_acc, frame_counts = evaluate_ckpt_offset_acc(
            ckpt_path=ckpt,
            model=model,
            test_ds=test_ds,
            device=device,
            batch_size=args.batch_size,
            switch_source=args.switch_target,
        )
        if args.switch_target == "fg":
            _save_outputs_fg(args.save_dir, ckpt, char_acc, pos_acc, frame_counts)
        else:
            _save_outputs_bg(args.save_dir, ckpt, char_acc, pos_acc, frame_counts)


if __name__ == "__main__":
    main()
