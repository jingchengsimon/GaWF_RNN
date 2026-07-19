"""Run inference-time feedback-component ablations for a frozen GaWF checkpoint.

This script evaluates a trained single-layer GaWF model on the same test frames under
different feedback lesions. At each time step, it computes char/sector logits, builds the
next feedback vector, then applies the requested lesion to the digit slice ``[0:10]`` and/or
sector slice ``[10:19]`` before the next gate computation.

Outputs (in --save_dir):
- ablation_metrics.json  — per-condition char/sector accuracy and switch recovery curves
- ablation_metrics.csv   — flat table with one row per condition
- frame_predictions.npz  — compressed per-frame predictions/labels for reproducibility
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils.clutter_train_helpers import set_seed
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset


DEFAULT_CONDITIONS = ["baseline", "clear_digit", "clear_sector", "clear_all"]
DIGIT_SLICE = slice(0, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-time GaWF feedback-component ablation on the test split."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to GaWF *_model.pth.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help=(
            "Ablation conditions. Supported: baseline clear_digit clear_sector clear_all "
            "shuffle_digit shuffle_sector. With --shuffle, shuffle_digit/shuffle_sector "
            "are appended if not already present."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Append shuffle_digit and shuffle_sector controls.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Post-fg-switch recovery offsets +1..+K.",
    )
    parser.add_argument(
        "--pre_K",
        type=int,
        default=5,
        help="Pre-fg-switch offsets -pre_K..-1 to include in switch curves.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("G_behaviour", "feedback_ablation", "data")),
        help="Directory for analysis outputs.",
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="40h-uint8")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 evaluates the full test split.",
    )
    parser.add_argument(
        "--debug_switch_map",
        type=int,
        default=0,
        help="Print this many fg_switch raw-frame to output-index mappings, then continue.",
    )
    return parser.parse_args()


def _validate_conditions(conditions: Sequence[str]) -> List[str]:
    valid = {
        "baseline",
        "clear_digit",
        "clear_sector",
        "clear_all",
        "shuffle_digit",
        "shuffle_sector",
    }
    out = []
    for condition in conditions:
        if condition not in valid:
            raise ValueError(f"Unknown condition {condition!r}; valid={sorted(valid)}")
        if condition not in out:
            out.append(condition)
    return out


def _condition_slices(condition: str, num_pos: int) -> Tuple[bool, bool]:
    if condition == "clear_digit":
        return True, False
    if condition == "clear_sector":
        return False, True
    if condition == "clear_all":
        return True, True
    if condition in ("baseline", "shuffle_digit", "shuffle_sector"):
        return False, False
    raise ValueError(condition)


def _apply_clear_feedback(fb: torch.Tensor, condition: str, num_pos: int) -> torch.Tensor:
    clear_digit, clear_sector = _condition_slices(condition, num_pos)
    if not clear_digit and not clear_sector:
        return fb

    fb = fb.clone()
    if clear_digit:
        fb[:, DIGIT_SLICE] = 0.0
    if clear_sector:
        fb[:, 10 : 10 + num_pos] = 0.0
    return fb


def _run_baseline_feedback_schedule(
    model,
    seq: torch.Tensor,
    *,
    num_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the unablated next-feedback schedule with shape (B, T, 10+num_pos)."""
    batch_size, frame_num, _ = seq.shape
    hidden_size = int(model.rnn.hidden_size)
    feedback_dim = int(model.feedback_dim)
    h = torch.zeros(batch_size, hidden_size, device=device, dtype=seq.dtype)
    fb = torch.zeros(batch_size, feedback_dim, device=device, dtype=seq.dtype)
    schedule = torch.empty(batch_size, frame_num, feedback_dim, device=device, dtype=seq.dtype)

    with torch.no_grad():
        for t in range(frame_num):
            x_t = seq[:, t, :]
            gated = model.middle_gawf(x_t, h, fb.clamp(-10, 10).unsqueeze(2))
            char_t, pos_t = model.classifier(gated)
            fb = model._compute_feedback(char_t, pos_t)
            schedule[:, t, :] = fb
            h = gated
    expected = 10 + num_pos
    if schedule.shape[-1] != expected:
        raise RuntimeError(f"Expected feedback_dim={expected}, got {schedule.shape[-1]}.")
    return schedule


def _shuffled_schedule_slice(
    schedule: torch.Tensor,
    condition: str,
    rng: np.random.Generator,
    num_pos: int,
) -> Optional[torch.Tensor]:
    if condition not in ("shuffle_digit", "shuffle_sector"):
        return None
    batch_size, frame_num, _ = schedule.shape
    shuffled = schedule.clone()
    if condition == "shuffle_digit":
        slc = DIGIT_SLICE
    else:
        slc = slice(10, 10 + num_pos)
    for b in range(batch_size):
        perm = torch.as_tensor(rng.permutation(frame_num), device=schedule.device)
        shuffled[b, :, slc] = schedule[b, perm, slc]
    return shuffled


def _rollout_condition(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    condition: str,
    num_pos: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one condition and return char/sector predictions with shape (B, T)."""
    batch_size, frame_num, channels, height, width = inputs.shape
    x = inputs.to(device=device, dtype=torch.float32)
    enc = model.encoder(x.reshape(batch_size * frame_num, channels, height, width))
    seq = enc.reshape(batch_size, frame_num, -1)

    shuffled_schedule = None
    if condition in ("shuffle_digit", "shuffle_sector"):
        base_schedule = _run_baseline_feedback_schedule(
            model, seq, num_pos=num_pos, device=device
        )
        shuffled_schedule = _shuffled_schedule_slice(
            base_schedule, condition, rng, num_pos
        )

    hidden_size = int(model.rnn.hidden_size)
    feedback_dim = int(model.feedback_dim)
    h = torch.zeros(batch_size, hidden_size, device=device, dtype=seq.dtype)
    fb = torch.zeros(batch_size, feedback_dim, device=device, dtype=seq.dtype)
    pred_char = torch.empty(batch_size, frame_num, device=device, dtype=torch.int64)
    pred_sector = torch.empty(batch_size, frame_num, device=device, dtype=torch.int64)

    with torch.no_grad():
        for t in range(frame_num):
            x_t = seq[:, t, :]
            gated = model.middle_gawf(x_t, h, fb.clamp(-10, 10).unsqueeze(2))
            char_t, pos_t = model.classifier(gated)
            pred_char[:, t] = char_t.argmax(dim=1)
            pred_sector[:, t] = pos_t.argmax(dim=1)

            fb_next = model._compute_feedback(char_t, pos_t)
            fb_next = _apply_clear_feedback(fb_next, condition, num_pos)
            if shuffled_schedule is not None:
                if condition == "shuffle_digit":
                    fb_next = fb_next.clone()
                    fb_next[:, DIGIT_SLICE] = shuffled_schedule[:, t, DIGIT_SLICE]
                else:
                    fb_next = fb_next.clone()
                    fb_next[:, 10 : 10 + num_pos] = shuffled_schedule[
                        :, t, 10 : 10 + num_pos
                    ]
            fb = fb_next
            h = gated

    return (
        pred_char.detach().cpu().numpy().astype(np.int64, copy=False),
        pred_sector.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def _global_frame_grid(
    *,
    seq_base: int,
    batch_size: int,
    frame_num: int,
    chan_num: int,
) -> np.ndarray:
    starts = (seq_base + np.arange(batch_size, dtype=np.int64)) * frame_num + chan_num
    return starts[:, None] + np.arange(frame_num, dtype=np.int64)[None, :]


def _build_switch_offset_targets(
    switch_arr: np.ndarray,
    *,
    post_k: int,
    pre_k: int,
) -> np.ndarray:
    switch_arr = np.asarray(switch_arr).astype(np.int32, copy=False)
    n_frames = int(switch_arr.shape[0])
    post_offsets = np.zeros(n_frames, dtype=np.int16)
    pre_offsets = np.zeros(n_frames, dtype=np.int16)
    switches = np.flatnonzero(switch_arr != 0)

    forbidden = np.zeros(n_frames, dtype=bool)
    close_threshold = int(pre_k) + int(post_k)
    if close_threshold > 0:
        for i in range(1, len(switches)):
            s_prev, s_curr = int(switches[i - 1]), int(switches[i])
            if s_curr - s_prev < close_threshold:
                forbidden[s_prev + 1 : s_curr] = True

    for s in switches:
        for dt in range(0, post_k):
            t = int(s + dt)
            if 0 <= t < n_frames:
                post_offsets[t] = np.int16(dt + 1)
        for dt in range(1, pre_k + 1):
            t = int(s - dt)
            if 0 <= t < n_frames:
                pre_offsets[t] = np.int16(-dt)

    post_offsets[forbidden] = 0
    pre_offsets[forbidden] = 0
    pre_offsets[post_offsets != 0] = 0

    offsets = np.zeros(n_frames, dtype=np.int16)
    offsets[pre_offsets != 0] = pre_offsets[pre_offsets != 0]
    offsets[post_offsets != 0] = post_offsets[post_offsets != 0]
    return offsets


def _debug_print_switch_map(
    switch_arr: np.ndarray,
    offset_targets: np.ndarray,
    *,
    frame_num: int,
    chan_num: int,
    limit: int,
    prefix: str,
) -> None:
    if limit <= 0:
        return
    print(f"[debug_switch_map] {prefix}: raw_frame -> sequence/output_t/offset")
    printed = 0
    for raw_frame in np.flatnonzero(np.asarray(switch_arr) != 0):
        seq_pos = int(raw_frame) - int(chan_num)
        if seq_pos < 0:
            sample_idx = -1
            output_t = -1
        else:
            sample_idx = seq_pos // int(frame_num)
            output_t = seq_pos % int(frame_num)
        assigned = int(offset_targets[int(raw_frame)])
        print(
            "[debug_switch_map] "
            f"raw_frame={int(raw_frame)} sample={sample_idx} "
            f"output_t={output_t} assigned_offset={assigned}"
        )
        printed += 1
        if printed >= limit:
            break


def _empty_condition_state(offset_values: np.ndarray) -> Dict[str, Any]:
    return {
        "char_correct": 0,
        "sector_correct": 0,
        "n_frames": 0,
        "offset_values": offset_values.astype(np.int16, copy=True),
        "offset_char_correct": np.zeros(offset_values.shape[0], dtype=np.int64),
        "offset_sector_correct": np.zeros(offset_values.shape[0], dtype=np.int64),
        "offset_counts": np.zeros(offset_values.shape[0], dtype=np.int64),
        "pred_char": [],
        "pred_sector": [],
    }


def _update_state(
    state: Dict[str, Any],
    pred_char: np.ndarray,
    pred_sector: np.ndarray,
    labels_np: np.ndarray,
    global_frames: np.ndarray,
    offset_targets: np.ndarray,
) -> None:
    true_char = labels_np[:, :, 0].astype(np.int64, copy=False)
    true_sector = labels_np[:, :, 1].astype(np.int64, copy=False)
    char_ok = pred_char == true_char
    sector_ok = pred_sector == true_sector
    state["char_correct"] += int(char_ok.sum())
    state["sector_correct"] += int(sector_ok.sum())
    state["n_frames"] += int(true_char.size)
    state["pred_char"].append(pred_char.reshape(-1).astype(np.int16, copy=False))
    state["pred_sector"].append(pred_sector.reshape(-1).astype(np.int16, copy=False))

    offsets = offset_targets[global_frames]
    for i, off_value in enumerate(state["offset_values"]):
        off = int(off_value)
        mask = offsets == off
        n = int(mask.sum())
        if n == 0:
            continue
        state["offset_counts"][i] += n
        state["offset_char_correct"][i] += int((char_ok & mask).sum())
        state["offset_sector_correct"][i] += int((sector_ok & mask).sum())


def _finalize_condition(state: Dict[str, Any]) -> Dict[str, Any]:
    n = int(state["n_frames"])
    counts = state["offset_counts"]
    char_acc = 100.0 * float(state["char_correct"]) / float(n) if n else 0.0
    sector_acc = 100.0 * float(state["sector_correct"]) / float(n) if n else 0.0
    offset_char = np.divide(
        state["offset_char_correct"] * 100.0,
        counts,
        out=np.zeros_like(counts, dtype=np.float64),
        where=counts > 0,
    )
    offset_sector = np.divide(
        state["offset_sector_correct"] * 100.0,
        counts,
        out=np.zeros_like(counts, dtype=np.float64),
        where=counts > 0,
    )
    offset_values = state["offset_values"].astype(np.int64)
    post_mask = offset_values > 0
    return {
        "char_acc": char_acc,
        "sector_acc": sector_acc,
        "n_frames": n,
        "switch_offsets": offset_values.astype(int).tolist(),
        "switch_char_acc": offset_char.astype(float).tolist(),
        "switch_sector_acc": offset_sector.astype(float).tolist(),
        "switch_counts": counts.astype(int).tolist(),
        "switch_post_offsets": offset_values[post_mask].astype(int).tolist(),
        "switch_post_char_acc": offset_char[post_mask].astype(float).tolist(),
        "switch_post_sector_acc": offset_sector[post_mask].astype(float).tolist(),
        "switch_post_counts": counts[post_mask].astype(int).tolist(),
    }


def _write_csv(path: str, metrics: Dict[str, Any], conditions: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "char_acc", "sector_acc", "n_frames"])
        for condition in conditions:
            row = metrics["conditions"][condition]
            writer.writerow(
                [
                    condition,
                    f"{float(row['char_acc']):.6f}",
                    f"{float(row['sector_acc']):.6f}",
                    int(row["n_frames"]),
                ]
            )


def _concat_state_arrays(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(state["pred_char"], axis=0).astype(np.int16, copy=False),
        np.concatenate(state["pred_sector"], axis=0).astype(np.int16, copy=False),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    conditions = list(args.conditions)
    if args.shuffle:
        conditions.extend(["shuffle_digit", "shuffle_sector"])
    conditions = _validate_conditions(conditions)
    if "baseline" not in conditions:
        conditions.insert(0, "baseline")

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    if num_pos != 9:
        raise RuntimeError(f"This ablation expects sector mode num_pos=9, got {num_pos}.")

    print(f"Loading model from: {args.ckpt}")
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    if not hasattr(model, "middle_gawf") or not hasattr(model, "_compute_feedback"):
        raise RuntimeError("feedback_ablation.py requires a GaWF checkpoint.")
    if getattr(model, "proj_out", None) is not None:
        raise RuntimeError("Feedback slice ablation expects legacy direct 19-d feedback.")
    if int(model.feedback_dim) != 10 + num_pos:
        raise RuntimeError(
            f"Expected feedback_dim={10 + num_pos}; got {int(model.feedback_dim)}."
        )

    frame_num = int(getattr(test_ds, "frame_num", 32))
    chan_num = int(getattr(test_ds, "chan_num", 2))
    fg_switch = getattr(test_ds, "fg_switch", None)
    if fg_switch is None:
        raise RuntimeError("Test dataset does not expose fg_switch.")
    offset_targets = _build_switch_offset_targets(
        fg_switch,
        post_k=args.K,
        pre_k=args.pre_K,
    )
    _debug_print_switch_map(
        fg_switch,
        offset_targets,
        frame_num=frame_num,
        chan_num=chan_num,
        limit=args.debug_switch_map,
        prefix="feedback_ablation",
    )
    offset_values = np.asarray(
        list(range(-args.pre_K, 0)) + list(range(1, args.K + 1)),
        dtype=np.int16,
    )

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.use_mmap else 4,
        pin_memory=False,
    )
    states = {
        condition: _empty_condition_state(offset_values) for condition in conditions
    }
    true_char_all = []
    true_sector_all = []
    global_frames_all = []

    model.eval()
    seq_base = 0
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            inputs = batch[0]
            labels = batch[1]
            labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            bs = int(labels_np.shape[0])
            global_frames = _global_frame_grid(
                seq_base=seq_base,
                batch_size=bs,
                frame_num=frame_num,
                chan_num=chan_num,
            )
            true_char_all.append(labels_np[:, :, 0].reshape(-1).astype(np.int16))
            true_sector_all.append(labels_np[:, :, 1].reshape(-1).astype(np.int16))
            global_frames_all.append(global_frames.reshape(-1).astype(np.int64))

            for condition in conditions:
                pred_char, pred_sector = _rollout_condition(
                    model,
                    inputs,
                    labels,
                    condition=condition,
                    num_pos=num_pos,
                    device=device,
                    rng=rng,
                )
                _update_state(
                    states[condition],
                    pred_char,
                    pred_sector,
                    labels_np,
                    global_frames,
                    offset_targets,
                )

            seq_base += bs
            if (bidx + 1) % 10 == 0 or bidx == 0:
                print(f"[eval] batches={bidx + 1} sequences={seq_base}")
            if args.max_batches > 0 and (bidx + 1) >= args.max_batches:
                break

    metrics: Dict[str, Any] = {
        "ckpt": os.path.abspath(args.ckpt),
        "conditions_order": conditions,
        "num_pos": int(num_pos),
        "feedback_dim": int(model.feedback_dim),
        "digit_slice": [0, 10],
        "sector_slice": [10, 10 + num_pos],
        "K": int(args.K),
        "pre_K": int(args.pre_K),
        "switch_offsets": offset_values.astype(int).tolist(),
        "clear_all_note": (
            "clear_all sets the recurrent feedback vector to zero at every step after "
            "readout, so GaWF gates are sigmoid(0)=0.5; it is not an RNN baseline."
        ),
        "conditions": {},
    }
    for condition in conditions:
        metrics["conditions"][condition] = _finalize_condition(states[condition])

    json_path = os.path.join(args.save_dir, "ablation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _write_csv(os.path.join(args.save_dir, "ablation_metrics.csv"), metrics, conditions)

    pred_npz: Dict[str, np.ndarray] = {
        "true_char": np.concatenate(true_char_all, axis=0).astype(np.int16),
        "true_sector": np.concatenate(true_sector_all, axis=0).astype(np.int16),
        "global_frame": np.concatenate(global_frames_all, axis=0).astype(np.int64),
    }
    for condition in conditions:
        pred_char, pred_sector = _concat_state_arrays(states[condition])
        pred_npz[f"{condition}_pred_char"] = pred_char
        pred_npz[f"{condition}_pred_sector"] = pred_sector
    np.savez_compressed(os.path.join(args.save_dir, "frame_predictions.npz"), **pred_npz)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
