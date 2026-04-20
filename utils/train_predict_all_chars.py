"""
Encapsulates training and evaluation logic for predict_all_chars mode.
Avoids scattered if-else in the main training script.

BG-only multiset: loss and metrics use ``labels[:, :, 1:]`` (slots >= 0) only;
foreground slot 0 is ignored.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _bg_true_chars_1d(labels_1d: torch.Tensor) -> torch.Tensor:
    """Return valid background class ids for one frame (slot 0 = fg excluded)."""
    bg = labels_1d[1:]
    return bg[bg >= 0]


def _greedy_match_bg_indices(
    frame_probs: torch.Tensor,
    valid_true_chars: torch.Tensor,
    max_slots: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Greedy slot-to-class assignment (same rule as legacy: pick unused slot with
    highest prob for each GT class in order).

    Args:
        frame_probs: (max_slots, num_classes) softmax rows
        valid_true_chars: (K,) int64 class ids on same device as frame_probs
        max_slots: S (must match frame_probs.size(0))

    Returns:
        (idx_buf, tgt_buf) each (K,) long, or None if K == 0.
    """
    k = int(valid_true_chars.numel())
    if k == 0:
        return None
    device = frame_probs.device
    used_mask = torch.zeros(max_slots, dtype=torch.bool, device=device)
    idx_buf = torch.empty(k, dtype=torch.long, device=device)
    tgt_buf = torch.empty(k, dtype=torch.long, device=device)
    for i in range(k):
        cid = valid_true_chars[i].long()
        char_p = frame_probs[:, cid].masked_fill(used_mask, -1.0)
        best = torch.argmax(char_p)
        idx_buf[i] = best
        tgt_buf[i] = cid
        used_mask[best] = True
    return idx_buf, tgt_buf


def bg_multiset_frame_exact(
    frame_logits: torch.Tensor,
    labels_1d: torch.Tensor,
    device: torch.device,
) -> Optional[bool]:
    """
    One-frame exact multiset match for **background** digits only.

    Args:
        frame_logits: (max_chars, num_classes)
        labels_1d: (max_chars,) int; slot 0 = fg (ignored for GT)
        device: unused; kept for call-site compatibility.

    Returns:
        None if there is no bg digit in GT (frame skipped for bg metrics).
        True/False if at least one bg digit exists.
    """
    _ = device
    valid_true_chars = _bg_true_chars_1d(labels_1d)
    if valid_true_chars.numel() == 0:
        return None

    pred_probs = F.softmax(frame_logits, dim=-1)
    max_slots = int(frame_logits.shape[0])
    pair = _greedy_match_bg_indices(pred_probs, valid_true_chars, max_slots)
    if pair is None:
        return None
    idx_buf, tgt_buf = pair
    pred_digits = torch.argmax(frame_logits[idx_buf], dim=-1)
    gt_sorted = torch.sort(tgt_buf)[0]
    pr_sorted = torch.sort(pred_digits)[0]
    return bool(torch.equal(gt_sorted, pr_sorted))


def loss_char_all_chars(out_char, labels, criterion_char, max_chars, device):
    """
    Character loss for all-chars mode: greedy matching + CE per matched pair.
    out_char: (B, T, max_chars, num_classes), labels: (B, T, max_chars).
    Returns: loss_char (scalar tensor), loss_pos is always 0 in this mode.
    """
    _ = max_chars
    batch_size, frame_num, max_chars_pred, _ = out_char.shape
    pred_probs = F.softmax(out_char, dim=-1)
    total_loss_sum: torch.Tensor | None = None
    total_valid_chars = 0

    for b in range(batch_size):
        for t in range(frame_num):
            valid_true_chars = _bg_true_chars_1d(labels[b, t])
            if valid_true_chars.numel() == 0:
                continue

            frame_probs = pred_probs[b, t]
            frame_logits = out_char[b, t]
            pair = _greedy_match_bg_indices(
                frame_probs, valid_true_chars, max_chars_pred
            )
            if pair is None:
                continue
            idx_buf, tgt_buf = pair
            matched_logits = frame_logits[idx_buf]
            batch_loss = criterion_char(matched_logits, tgt_buf)
            total_loss_sum = batch_loss if total_loss_sum is None else total_loss_sum + batch_loss
            total_valid_chars += int(idx_buf.numel())

    if total_valid_chars == 0:
        loss_char = torch.tensor(0.0, device=device)
    else:
        loss_char = total_loss_sum / float(total_valid_chars)  # type: ignore[operator]
    loss_pos = torch.tensor(0.0, device=device)
    return loss_char, loss_pos


def batch_metrics_all_chars(out_char, labels, max_chars, device):
    """
    Training batch metrics: exact frame match (scheme B). Call only when batch_idx % 50 == 0 or last batch.
    Returns: (batch_exact, batch_frames_eval) to add to epoch_train_acc_char and epoch_train_frames_eval.
    """
    batch_size, frame_num = labels.shape[:2]
    batch_exact = 0
    batch_frames_eval = 0
    pred_probs = F.softmax(out_char, dim=-1)

    for b in range(batch_size):
        for t in range(frame_num):
            ex = bg_multiset_frame_exact(out_char[b, t], labels[b, t], device)
            if ex is None:
                continue
            batch_frames_eval += 1
            if ex:
                batch_exact += 1

    return batch_exact, batch_frames_eval


def eval_accumulate_batch_all_chars(out_char, labels, device):
    """
    Eval loop: accumulate (total_frames_exact, total_frames_eval) for one batch.
    Returns: (batch_exact, batch_frames_eval).
    """
    batch_size, frame_num = labels.shape[:2]
    pred_probs = F.softmax(out_char, dim=-1)
    batch_exact = 0
    batch_frames_eval = 0

    for b in range(batch_size):
        for t in range(frame_num):
            ex = bg_multiset_frame_exact(out_char[b, t], labels[b, t], device)
            if ex is None:
                continue
            batch_frames_eval += 1
            if ex:
                batch_exact += 1

    return batch_exact, batch_frames_eval


def finalize_acc_all_chars(total_exact, total_eval):
    """Convert accumulated (exact, eval) to accuracy percentage. metric_pos is 0."""
    acc_char = (total_exact / total_eval) * 100 if total_eval > 0 else 0.0
    metric_pos = 0.0
    return acc_char, metric_pos


def loss_weights_all_chars():
    return [1, 0]


def build_loss_fn_all_chars(mdl, criterion_char, max_chars, device, loss_weights, rnn_diag_lambda):
    """Build a single loss_fn(out_char, out_pos, labels) for all-chars mode (includes RNN diag and weights)."""
    def loss_fn(out_char, out_pos, labels):
        loss_char, loss_pos = loss_char_all_chars(out_char, labels, criterion_char, max_chars, device)
        if hasattr(mdl, 'rnn') and mdl.rnn is not None:
            rnn_hh_diag = mdl.rnn.weight_hh_l0.diagonal().abs().mean()
        else:
            rnn_hh_diag = torch.tensor(0.0, device=device)
        return (loss_weights[0] * loss_char + loss_weights[1] * loss_pos +
                rnn_diag_lambda * rnn_hh_diag)
    return loss_fn


class AllCharsMetricsMode:
    """
    Encapsulates all predict_all_chars metrics state and logic.
    Single place for total_frames_exact / total_frames_eval and train/eval accumulation.
    """

    def __init__(self, max_chars, device):
        self.max_chars = max_chars
        self.device = device

    def init_epoch_train(self):
        return {"exact": 0, "eval": 0}

    def update_train_batch(self, acc, out_char, labels, batch_idx, len_train_dl, out_pos=None):
        if batch_idx % 50 != 0 and batch_idx != len_train_dl - 1:
            return acc
        batch_exact, batch_frames_eval = batch_metrics_all_chars(
            out_char, labels, self.max_chars, self.device
        )
        if batch_frames_eval > 0:
            acc["exact"] += batch_exact
            acc["eval"] += batch_frames_eval
        return acc

    def finalize_train_epoch(self, acc, num_batches=None):
        train_acc_char = (acc["exact"] / acc["eval"]) * 100 if acc["eval"] > 0 else 0.0
        train_metric_pos = 0.0
        return train_acc_char, train_metric_pos

    def init_eval(self):
        return {"exact": 0, "eval": 0}

    def update_eval_batch(self, acc, out_char, labels):
        be, bf = eval_accumulate_batch_all_chars(out_char, labels, self.device)
        acc["exact"] += be
        acc["eval"] += bf
        return acc

    def finalize_eval(self, acc, num_batches):
        return finalize_acc_all_chars(acc["exact"], acc["eval"])

    def format_train_str(self, epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
        return f"Epoch {epoch + 1}/{num_epochs} - Train (all chars acc): {acc_char:.2f}%{gpu_info}"

    def format_val_str(self, acc_char, metric_pos):
        return f" Validation (all chars acc): {acc_char:.2f}%"

    def postfix_for_pbar(self, current_loss, out_char, out_pos, labels):
        return {"loss": f"{current_loss:.4f}"}

    def add_pos_to_result_dict(self, base, train_metric_pos, val_metric_pos, actual_epochs,
                                train_loss_pos=None, val_loss_pos=None,
                                train_loss_char=None, val_loss_char=None):
        """All-chars has no pos/char-loss keys; return base unchanged."""
        return base
