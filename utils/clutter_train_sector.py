"""
Encapsulates training and evaluation logic for use_sector vs coordinate (single-char + position).
Avoids scattered if-else in the main training script.
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def loss_char_single(out_char, labels, criterion_char):
    """Character loss for single-char mode. labels[:,:,0] is char id."""
    labels_char = labels[:, :, 0].long().view(-1)
    outputs_char = out_char.reshape(-1, out_char.shape[-1])
    return criterion_char(outputs_char, labels_char)


def loss_pos_single(out_pos, labels, criterion_pos):
    """Position loss for sector classification."""
    labels_pos = labels[:, :, 1].long().view(-1)
    outputs_pos = out_pos.reshape(-1, out_pos.shape[-1])
    return criterion_pos(outputs_pos, labels_pos)


def batch_metric_char_single(out_char, labels):
    """Mean character correctness for one batch (single-char mode)."""
    return (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()


def batch_metric_pos_single(out_pos, labels):
    """Position metric for one batch: accuracy (sector)."""
    return (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()


def batch_loss_pos_sector(out_pos, labels):
    """Sector position cross-entropy loss for one batch (for logging/saving, like MSE in coord mode)."""
    labels_pos = labels[:, :, 1].long().view(-1)
    outputs_pos = out_pos.reshape(-1, out_pos.shape[-1])
    return F.cross_entropy(outputs_pos, labels_pos, reduction="mean").item()


def batch_loss_char_single(out_char, labels):
    """
    Character cross-entropy loss for one batch (single-char mode).
    Used only for logging/saving curves; always uses standard CE (same as criterion_char).
    """
    labels_char = labels[:, :, 0].long().view(-1)
    outputs_char = out_char.reshape(-1, out_char.shape[-1])
    return F.cross_entropy(outputs_char, labels_char, reduction="mean").item()


def compute_fg_transition_masks(fg_switch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-frame pre5 / post5 masks from a global ``fg_switch`` sequence (1 = fg switch frame).

    - post5: frames ``[s, s+4]`` for each switch ``s`` (5 frames including switch).
    - pre5: frames ``[s-5, s-1]`` for each switch ``s`` (5 frames before switch, excluding ``s``).
    - If consecutive switches ``s_prev, s_curr`` satisfy ``s_curr - s_prev < 10``, every frame ``t``
      with ``s_prev < t < s_curr`` is excluded from both pre5 and post5.
    - Where ``post5`` and ``pre5`` would overlap (different switches), **post5 wins** (frame is
      counted only in post5 for global metrics).
    """
    fg_switch = np.asarray(fg_switch).astype(np.int32)
    num_frames = int(fg_switch.shape[0])
    fg = fg_switch != 0
    switches = np.where(fg)[0].tolist()

    forbidden = np.zeros(num_frames, dtype=bool)
    for i in range(1, len(switches)):
        s_prev, s_curr = switches[i - 1], switches[i]
        if s_curr - s_prev < 10:
            for t in range(s_prev + 1, s_curr):
                if 0 <= t < num_frames:
                    forbidden[t] = True

    post5_nom = np.zeros(num_frames, dtype=bool)
    pre5_nom = np.zeros(num_frames, dtype=bool)
    for s in switches:
        for t in range(s, min(s + 5, num_frames)):
            post5_nom[t] = True
        for t in range(max(0, s - 5), s):
            pre5_nom[t] = True

    post5 = post5_nom & ~forbidden
    pre5 = pre5_nom & ~forbidden
    pre5 = pre5 & ~post5
    return pre5, post5


def single_char_global_eval_init() -> Dict[str, int]:
    """Counters for strict global accuracy (all frames + pre5 / post5 subsets)."""
    return {
        "char_correct": 0,
        "char_total": 0,
        "pos_correct": 0,
        "pos_total": 0,
        "pre5_char_correct": 0,
        "pre5_char_total": 0,
        "pre5_pos_correct": 0,
        "pre5_pos_total": 0,
        "post5_char_correct": 0,
        "post5_char_total": 0,
        "post5_pos_correct": 0,
        "post5_pos_total": 0,
    }


def single_char_global_eval_update(
    state: Dict[str, int],
    out_char: torch.Tensor,
    out_pos: torch.Tensor,
    labels: torch.Tensor,
    pre5_mask: torch.Tensor,
    post5_mask: torch.Tensor,
) -> Dict[str, int]:
    """Accumulate per-batch strict counts for global / pre5 / post5 accuracies."""
    pred_char = torch.argmax(out_char, dim=2) == labels[:, :, 0].long()
    pred_pos = torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()
    pre5 = pre5_mask.bool()
    post5 = post5_mask.bool()

    char_total = int(pred_char.numel())
    state["char_correct"] += int(pred_char.sum().item())
    state["char_total"] += char_total
    state["pos_correct"] += int(pred_pos.sum().item())
    state["pos_total"] += char_total

    state["pre5_char_correct"] += int((pred_char & pre5).sum().item())
    state["pre5_char_total"] += int(pre5.sum().item())
    state["pre5_pos_correct"] += int((pred_pos & pre5).sum().item())
    state["pre5_pos_total"] += int(pre5.sum().item())

    state["post5_char_correct"] += int((pred_char & post5).sum().item())
    state["post5_char_total"] += int(post5.sum().item())
    state["post5_pos_correct"] += int((pred_pos & post5).sum().item())
    state["post5_pos_total"] += int(post5.sum().item())
    return state


def _pct(correct: int, total: int) -> float:
    return 100.0 * float(correct) / float(total) if total > 0 else 0.0


def single_char_global_eval_finalize(state: Dict[str, int]) -> Dict[str, float]:
    """Return percentage accuracies for one eval pass."""
    return {
        "glob_acc_char": _pct(state["char_correct"], state["char_total"]),
        "glob_acc_pos": _pct(state["pos_correct"], state["pos_total"]),
        "fg_switch_pre5_acc_char": _pct(state["pre5_char_correct"], state["pre5_char_total"]),
        "fg_switch_pre5_acc_pos": _pct(state["pre5_pos_correct"], state["pre5_pos_total"]),
        "fg_switch_post5_acc_char": _pct(state["post5_char_correct"], state["post5_char_total"]),
        "fg_switch_post5_acc_pos": _pct(state["post5_pos_correct"], state["post5_pos_total"]),
    }


def eval_accumulate_batch_single(out_char, out_pos, labels):
    """
    Eval loop: accumulate total_acc_char and total_metric_pos for one batch (single-char).
    Returns: (acc_char_sum_delta, metric_pos_sum_delta).
    """
    acc_char = (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
    metric_pos = (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
    return acc_char, metric_pos


def finalize_metrics_single(total_acc_char, total_metric_pos, num_batches):
    """Convert accumulated sums to (acc_char %, metric_pos %)."""
    acc_char = total_acc_char * 100 / num_batches
    metric_pos = total_metric_pos * 100 / num_batches
    return acc_char, metric_pos


def finalize_eval_metrics_single(total_acc_char, total_metric_pos, num_batches):
    """Eval: same as finalize_metrics_single."""
    return finalize_metrics_single(total_acc_char, total_metric_pos, num_batches)


def format_train_str_single(epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
    return f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%){gpu_info}"


def format_val_str_single(acc_char, metric_pos):
    return f" Validation (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%)"


def result_dict_keys_single():
    """Return the key names for the result dict (sector accuracy keys)."""
    return "train_acc_pos", "val_acc_pos"


def build_loss_fn_single(mdl, criterion_char, criterion_pos, loss_weights, rnn_diag_lambda, device):
    """Build a single loss_fn(out_char, out_pos, labels) for single-char + sector/coordinate (includes RNN diag)."""
    def loss_fn(out_char, out_pos, labels):
        loss_char = loss_char_single(out_char, labels, criterion_char)
        loss_pos = loss_pos_single(out_pos, labels, criterion_pos)
        if (
            hasattr(mdl, "rnn")
            and mdl.rnn is not None
            and hasattr(mdl.rnn, "weight_hh_l0")
        ):
            rnn_hh_diag = mdl.rnn.weight_hh_l0.diagonal().abs().mean()
        else:
            rnn_hh_diag = torch.tensor(0.0, device=device)
        return (loss_weights[0] * loss_char + loss_weights[1] * loss_pos +
                rnn_diag_lambda * rnn_hh_diag)
    return loss_fn


class SingleCharMetricsMode:
    """
    Encapsulates single-char + sector/coordinate metrics state and logic.
    """

    def __init__(self, use_sector):
        self.use_sector = use_sector

    def init_epoch_train(self):
        d = {
            "acc_char_sum": 0.0,
            "metric_pos_sum": 0.0,
            "loss_char_sum": 0.0,
        }
        d["loss_pos_sum"] = 0.0
        return d

    def update_train_batch(self, acc, out_char, labels, batch_idx, len_train_dl, out_pos=None):
        acc["acc_char_sum"] += batch_metric_char_single(out_char, labels)
        acc["metric_pos_sum"] += batch_metric_pos_single(out_pos, labels)
        acc["loss_char_sum"] += batch_loss_char_single(out_char, labels)
        acc["loss_pos_sum"] += batch_loss_pos_sector(out_pos, labels)
        return acc

    def finalize_train_epoch(self, acc, num_batches):
        acc_char, metric_pos = finalize_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches
        )
        loss_pos = (acc.get("loss_pos_sum", 0.0) / num_batches) if num_batches else None
        loss_char = (acc["loss_char_sum"] / num_batches) if num_batches else None
        return acc_char, metric_pos, loss_pos, loss_char

    def init_eval(self):
        d = {
            "acc_char_sum": 0.0,
            "metric_pos_sum": 0.0,
            "loss_char_sum": 0.0,
        }
        d["loss_pos_sum"] = 0.0
        return d

    def update_eval_batch(self, acc, out_char, labels, out_pos=None):
        ac, mp = eval_accumulate_batch_single(out_char, out_pos, labels)
        acc["acc_char_sum"] += ac
        acc["metric_pos_sum"] += mp
        acc["loss_pos_sum"] += batch_loss_pos_sector(out_pos, labels)
        acc["loss_char_sum"] += batch_loss_char_single(out_char, labels)
        return acc

    def finalize_eval(self, acc, num_batches):
        acc_char, metric_pos = finalize_eval_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches
        )
        loss_pos = (acc.get("loss_pos_sum", 0.0) / num_batches) if num_batches else None
        loss_char = (acc["loss_char_sum"] / num_batches) if num_batches else None
        return acc_char, metric_pos, loss_pos, loss_char

    def format_train_str(self, epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
        return format_train_str_single(epoch, num_epochs, acc_char, metric_pos, gpu_info)

    def format_val_str(self, acc_char, metric_pos):
        return format_val_str_single(acc_char, metric_pos)

    def postfix_for_pbar(self, current_loss, out_char, out_pos, labels):
        postfix = {"loss": f"{current_loss:.4f}"}
        pos_val = batch_metric_pos_single(out_pos, labels)
        postfix["pos_acc"] = f"{pos_val * 100:.2f}%"
        return postfix

    def add_pos_to_result_dict(self, base, train_metric_pos, val_metric_pos, actual_epochs,
                                train_loss_pos=None, val_loss_pos=None,
                                train_loss_char=None, val_loss_char=None):
        k1, k2 = result_dict_keys_single()
        base[k1] = train_metric_pos[:actual_epochs]
        base[k2] = val_metric_pos[:actual_epochs]
        if train_loss_pos is not None and val_loss_pos is not None:
            base["train_loss_pos"] = train_loss_pos[:actual_epochs]
            base["val_loss_pos"] = val_loss_pos[:actual_epochs]
        if train_loss_char is not None and val_loss_char is not None:
            base["train_loss_char"] = train_loss_char[:actual_epochs]
            base["val_loss_char"] = val_loss_char[:actual_epochs]
        return base
