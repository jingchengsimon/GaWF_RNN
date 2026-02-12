"""
Encapsulates training and evaluation logic for use_sector vs coordinate (single-char + position).
Avoids scattered if-else in the main training script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_weights(predict_all_chars, use_sector):
    """Default loss weights: [char_weight, pos_weight]."""
    if predict_all_chars:
        return [1, 0]
    if use_sector:
        return [1, 1]
    return [1, 0.001]


def get_criterion_pos(use_sector):
    """Position criterion: CrossEntropyLoss for sector, MSELoss for coordinate."""
    if use_sector:
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def loss_char_single(out_char, labels, criterion_char):
    """Character loss for single-char mode. labels[:,:,0] is char id."""
    labels_char = labels[:, :, 0].long().view(-1)
    outputs_char = out_char.reshape(-1, out_char.shape[-1])
    return criterion_char(outputs_char, labels_char)


def loss_pos_single(out_pos, labels, use_sector, criterion_pos):
    """Position loss: classification (sector) or regression (coordinate)."""
    if use_sector:
        labels_pos = labels[:, :, 1].long().view(-1)
        outputs_pos = out_pos.reshape(-1, out_pos.shape[-1])
        return criterion_pos(outputs_pos, labels_pos)
    labels_pos = labels[:, :, 1:].float()
    return criterion_pos(out_pos, labels_pos)


def batch_metric_char_single(out_char, labels):
    """Mean character correctness for one batch (single-char mode)."""
    return (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()


def batch_metric_pos_single(out_pos, labels, use_sector):
    """Position metric for one batch: accuracy (sector) or MSE (coordinate)."""
    if use_sector:
        return (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
    labels_pos = labels[:, :, 1:].float()
    return F.mse_loss(out_pos, labels_pos, reduction='mean').item()


def eval_accumulate_batch_single(out_char, out_pos, labels, use_sector):
    """
    Eval loop: accumulate total_acc_char and total_metric_pos for one batch (single-char).
    Returns: (acc_char_sum_delta, metric_pos_sum_delta).
    """
    acc_char = (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
    if use_sector:
        metric_pos = (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
    else:
        labels_pos = labels[:, :, 1:].float()
        metric_pos = F.mse_loss(out_pos, labels_pos, reduction='mean').item()
    return acc_char, metric_pos


def finalize_metrics_single(total_acc_char, total_metric_pos, num_batches, use_sector):
    """Convert accumulated sums to (acc_char %, metric_pos). metric_pos is % for sector, raw MSE for coordinate."""
    acc_char = total_acc_char * 100 / num_batches
    if use_sector:
        metric_pos = total_metric_pos * 100 / num_batches
    else:
        metric_pos = total_metric_pos / num_batches
    return acc_char, metric_pos


def finalize_eval_metrics_single(total_acc_char, total_metric_pos, num_batches, use_sector):
    """Eval: same as finalize_metrics_single."""
    return finalize_metrics_single(total_acc_char, total_metric_pos, num_batches, use_sector)


def format_train_str_single(epoch, num_epochs, acc_char, metric_pos, use_sector, gpu_info=""):
    if use_sector:
        return f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%){gpu_info}"
    return f"Epoch {epoch + 1}/{num_epochs} - Train (char, pos): ({acc_char:.2f}%, {metric_pos:.2f} pix^2){gpu_info}"


def format_val_str_single(acc_char, metric_pos, use_sector):
    if use_sector:
        return f" Validation (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%)"
    return f" Validation (char, pos): ({acc_char:.2f}%, {metric_pos:.2f} pix^2)"


def result_dict_keys_single(use_sector):
    """Return the key names for the result dict (train_acc_pos/val_acc_pos vs train_err_pos/val_err_pos)."""
    if use_sector:
        return "train_acc_pos", "val_acc_pos"
    return "train_err_pos", "val_err_pos"


def build_loss_fn_single(mdl, criterion_char, criterion_pos, use_sector, loss_weights, rnn_diag_lambda, device):
    """Build a single loss_fn(out_char, out_pos, labels) for single-char + sector/coordinate (includes RNN diag)."""
    def loss_fn(out_char, out_pos, labels):
        loss_char = loss_char_single(out_char, labels, criterion_char)
        loss_pos = loss_pos_single(out_pos, labels, use_sector, criterion_pos)
        if hasattr(mdl, 'rnn') and mdl.rnn is not None:
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
        return {"acc_char_sum": 0.0, "metric_pos_sum": 0.0}

    def update_train_batch(self, acc, out_char, labels, batch_idx, len_train_dl, out_pos=None):
        acc["acc_char_sum"] += batch_metric_char_single(out_char, labels)
        acc["metric_pos_sum"] += batch_metric_pos_single(out_pos, labels, self.use_sector)
        return acc

    def finalize_train_epoch(self, acc, num_batches):
        return finalize_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches, self.use_sector
        )

    def init_eval(self):
        return {"acc_char_sum": 0.0, "metric_pos_sum": 0.0}

    def update_eval_batch(self, acc, out_char, labels, out_pos=None):
        ac, mp = eval_accumulate_batch_single(out_char, out_pos, labels, self.use_sector)
        acc["acc_char_sum"] += ac
        acc["metric_pos_sum"] += mp
        return acc

    def finalize_eval(self, acc, num_batches):
        return finalize_eval_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches, self.use_sector
        )

    def format_train_str(self, epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
        return format_train_str_single(epoch, num_epochs, acc_char, metric_pos, self.use_sector, gpu_info)

    def format_val_str(self, acc_char, metric_pos):
        return format_val_str_single(acc_char, metric_pos, self.use_sector)

    def postfix_for_pbar(self, current_loss, out_char, out_pos, labels):
        postfix = {"loss": f"{current_loss:.4f}"}
        pos_val = batch_metric_pos_single(out_pos, labels, self.use_sector)
        if self.use_sector:
            postfix["pos_acc"] = f"{pos_val * 100:.2f}%"
        else:
            postfix["pos_mse"] = f"{pos_val:.2f}"
        return postfix

    def add_pos_to_result_dict(self, base, train_metric_pos, val_metric_pos, actual_epochs):
        k1, k2 = result_dict_keys_single(self.use_sector)
        base[k1] = train_metric_pos[:actual_epochs]
        base[k2] = val_metric_pos[:actual_epochs]
        return base
