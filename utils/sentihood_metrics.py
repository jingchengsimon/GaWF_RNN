"""Evaluation metrics for the SentiHood benchmark.

Implements the four metrics reported in the SentiHood paper for flattened query
examples: Aspect(F1), Sentiment(Accuracy), Aspect(AUC), and Sentiment(AUC).
Labels use 0=None, 1=Positive, 2=Negative.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def _rank_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Binary ROC AUC via average ranks; returns NaN when one class is absent."""
    n = len(y_true)
    pos = sum(1 for y in y_true if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda idx: y_score[idx])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _nanmean(values: Iterable[float]) -> float:
    valid = [v for v in values if v == v]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def aspect_macro_f1(
    labels: torch.Tensor,
    preds: torch.Tensor,
    aspect_ids: torch.Tensor,
    target_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
) -> float:
    """Macro-F1 over aspect sets per ``(sentence, target)`` group."""
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (sent_id, target_id) in enumerate(zip(sentence_ids.tolist(), target_ids.tolist())):
        groups[(int(sent_id), int(target_id))].append(idx)

    p_all = 0.0
    r_all = 0.0
    count = 0
    labels_l = labels.tolist()
    preds_l = preds.tolist()
    aspects_l = aspect_ids.tolist()
    for indices in groups.values():
        gold_aspects = {aspects_l[i] for i in indices if labels_l[i] != 0}
        if not gold_aspects:
            continue
        pred_aspects = {aspects_l[i] for i in indices if preds_l[i] != 0}
        overlap = gold_aspects.intersection(pred_aspects)
        precision = len(overlap) / len(pred_aspects) if pred_aspects else 0.0
        recall = len(overlap) / len(gold_aspects)
        p_all += precision
        r_all += recall
        count += 1
    if count == 0:
        return 0.0
    macro_p = p_all / count
    macro_r = r_all / count
    if macro_p + macro_r == 0:
        return 0.0
    return 2 * macro_p * macro_r / (macro_p + macro_r)


def sentiment_accuracy(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """Positive/Negative accuracy over gold non-None query examples."""
    mask = labels != 0
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    pos = probs[:, 1]
    neg = probs[:, 2]
    pred_negative = neg / (pos + neg).clamp(min=1e-12) > 0.5
    pred_sent = torch.where(pred_negative, torch.full_like(labels, 2), torch.ones_like(labels))
    return float((pred_sent[mask] == labels[mask]).float().mean().item())


def aspect_macro_auc(labels: torch.Tensor, probs: torch.Tensor, aspect_ids: torch.Tensor) -> float:
    """Macro AUC for aspect absence/presence, averaged over aspects."""
    aucs = []
    for aspect in sorted(set(aspect_ids.tolist())):
        mask = aspect_ids == aspect
        # Match common SentiHood eval: None is the positive class scored by p(None).
        y_true = (labels[mask] == 0).long().tolist()
        y_score = probs[mask, 0].tolist()
        aucs.append(_rank_auc(y_true, y_score))
    return _nanmean(aucs)


def sentiment_macro_auc(
    labels: torch.Tensor, probs: torch.Tensor, aspect_ids: torch.Tensor
) -> float:
    """Macro AUC for Positive/Negative polarity over gold non-None examples."""
    aucs = []
    mask_present = labels != 0
    pos = probs[:, 1]
    neg = probs[:, 2]
    neg_score = neg / (pos + neg).clamp(min=1e-12)
    for aspect in sorted(set(aspect_ids.tolist())):
        mask = (aspect_ids == aspect) & mask_present
        if int(mask.sum().item()) == 0:
            aucs.append(float("nan"))
            continue
        y_true = (labels[mask] == 2).long().tolist()
        y_score = neg_score[mask].tolist()
        aucs.append(_rank_auc(y_true, y_score))
    return _nanmean(aucs)


def compute_sentihood_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    aspect_ids: torch.Tensor,
    target_ids: torch.Tensor,
    sentence_ids: torch.Tensor,
) -> Dict[str, float]:
    """Return SentiHood paper metrics plus query-level accuracy."""
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return {
        "query_acc": float((preds == labels).float().mean().item()),
        "aspect_f1": aspect_macro_f1(labels, preds, aspect_ids, target_ids, sentence_ids),
        "sentiment_acc": sentiment_accuracy(labels, probs),
        "aspect_auc": aspect_macro_auc(labels, probs, aspect_ids),
        "sentiment_auc": sentiment_macro_auc(labels, probs, aspect_ids),
    }
