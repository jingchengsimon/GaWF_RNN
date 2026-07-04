#!/usr/bin/env python3
"""Train shared text recurrent models on the SentiHood benchmark.

SentiHood is prepared as flattened query-pair examples:

    sentence + <sep> + location-aspect query -> None / Positive / Negative

This entry point intentionally reuses ``utils.text_models`` from IMDB. The
embedding and recurrent modules are unchanged; the task difference enters through
offline query-pair tensors and ``num_classes=3``.

Paper-style LSTM-Final smoke/repro command after preprocessing::

    python train_sentihood.py --model_types lstm --hidden_sizes 50 \
        --embed_dim 50 --lrs 0.01 --wds 0.0 --batch_size 150 \
        --embed_dropout 0.001 --rnn_dropout 0.001 --pooling last \
        --num_epochs 20 --patience 5 --balance_train_labels \
        --result_suffix sentihood_lstm_final
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from contextlib import nullcontext
from itertools import product
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.sentihood_data import build_sentihood_loaders, load_meta
from utils.sentihood_metrics import compute_sentihood_metrics
from utils.text_models import get_text_model_classes
from utils.text_train_utils import (
    build_optimizer,
    count_core_params,
    maybe_subset,
    select_device,
)
from utils.train_helpers import set_seed

DISABLE_TQDM = os.environ.get("DISABLE_TQDM", "0").lower() in ("1", "true", "yes")
SENTIHOOD_MODEL_CHOICES = sorted(
    model_name for model_name in get_text_model_classes() if model_name != "gawf_multi"
)


def _autocast(device: str, use_acceleration: bool):
    if device == "cuda" and use_acceleration:
        return lambda: torch.amp.autocast(device_type="cuda")
    return lambda: nullcontext()


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device: str, autocast_fn) -> Dict[str, float]:
    model.eval()
    total, loss_sum = 0, 0.0
    all_logits = []
    all_labels = []
    all_aspects = []
    all_targets = []
    all_sentences = []
    for ids, lengths, labels, aspect_ids, target_ids, sentence_ids in loader:
        labels = labels.to(device).long()
        with autocast_fn():
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
        loss_sum += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_aspects.append(aspect_ids.detach().cpu())
        all_targets.append(target_ids.detach().cpu())
        all_sentences.append(sentence_ids.detach().cpu())

    if not all_labels:
        return {
            "loss": 0.0,
            "query_acc": 0.0,
            "aspect_f1": 0.0,
            "sentiment_acc": 0.0,
            "aspect_auc": float("nan"),
            "sentiment_auc": float("nan"),
        }
    metrics = compute_sentihood_metrics(
        labels=torch.cat(all_labels),
        logits=torch.cat(all_logits),
        aspect_ids=torch.cat(all_aspects),
        target_ids=torch.cat(all_targets),
        sentence_ids=torch.cat(all_sentences),
    )
    metrics["loss"] = loss_sum / max(total, 1)
    return metrics


def _score_for_selection(metrics: Dict[str, float], key: str) -> float:
    value = metrics.get(key)
    if value is None or value != value:
        return -1.0
    return float(value)


def train_one_config(
    args, cfg: Dict, loaders: Dict[str, DataLoader], device: str, meta: Dict
) -> Dict:
    set_seed(args.seed)
    model_classes = get_text_model_classes()
    ModelClass = model_classes[cfg["model"]]
    model = ModelClass(
        vocab_size=meta["vocab_size"],
        embed_dim=args.embed_dim,
        hidden_size=cfg["hidden"],
        num_classes=len(meta["labels"]),
        embed_dropout=args.embed_dropout,
        rnn_dropout=args.rnn_dropout,
        pooling=args.pooling,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["lr"], cfg["wd"], args.optim)
    autocast_fn = _autocast(device, args.use_acceleration)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" and args.use_acceleration else None

    history = {
        "train_loss": [],
        "train_query_acc": [],
        "val_loss": [],
        "val_query_acc": [],
        "val_aspect_f1": [],
        "val_sentiment_acc": [],
        "val_aspect_auc": [],
        "val_sentiment_auc": [],
    }
    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_val_metrics: Dict[str, float] = {}
    epochs_without_improve = 0
    actual_epochs = 0

    for epoch in range(args.num_epochs):
        actual_epochs = epoch + 1
        model.train()
        run_total, run_correct, run_loss = 0, 0, 0.0
        for ids, lengths, labels, _aspect_ids, _target_ids, _sentence_ids in loaders["train"]:
            labels = labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast_fn():
                logits = model(ids, lengths)
                loss = criterion(logits, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            run_loss += float(loss.item()) * labels.size(0)
            run_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            run_total += labels.size(0)

        train_loss = run_loss / max(run_total, 1)
        train_query_acc = run_correct / max(run_total, 1)
        val_metrics = evaluate(model, loaders["val"], criterion, device, autocast_fn)
        history["train_loss"].append(train_loss)
        history["train_query_acc"].append(train_query_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_query_acc"].append(val_metrics["query_acc"])
        history["val_aspect_f1"].append(val_metrics["aspect_f1"])
        history["val_sentiment_acc"].append(val_metrics["sentiment_acc"])
        history["val_aspect_auc"].append(val_metrics["aspect_auc"])
        history["val_sentiment_auc"].append(val_metrics["sentiment_auc"])

        if not DISABLE_TQDM:
            print(
                f"[{cfg['model']} h{cfg['hidden']} lr{cfg['lr']} wd{cfg['wd']}] "
                f"epoch {actual_epochs}/{args.num_epochs} "
                f"train_qacc={train_query_acc:.4f} "
                f"val_aspect_f1={val_metrics['aspect_f1']:.4f} "
                f"val_sent_acc={val_metrics['sentiment_acc']:.4f}",
                flush=True,
            )

        current_score = _score_for_selection(val_metrics, args.selection_metric)
        if current_score > best_score:
            best_score = current_score
            best_epoch = actual_epochs
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                break

    stopped_by_patience = epochs_without_improve >= args.patience
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device, autocast_fn)

    metrics = {
        "model_type": cfg["model"],
        "dataset": "sentihood",
        "vocab_size": meta["vocab_size"],
        "max_len": meta["max_len"],
        "labels": meta["labels"],
        "aspects": meta["aspects"],
        "targets": meta["targets"],
        "n_query": meta["n_query"],
        "embed_dim": args.embed_dim,
        "hidden_size": cfg["hidden"],
        "lr": cfg["lr"],
        "weight_decay": cfg["wd"],
        "embed_dropout": args.embed_dropout,
        "rnn_dropout": args.rnn_dropout,
        "pooling": args.pooling,
        "optimizer": args.optim,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "balance_train_labels": args.balance_train_labels,
        "core_param_count": count_core_params(model),
        "total_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "actual_epochs": actual_epochs,
        "best_epoch_1based": best_epoch,
        "best_val_score": best_score,
        "stopped_by_patience": stopped_by_patience,
        **history,
        "best_val_metrics": best_val_metrics,
        "test_loss_at_best": test_metrics["loss"],
        "test_query_acc_at_best": test_metrics["query_acc"],
        "test_aspect_f1_at_best": test_metrics["aspect_f1"],
        "test_sentiment_acc_at_best": test_metrics["sentiment_acc"],
        "test_aspect_auc_at_best": test_metrics["aspect_auc"],
        "test_sentiment_auc_at_best": test_metrics["sentiment_auc"],
    }
    return {"metrics": metrics, "model": model}


def result_stem(cfg: Dict, embed_dim: int, edo: float, rdo: float) -> str:
    return (
        f"{cfg['model']}_sentihood_h{cfg['hidden']}_emb{embed_dim}"
        f"_lr{cfg['lr']}_wd{cfg['wd']}_edo{edo}_rdo{rdo}"
    )


def save_outputs(args, cfg: Dict, out: Dict) -> str:
    result_dir = os.path.join(args.result_dir, "results", "train_data", args.result_suffix)
    os.makedirs(result_dir, exist_ok=True)
    stem = result_stem(cfg, args.embed_dim, args.embed_dropout, args.rnn_dropout)
    metrics_path = os.path.join(result_dir, f"{stem}_metrics.json")
    pkl_path = os.path.join(result_dir, f"{stem}.pkl")
    model_path = os.path.join(result_dir, f"{stem}_model.pth")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out["metrics"], f, indent=2)
    torch.save(out["model"].state_dict(), model_path)
    with open(pkl_path, "wb") as f:
        pickle.dump({k: v for k, v in out["metrics"].items()}, f)
    return metrics_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model_types",
        nargs="+",
        default=["lstm"],
        choices=SENTIHOOD_MODEL_CHOICES,
    )
    p.add_argument("--data_dir", default=None, help="Base data dir (CLI -> env -> <repo>/stimuli).")
    p.add_argument("--result_dir", default=".", help="Repo root under which results/ is written.")
    p.add_argument("--result_suffix", default="sentihood_hparam")
    p.add_argument("--embed_dim", type=int, default=50)
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[50])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.01])
    p.add_argument("--wds", type=float, nargs="+", default=[0.0])
    p.add_argument("--embed_dropout", type=float, default=0.001)
    p.add_argument("--rnn_dropout", type=float, default=0.001)
    p.add_argument("--pooling", default="last", choices=["last", "mean"])
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--selection_metric",
        default="aspect_f1",
        choices=["aspect_f1", "sentiment_acc", "query_acc", "aspect_auc", "sentiment_auc"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=150)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--balance_train_labels",
        action="store_true",
        help="Use inverse-frequency sampling over None/Positive/Negative train labels.",
    )
    p.add_argument("--optim", default="adam", choices=["adam", "adamw"])
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--use_acceleration", action="store_true", help="Enable CUDA AMP.")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    meta = load_meta(args.data_dir)
    loaders = build_sentihood_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        seed=args.seed,
        balance_labels=args.balance_train_labels,
    )
    if args.max_train_samples:
        loaders["train"] = maybe_subset(
            loaders["train"], args.max_train_samples, args.batch_size, True
        )
    if args.max_eval_samples:
        for split in ("val", "test"):
            loaders[split] = maybe_subset(
                loaders[split], args.max_eval_samples, args.batch_size, False
            )
    print(
        f"[train_sentihood] device={device} vocab={meta['vocab_size']} "
        f"max_len={meta['max_len']} query={meta['n_query']} aspects={meta['aspects']} "
        f"balance_train_labels={args.balance_train_labels}",
        flush=True,
    )
    configs = [
        {"model": m, "hidden": h, "lr": lr, "wd": wd}
        for m, h, lr, wd in product(args.model_types, args.hidden_sizes, args.lrs, args.wds)
    ]
    for cfg in configs:
        t0 = time.time()
        out = train_one_config(args, cfg, loaders, device, meta)
        metrics_path = save_outputs(args, cfg, out)
        m = out["metrics"]
        print(
            f"[train_sentihood] done {cfg} core_params={m['core_param_count']} "
            f"best_val_{args.selection_metric}={m['best_val_score']:.4f} "
            f"test_aspect_f1={m['test_aspect_f1_at_best']:.4f} "
            f"test_sent_acc={m['test_sentiment_acc_at_best']:.4f} "
            f"test_aspect_auc={m['test_aspect_auc_at_best']:.4f} "
            f"test_sent_auc={m['test_sentiment_auc_at_best']:.4f} "
            f"({time.time() - t0:.1f}s) -> {metrics_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
