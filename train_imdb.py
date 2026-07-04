#!/usr/bin/env python3
"""Train text sequence-classification models on the IMDB sentiment benchmark.

A self-contained entry point parallel to ``train_model.py`` (vision). It reuses the
lightweight helpers (``set_seed``) and the text subsystem (``utils.imdb_data``,
``utils.text_models``) but keeps its own compact train loop because the vision
engine is fused to the dual (char, pos) head structure.

Sweeps the cartesian product of ``--model_types x --hidden_sizes x --lrs x --wds``
(dropout fixed, not swept) and writes one metrics JSON + model + pkl per config,
mirroring the vision result layout under ``results/train_data/<result_suffix>/``.

Example (local smoke)::

    python train_imdb.py --model_types lstm gawf --hidden_sizes 128 \
        --lrs 1e-3 --wds 0.0 --num_epochs 2 --max_train_samples 512 \
        --result_suffix imdb_smoke
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

from utils.imdb_data import build_imdb_loaders, load_meta
from utils.text_models import get_text_model_classes
from utils.text_train_utils import (
    build_optimizer,
    count_core_params,
    maybe_subset,
    select_device,
)
from utils.train_helpers import set_seed

DISABLE_TQDM = os.environ.get("DISABLE_TQDM", "0").lower() in ("1", "true", "yes")


@torch.no_grad()
def evaluate(model, loader, criterion, device, autocast_fn) -> Dict[str, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for ids, lengths, labels in loader:
        labels = labels.to(device).long()
        with autocast_fn():
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
        loss_sum += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += labels.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_one_config(
    args, cfg: Dict, loaders: Dict[str, DataLoader], device: str, meta: Dict
) -> Dict:
    set_seed(args.seed)
    model_classes = get_text_model_classes()
    ModelClass = model_classes[cfg["model"]]
    model_kwargs = {}
    if cfg["model"] == "gawf_multi":
        model_kwargs["num_layers"] = args.gawf_layers
        model_kwargs["feedback_dim"] = args.feedback_dim
    model = ModelClass(
        vocab_size=meta["vocab_size"],
        embed_dim=args.embed_dim,
        hidden_size=cfg["hidden"],
        num_classes=2,
        embed_dropout=args.embed_dropout,
        rnn_dropout=args.rnn_dropout,
        pooling=args.pooling,
        device=device,
        **model_kwargs,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        cfg["lr"],
        cfg["wd"],
        args.optim,
        gawf_feedback_lr_scale=args.gawf_multi_feedback_lr_scale,
    )

    use_amp = device == "cuda" and args.use_acceleration
    autocast_fn = (
        (lambda: torch.amp.autocast(device_type="cuda")) if use_amp else (lambda: nullcontext())
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    best_train_acc = None
    epochs_without_improve = 0
    actual_epochs = 0

    for epoch in range(args.num_epochs):
        actual_epochs = epoch + 1
        model.train()
        run_total, run_correct, run_loss = 0, 0, 0.0
        for ids, lengths, labels in train_loader:
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

        train_metrics = {
            "loss": run_loss / max(run_total, 1),
            "acc": run_correct / max(run_total, 1),
        }
        val_metrics = evaluate(model, val_loader, criterion, device, autocast_fn)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        if not DISABLE_TQDM:
            print(
                f"[{cfg['model']} h{cfg['hidden']} lr{cfg['lr']} wd{cfg['wd']}] "
                f"epoch {actual_epochs}/{args.num_epochs} "
                f"train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f}",
                flush=True,
            )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = actual_epochs
            best_train_acc = train_metrics["acc"]
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
        "dataset": "imdb",
        "vocab_size": meta["vocab_size"],
        "max_len": meta["max_len"],
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
        "seed": args.seed,
        "batch_size": args.batch_size,
        "core_param_count": count_core_params(model),
        "total_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "best_val_acc": best_val_acc,
        "val_acc_at_best": best_val_acc,
        "best_epoch_val_acc_1based": best_epoch,
        "train_acc_at_best_val": best_train_acc,
        "overfit_gap": (best_train_acc - best_val_acc) if best_train_acc is not None else None,
        "test_acc_at_best": test_metrics["acc"],
        "test_loss_at_best": test_metrics["loss"],
        "actual_epochs": actual_epochs,
        "early_stop_epoch_1based": actual_epochs,
        "stopped_by_patience": stopped_by_patience,
    }
    if cfg["model"] == "gawf_multi":
        metrics.update(
            {
                "gawf_layers": int(model.num_layers),
                "feedback_dim": int(model.feedback_dim),
                "use_feedback_projector": bool(model.use_feedback_projector),
                "gawf_multi_feedback_lr_scale": args.gawf_multi_feedback_lr_scale,
                "layer_feedback_dims": [int(x) for x in model.layer_feedback_dims],
            }
        )
    return {"metrics": metrics, "model": model}


def result_stem(
    cfg: Dict,
    embed_dim: int,
    edo: float,
    rdo: float,
    gawf_layers: int = 2,
    feedback_dim: int | None = None,
) -> str:
    stem = (
        f"{cfg['model']}_imdb_h{cfg['hidden']}_emb{embed_dim}"
        f"_lr{cfg['lr']}_wd{cfg['wd']}_edo{edo}_rdo{rdo}"
    )
    if cfg["model"] == "gawf_multi":
        stem = f"{stem}_L{gawf_layers}"
        if feedback_dim is not None and feedback_dim > 0:
            stem = f"{stem}_dz{feedback_dim}"
    return stem


def save_outputs(args, cfg: Dict, out: Dict) -> str:
    result_dir = os.path.join(args.result_dir, "results", "train_data", args.result_suffix)
    os.makedirs(result_dir, exist_ok=True)
    stem = result_stem(
        cfg,
        args.embed_dim,
        args.embed_dropout,
        args.rnn_dropout,
        gawf_layers=args.gawf_layers,
        feedback_dim=args.feedback_dim,
    )
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
        choices=sorted(get_text_model_classes()),
    )
    p.add_argument("--data_dir", default=None, help="Base data dir (CLI -> env -> <repo>/stimuli).")
    p.add_argument("--result_dir", default=".", help="Repo root under which results/ is written.")
    p.add_argument(
        "--result_suffix", default="imdb_hparam", help="Subdir under results/train_data/."
    )
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[256])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    p.add_argument("--wds", type=float, nargs="+", default=[0.0])
    p.add_argument(
        "--embed_dropout", type=float, default=0.0, help="Fixed (vision config); not swept."
    )
    p.add_argument(
        "--rnn_dropout", type=float, default=0.5, help="Fixed (vision config); not swept."
    )
    p.add_argument("--pooling", default="last", choices=["last", "mean"])
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--gawf_layers",
        type=int,
        default=2,
        help="gawf_multi only: number of recurrent GaWF layers.",
    )
    p.add_argument(
        "--feedback_dim",
        "--dz",
        dest="feedback_dim",
        type=int,
        default=0,
        help=(
            "gawf_multi only: 0 uses direct feedback; >0 enables per-layer "
            "projected feedback with this dimension."
        ),
    )
    p.add_argument(
        "--gawf_multi_feedback_lr_scale",
        type=float,
        default=0.1,
        help="gawf_multi only: learning-rate scale for U/V and feedback projectors.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--optim", default="adam", choices=["adam", "adamw"])
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--use_acceleration", action="store_true", help="Enable CUDA AMP.")
    p.add_argument("--max_train_samples", type=int, default=0, help="Subset train (smoke tests).")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Subset val/test (smoke tests).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.gawf_layers < 2:
        raise SystemExit("--gawf_layers must be >= 2")
    if args.feedback_dim < 0:
        raise SystemExit("--feedback_dim/--dz must be >= 0")
    if args.gawf_multi_feedback_lr_scale <= 0:
        raise SystemExit("--gawf_multi_feedback_lr_scale must be > 0")
    device = select_device(args.device)
    set_seed(args.seed)
    meta = load_meta(args.data_dir)
    pin_memory = device == "cuda"
    loaders = build_imdb_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
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
        f"[train_imdb] device={device} vocab={meta['vocab_size']} max_len={meta['max_len']} "
        f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
        f"test={len(loaders['test'].dataset)}",
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
            f"[train_imdb] done {cfg} core_params={m['core_param_count']} "
            f"best_val_acc={m['best_val_acc']:.4f} test_acc={m['test_acc_at_best']:.4f} "
            f"({time.time() - t0:.1f}s) -> {metrics_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
