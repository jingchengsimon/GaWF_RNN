"""Run deterministic short optimization checks for Clutter feedback controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Type

import torch
import torch.nn.functional as F

from utils.training.clutter.clutter_task_models import (
    GaWFAdditiveConv,
    GRUFeedbackConv,
    LSTMFeedbackConv,
    RNNFeedbackConv,
)
from utils.training.clutter.clutter_train_acceleration import run_forward_with_feedback


MODEL_SPECS: tuple[tuple[str, Type[torch.nn.Module], int, float, float, int], ...] = (
    ("gawf_additive", GaWFAdditiveConv, 271, 0.005, 0.001, 585_401),
    ("rnn_fb", RNNFeedbackConv, 272, 0.001, 0.00001, 586_867),
    ("gru_fb", GRUFeedbackConv, 103, 0.005, 0.001, 584_562),
    ("lstm_fb", LSTMFeedbackConv, 79, 0.001, 0.001, 585_406),
)


def parse_args() -> argparse.Namespace:
    """Parse the bounded synthetic optimization check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def run_sanity(args: argparse.Namespace) -> list[dict[str, object]]:
    """Overfit one fixed mini-batch and require a finite decreasing loss per model."""
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite sanity output: {args.output}")
    if args.steps < 20:
        raise ValueError("--steps must be at least 20")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA sanity requested but unavailable")

    torch.manual_seed(args.seed)
    inputs = torch.randn(4, 4, 2, 96, 96, device=device)
    char_targets = torch.randint(0, 10, (4, 4), device=device)
    sector_targets = torch.randint(0, 9, (4, 4), device=device)
    reports: list[dict[str, object]] = []

    for name, model_class, width, lr, weight_decay, expected_count in MODEL_SPECS:
        torch.manual_seed(args.seed)
        model = model_class(
            10,
            9,
            kernel_size=5,
            hidden_size=width,
            cnn_dropout=0.0,
            rnn_dropout=0.5,
            device=str(device),
        )
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != expected_count:
            raise RuntimeError(f"{name} parameter count {count} != {expected_count}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        losses: list[float] = []
        model.train()
        for _step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            char_logits, sector_logits = run_forward_with_feedback(model, inputs)
            loss = F.cross_entropy(char_logits.flatten(0, 1), char_targets.flatten())
            loss = loss + F.cross_entropy(sector_logits.flatten(0, 1), sector_targets.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} produced non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        first_mean = sum(losses[:10]) / 10.0
        last_mean = sum(losses[-10:]) / 10.0
        if not math.isfinite(last_mean) or last_mean >= first_mean:
            raise RuntimeError(
                f"{name} loss did not decrease: first10={first_mean}, last10={last_mean}"
            )
        reports.append(
            {
                "model": name,
                "seed": args.seed,
                "steps": args.steps,
                "parameter_count": count,
                "first_10_loss_mean": first_mean,
                "last_10_loss_mean": last_mean,
                "minimum_loss": min(losses),
                "all_finite": True,
            }
        )
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": reports}, indent=2) + "\n", encoding="utf-8")
    return reports


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    for report in run_sanity(args):
        print(json.dumps(report, sort_keys=True))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
