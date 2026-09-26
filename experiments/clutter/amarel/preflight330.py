"""GPU smoke for the nine Clutter model types in the 330-unit output-dropout campaign.

Input: CUDA device and the fixed model definitions. Output: JSON with one finite optimizer
step per model; this is a protocol gate, not a formal training result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.training.clutter.clutter_task_models import (
    GaWFRNNConv, GRUAdditiveFeedbackConv, GRUConv, LSTMAdditiveFeedbackConv, LSTMConv,
    MambaConv, RNNAdditiveFeedbackConv, RNNConv, S5Conv,
)

_ = np.__version__


def run_smoke(output: Path) -> None:
    """Train one synthetic two-frame batch per model and save finite-loss evidence."""

    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires CUDA on an Amarel compute node")
    torch.manual_seed(330)
    models = (
        ("rnn", RNNConv, {"hidden_size": 275}),
        ("lstm", LSTMConv, {"hidden_size": 80}),
        ("gru", GRUConv, {"hidden_size": 105}),
        ("gawf", GaWFRNNConv, {"hidden_size": 256}),
        ("mamba", MambaConv, {"mamba_d_model": 170}),
        ("s5", S5Conv, {"s5_d_model": 256, "s5_state_size": 128}),
        ("rnn_fb_add", RNNAdditiveFeedbackConv, {"hidden_size": 271}),
        ("gru_fb_add", GRUAdditiveFeedbackConv, {"hidden_size": 103}),
        ("lstm_fb_add", LSTMAdditiveFeedbackConv, {"hidden_size": 79}),
    )
    report: dict[str, object] = {"status": "running", "models": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, model_class, width_args in models:
        model = model_class(
            10, 9, kernel_size=5, device="cuda", cnn_dropout=0.0,
            rnn_dropout=0.5, **width_args,
        ).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        frames = torch.randn(2, 2, 2, 96, 96, device="cuda")
        char_target = torch.tensor([1, 2], device="cuda")
        pos_target = torch.tensor([3, 4], device="cuda")
        char_logits, pos_logits = model(frames)
        loss = F.cross_entropy(char_logits[:, -1], char_target)
        loss = loss + F.cross_entropy(pos_logits[:, -1], pos_target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss for {name}")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"Non-finite smoke gradient for {name}")
        optimizer.step()
        report["models"][name] = {"loss": float(loss.detach().cpu())}
        del model, optimizer, frames, char_logits, pos_logits, loss
        torch.cuda.empty_cache()
    report["status"] = "passed"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Run the nine-model smoke and write its structured report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(args.output)


if __name__ == "__main__":
    main()
