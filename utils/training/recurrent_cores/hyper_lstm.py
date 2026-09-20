"""Adapter from the vendored labml HyperLSTM to the shared batch-first interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.labml_nn.hyper_lstm import HyperLSTM


class HyperLSTMCore(nn.Module):
    """Single-layer HyperLSTM with Clutter-compatible state and post-processing."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_hidden_size: int,
        hyper_embedding_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hyper_hidden_size <= 0:
            raise ValueError("hyper_hidden_size must be positive")
        if hyper_embedding_size <= 0:
            raise ValueError("hyper_embedding_size must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.hyper_hidden_size = int(hyper_hidden_size)
        self.hyper_embedding_size = int(hyper_embedding_size)
        self.dropout = float(dropout)
        self.hyper_lstm = HyperLSTM(
            self.input_size,
            self.hidden_size,
            self.hyper_hidden_size,
            self.hyper_embedding_size,
            n_layers=1,
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run ``(B, T, F)`` input while preserving labml's four-tensor state."""
        time_major = x.transpose(0, 1)
        output, next_state = self.hyper_lstm(time_major, state)
        output = output.transpose(0, 1)
        output = self.norm(output)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, next_state
