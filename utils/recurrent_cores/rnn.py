"""Task-agnostic wrappers around PyTorch RNN, GRU, and LSTM sequence layers."""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchRecurrentCore(nn.Module):
    """Batch-first recurrent core with LayerNorm, ReLU, and output dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_class: Type[nn.RNNBase],
        dropout: float = 0.0,
        num_layers: int = 1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor, state=None):
        """Run the recurrent core over an encoded sequence shaped ``(B, T, F)``."""
        out, next_state = self.rnn(x, state) if state is not None else self.rnn(x)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out, next_state


class RNNCore(TorchRecurrentCore):
    """Task-agnostic vanilla RNN core."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__(input_size, hidden_size, nn.RNN, dropout=dropout)


class GRUCore(TorchRecurrentCore):
    """Task-agnostic GRU core."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__(input_size, hidden_size, nn.GRU, dropout=dropout)


class LSTMCore(TorchRecurrentCore):
    """Task-agnostic LSTM core."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__(input_size, hidden_size, nn.LSTM, dropout=dropout)
