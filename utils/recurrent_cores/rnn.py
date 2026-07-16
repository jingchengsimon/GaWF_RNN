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
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.uses_tuple_state = rnn_class is nn.LSTM
        if self.num_layers == 1:
            # Keep the legacy module names exactly so existing checkpoints load.
            self.rnn = rnn_class(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=self.batch_first,
            )
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            layer_input_sizes = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            self.rnns = nn.ModuleList(
                [
                    rnn_class(
                        input_size=layer_input_size,
                        hidden_size=self.hidden_size,
                        num_layers=1,
                        batch_first=self.batch_first,
                    )
                    for layer_input_size in layer_input_sizes
                ]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
            )

    def forward(self, x: torch.Tensor, state=None):
        """Run the recurrent core over an encoded sequence shaped ``(B, T, F)``."""
        if self.num_layers == 1:
            out, next_state = self.rnn(x, state) if state is not None else self.rnn(x)
            out = self.norm(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out, next_state

        layer_output = x
        next_h: list[torch.Tensor] = []
        next_c: list[torch.Tensor] = []
        for layer_idx, (rnn, norm) in enumerate(zip(self.rnns, self.norms)):
            layer_state = None
            if state is not None:
                if self.uses_tuple_state:
                    layer_state = (
                        state[0][layer_idx : layer_idx + 1],
                        state[1][layer_idx : layer_idx + 1],
                    )
                else:
                    layer_state = state[layer_idx : layer_idx + 1]
            layer_output, layer_next = (
                rnn(layer_output, layer_state) if layer_state is not None else rnn(layer_output)
            )
            layer_output = norm(layer_output)
            layer_output = F.relu(layer_output)
            layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            if self.uses_tuple_state:
                next_h.append(layer_next[0])
                next_c.append(layer_next[1])
            else:
                next_h.append(layer_next)

        if self.uses_tuple_state:
            next_state = torch.cat(next_h, dim=0), torch.cat(next_c, dim=0)
        else:
            next_state = torch.cat(next_h, dim=0)
        return layer_output, next_state


class RNNCore(TorchRecurrentCore):
    """Task-agnostic vanilla RNN core."""

    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__(input_size, hidden_size, nn.RNN, dropout=dropout, num_layers=num_layers)


class GRUCore(TorchRecurrentCore):
    """Task-agnostic GRU core."""

    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__(input_size, hidden_size, nn.GRU, dropout=dropout, num_layers=num_layers)


class LSTMCore(TorchRecurrentCore):
    """Task-agnostic LSTM core."""

    def __init__(
        self, input_size: int, hidden_size: int, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__(input_size, hidden_size, nn.LSTM, dropout=dropout, num_layers=num_layers)
