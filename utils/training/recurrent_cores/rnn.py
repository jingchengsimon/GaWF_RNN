"""Canonical RNN, GRU, and LSTM cores.

The Elman RNN uses the same in-loop activity definition as GaWF and has no tanh:
`h_t = dropout(relu(layer_norm(W_ih x_t + W_hh h_(t-1) + b_ih + b_hh)))`.
GRU and LSTM use the native PyTorch sequence layers without an external wrap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNCore(nn.Module):
    """Ungated GaWF-matched recurrence with ReLU activity inside the loop."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.rnn_activation = "identity"
        self.output_wrap = "in_loop_ln_relu_dropout"
        self.wrap_recurrent_state = True

        if self.num_layers == 1:
            self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            self.rnns = nn.ModuleList(
                [nn.RNN(size, self.hidden_size, batch_first=True) for size in layer_inputs]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
            )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a PyTorch-shaped zero state `(layers, batch, hidden)`."""

        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
        )

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep and return `(top_output, next_layer_states)`."""

        if state.ndim == 2 and self.num_layers == 1:
            state = state.unsqueeze(0)
        expected = (self.num_layers, x_t.size(0), self.hidden_size)
        if tuple(state.shape) != expected:
            raise ValueError(f"Expected RNN state {expected}, got {tuple(state.shape)}")

        layer_input = x_t
        next_states: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            if self.num_layers == 1:
                rnn, norm = self.rnn, self.norm
            else:
                rnn, norm = self.rnns[layer_idx], self.norms[layer_idx]
            preactivation = F.linear(
                layer_input, rnn.weight_ih_l0, rnn.bias_ih_l0
            ) + F.linear(state[layer_idx], rnn.weight_hh_l0, rnn.bias_hh_l0)
            layer_input = F.dropout(
                F.relu(norm(preactivation)), p=self.dropout, training=self.training
            )
            next_states.append(layer_input)
        return layer_input, torch.stack(next_states, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the batch-first sequence through the fixed in-loop ReLU recurrence."""

        if state is None:
            state = self.initial_state(x.size(0), x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        for time_idx in range(x.size(1)):
            output, state = self.step(x[:, time_idx, :], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class GRUCore(nn.Module):
    """Native PyTorch GRU with no external LayerNorm/ReLU/dropout wrap."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_wrap = "none"
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native GRU sequence path."""

        return self.rnn(x, state) if state is not None else self.rnn(x)


class LSTMCore(nn.Module):
    """Native PyTorch LSTM with no external LayerNorm/ReLU/dropout wrap."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_wrap = "none"
        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the native LSTM sequence path."""

        return self.rnn(x, state) if state is not None else self.rnn(x)
