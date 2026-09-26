"""Canonical RNN, GRU, and LSTM cores.

The Elman RNN carries clean ReLU activity and drops only the layer output:
`h_t = relu(layer_norm(W_ih x_t + W_hh h_(t-1) + b_ih + b_hh))`,
`out_t = dropout(h_t)`.
Its four affine parameters retain the historical ``nn.RNN`` names for checkpoint compatibility,
but no native ``nn.RNN`` forward or implicit activation is present.
GRU and LSTM use native PyTorch layers with explicit output dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElmanAffine(nn.Module):
    """Checkpoint-compatible parameters for one activation-free Elman affine transform."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih_l0 = nn.Parameter(torch.empty(self.hidden_size, self.input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(self.hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match the default initialization used by ``nn.RNN``."""

        bound = self.hidden_size**-0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)


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
        self.output_wrap = "in_loop_ln_relu_output_dropout"
        self.wrap_recurrent_state = False
        self.output_dropouts = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(num_layers)])

        if self.num_layers == 1:
            self.rnn = ElmanAffine(self.input_size, self.hidden_size)
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            self.rnns = nn.ModuleList(
                [ElmanAffine(size, self.hidden_size) for size in layer_inputs]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers)])

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
            preactivation = F.linear(layer_input, rnn.weight_ih_l0, rnn.bias_ih_l0) + F.linear(
                state[layer_idx], rnn.weight_hh_l0, rnn.bias_hh_l0
            )
            hidden = F.relu(norm(preactivation))
            next_states.append(hidden)
            layer_input = self.output_dropouts[layer_idx](hidden)
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
    """Native GRU layers with clean hidden state and explicit output dropout."""

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
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_wrap = "none"
        self.output_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
        if self.num_layers == 1:
            self.rnn = nn.GRU(layer_inputs[0], self.hidden_size, batch_first=True, dropout=0.0)
        else:
            self.rnns = nn.ModuleList(
                [nn.GRU(size, self.hidden_size, batch_first=True, dropout=0.0)
                 for size in layer_inputs]
            )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native GRU sequence path."""

        next_states: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            layer = self.rnn if self.num_layers == 1 else self.rnns[layer_idx]
            layer_state = None if state is None else state[layer_idx : layer_idx + 1]
            hidden, next_state = layer(x, layer_state)
            next_states.append(next_state)
            x = self.output_dropouts[layer_idx](hidden)
        return x, torch.cat(next_states, dim=0)


class LSTMCore(nn.Module):
    """Native LSTM layers with clean hidden/cell states and output dropout."""

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
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.output_wrap = "none"
        self.output_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
        if self.num_layers == 1:
            self.rnn = nn.LSTM(layer_inputs[0], self.hidden_size, batch_first=True, dropout=0.0)
        else:
            self.rnns = nn.ModuleList(
                [nn.LSTM(size, self.hidden_size, batch_first=True, dropout=0.0)
                 for size in layer_inputs]
            )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the native LSTM sequence path."""

        next_h: list[torch.Tensor] = []
        next_c: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            layer = self.rnn if self.num_layers == 1 else self.rnns[layer_idx]
            layer_state = None if state is None else (
                state[0][layer_idx : layer_idx + 1],
                state[1][layer_idx : layer_idx + 1],
            )
            hidden, (h, c) = layer(x, layer_state)
            next_h.append(h)
            next_c.append(c)
            x = self.output_dropouts[layer_idx](hidden)
        return x, (torch.cat(next_h, dim=0), torch.cat(next_c, dim=0))
