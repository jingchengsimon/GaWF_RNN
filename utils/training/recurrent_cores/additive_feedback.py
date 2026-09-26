"""Single-layer RNN/GRU/LSTM cores with an independent additive logit-feedback affine.

Input: one encoded frame, clean recurrent state, and previous detached task logits.
Output: dropped readout activity and the clean next recurrent state.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveFeedbackCellCore(nn.Module):
    """Add ``W_fb f_(t-1) + b_fb`` to the cell preactivation before its nonlinearity."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cell_classes = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}
        gate_counts = {"rnn": 1, "gru": 3, "lstm": 4}
        if cell_type not in cell_classes:
            raise ValueError(f"Unsupported feedback cell type: {cell_type!r}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = self.hidden_size
        self.feedback_dim = int(feedback_dim)
        self.cell_type = cell_type
        self.rnn_activation = "relu" if cell_type == "rnn" else None
        self.output_wrap = "in_loop_ln_relu_output_dropout" if cell_type == "rnn" else "none"
        self.wrap_recurrent_state = False
        self.cell = (
            nn.RNNCell(self.input_size, self.hidden_size, nonlinearity="relu")
            if cell_type == "rnn"
            else cell_classes[cell_type](self.input_size, self.hidden_size)
        )
        self.norm = nn.LayerNorm(self.hidden_size) if cell_type == "rnn" else None
        self.feedback_linear = nn.Linear(
            self.feedback_dim, gate_counts[cell_type] * self.hidden_size, bias=True
        )
        bound = self.hidden_size**-0.5
        nn.init.uniform_(self.feedback_linear.weight, -bound, bound)
        nn.init.uniform_(self.feedback_linear.bias, -bound, bound)
        self.output_dropout = nn.Dropout(dropout)

    def initial_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return a zero hidden state, plus a zero cell state for LSTM."""

        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.cell_type == "lstm":
            return hidden, torch.zeros_like(hidden)
        return hidden

    def _advance(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        feedback_affine: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the clean next state; omit the whole feedback affine when disabled."""

        if self.cell_type == "lstm":
            hidden, cell_state = state
            gates = F.linear(x_t, self.cell.weight_ih, self.cell.bias_ih)
            gates = gates + F.linear(hidden, self.cell.weight_hh, self.cell.bias_hh)
            if feedback_affine is not None:
                gates = gates + feedback_affine
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            next_cell = torch.sigmoid(forget_gate) * cell_state
            next_cell = next_cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
            next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
            return next_hidden, next_cell

        hidden = state
        input_affine = F.linear(x_t, self.cell.weight_ih, self.cell.bias_ih)
        if feedback_affine is not None:
            input_affine = input_affine + feedback_affine
        hidden_affine = F.linear(hidden, self.cell.weight_hh, self.cell.bias_hh)
        if self.cell_type == "rnn":
            return F.relu(self.norm(input_affine + hidden_affine))

        input_reset, input_update, input_candidate = input_affine.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = hidden_affine.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset_gate * hidden_candidate)
        return candidate + update_gate * (hidden - candidate)

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Return `(readout_output, clean_state)` for one time step."""

        feedback = feedback.to(device=x_t.device, dtype=x_t.dtype).clamp(-10, 10)
        next_state = self._advance(x_t, state, self.feedback_linear(feedback))
        hidden = next_state[0] if self.cell_type == "lstm" else next_state
        return self.output_dropout(hidden), next_state

    def forward_no_feedback(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Run the same clean cell while omitting feedback weights and bias."""

        state = self.initial_state(x.size(0), x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        for time_idx in range(x.size(1)):
            state = self._advance(x[:, time_idx, :], state, feedback_affine=None)
            hidden = state[0] if self.cell_type == "lstm" else state
            outputs.append(self.output_dropout(hidden))
        return torch.stack(outputs, dim=1), state
