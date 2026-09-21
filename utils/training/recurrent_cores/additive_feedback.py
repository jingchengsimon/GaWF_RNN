"""Task-agnostic recurrent cores for non-multiplicative output-feedback controls."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveFeedbackRNNCore(nn.Module):
    """GaWF-shaped RNN core with additive, rather than multiplicative, feedback.

    Follows the RNN-aligned GaWF contract: the recurrent state that is carried and fed back is the
    raw activation ``tanh(preactivation)``, and the external ``LayerNorm -> ReLU -> dropout`` wrap
    is applied to the readout only. ``step`` therefore returns ``(readout, next_state)`` exactly
    like :class:`ConcatenatedFeedbackCellCore`, so the two feedback controls differ only in how the
    feedback vector enters the preactivation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int,
        dropout: float = 0.0,
        initial_weight_scale: float = 0.5,
        state_semantics: str = "aligned",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.feedback_dim = int(feedback_dim)
        self.dropout = float(dropout)
        self.initial_weight_scale = float(initial_weight_scale)
        if state_semantics not in ("aligned", "legacy"):
            raise ValueError(f"Unsupported state_semantics: {state_semantics!r}")
        self.state_semantics = str(state_semantics)
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.feedback_linear = nn.Linear(self.feedback_dim, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        with torch.no_grad():
            self.rnn.weight_ih_l0.mul_(self.initial_weight_scale)
            self.rnn.weight_hh_l0.mul_(self.initial_weight_scale)
            self.feedback_linear.bias.zero_()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the zero recurrent state for one independent sequence batch."""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep and return the wrapped readout and the raw next state."""
        fb = feedback.to(device=x_t.device, dtype=x_t.dtype).clamp(-10, 10)
        preactivation = F.linear(x_t, self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
        preactivation = preactivation + F.linear(
            state,
            self.rnn.weight_hh_l0,
            self.rnn.bias_hh_l0,
        )
        preactivation = preactivation + self.feedback_linear(fb)
        hidden = torch.tanh(preactivation)
        readout = torch.relu(self.norm(hidden))
        readout = F.dropout(readout, p=self.dropout, training=self.training)
        if self.state_semantics == "legacy":
            # Pre-alignment behavior: the wrapped value itself was carried and fed back.
            return readout, readout
        return readout, hidden

    def forward_no_feedback(self, x: torch.Tensor):
        """Run the same recurrence while omitting the additive feedback pathway."""
        batch_size, frame_num = x.shape[:2]
        state = self.initial_state(batch_size, x.device, x.dtype)
        outputs = []
        for t in range(frame_num):
            preactivation = F.linear(
                x[:, t, :],
                self.rnn.weight_ih_l0,
                self.rnn.bias_ih_l0,
            )
            preactivation = preactivation + F.linear(
                state,
                self.rnn.weight_hh_l0,
                self.rnn.bias_hh_l0,
            )
            raw_state = torch.tanh(preactivation)
            output = F.relu(self.norm(raw_state))
            output = F.dropout(output, p=self.dropout, training=self.training)
            state = output if self.state_semantics == "legacy" else raw_state
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class ConcatenatedFeedbackCellCore(nn.Module):
    """Single-layer RNN/GRU/LSTM cell with feedback concatenated to each frame input."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int,
        cell_type: Literal["rnn", "gru", "lstm"],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.feedback_dim = int(feedback_dim)
        self.cell_type = cell_type
        self.dropout = float(dropout)
        cell_classes = {
            "rnn": nn.RNNCell,
            "gru": nn.GRUCell,
            "lstm": nn.LSTMCell,
        }
        if cell_type not in cell_classes:
            raise ValueError(f"Unsupported feedback cell type: {cell_type!r}")
        self.cell = cell_classes[cell_type](
            input_size=self.input_size + self.feedback_dim,
            hidden_size=self.hidden_size,
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the zero recurrent state for one independent sequence batch."""
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.cell_type == "lstm":
            return hidden, torch.zeros_like(hidden)
        return hidden

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        feedback: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ]:
        """Advance one timestep and keep the raw cell state separate from readout activation."""
        fb = feedback.to(device=x_t.device, dtype=x_t.dtype).clamp(-10, 10)
        cell_input = torch.cat([x_t, fb], dim=-1)
        next_state = self.cell(cell_input, state)
        hidden = next_state[0] if self.cell_type == "lstm" else next_state
        output = F.relu(self.norm(hidden))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, next_state

    def forward_no_feedback(self, x: torch.Tensor):
        """Run the cell loop with all-zero feedback for open-loop equivalence checks."""
        batch_size, frame_num = x.shape[:2]
        state = self.initial_state(batch_size, x.device, x.dtype)
        feedback = torch.zeros(
            batch_size,
            self.feedback_dim,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []
        for t in range(frame_num):
            output, state = self.step(x[:, t, :], state, feedback)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state
