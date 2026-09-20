"""Multiplicative LSTM core from Krause et al. (2017), not xLSTM mLSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSTMCore(nn.Module):
    """Batch-first multiplicative LSTM with the shared Clutter post-processing.

    This implements Krause et al., arXiv:1609.07959, Equations (17)--(21):
    ``m_t = (W_mx x_t) * (W_mh h_{t-1})`` and the input, forget, output,
    and candidate preactivations use ``m_t`` in place of the usual recurrent
    hidden-state term. This is unrelated to the matrix-memory mLSTM in xLSTM
    (Beck et al., 2024).
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.mx = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.mh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.x_gates = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.m_gates = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an LSTM-compatible ``(h, c)`` state shaped ``(1, B, H)``."""
        zeros = torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def cell_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the raw mLSTM recurrence before output normalization/dropout."""
        m_t = self.mx(x_t) * self.mh(h_prev)
        i_t, f_t, g_t, o_t = (self.x_gates(x_t) + self.m_gates(m_t)).chunk(4, dim=-1)
        c_t = torch.sigmoid(f_t) * c_prev + torch.sigmoid(i_t) * torch.tanh(g_t)
        h_t = torch.sigmoid(o_t) * torch.tanh(c_t)
        return h_t, c_t

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run a sequence shaped ``(B, T, F)`` and return LSTM-compatible state."""
        batch_size = x.size(0)
        if state is None:
            h_state, c_state = self.initial_state(batch_size, x.device, x.dtype)
        else:
            h_state, c_state = state
        h_t, c_t = h_state[0], c_state[0]
        outputs = []
        for timestep in range(x.size(1)):
            h_t, c_t = self.cell_step(x[:, timestep], h_t, c_t)
            outputs.append(h_t)
        output = torch.stack(outputs, dim=1)
        output = self.norm(output)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, (h_t.unsqueeze(0), c_t.unsqueeze(0))
