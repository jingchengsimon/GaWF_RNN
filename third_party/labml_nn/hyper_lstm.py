"""Minimal vendored labml-nn HyperLSTM implementation.

This file retains the equations and module layout of
``labml_nn.hypernetworks.hyper_lstm`` at the commit recorded in ``UPSTREAM.md``.
The only compatibility change is inlining the small labml ``LSTMCell`` dependency
so the project does not need the full ``labml-nn`` dependency tree.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class _LSTMCell(nn.Module):
    """The labml LSTM cell used by the HyperLSTM hypernetwork."""

    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False) -> None:
        super().__init__()
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = (self.hidden_lin(h) + self.input_lin(x)).chunk(4, dim=-1)
        i, f, g, o = [norm(gate) for norm, gate in zip(self.layer_norm, gates)]
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next


class HyperLSTMCell(nn.Module):
    """HyperLSTM cell with row-wise dynamic scaling of main-LSTM parameters."""

    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int) -> None:
        super().__init__()
        self.hyper = _LSTMCell(hidden_size + input_size, hyper_size, layer_norm=True)
        self.z_h = nn.Linear(hyper_size, 4 * n_z)
        self.z_x = nn.Linear(hyper_size, 4 * n_z)
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)
        self.d_h = nn.ModuleList([nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)])
        self.d_x = nn.ModuleList([nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)])
        self.d_b = nn.ModuleList([nn.Linear(n_z, hidden_size) for _ in range(4)])
        self.w_h = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)]
        )
        self.w_x = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)]
        )
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        h_hat: torch.Tensor,
        c_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_hat = torch.cat((h, x), dim=-1)
        h_hat, c_hat = self.hyper(x_hat, h_hat, c_hat)
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        gates = []
        for gate_idx in range(4):
            d_h = self.d_h[gate_idx](z_h[gate_idx])
            d_x = self.d_x[gate_idx](z_x[gate_idx])
            preactivation = (
                d_h * torch.einsum("ij,bj->bi", self.w_h[gate_idx], h)
                + d_x * torch.einsum("ij,bj->bi", self.w_x[gate_idx], x)
                + self.d_b[gate_idx](z_b[gate_idx])
            )
            gates.append(self.layer_norm[gate_idx](preactivation))

        i, f, g, o = gates
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next, h_hat, c_hat


class HyperLSTM(nn.Module):
    """Time-major stack of labml HyperLSTM cells."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.hyper_size = int(hyper_size)
        self.cells = nn.ModuleList(
            [HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)]
            + [
                HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z)
                for _ in range(n_layers - 1)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        n_steps, batch_size = x.shape[:2]
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
        else:
            h, c, h_hat, c_hat = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        outputs = []
        for timestep in range(n_steps):
            layer_input = x[timestep]
            for layer in range(self.n_layers):
                h[layer], c[layer], h_hat[layer], c_hat[layer] = self.cells[layer](
                    layer_input,
                    h[layer],
                    c[layer],
                    h_hat[layer],
                    c_hat[layer],
                )
                layer_input = h[layer]
            outputs.append(h[-1])

        return torch.stack(outputs), (
            torch.stack(h),
            torch.stack(c),
            torch.stack(h_hat),
            torch.stack(c_hat),
        )
