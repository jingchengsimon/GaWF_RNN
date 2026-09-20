"""Clean-room BRIMs core from Mittal et al. (2020), arXiv:2006.16981."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedLinear(nn.Module):
    """Independent linear transform for each recurrent module."""

    def __init__(self, groups: int, input_size: int, output_size: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(groups, output_size, input_size))
        self.bias = nn.Parameter(torch.empty(groups, output_size)) if bias else None
        with torch.no_grad():
            for group_weight in self.weight:
                nn.init.kaiming_uniform_(group_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("bni,noi->bno", x, self.weight)
        return output if self.bias is None else output + self.bias.unsqueeze(0)


class GroupedLSTMCell(nn.Module):
    """A bank of independent LSTM cells represented as grouped tensors."""

    def __init__(self, groups: int, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_linear = GroupedLinear(groups, input_size, 4 * hidden_size)
        self.hidden_linear = GroupedLinear(groups, hidden_size, 4 * hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i, f, g, o = (self.input_linear(x) + self.hidden_linear(h)).chunk(4, dim=-1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class BRIMsLayer(nn.Module):
    """One BRIMs layer: input selection, sparse recurrence, and communication."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        topk: int,
        num_sources: int,
        input_attention_heads: int,
        input_attention_key_size: int,
        communication_attention_heads: int,
        communication_attention_key_size: int,
        communication_attention_value_size: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_blocks:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_blocks={num_blocks}"
            )
        if not 0 < topk <= num_blocks:
            raise ValueError(f"topk must be in [1, {num_blocks}], got {topk}")
        self.hidden_size = int(hidden_size)
        self.num_blocks = int(num_blocks)
        self.topk = int(topk)
        self.module_size = self.hidden_size // self.num_blocks
        self.num_sources = int(num_sources)
        self.input_attention_heads = int(input_attention_heads)
        self.input_attention_key_size = int(input_attention_key_size)
        self.communication_attention_heads = int(communication_attention_heads)
        self.communication_attention_key_size = int(communication_attention_key_size)
        self.communication_attention_value_size = int(communication_attention_value_size)
        self.attention_dropout = float(attention_dropout)

        input_query_size = self.input_attention_heads * self.input_attention_key_size
        input_value_size = self.input_attention_heads * self.module_size
        self.input_queries = GroupedLinear(self.num_blocks, self.module_size, input_query_size)
        self.source_keys = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, input_query_size, bias=False)
                for _ in range(num_sources - 1)
            ]
        )
        self.source_values = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, input_value_size, bias=False)
                for _ in range(num_sources - 1)
            ]
        )
        self.recurrent = GroupedLSTMCell(self.num_blocks, input_value_size, self.module_size)

        communication_query_size = (
            self.communication_attention_heads * self.communication_attention_key_size
        )
        communication_value_size = (
            self.communication_attention_heads * self.communication_attention_value_size
        )
        self.comm_queries = GroupedLinear(
            self.num_blocks, self.module_size, communication_query_size
        )
        self.comm_keys = GroupedLinear(self.num_blocks, self.module_size, communication_query_size)
        self.comm_values = GroupedLinear(
            self.num_blocks, self.module_size, communication_value_size
        )
        self.comm_output = GroupedLinear(
            self.num_blocks, communication_value_size, self.module_size
        )

    def _input_attention(
        self,
        h_prev: torch.Tensor,
        sources: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_prev.size(0)
        query = self.input_queries(h_prev).view(
            batch_size,
            self.num_blocks,
            self.input_attention_heads,
            self.input_attention_key_size,
        )
        zero_key = query.new_zeros(
            batch_size,
            self.input_attention_heads,
            self.input_attention_key_size,
        )
        zero_value = query.new_zeros(
            batch_size,
            self.input_attention_heads,
            self.module_size,
        )
        keys = [zero_key]
        values = [zero_value]
        for source, key_layer, value_layer in zip(
            sources,
            self.source_keys,
            self.source_values,
        ):
            keys.append(
                key_layer(source).view(
                    batch_size,
                    self.input_attention_heads,
                    self.input_attention_key_size,
                )
            )
            values.append(
                value_layer(source).view(
                    batch_size,
                    self.input_attention_heads,
                    self.module_size,
                )
            )
        key_tensor = torch.stack(keys, dim=1)
        value_tensor = torch.stack(values, dim=1)
        scores = torch.einsum("bnhk,bshk->bnhs", query, key_tensor)
        scores = scores / math.sqrt(self.input_attention_key_size)
        selection_attention = torch.softmax(scores, dim=-1)
        context_attention = F.dropout(
            selection_attention,
            p=self.attention_dropout,
            training=self.training,
        )
        context = torch.einsum("bnhs,bshd->bnhd", context_attention, value_tensor)
        context = context.reshape(batch_size, self.num_blocks, -1)
        null_score = selection_attention[..., 0].mean(dim=-1)
        active_indices = torch.topk(null_score, self.topk, dim=-1, largest=False).indices
        active = torch.zeros_like(null_score, dtype=torch.bool)
        active.scatter_(1, active_indices, True)
        return context, active

    def _communicate(self, proposed: torch.Tensor) -> torch.Tensor:
        batch_size = proposed.size(0)
        query = self.comm_queries(proposed).view(
            batch_size,
            self.num_blocks,
            self.communication_attention_heads,
            self.communication_attention_key_size,
        )
        key = self.comm_keys(proposed).view_as(query)
        value = self.comm_values(proposed).view(
            batch_size,
            self.num_blocks,
            self.communication_attention_heads,
            self.communication_attention_value_size,
        )
        scores = torch.einsum("bnhk,bmhk->bnhm", query, key)
        scores = scores / math.sqrt(self.communication_attention_key_size)
        attention = torch.softmax(scores, dim=-1)
        attention = F.dropout(
            attention,
            p=self.attention_dropout,
            training=self.training,
        )
        message = torch.einsum("bnhm,bmhd->bnhd", attention, value)
        message = message.reshape(batch_size, self.num_blocks, -1)
        return proposed + self.comm_output(message)

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sources) != self.num_sources - 1:
            raise ValueError(
                f"expected {self.num_sources - 1} non-null sources, got {len(sources)}"
            )
        h_modules = h_prev.view(h_prev.size(0), self.num_blocks, self.module_size)
        c_modules = c_prev.view(c_prev.size(0), self.num_blocks, self.module_size)
        attended, active = self._input_attention(h_modules, sources)
        proposed_h, proposed_c = self.recurrent(attended, h_modules, c_modules)
        active = active.unsqueeze(-1)
        recurrent_h = torch.where(active, proposed_h, h_modules)
        next_c = torch.where(active, proposed_c, c_modules)
        communicated_h = self._communicate(recurrent_h)
        next_h = torch.where(active, communicated_h, recurrent_h)
        return next_h.reshape(h_prev.shape), next_c.reshape(c_prev.shape)


class BRIMsCore(nn.Module):
    """Two-layer, batch-first clean-room BRIMs recurrent core.

    The bottom layer attends to a null source, current encoded input, and the
    previous-timestep upper-layer state. The upper layer attends to a null source
    and the current bottom-layer state. Only the selected top-k modules update;
    within-layer attention communication is retained as part of BRIMs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_blocks: Sequence[int] = (6, 3),
        topk: Sequence[int] = (4, 2),
        input_attention_heads: int = 4,
        input_attention_key_size: int = 64,
        communication_attention_heads: int = 4,
        communication_attention_key_size: int = 32,
        communication_attention_value_size: int = 32,
        attention_dropout: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 2 or len(topk) != 2:
            raise ValueError("BRIMs requires exactly two num_blocks and topk values")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.num_layers = 2
        self.num_blocks = tuple(int(value) for value in num_blocks)
        self.topk = tuple(int(value) for value in topk)
        self.input_attention_heads = int(input_attention_heads)
        self.input_attention_key_size = int(input_attention_key_size)
        self.communication_attention_heads = int(communication_attention_heads)
        self.communication_attention_key_size = int(communication_attention_key_size)
        self.communication_attention_value_size = int(communication_attention_value_size)
        self.attention_dropout = float(attention_dropout)
        self.dropout = float(dropout)
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        self.layers = nn.ModuleList(
            [
                BRIMsLayer(
                    self.hidden_size,
                    self.num_blocks[0],
                    self.topk[0],
                    num_sources=3,
                    input_attention_heads=self.input_attention_heads,
                    input_attention_key_size=self.input_attention_key_size,
                    communication_attention_heads=self.communication_attention_heads,
                    communication_attention_key_size=self.communication_attention_key_size,
                    communication_attention_value_size=(self.communication_attention_value_size),
                    attention_dropout=self.attention_dropout,
                ),
                BRIMsLayer(
                    self.hidden_size,
                    self.num_blocks[1],
                    self.topk[1],
                    num_sources=2,
                    input_attention_heads=self.input_attention_heads,
                    input_attention_key_size=self.input_attention_key_size,
                    communication_attention_heads=self.communication_attention_heads,
                    communication_attention_key_size=self.communication_attention_key_size,
                    communication_attention_value_size=(self.communication_attention_value_size),
                    attention_dropout=self.attention_dropout,
                ),
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two-layer state as ``(h, c)``, each shaped ``(2, B, H)``."""
        zeros = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        return zeros, zeros.clone()

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run a ``(B, T, F)`` sequence with paper-defined BRIMs timing."""
        batch_size = x.size(0)
        if state is None:
            h, c = self.initial_state(batch_size, x.device, x.dtype)
        else:
            h, c = state
        h0, h1 = h[0], h[1]
        c0, c1 = c[0], c[1]
        outputs = []
        projected = self.input_projection(x)
        for timestep in range(x.size(1)):
            previous_upper = h1
            h0, c0 = self.layers[0]([projected[:, timestep], previous_upper], h0, c0)
            h1, c1 = self.layers[1]([h0], h1, c1)
            outputs.append(h1)
        output = torch.stack(outputs, dim=1)
        output = self.norm(output)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output, (torch.stack((h0, h1)), torch.stack((c0, c1)))
