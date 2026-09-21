"""Task-agnostic wrappers around PyTorch RNN, GRU, and LSTM sequence layers."""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchRecurrentCore(nn.Module):
    """Batch-first recurrent core with an optional LayerNorm, ReLU, and output dropout wrap.

    ``output_wrap="ln_relu_dropout"`` (default) keeps the historical behaviour: the built-in
    recurrence is followed by ``LayerNorm -> ReLU -> dropout`` on the output sequence.
    ``output_wrap="none"`` removes that wrap so the core returns the raw built-in output.
    ``rnn_activation`` only applies to the vanilla :class:`torch.nn.RNN` core and selects the
    activation applied inside the recurrence: the built-in ``tanh`` (default), the built-in
    ``relu``, or ``identity``, which removes the activation entirely and yields a linear
    first-order recurrence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_class: Type[nn.RNNBase],
        dropout: float = 0.0,
        num_layers: int = 1,
        batch_first: bool = True,
        output_wrap: str = "ln_relu_dropout",
        rnn_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if output_wrap not in ("ln_relu_dropout", "none"):
            raise ValueError(f"Unsupported output_wrap: {output_wrap!r}")
        if rnn_activation not in ("tanh", "relu", "identity"):
            raise ValueError(f"Unsupported rnn_activation: {rnn_activation!r}")
        if rnn_activation != "tanh":
            if rnn_class is not nn.RNN:
                raise ValueError(
                    "rnn_activation variants are supported only for the vanilla RNN core"
                )
            if num_layers != 1:
                raise ValueError("rnn_activation variants require num_layers == 1")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.uses_tuple_state = rnn_class is nn.LSTM
        self.output_wrap = str(output_wrap)
        self.rnn_activation = str(rnn_activation)
        self.identity_recurrence = self.rnn_activation == "identity"
        wrap_enabled = self.output_wrap == "ln_relu_dropout"
        if self.num_layers == 1:
            # Keep the legacy module names exactly so existing checkpoints load.
            rnn_kwargs = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": 1,
                "batch_first": self.batch_first,
            }
            if rnn_class is nn.RNN:
                rnn_kwargs["nonlinearity"] = (
                    "relu" if self.rnn_activation == "relu" else "tanh"
                )
            self.rnn = rnn_class(**rnn_kwargs)
            self.norm = nn.LayerNorm(self.hidden_size) if wrap_enabled else None
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
            self.norms = (
                nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)])
                if wrap_enabled
                else None
            )

    def _wrap_sequence(self, out: torch.Tensor, norm: nn.Module | None) -> torch.Tensor:
        """Apply the configured outer LayerNorm, ReLU, and dropout wrap to a sequence output."""

        if norm is None:
            return out
        out = norm(out)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout, training=self.training)

    def _forward_identity_recurrence(
        self, x: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vanilla recurrence with no activation, i.e. a linear first-order recurrence."""

        batch_size, frame_num = x.shape[:2]
        hidden = (
            torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            if state is None
            else state
        )
        outputs = []
        for step in range(frame_num):
            hidden = F.linear(
                x[:, step, :], self.rnn.weight_ih_l0, self.rnn.bias_ih_l0
            ) + F.linear(hidden, self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)
            outputs.append(hidden)
        stacked = torch.stack(outputs, dim=1)
        return self._wrap_sequence(stacked, self.norm), hidden.unsqueeze(0)

    def forward(self, x: torch.Tensor, state=None):
        """Run the recurrent core over an encoded sequence shaped ``(B, T, F)``."""
        if self.identity_recurrence:
            return self._forward_identity_recurrence(x, state)
        if self.num_layers == 1:
            out, next_state = self.rnn(x, state) if state is not None else self.rnn(x)
            out = self._wrap_sequence(out, self.norm)
            return out, next_state

        layer_output = x
        next_h: list[torch.Tensor] = []
        next_c: list[torch.Tensor] = []
        norms = self.norms if self.norms is not None else [None] * self.num_layers
        for layer_idx, (rnn, norm) in enumerate(zip(self.rnns, norms)):
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
            layer_output = self._wrap_sequence(layer_output, norm)
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
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        **core_kwargs,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            nn.RNN,
            dropout=dropout,
            num_layers=num_layers,
            **core_kwargs,
        )


class GRUCore(TorchRecurrentCore):
    """Task-agnostic GRU core."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        **core_kwargs,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            nn.GRU,
            dropout=dropout,
            num_layers=num_layers,
            **core_kwargs,
        )


class LSTMCore(TorchRecurrentCore):
    """Task-agnostic LSTM core."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        **core_kwargs,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            nn.LSTM,
            dropout=dropout,
            num_layers=num_layers,
            **core_kwargs,
        )
