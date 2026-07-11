"""Task-agnostic S5 sequence core."""

from __future__ import annotations

import importlib

# Import NumPy before torch on macOS/conda to avoid duplicate libomp crashes in
# S5's HiPPO initialization path.
importlib.import_module("numpy")

import torch
import torch.nn as nn
import torch.nn.functional as F

S5_DEFAULT_D_MODEL = 256
S5_DEFAULT_STATE_SIZE = 128


def _get_s5_layer():
    try:
        from s5 import S5
    except ImportError as exc:
        raise ImportError(
            "S5Core requires the optional dependency 's5-pytorch'. "
            "Install it with: pip install s5-pytorch"
        ) from exc
    return S5


class S5Core(nn.Module):
    """Batch-first stack of S5 layers with a recurrent-core style interface."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        state_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dropout: float | None = None,
        residual: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("S5Core does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_size = int(input_size)
        self.hidden_size = int(d_model)
        self.output_size = int(d_model)
        self.d_model = int(d_model)
        self.state_size = int(state_size)
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.residual = bool(residual)
        self.output_dropout = float(dropout if output_dropout is None else output_dropout)

        self.input_proj = (
            nn.Linear(self.input_size, self.d_model)
            if self.input_size != self.d_model
            else nn.Identity()
        )
        layer_cls = _get_s5_layer()
        self.layers = nn.ModuleList(
            [layer_cls(self.d_model, self.state_size) for _ in range(self.num_layers)]
        )
        self.layer_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.d_model)

    @staticmethod
    def _is_autocast_enabled(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type=device_type))
        except TypeError:
            return bool(torch.is_autocast_enabled())

    def forward(self, x: torch.Tensor, state=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        x = self.input_proj(x)
        layer_finals = []
        autocast_active = self._is_autocast_enabled(x.device.type)
        for layer_idx, layer in enumerate(self.layers):
            residual = x
            if autocast_active:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    x = layer(x.float())
                x = x.to(residual.dtype)
            else:
                x = layer(x)
            if self.residual:
                x = x + residual
            layer_finals.append(x[:, -1, :])
            if layer_idx < self.num_layers - 1:
                x = self.layer_dropout(x)

        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.output_dropout, training=self.training)
        h_n = torch.stack(layer_finals, dim=0)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, h_n
