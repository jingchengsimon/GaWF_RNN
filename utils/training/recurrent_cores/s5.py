"""Canonical batch-first S5 core with projection and residual connections."""

from __future__ import annotations

import importlib

importlib.import_module("numpy")

import torch
import torch.nn as nn

S5_DEFAULT_D_MODEL = 256
S5_DEFAULT_STATE_SIZE = 128


def _s5_class():
    try:
        from s5 import S5
    except ImportError as exc:
        raise ImportError("S5Core requires 's5-pytorch'.") from exc
    return S5


class S5Core(nn.Module):
    """Projected S5 stack; each layer output is added to its input."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        state_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = int(input_size)
        self.hidden_size = int(d_model)
        self.output_size = int(d_model)
        self.d_model = int(d_model)
        self.state_size = int(state_size)
        self.num_layers = int(num_layers)
        self.output_wrap = "none"
        self.input_proj = (
            nn.Linear(self.input_size, self.d_model)
            if self.input_size != self.d_model
            else nn.Identity()
        )
        layer = _s5_class()
        self.layers = nn.ModuleList(
            [layer(self.d_model, self.state_size) for _ in range(self.num_layers)]
        )
        self.output_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    @staticmethod
    def _autocast_enabled(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type=device_type))
        except TypeError:
            return bool(torch.is_autocast_enabled())

    def forward(
        self,
        x: torch.Tensor,
        state: None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native S5 layers without an external activation wrap."""

        del state
        x = self.input_proj(x)
        layer_finals: list[torch.Tensor] = []
        autocast_active = self._autocast_enabled(x.device.type)
        for layer_idx, layer in enumerate(self.layers):
            residual = x
            if autocast_active:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    branch = layer(x.float())
                branch = branch.to(residual.dtype)
            else:
                branch = layer(x)
            x = residual + self.output_dropouts[layer_idx](branch)
            layer_finals.append(x[:, -1, :])
        return x, torch.stack(layer_finals, dim=0)
