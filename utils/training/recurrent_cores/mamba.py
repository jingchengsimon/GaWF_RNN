"""Canonical batch-first Mamba core with projection and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn

MAMBA_DEFAULT_D_MODEL = 170


def _mamba_class():
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "MambaCore requires 'mamba-ssm' and 'causal-conv1d'."
        ) from exc
    return Mamba


class MambaCore(nn.Module):
    """Projected Mamba stack; each block output is added to its input."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = int(input_size)
        self.hidden_size = int(d_model)
        self.output_size = int(d_model)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.output_wrap = "none"
        self.input_proj = (
            nn.Linear(self.input_size, self.d_model)
            if self.input_size != self.d_model
            else nn.Identity()
        )
        block = _mamba_class()
        self.layers = nn.ModuleList(
            [
                block(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(self.num_layers)
            ]
        )
        self.layer_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native Mamba blocks without an external activation wrap."""

        del state
        x = self.input_proj(x)
        layer_finals: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x) + x
            layer_finals.append(x[:, -1, :])
            if layer_idx < self.num_layers - 1:
                x = self.layer_dropout(x)
        return x, torch.stack(layer_finals, dim=0)
