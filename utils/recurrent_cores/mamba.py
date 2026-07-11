"""Task-agnostic Mamba sequence core."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA_DEFAULT_D_MODEL = 170


def _get_mamba_block(block_type: str):
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "MambaCore requires the optional dependency 'mamba-ssm'. "
            "Install it with: pip install mamba-ssm causal-conv1d"
        ) from exc

    if block_type == "mamba":
        return Mamba
    if block_type == "mamba2":
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "mamba_block_type='mamba2' requires a mamba-ssm version that exports Mamba2."
            ) from exc
        return Mamba2
    raise ValueError("block_type must be 'mamba' or 'mamba2'")


class MambaCore(nn.Module):
    """Batch-first stack of Mamba blocks with a recurrent-core style interface."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_dropout: float | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        block_type: str = "mamba",
        residual: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("MambaCore does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = int(input_size)
        self.hidden_size = int(d_model)
        self.output_size = int(d_model)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.residual = bool(residual)
        self.output_dropout = float(dropout if output_dropout is None else output_dropout)

        self.input_proj = (
            nn.Linear(self.input_size, self.d_model)
            if self.input_size != self.d_model
            else nn.Identity()
        )
        block_cls = _get_mamba_block(block_type)
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=self.d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, state=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        x = self.input_proj(x)
        layer_finals = []
        for layer_idx, layer in enumerate(self.layers):
            residual = x
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
