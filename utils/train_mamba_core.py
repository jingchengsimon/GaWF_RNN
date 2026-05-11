import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


MAMBA_DEFAULT_D_MODEL = 170


def _get_mamba_block(block_type: str):
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "MambaConv requires the optional dependency 'mamba-ssm'. "
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


class MambaRNNWrapper(nn.Module):
    """
    Adapter that makes a stack of Mamba blocks look like an nn.RNN-style module.

    The forward interface matches the part of nn.RNN/nn.GRU used by BaseRNNConv:
    input is (B, T, input_size) when batch_first=True and output is
    (sequence_output, h_n). Mamba does not expose a training-time recurrent state,
    so h_n is represented by each layer's last output token.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        block_type: str = "mamba",
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if bidirectional:
            raise ValueError("MambaRNNWrapper does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_size = d_model
        self.d_model = d_model
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.residual = residual

        self.input_proj = (
            nn.Linear(input_size, d_model)
            if input_size != d_model
            else nn.Identity()
        )
        block_cls = _get_mamba_block(block_type)
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, h_0=None):
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
                x = self.dropout(x)

        h_n = torch.stack(layer_finals, dim=0)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, h_n


class MambaConv(BaseConvSequenceModel):
    """
    CNN encoder + Mamba sequence model + existing classifier heads.

    This preserves the same external model I/O as RNNConv:
    forward(frames) -> (char_out, pos_out), where frames has shape (B, T, C, H, W).
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        mamba_d_model=MAMBA_DEFAULT_D_MODEL,
        max_chars=15,
        predict_all_chars=False,
        mamba_num_layers=1,
        mamba_dropout=0.0,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_block_type="mamba",
        mamba_residual=True,
    ):
        super(MambaConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=mamba_d_model,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.mamba_d_model = mamba_d_model
        self.rnn = MambaRNNWrapper(
            input_size=self.encoder_flatten_size,
            d_model=mamba_d_model,
            num_layers=mamba_num_layers,
            batch_first=True,
            dropout=mamba_dropout,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            block_type=mamba_block_type,
            residual=mamba_residual,
        )
        self.LNormRNN = nn.LayerNorm(mamba_d_model)
        self.to(self.device)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.rnn_dropout, training=self.training)
        return x
