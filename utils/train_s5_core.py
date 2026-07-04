import importlib

# Import NumPy before torch on macOS/conda to avoid duplicate libomp crashes in
# S5's HiPPO initialization path.
importlib.import_module("numpy")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


S5_DEFAULT_D_MODEL = 256
S5_DEFAULT_STATE_SIZE = 128


def _get_s5_layer():
    try:
        from s5 import S5
    except ImportError as exc:
        raise ImportError(
            "S5Conv requires the optional dependency 's5-pytorch'. "
            "Install it with: pip install s5-pytorch"
        ) from exc
    return S5


class S5RNNWrapper(nn.Module):
    """
    Adapter that makes a stack of S5 layers look like an nn.RNN-style module.

    S5 consumes and returns batch-first tensors shaped (B, T, d_model). It does
    not expose a training-time recurrent state, so h_n is represented by each
    layer's last output token, matching MambaRNNWrapper.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        state_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if bidirectional:
            raise ValueError("S5RNNWrapper does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_size = d_model
        self.d_model = d_model
        self.state_size = state_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.residual = residual

        self.input_proj = (
            nn.Linear(input_size, d_model)
            if input_size != d_model
            else nn.Identity()
        )
        layer_cls = _get_s5_layer()
        self.layers = nn.ModuleList(
            [layer_cls(d_model, state_size) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _is_autocast_enabled(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type=device_type))
        except TypeError:
            return bool(torch.is_autocast_enabled())

    def forward(self, x: torch.Tensor, h_0=None):
        if not self.batch_first:
            x = x.transpose(0, 1)

        x = self.input_proj(x)
        layer_finals = []
        autocast_active = self._is_autocast_enabled(x.device.type)
        for layer_idx, layer in enumerate(self.layers):
            residual = x
            if autocast_active:
                # S5 uses vmap + associative_scan internally and is not stable under autocast.
                # Run only the S5 kernel in fp32, then cast back for the surrounding AMP graph.
                with torch.autocast(device_type=x.device.type, enabled=False):
                    x = layer(x.float())
                x = x.to(residual.dtype)
            else:
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


class S5Conv(BaseConvSequenceModel):
    """
    CNN encoder + S5 sequence model + existing classifier heads.

    This preserves the same external model I/O as RNNConv and MambaConv:
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
        s5_d_model=S5_DEFAULT_D_MODEL,
        max_chars=15,
        predict_all_chars=False,
        s5_num_layers=1,
        s5_dropout=0.0,
        s5_state_size=S5_DEFAULT_STATE_SIZE,
        s5_residual=True,
    ):
        super(S5Conv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=s5_d_model,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.s5_d_model = s5_d_model
        self.s5_state_size = s5_state_size
        self.rnn = S5RNNWrapper(
            input_size=self.encoder_flatten_size,
            d_model=s5_d_model,
            state_size=s5_state_size,
            num_layers=s5_num_layers,
            batch_first=True,
            dropout=s5_dropout,
            residual=s5_residual,
        )
        self.LNormRNN = nn.LayerNorm(s5_d_model)
        self.to(self.device)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.rnn_dropout, training=self.training)
        return x
