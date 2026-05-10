import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


class DiagonalSSMLayer(nn.Module):
    """
    Real-valued diagonal state-space layer with an nn.RNN-like sequence contract.

    This is a lightweight SSM/LRU-style recurrent layer implemented with standard
    PyTorch ops. It is intended as a dependency-free baseline for comparing SSM
    dynamics against RNN/GRU/LSTM under the existing training pipeline.
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        min_decay: float = 0.1,
        max_decay: float = 0.99,
        activation: str = "silu",
    ):
        super().__init__()
        if not 0.0 < min_decay < max_decay < 1.0:
            raise ValueError("Expected 0 < min_decay < max_decay < 1")

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.activation = activation

        self.in_proj = nn.Linear(hidden_size, state_size)
        self.out_proj = nn.Linear(state_size, hidden_size)
        self.skip_scale = nn.Parameter(torch.ones(hidden_size))

        init_decay = torch.linspace(min_decay, max_decay, state_size)
        init_logit = torch.logit((init_decay - min_decay) / (max_decay - min_decay))
        self.decay_logit = nn.Parameter(init_logit)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "identity":
            return x
        raise ValueError("activation must be one of: silu, gelu, tanh, identity")

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, _ = x.shape
        if state is None:
            state = x.new_zeros(B, self.state_size)

        decay = self.min_decay + (self.max_decay - self.min_decay) * torch.sigmoid(
            self.decay_logit
        )
        input_scale = 1.0 - decay
        outputs = []

        for t in range(T):
            u_t = self._activate(self.in_proj(x[:, t, :]))
            state = decay.unsqueeze(0) * state + input_scale.unsqueeze(0) * u_t
            y_t = self.out_proj(state) + self.skip_scale.unsqueeze(0) * x[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), state


class SSMRNNWrapper(nn.Module):
    """
    Stack diagonal SSM layers behind the same forward shape used by nn.RNN.

    Input/Output:
      - batch_first=True:  (B, T, input_size) -> ((B, T, hidden_size), h_n)
      - batch_first=False: (T, B, input_size) -> ((T, B, hidden_size), h_n)

    h_n has shape (num_layers, B, hidden_size), matching the common RNN/GRU
    convention closely enough for code that only consumes the first return value.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        state_size: int | None = None,
        min_decay: float = 0.1,
        max_decay: float = 0.99,
        activation: str = "silu",
        residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        if bidirectional:
            raise ValueError("SSMRNNWrapper does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.residual = residual
        state_size = hidden_size if state_size is None else state_size

        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                DiagonalSSMLayer(
                    hidden_size=hidden_size,
                    state_size=state_size,
                    min_decay=min_decay,
                    max_decay=max_decay,
                    activation=activation,
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
            x, _state = layer(x)
            if self.residual:
                x = x + residual
            layer_finals.append(x[:, -1, :])
            if layer_idx < self.num_layers - 1:
                x = self.dropout(x)

        h_n = torch.stack(layer_finals, dim=0)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x, h_n


class SSMConv(BaseConvSequenceModel):
    """
    CNN encoder + diagonal SSM sequence model + existing classifier heads.

    This class keeps the same external forward I/O as RNNConv while replacing the
    middle recurrent computation with an SSM/LRU-style state update.
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        ssm_num_layers=1,
        ssm_dropout=0.0,
        ssm_state_size=None,
        ssm_min_decay=0.1,
        ssm_max_decay=0.99,
        ssm_activation="silu",
        ssm_residual=True,
    ):
        super(SSMConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.rnn = SSMRNNWrapper(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            num_layers=ssm_num_layers,
            batch_first=True,
            dropout=ssm_dropout,
            state_size=ssm_state_size,
            min_decay=ssm_min_decay,
            max_decay=ssm_max_decay,
            activation=ssm_activation,
            residual=ssm_residual,
        )
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.to(self.device)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.rnn_dropout, training=self.training)
        return x
