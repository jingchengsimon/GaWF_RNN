import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


DIAGLTI_DEFAULT_D_MODEL = 256
DIAGLTI_DEFAULT_STATE_SIZE = 189


class DiagonalLTILayer(nn.Module):
    """
    Real-valued diagonal LTI / EMA layer with an nn.RNN-like sequence contract.

    This is a lightweight dependency-free baseline built from learnable
    multi-timescale exponential moving averages. It is NOT an S4/S5-family SSM:
    it has real diagonal dynamics, a serial for-loop update, and no HiPPO or
    complex eigenvalue parameterization.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int,
        min_decay: float = 0.1,
        max_decay: float = 0.99,
        activation: str = "silu",
    ):
        super().__init__()
        if not 0.0 < min_decay < max_decay < 1.0:
            raise ValueError("Expected 0 < min_decay < max_decay < 1")

        self.d_model = d_model
        self.state_size = state_size
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.activation = activation

        self.in_proj = nn.Linear(d_model, state_size)
        self.out_proj = nn.Linear(state_size, d_model)
        self.skip_scale = nn.Parameter(torch.ones(d_model))

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


class DiagLTIWrapper(nn.Module):
    """
    Stack diagonal LTI / EMA layers behind the same forward shape used by nn.RNN.

    Input/Output:
      - batch_first=True:  (B, T, input_size) -> ((B, T, hidden_size), h_n)
      - batch_first=False: (T, B, input_size) -> ((T, B, hidden_size), h_n)

    h_n has shape (num_layers, B, hidden_size), matching the common RNN/GRU
    convention closely enough for code that only consumes the first return value.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
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
            raise ValueError("DiagLTIWrapper does not support bidirectional=True")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden_size = d_model
        self.d_model = d_model
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.residual = residual
        state_size = d_model if state_size is None else state_size

        self.input_proj = (
            nn.Linear(input_size, d_model)
            if input_size != d_model
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                DiagonalLTILayer(
                    d_model=d_model,
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


class DiagLTIConv(BaseConvSequenceModel):
    """
    CNN encoder + diagonal LTI / EMA sequence model + existing classifier heads.

    This class keeps the same external forward I/O as RNNConv while replacing the
    middle recurrent computation with a real-valued diagonal LTI baseline. It is
    NOT an S4/S5-family SSM.
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        diaglti_d_model=DIAGLTI_DEFAULT_D_MODEL,
        max_chars=15,
        predict_all_chars=False,
        diaglti_num_layers=1,
        diaglti_dropout=0.0,
        diaglti_state_size=DIAGLTI_DEFAULT_STATE_SIZE,
        diaglti_min_decay=0.1,
        diaglti_max_decay=0.99,
        diaglti_activation="silu",
        diaglti_residual=True,
    ):
        super(DiagLTIConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=diaglti_d_model,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.diaglti_d_model = diaglti_d_model
        self.diaglti_state_size = diaglti_state_size
        self.rnn = DiagLTIWrapper(
            input_size=self.encoder_flatten_size,
            d_model=diaglti_d_model,
            num_layers=diaglti_num_layers,
            batch_first=True,
            dropout=diaglti_dropout,
            state_size=diaglti_state_size,
            min_decay=diaglti_min_decay,
            max_decay=diaglti_max_decay,
            activation=diaglti_activation,
            residual=diaglti_residual,
        )
        self.LNormRNN = nn.LayerNorm(diaglti_d_model)
        self.to(self.device)

    def middle(self, x):
        x = self.rnn(x)[0]
        x = self.LNormRNN(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.rnn_dropout, training=self.training)
        return x
