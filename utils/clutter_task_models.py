"""Clutter sequence models built from a task encoder, recurrent core, and heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.rnn import (
    GRUCore,
    LSTMCore,
    RNNCore,
)

MAMBA_DEFAULT_D_MODEL = 170
S5_DEFAULT_D_MODEL = 256
S5_DEFAULT_STATE_SIZE = 128


class ClutterCNNEncoder(nn.Module):
    """CNN encoder for clutter frame sequences."""

    def __init__(
        self,
        kernel_size: int = 3,
        cnn_dropout: float = 0.0,
        input_channels: int = 2,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        self.cnn_dropout = float(cnn_dropout)
        self.input_channels = int(input_channels)
        out_ch, out_h, out_w = 64, 12, 12
        mp2_k, mp2_s = 4, 4
        self.conv1 = nn.Conv2d(
            self.input_channels,
            32,
            kernel_size=kernel_size,
            padding="same",
        )
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=mp2_k, stride=mp2_s)
        self.LNorm2 = nn.LayerNorm([out_ch, out_h, out_w])
        reduced_ch = max(8, out_ch // 2)
        reduced_h, reduced_w = out_h // 2, out_w // 2
        self.conv_reduce = nn.Conv2d(out_ch, reduced_ch, kernel_size=1)
        self.pool_reduce = nn.AdaptiveAvgPool2d((reduced_h, reduced_w))
        self.output_size = reduced_ch * reduced_h * reduced_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.MP1(x)
        x = self.LNorm1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=self.cnn_dropout, training=self.training)
        x = self.conv2(x)
        x = self.MP2(x)
        x = self.LNorm2(x)
        x = self.conv_reduce(x)
        x = F.relu(x)
        x = self.pool_reduce(x)
        x = F.dropout2d(x, p=self.cnn_dropout, training=self.training)
        return x


class ClutterCharPosHead(nn.Module):
    """Task heads for foreground char/position or all-character prediction."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_pos: int,
        max_chars: int = 15,
        predict_all_chars: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.num_pos = int(num_pos)
        self.max_chars = int(max_chars)
        self.predict_all_chars = bool(predict_all_chars)
        if self.predict_all_chars:
            self.fcchars = nn.Linear(self.hidden_size, self.max_chars * self.num_classes)
            self.fcpos = None
        else:
            self.fcchar = nn.Linear(self.hidden_size, self.num_classes)
            self.fcpos = nn.Linear(self.hidden_size, self.num_pos)

    def forward(self, x: torch.Tensor):
        if self.predict_all_chars:
            chars_out = self.fcchars(x)
            batch_size, frame_num = chars_out.shape[:2]
            num_classes = chars_out.shape[-1] // self.max_chars
            chars_out = chars_out.view(batch_size, frame_num, self.max_chars, num_classes)
            return chars_out, None
        return self.fcchar(x), self.fcpos(x)


class ClutterSequenceModel(nn.Module):
    """Base clutter model: frames -> CNN encoder -> recurrent core -> task heads."""

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        sequence_width: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        max_chars: int = 15,
        predict_all_chars: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_channels = int(input_channels)
        self.cnn_dropout = float(cnn_dropout)
        self.rnn_dropout = float(rnn_dropout)
        self.max_chars = int(max_chars)
        self.predict_all_chars = bool(predict_all_chars)
        self.num_classes = int(num_classes)
        self.num_pos = int(num_pos)
        self.hidden_size = int(sequence_width)
        self.encoder_module = ClutterCNNEncoder(
            kernel_size=kernel_size,
            cnn_dropout=cnn_dropout,
            input_channels=self.input_channels,
        )
        self.encoder_flatten_size = self.encoder_module.output_size
        self.head = ClutterCharPosHead(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            num_pos=self.num_pos,
            max_chars=self.max_chars,
            predict_all_chars=self.predict_all_chars,
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_module(x)

    def classifier(self, x: torch.Tensor):
        return self.head(x)

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        return x.view(batch_size, frame_num, -1)

    def middle(self, x: torch.Tensor) -> torch.Tensor:
        out, _state = self.core(x)
        return out

    def forward(self, x: torch.Tensor):
        encoded = self.encode_frames(x)
        hidden = self.middle(encoded)
        return self.classifier(hidden)


class RNNConv(ClutterSequenceModel):
    """Clutter CNN encoder + task-agnostic RNN core + char/position heads."""

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        num_layers=1,
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            hidden_size,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.num_layers = int(num_layers)
        self.core = RNNCore(
            self.encoder_flatten_size,
            hidden_size,
            dropout=rnn_dropout,
            num_layers=self.num_layers,
        )
        self.to(self.device)

    @property
    def rnn(self):
        return self.core.rnn if self.num_layers == 1 else self.core.rnns

    @property
    def LNormRNN(self):
        return self.core.norm if self.num_layers == 1 else self.core.norms


class GRUConv(RNNConv):
    """Clutter CNN encoder + task-agnostic GRU core + char/position heads."""

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        num_layers=1,
    ) -> None:
        ClutterSequenceModel.__init__(
            self,
            num_classes,
            num_pos,
            hidden_size,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.num_layers = int(num_layers)
        self.core = GRUCore(
            self.encoder_flatten_size,
            hidden_size,
            dropout=rnn_dropout,
            num_layers=self.num_layers,
        )
        self.to(self.device)


class LSTMConv(RNNConv):
    """Clutter CNN encoder + task-agnostic LSTM core + char/position heads."""

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        num_layers=1,
    ) -> None:
        ClutterSequenceModel.__init__(
            self,
            num_classes,
            num_pos,
            hidden_size,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.num_layers = int(num_layers)
        self.core = LSTMCore(
            self.encoder_flatten_size,
            hidden_size,
            dropout=rnn_dropout,
            num_layers=self.num_layers,
        )
        self.to(self.device)


class MambaConv(ClutterSequenceModel):
    """Clutter CNN encoder + task-agnostic Mamba core + char/position heads."""

    uses_mamba_core = True

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
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
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            mamba_d_model,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.mamba_d_model = int(mamba_d_model)
        from .recurrent_cores.mamba import MambaCore

        self.core = MambaCore(
            input_size=self.encoder_flatten_size,
            d_model=mamba_d_model,
            num_layers=mamba_num_layers,
            dropout=mamba_dropout,
            output_dropout=rnn_dropout,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            block_type=mamba_block_type,
            residual=mamba_residual,
        )
        self.to(self.device)

    @property
    def rnn(self):
        return self.core

    @property
    def LNormRNN(self):
        return self.core.norm


class S5Conv(ClutterSequenceModel):
    """Clutter CNN encoder + task-agnostic S5 core + char/position heads."""

    uses_s5_core = True

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        s5_d_model=S5_DEFAULT_D_MODEL,
        max_chars=15,
        predict_all_chars=False,
        s5_num_layers=1,
        s5_dropout=0.0,
        s5_state_size=S5_DEFAULT_STATE_SIZE,
        s5_residual=True,
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            s5_d_model,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.s5_d_model = int(s5_d_model)
        self.s5_state_size = int(s5_state_size)
        from .recurrent_cores.s5 import S5Core

        self.core = S5Core(
            input_size=self.encoder_flatten_size,
            d_model=s5_d_model,
            state_size=s5_state_size,
            num_layers=s5_num_layers,
            dropout=s5_dropout,
            output_dropout=rnn_dropout,
            residual=s5_residual,
        )
        self.to(self.device)

    @property
    def rnn(self):
        return self.core

    @property
    def LNormRNN(self):
        return self.core.norm


class GaWFRNNConv(ClutterSequenceModel):
    """Clutter CNN encoder + task-agnostic single-layer GaWF core."""

    is_gawf_model = True
    is_gawf_multi_model = False

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        feedback_dim=None,
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            hidden_size,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=False,
        )
        self.output_feedback_dim = self.num_classes + self.num_pos
        requested_feedback_dim = (
            self.output_feedback_dim if feedback_dim is None else int(feedback_dim)
        )
        self.core = GaWFCore(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            feedback_dim=requested_feedback_dim,
            dropout=rnn_dropout,
        )
        self.proj_out = (
            nn.Linear(self.output_feedback_dim, requested_feedback_dim)
            if feedback_dim is not None
            else None
        )
        self.register_buffer("prev_feedback", None)
        self.to(self.device)

    @property
    def feedback_dim(self) -> int:
        return self.core.feedback_dim

    @property
    def gate_tau(self) -> float:
        return self.core.gate_tau

    @property
    def rnn(self):
        return self.core.rnn

    @property
    def LNormRNN(self):
        return self.core.norm

    @property
    def U(self):
        return self.core.U

    @property
    def V(self):
        return self.core.V

    def begin_gawf_diagnostics(self, gate_saturation_eps: float = 0.01) -> None:
        self.core.begin_gawf_diagnostics(gate_saturation_eps)

    def pop_gawf_diagnostics(self):
        return self.core.pop_gawf_diagnostics()

    def set_feedback_frozen(self, freeze: bool) -> None:
        self.core.set_feedback_frozen(freeze)
        if self.proj_out is not None:
            for param in self.proj_out.parameters():
                param.requires_grad = not freeze

    def _compute_feedback(self, char_t: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
        y_t = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is None:
            return y_t
        return self.proj_out(y_t)

    def middle_gawf(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        fb_t: torch.Tensor,
    ) -> torch.Tensor:
        feedback = fb_t.squeeze(2) if fb_t.ndim == 3 else fb_t
        return self.core.step(x_t, h_prev, feedback)

    def forward(self, x: torch.Tensor, use_feedback=True, reset_feedback=False):
        encoded = self.encode_frames(x)
        batch_size, frame_num = encoded.shape[:2]
        if not use_feedback:
            self.prev_feedback = None
            hidden, _state = self.core.forward_no_feedback(encoded)
            return self.classifier(hidden)

        if reset_feedback or self.prev_feedback is None:
            fb = torch.zeros(
                batch_size,
                self.feedback_dim,
                device=encoded.device,
                dtype=torch.float32,
            )
        else:
            fb = self.prev_feedback.to(device=encoded.device, dtype=torch.float32)

        char_out = torch.empty(
            batch_size,
            frame_num,
            self.num_classes,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        pos_out = torch.empty(
            batch_size,
            frame_num,
            self.num_pos,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        h = self.core.initial_state(batch_size, encoded.device, encoded.dtype)
        for t in range(frame_num):
            h = self.core.step(encoded[:, t, :], h, fb)
            char_t, pos_t = self.classifier(h)
            if self.proj_out is None:
                with torch.no_grad():
                    fb = self._compute_feedback(char_t, pos_t)
            else:
                fb = self._compute_feedback(char_t, pos_t)
            char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

        self.prev_feedback = fb.detach().to(dtype=torch.float32)
        return char_out, pos_out


class MultiLayerGaWFRNNConv(ClutterSequenceModel):
    """Clutter CNN encoder + task-agnostic multi-layer GaWF core."""

    is_gawf_model = True
    is_gawf_multi_model = True

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        input_channels=2,
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        feedback_dim=None,
        num_layers=2,
    ) -> None:
        if predict_all_chars:
            raise ValueError(
                "MultiLayerGaWFRNNConv currently supports single-character heads only."
            )
        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError(f"MultiLayerGaWFRNNConv requires num_layers >= 2, got {num_layers}")
        super().__init__(
            num_classes,
            num_pos,
            hidden_size,
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.output_feedback_dim = self.num_classes + self.num_pos
        requested_feedback_dim = None if feedback_dim is None else int(feedback_dim)
        if requested_feedback_dim is not None and requested_feedback_dim < 0:
            raise ValueError(f"feedback_dim must be >= 0, got {requested_feedback_dim}")
        self.use_feedback_projector = (
            requested_feedback_dim is not None and requested_feedback_dim > 0
        )
        self.feedback_dim = requested_feedback_dim if self.use_feedback_projector else 0
        self.layer_feedback_dims = (
            [self.feedback_dim] * self.num_layers
            if self.use_feedback_projector
            else [hidden_size] * (self.num_layers - 1) + [self.output_feedback_dim]
        )
        self.top_feedback_dim = self.layer_feedback_dims[-1]
        self.core = GaWFCore(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            feedback_dim=self.layer_feedback_dims[-1],
            layer_feedback_dims=self.layer_feedback_dims,
            dropout=rnn_dropout,
            num_layers=self.num_layers,
        )
        if self.use_feedback_projector:
            self.hidden_projectors = nn.ModuleList(
                [nn.Linear(hidden_size, self.feedback_dim) for _ in range(self.num_layers - 1)]
            )
            self.proj_out = nn.Linear(self.output_feedback_dim, self.feedback_dim)
        else:
            self.hidden_projectors = nn.ModuleList()
            self.proj_out = None
        self.register_buffer("prev_feedback", None)
        self.to(self.device)

    @property
    def rnns(self):
        return self.core.rnns

    @property
    def LNormRNN(self):
        return self.core.norms

    @property
    def U_layers(self):
        return self.core.U_layers

    @property
    def V_layers(self):
        return self.core.V_layers

    def begin_gawf_diagnostics(self, gate_saturation_eps: float = 0.01) -> None:
        self.core.begin_gawf_diagnostics(gate_saturation_eps)

    def pop_gawf_diagnostics(self):
        return self.core.pop_gawf_diagnostics()

    def set_feedback_frozen(self, freeze: bool) -> None:
        self.core.set_feedback_frozen(freeze)
        for param in self.hidden_projectors.parameters():
            param.requires_grad = not freeze
        if self.proj_out is not None:
            for param in self.proj_out.parameters():
                param.requires_grad = not freeze

    def _compute_output_feedback(self, char_t: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
        y_t = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is None:
            return y_t
        return self.proj_out(y_t)

    def _feedbacks_for_layers(
        self,
        h_states: list[torch.Tensor],
        fb_top: torch.Tensor,
    ) -> list[torch.Tensor]:
        feedbacks = []
        for layer_idx in range(self.num_layers):
            if layer_idx == self.num_layers - 1:
                feedbacks.append(fb_top)
            elif self.use_feedback_projector:
                feedbacks.append(self.hidden_projectors[layer_idx](h_states[layer_idx + 1]))
            else:
                feedbacks.append(h_states[layer_idx + 1].detach())
        return feedbacks

    def forward(self, x: torch.Tensor, use_feedback=True, reset_feedback=False):
        encoded = self.encode_frames(x)
        batch_size, frame_num = encoded.shape[:2]
        if not use_feedback:
            self.prev_feedback = None
            hidden, _states = self.core.forward_no_feedback(encoded)
            return self.classifier(hidden)

        if reset_feedback or self.prev_feedback is None:
            fb_top = torch.zeros(
                batch_size,
                self.top_feedback_dim,
                device=encoded.device,
                dtype=torch.float32,
            )
        else:
            fb_top = self.prev_feedback.to(device=encoded.device, dtype=torch.float32)

        char_out = torch.empty(
            batch_size,
            frame_num,
            self.num_classes,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        pos_out = torch.empty(
            batch_size,
            frame_num,
            self.num_pos,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        h_states = self.core.initial_state(batch_size, encoded.device, encoded.dtype)
        for t in range(frame_num):
            feedbacks = self._feedbacks_for_layers(h_states, fb_top)
            layer_output, h_states = self.core.step(encoded[:, t, :], h_states, feedbacks)
            char_t, pos_t = self.classifier(layer_output)
            if self.proj_out is None:
                with torch.no_grad():
                    fb_top = self._compute_output_feedback(char_t, pos_t)
            else:
                fb_top = self._compute_output_feedback(char_t, pos_t)
            char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

        self.prev_feedback = fb_top.detach().to(dtype=torch.float32)
        return char_out, pos_out
