"""Clutter sequence models built from a task encoder, recurrent core, and heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..recurrent_cores.additive_feedback import (
    AdditiveFeedbackRNNCore,
    ConcatenatedFeedbackCellCore,
)
from ..recurrent_cores.gawf import GaWFCore
from ..recurrent_cores.gawf_legacy import GaWFCoreLegacy
from ..recurrent_cores.brims import BRIMsCore
from ..recurrent_cores.hyper_lstm import HyperLSTMCore
from ..recurrent_cores.mlstm import MLSTMCore
from ..recurrent_cores.rnn import (
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
        output_wrap="ln_relu_dropout",
        rnn_activation="tanh",
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
            output_wrap=output_wrap,
            rnn_activation=rnn_activation,
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
        output_wrap="ln_relu_dropout",
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
            output_wrap=output_wrap,
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
        output_wrap="ln_relu_dropout",
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
            output_wrap=output_wrap,
        )
        self.to(self.device)


class MLSTMConv(ClutterSequenceModel):
    """Clutter CNN encoder + multiplicative LSTM core + shared heads."""

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        hidden_size: int = 256,
        max_chars: int = 15,
        predict_all_chars: bool = False,
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
        self.num_layers = 1
        self.core = MLSTMCore(self.encoder_flatten_size, hidden_size, dropout=rnn_dropout)
        self.to(self.device)


class HyperLSTMConv(ClutterSequenceModel):
    """Clutter CNN encoder + vendored labml HyperLSTM core + shared heads."""

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        hidden_size: int = 256,
        hyper_hidden_size: int = 10,
        hyper_embedding_size: int = 4,
        max_chars: int = 15,
        predict_all_chars: bool = False,
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
        self.num_layers = 1
        self.hyper_hidden_size = int(hyper_hidden_size)
        self.hyper_embedding_size = int(hyper_embedding_size)
        self.core = HyperLSTMCore(
            self.encoder_flatten_size,
            hidden_size,
            hyper_hidden_size=self.hyper_hidden_size,
            hyper_embedding_size=self.hyper_embedding_size,
            dropout=rnn_dropout,
        )
        self.to(self.device)


class BRIMsConv(ClutterSequenceModel):
    """Clutter CNN encoder + clean-room two-layer BRIMs core + shared heads."""

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        hidden_size: int = 84,
        brims_num_blocks: tuple[int, int] = (6, 3),
        brims_topk: tuple[int, int] = (4, 2),
        brims_input_attention_heads: int = 4,
        brims_input_attention_key_size: int = 64,
        brims_communication_attention_heads: int = 4,
        brims_communication_attention_key_size: int = 32,
        brims_communication_attention_value_size: int = 32,
        brims_attention_dropout: float = 0.1,
        max_chars: int = 15,
        predict_all_chars: bool = False,
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
        self.num_layers = 2
        self.brims_num_blocks = tuple(int(value) for value in brims_num_blocks)
        self.brims_topk = tuple(int(value) for value in brims_topk)
        self.core = BRIMsCore(
            self.encoder_flatten_size,
            hidden_size,
            num_blocks=self.brims_num_blocks,
            topk=self.brims_topk,
            input_attention_heads=brims_input_attention_heads,
            input_attention_key_size=brims_input_attention_key_size,
            communication_attention_heads=brims_communication_attention_heads,
            communication_attention_key_size=brims_communication_attention_key_size,
            communication_attention_value_size=(
                brims_communication_attention_value_size
            ),
            attention_dropout=brims_attention_dropout,
            dropout=rnn_dropout,
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
        output_wrap="ln_relu_dropout",
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
        from ..recurrent_cores.mamba import MambaCore

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
            output_wrap=output_wrap,
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
        output_wrap="ln_relu_dropout",
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
        from ..recurrent_cores.s5 import S5Core

        self.core = S5Core(
            input_size=self.encoder_flatten_size,
            d_model=s5_d_model,
            state_size=s5_state_size,
            num_layers=s5_num_layers,
            dropout=s5_dropout,
            output_dropout=rnn_dropout,
            residual=s5_residual,
            output_wrap=output_wrap,
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
        output_wrap="ln_relu_dropout",
        rnn_activation="tanh",
        gawf_core="rnn_aligned",
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
        if gawf_core not in ("rnn_aligned", "legacy"):
            raise ValueError(f"Unsupported gawf_core: {gawf_core!r}")
        self.gawf_core = str(gawf_core)
        core_class = GaWFCore if self.gawf_core == "rnn_aligned" else GaWFCoreLegacy
        self.core = core_class(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            feedback_dim=requested_feedback_dim,
            dropout=rnn_dropout,
            output_wrap=output_wrap,
            rnn_activation=rnn_activation,
        )
        self.proj_out = (
            nn.Linear(self.output_feedback_dim, requested_feedback_dim)
            if feedback_dim is not None
            else None
        )
        self.register_buffer("prev_feedback", None, persistent=False)
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
            char_t, pos_t = self.classifier(self.core.project_readout(h))
            if self.proj_out is None:
                with torch.no_grad():
                    fb = self._compute_feedback(char_t, pos_t)
            else:
                fb = self._compute_feedback(char_t, pos_t)
            char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

        self.prev_feedback = fb.detach().to(dtype=torch.float32)
        return char_out, pos_out


class FeedbackControlConv(GaWFRNNConv):
    """Shared closed-loop wrapper for non-multiplicative output-feedback controls."""

    is_gawf_model = False
    is_gawf_multi_model = False
    is_feedback_control_model = True

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        *,
        hidden_size: int,
        core_class: type[nn.Module],
        core_kwargs: dict[str, object] | None = None,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        max_chars: int = 15,
        predict_all_chars: bool = False,
    ) -> None:
        if predict_all_chars:
            raise ValueError("Feedback controls require char and sector prediction heads.")
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
            predict_all_chars=False,
        )
        self.output_feedback_dim = self.num_classes + self.num_pos
        self.core = core_class(
            input_size=self.encoder_flatten_size,
            hidden_size=hidden_size,
            feedback_dim=self.output_feedback_dim,
            dropout=rnn_dropout,
            **(core_kwargs or {}),
        )
        if int(self.core.feedback_dim) != self.output_feedback_dim:
            raise ValueError(
                "Feedback control core dimension must equal the concatenated head logits: "
                f"expected {self.output_feedback_dim}, got {self.core.feedback_dim}."
            )
        self.proj_out = None
        self.register_buffer("prev_feedback", None, persistent=False)
        self.to(self.device)

    @property
    def feedback_dim(self) -> int:
        return self.output_feedback_dim

    @property
    def rnn(self) -> nn.Module:
        if hasattr(self.core, "rnn"):
            return self.core.rnn
        return self.core.cell

    @property
    def LNormRNN(self) -> nn.LayerNorm:
        return self.core.norm

    def feedback_initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Any:
        """Return the recurrent state used by feedback-ablation rollouts."""
        return self.core.initial_state(batch_size, device, dtype)

    def feedback_step(
        self,
        x_t: torch.Tensor,
        state: Any,
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        """Advance one recurrent step for feedback-ablation rollouts."""
        return self.core.step(x_t, state, feedback)

    def forward(
        self,
        x: torch.Tensor,
        use_feedback: bool = True,
        reset_feedback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_frames(x)
        batch_size, frame_num = encoded.shape[:2]
        if not use_feedback:
            self.prev_feedback = None
            hidden, _state = self.core.forward_no_feedback(encoded)
            return self.classifier(hidden)

        if reset_feedback or self.prev_feedback is None:
            feedback = torch.zeros(
                batch_size,
                self.feedback_dim,
                device=encoded.device,
                dtype=torch.float32,
            )
        else:
            feedback = self.prev_feedback.to(device=encoded.device, dtype=torch.float32)

        state = self.core.initial_state(batch_size, encoded.device, encoded.dtype)
        char_outputs: list[torch.Tensor] = []
        pos_outputs: list[torch.Tensor] = []
        for t in range(frame_num):
            hidden, state = self.core.step(encoded[:, t, :], state, feedback)
            char_t, pos_t = self.classifier(hidden)
            with torch.no_grad():
                feedback = self._compute_feedback(char_t, pos_t)
            char_outputs.append(char_t)
            pos_outputs.append(pos_t)

        self.prev_feedback = feedback.detach().to(dtype=torch.float32)
        return torch.stack(char_outputs, dim=1), torch.stack(pos_outputs, dim=1)


class GaWFAdditiveConv(FeedbackControlConv):
    """GaWF control with additive output feedback and no multiplicative gate."""

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        hidden_size: int = 271,
        max_chars: int = 15,
        predict_all_chars: bool = False,
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            hidden_size=hidden_size,
            core_class=AdditiveFeedbackRNNCore,
            core_kwargs={"initial_weight_scale": 0.5},
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )


class ConcatenatedFeedbackConv(FeedbackControlConv):
    """Shared wrapper for RNN/GRU/LSTM controls receiving concatenated feedback."""

    cell_type = "rnn"

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        hidden_size: int = 256,
        max_chars: int = 15,
        predict_all_chars: bool = False,
    ) -> None:
        super().__init__(
            num_classes,
            num_pos,
            hidden_size=hidden_size,
            core_class=ConcatenatedFeedbackCellCore,
            core_kwargs={"cell_type": self.cell_type},
            kernel_size=kernel_size,
            device=device,
            input_channels=input_channels,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )


class RNNFeedbackConv(ConcatenatedFeedbackConv):
    """Vanilla RNN control with previous detached logits concatenated to each input."""

    cell_type = "rnn"


class GRUFeedbackConv(ConcatenatedFeedbackConv):
    """GRU control with previous detached logits concatenated to each input."""

    cell_type = "gru"


class LSTMFeedbackConv(ConcatenatedFeedbackConv):
    """LSTM control with previous detached logits concatenated to each input."""

    cell_type = "lstm"


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
        gawf_core="rnn_aligned",
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
        if gawf_core not in ("rnn_aligned", "legacy"):
            raise ValueError(f"Unsupported gawf_core: {gawf_core!r}")
        self.gawf_core = str(gawf_core)
        core_class = GaWFCore if self.gawf_core == "rnn_aligned" else GaWFCoreLegacy
        self.core = core_class(
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
        self.register_buffer("prev_feedback", None, persistent=False)
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


# --- Nonlinearity-placement ablation variants -----------------------------------------------
#
# These wrappers keep the shared encoder and heads and only change where nonlinearities sit:
#   *_nowrap  removes the outer LayerNorm -> ReLU -> dropout wrap from the sequence core.
#   *notanh   keeps the outer wrap and removes the activation inside the recurrence, so the
#             vanilla-RNN recurrence becomes linear.
# The default model types are untouched, and every variant only sets a non-default core mode.


class RNNNoWrapConv(RNNConv):
    """Vanilla RNN with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class GRUNoWrapConv(GRUConv):
    """GRU with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class LSTMNoWrapConv(LSTMConv):
    """LSTM with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class MambaNoWrapConv(MambaConv):
    """Mamba with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class S5NoWrapConv(S5Conv):
    """S5 with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class GaWFNoWrapConv(GaWFRNNConv):
    """GaWF with the outer LayerNorm, ReLU, and dropout wrap removed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("output_wrap", "none")
        super().__init__(*args, **kwargs)


class GaWFNoTanhConv(GaWFRNNConv):
    """GaWF that keeps the outer wrap and removes the tanh activation inside the recurrence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rnn_activation", "identity")
        super().__init__(*args, **kwargs)


class RNNNoTanhConv(RNNConv):
    """Vanilla RNN with the built-in activation removed, leaving a linear recurrence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rnn_activation", "identity")
        super().__init__(*args, **kwargs)


class GaWFLegacyConv(GaWFRNNConv):
    """Historical GaWF: the wrapped value was fed back into the recurrence.

    This type exists only to reproduce artifacts trained before the RNN alignment; it must not be
    used for new comparisons against ``rnn``/``lstm``/``gru``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("gawf_core", "legacy")
        super().__init__(*args, **kwargs)
