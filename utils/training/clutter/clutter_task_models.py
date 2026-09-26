"""Canonical Clutter models: encoder, one recurrent/state-space core, and task heads.

Each core drops its layer output while preserving its recurrent state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..recurrent_cores.gawf import GaWFCore
from ..recurrent_cores.additive_feedback import AdditiveFeedbackCellCore
from ..recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

MAMBA_DEFAULT_D_MODEL = 170
S5_DEFAULT_D_MODEL = 256
S5_DEFAULT_STATE_SIZE = 128


class ClutterCNNEncoder(nn.Module):
    """Encode each two-channel Clutter frame into 1152 features."""

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
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding="same")
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LNorm1 = nn.LayerNorm([32, 48, 48])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.MP2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.LNorm2 = nn.LayerNorm([64, 12, 12])
        self.conv_reduce = nn.Conv2d(64, 32, kernel_size=1)
        self.pool_reduce = nn.AdaptiveAvgPool2d((6, 6))
        self.output_size = 32 * 6 * 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoded frame tensor `(batch, 32, 6, 6)`."""

        x = F.dropout2d(
            F.relu(self.LNorm1(self.MP1(self.conv1(x)))),
            p=self.cnn_dropout,
            training=self.training,
        )
        x = self.conv_reduce(self.LNorm2(self.MP2(self.conv2(x))))
        return F.dropout2d(
            self.pool_reduce(F.relu(x)), p=self.cnn_dropout, training=self.training
        )


class ClutterCharPosHead(nn.Module):
    """Map sequence activity to identity and location logits."""

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return character logits and optional position logits."""

        if self.predict_all_chars:
            chars = self.fcchars(x)
            batch_size, frame_num = chars.shape[:2]
            chars = chars.view(batch_size, frame_num, self.max_chars, self.num_classes)
            return chars, None
        return self.fcchar(x), self.fcpos(x)


class ClutterSequenceModel(nn.Module):
    """Shared encoder/head shell for a batch-first sequence core."""

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
            input_channels=input_channels,
        )
        self.encoder_flatten_size = self.encoder_module.output_size
        self.head = ClutterCharPosHead(
            hidden_size=sequence_width,
            num_classes=num_classes,
            num_pos=num_pos,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one flattened batch of frames."""

        return self.encoder_module(x)

    def classifier(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the task heads to recurrent activity."""

        return self.head(x)

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `(batch, time, channels, height, width)` into a sequence."""

        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.shape
        encoded = self.encoder(x.reshape(batch_size * frame_num, channels, height, width))
        return encoded.reshape(batch_size, frame_num, -1)

    def middle(self, x: torch.Tensor) -> torch.Tensor:
        """Return the recurrent/state-space layer output sequence."""

        output, _state = self.core(x)
        return output

    def reset_sequence_state(self) -> None:
        """Reset cross-call runtime state; stateless cores need no action."""

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run encoder, sequence core, and task heads."""

        return self.classifier(self.middle(self.encode_frames(x)))


class RNNConv(ClutterSequenceModel):
    """Elman recurrence with LayerNorm-ReLU state and output dropout."""

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
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            num_classes, num_pos, hidden_size, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, predict_all_chars,
        )
        self.num_layers = int(num_layers)
        self.core = RNNCore(
            self.encoder_flatten_size,
            hidden_size,
            dropout=rnn_dropout,
            num_layers=num_layers,
        )
        self.to(self.device)

    @property
    def rnn(self) -> nn.Module:
        """Expose recurrent weights for checkpoint/analysis compatibility."""

        return self.core.rnn if self.num_layers == 1 else self.core.rnns

    @property
    def LNormRNN(self) -> nn.Module:
        """Expose in-loop LayerNorm for checkpoint/analysis compatibility."""

        return self.core.norm if self.num_layers == 1 else self.core.norms


class GRUConv(ClutterSequenceModel):
    """Native PyTorch GRU with explicit output dropout."""

    core_class = GRUCore

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
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            num_classes, num_pos, hidden_size, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, predict_all_chars,
        )
        self.num_layers = int(num_layers)
        self.core = self.core_class(
            self.encoder_flatten_size,
            hidden_size,
            dropout=rnn_dropout,
            num_layers=num_layers,
        )
        self.to(self.device)

    @property
    def rnn(self) -> nn.Module:
        """Expose the native recurrent module."""

        return self.core.rnn if self.num_layers == 1 else self.core.rnns


class LSTMConv(GRUConv):
    """Native PyTorch LSTM with explicit output dropout."""

    core_class = LSTMCore


class AdditiveFeedbackConv(ClutterSequenceModel):
    """Single-layer recurrent control with detached additive task-logit feedback."""

    is_gawf_model = False
    is_feedback_control_model = True
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
        if predict_all_chars:
            raise ValueError("Additive feedback requires character and position heads")
        super().__init__(
            num_classes, num_pos, hidden_size, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, False,
        )
        self.output_feedback_dim = self.num_classes + self.num_pos
        self.core = AdditiveFeedbackCellCore(
            self.encoder_flatten_size,
            hidden_size,
            self.output_feedback_dim,
            cell_type=self.cell_type,
            dropout=rnn_dropout,
        )
        self.register_buffer("prev_feedback", None, persistent=False)
        self.to(self.device)

    @property
    def feedback_dim(self) -> int:
        """Return the number of raw task logits fed back to the cell."""

        return self.output_feedback_dim

    @property
    def rnn(self) -> nn.Module:
        """Expose native cell parameters for feedback-control analyses."""

        return self.core.cell

    @property
    def LNormRNN(self) -> nn.LayerNorm | None:
        """Expose the RNN LayerNorm when present."""

        return self.core.norm

    def reset_sequence_state(self) -> None:
        """Start the next sequence with zero feedback."""

        self.prev_feedback = None

    def _compute_feedback(
        self, char_t: torch.Tensor, pos_t: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate raw character and position logits for the next step."""

        return torch.cat((char_t, pos_t), dim=-1)

    def feedback_initial_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the clean initial state for feedback-control rollouts."""

        return self.core.initial_state(batch_size, device, dtype)

    def feedback_step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Advance one recurrent step for feedback-control rollouts."""

        return self.core.step(x_t, state, feedback)

    def forward(
        self, x: torch.Tensor, use_feedback: bool = True, reset_feedback: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the detached previous-logit feedback loop."""

        encoded = self.encode_frames(x)
        batch_size, frame_num = encoded.shape[:2]
        if not use_feedback:
            self.prev_feedback = None
            output, _state = self.core.forward_no_feedback(encoded)
            char_logits, pos_logits = self.classifier(output)
            return char_logits, pos_logits

        if reset_feedback or self.prev_feedback is None:
            feedback = torch.zeros(
                batch_size, self.feedback_dim, device=encoded.device, dtype=torch.float32
            )
        else:
            feedback = self.prev_feedback.to(device=encoded.device, dtype=torch.float32)
            if feedback.size(0) != batch_size:
                raise ValueError("Cached feedback batch size differs; call reset_sequence_state")

        state = self.core.initial_state(batch_size, encoded.device, encoded.dtype)
        char_outputs: list[torch.Tensor] = []
        pos_outputs: list[torch.Tensor] = []
        for time_idx in range(frame_num):
            output, state = self.core.step(encoded[:, time_idx, :], state, feedback)
            char_t, pos_t = self.classifier(output)
            feedback = self._compute_feedback(char_t, pos_t).detach()
            char_outputs.append(char_t)
            pos_outputs.append(pos_t)
        self.prev_feedback = feedback.to(dtype=torch.float32)
        return torch.stack(char_outputs, dim=1), torch.stack(pos_outputs, dim=1)


class RNNAdditiveFeedbackConv(AdditiveFeedbackConv):
    """LayerNorm-ReLU RNN with additive feedback and output-only dropout."""

    cell_type = "rnn"


class GRUAdditiveFeedbackConv(AdditiveFeedbackConv):
    """Native GRU gates with additive feedback and output-only dropout."""

    cell_type = "gru"


class LSTMAdditiveFeedbackConv(AdditiveFeedbackConv):
    """Native LSTM gates with additive feedback and output-only dropout."""

    cell_type = "lstm"


class MambaConv(ClutterSequenceModel):
    """Projected residual Mamba stack with branch output dropout."""

    uses_mamba_core = True

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        mamba_d_model: int = MAMBA_DEFAULT_D_MODEL,
        max_chars: int = 15,
        predict_all_chars: bool = False,
        mamba_num_layers: int = 1,
        mamba_dropout: float = 0.0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__(
            num_classes, num_pos, mamba_d_model, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, predict_all_chars,
        )
        from ..recurrent_cores.mamba import MambaCore

        if mamba_dropout != 0.0:
            raise ValueError("mamba_dropout is obsolete; use rnn_dropout for every core")

        self.mamba_d_model = int(mamba_d_model)
        self.core = MambaCore(
            self.encoder_flatten_size,
            mamba_d_model,
            num_layers=mamba_num_layers,
            dropout=rnn_dropout,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.to(self.device)

    @property
    def rnn(self) -> nn.Module:
        """Expose the Mamba core for common analysis code."""

        return self.core


class S5Conv(ClutterSequenceModel):
    """Projected residual S5 stack with branch output dropout."""

    uses_s5_core = True

    def __init__(
        self,
        num_classes: int,
        num_pos: int,
        kernel_size: int = 3,
        device: str = "cuda",
        input_channels: int = 2,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        s5_d_model: int = S5_DEFAULT_D_MODEL,
        max_chars: int = 15,
        predict_all_chars: bool = False,
        s5_num_layers: int = 1,
        s5_dropout: float = 0.0,
        s5_state_size: int = S5_DEFAULT_STATE_SIZE,
    ) -> None:
        super().__init__(
            num_classes, num_pos, s5_d_model, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, predict_all_chars,
        )
        from ..recurrent_cores.s5 import S5Core

        if s5_dropout != 0.0:
            raise ValueError("s5_dropout is obsolete; use rnn_dropout for every core")

        self.s5_d_model = int(s5_d_model)
        self.s5_state_size = int(s5_state_size)
        self.core = S5Core(
            self.encoder_flatten_size,
            s5_d_model,
            s5_state_size,
            num_layers=s5_num_layers,
            dropout=rnn_dropout,
        )
        self.to(self.device)

    @property
    def rnn(self) -> nn.Module:
        """Expose the S5 core for common analysis code."""

        return self.core


class GaWFRNNConv(ClutterSequenceModel):
    """GaWF with feedback-gated weights and ReLU activity inside the loop."""

    is_gawf_model = True

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
        feedback_dim: int | None = None,
        num_layers: int = 1,
    ) -> None:
        if predict_all_chars:
            raise ValueError("GaWF requires character and position feedback heads")
        super().__init__(
            num_classes, num_pos, hidden_size, kernel_size, device, input_channels,
            cnn_dropout, rnn_dropout, max_chars, False,
        )
        self.num_layers = int(num_layers)
        self.output_feedback_dim = self.num_classes + self.num_pos
        self.use_feedback_projector = feedback_dim is not None and int(feedback_dim) > 0
        projected_dim = int(feedback_dim) if self.use_feedback_projector else 0
        layer_feedback_dims = (
            [projected_dim] * self.num_layers
            if self.use_feedback_projector
            else [hidden_size] * (self.num_layers - 1) + [self.output_feedback_dim]
        )
        self.core = GaWFCore(
            self.encoder_flatten_size,
            hidden_size,
            feedback_dim=layer_feedback_dims[-1],
            dropout=rnn_dropout,
            num_layers=self.num_layers,
            layer_feedback_dims=layer_feedback_dims,
        )
        self.hidden_projectors = nn.ModuleList(
            [nn.Linear(hidden_size, projected_dim) for _ in range(self.num_layers - 1)]
            if self.use_feedback_projector
            else []
        )
        self.proj_out = (
            nn.Linear(self.output_feedback_dim, projected_dim)
            if self.use_feedback_projector
            else None
        )
        self.register_buffer("prev_feedback", None, persistent=False)
        self.to(self.device)

    @property
    def feedback_dim(self) -> int:
        """Return the top-layer feedback dimension."""

        return self.core.layer_feedback_dims[-1]

    @property
    def gate_tau(self) -> float:
        """Return the gate sigmoid temperature."""

        return self.core.gate_tau

    @property
    def rnn(self) -> nn.Module:
        """Expose recurrent weights for checkpoint/analysis compatibility."""

        return self.core.rnn if self.num_layers == 1 else self.core.rnns

    @property
    def LNormRNN(self) -> nn.Module:
        """Expose in-loop LayerNorm for checkpoint/analysis compatibility."""

        return self.core.norm if self.num_layers == 1 else self.core.norms

    @property
    def U(self) -> nn.Parameter:
        """Expose the single-layer U matrix."""

        if self.num_layers != 1:
            raise AttributeError("Multi-layer GaWF stores U in U_layers")
        return self.core.U

    @property
    def V(self) -> nn.Parameter:
        """Expose the single-layer V matrix."""

        if self.num_layers != 1:
            raise AttributeError("Multi-layer GaWF stores V in V_layers")
        return self.core.V

    def begin_gawf_diagnostics(self, gate_saturation_eps: float = 0.01) -> None:
        """Start core gate and feedback diagnostics."""

        self.core.begin_gawf_diagnostics(gate_saturation_eps)

    def pop_gawf_diagnostics(self) -> dict[str, float | None]:
        """Return and clear core diagnostics."""

        return self.core.pop_gawf_diagnostics()

    def reset_sequence_state(self) -> None:
        """Start the next batch with zero task feedback."""

        self.prev_feedback = None

    def _compute_feedback(
        self, char_t: torch.Tensor, pos_t: torch.Tensor
    ) -> torch.Tensor:
        feedback = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is not None:
            return self.proj_out(feedback)
        return feedback

    def _layer_feedbacks(
        self,
        states: list[torch.Tensor],
        output_feedback: torch.Tensor,
    ) -> list[torch.Tensor]:
        feedbacks: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers - 1):
            if self.use_feedback_projector:
                source = self.hidden_projectors[layer_idx](states[layer_idx + 1])
            else:
                source = states[layer_idx + 1].detach()
            feedbacks.append(source)
        feedbacks.append(output_feedback)
        return feedbacks

    def forward(
        self,
        x: torch.Tensor,
        reset_feedback: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the always-on closed feedback loop over one frame sequence."""

        encoded = self.encode_frames(x)
        batch_size, frame_num = encoded.shape[:2]
        if reset_feedback or self.prev_feedback is None:
            output_feedback = torch.zeros(
                batch_size,
                self.feedback_dim,
                device=encoded.device,
                dtype=torch.float32,
            )
        else:
            output_feedback = self.prev_feedback.to(encoded.device, torch.float32)

        state = self.core.initial_state(batch_size, encoded.device, encoded.dtype)
        char_outputs: list[torch.Tensor] = []
        pos_outputs: list[torch.Tensor] = []
        for time_idx in range(frame_num):
            if self.num_layers == 1:
                hidden, state = self.core.step_with_state(
                    encoded[:, time_idx, :], state, output_feedback
                )
            else:
                hidden, state = self.core.step_with_state(
                    encoded[:, time_idx, :],
                    state,
                    self._layer_feedbacks(state, output_feedback),
                )
            char_t, pos_t = self.classifier(hidden)
            if pos_t is None:
                raise RuntimeError("GaWF position head cannot be disabled")
            output_feedback = self._compute_feedback(char_t, pos_t)
            if self.proj_out is None:
                output_feedback = output_feedback.detach()
            char_outputs.append(char_t)
            pos_outputs.append(pos_t)

        self.prev_feedback = output_feedback.detach().to(torch.float32)
        return torch.stack(char_outputs, dim=1), torch.stack(pos_outputs, dim=1)
