"""Atari Q-networks in the DRQN framework (Hausknecht & Stone, 2015).

The Nature-DQN convolutional stack produces flattened features that feed a
single "readout slot". The classic ``cnn`` baseline fills that slot with the
Nature-DQN dense layer (``FC(conv->feature_dim)+ReLU``); the recurrent variants
replace that dense layer with a recurrent core (RNN/GRU/LSTM/GaWF) exactly as
DRQN replaces DQN's first fully connected layer. All variants share the conv
stack and a final linear Q-head, so they differ only in the readout slot.

The recurrent input is the raw flattened conv features only (no previous
action/reward), matching the classic DQN observation. The GaWF variant
additionally gates its recurrence with the detached previous-step Q-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

AtariDQNModelType = Literal["cnn", "rnn", "gru", "lstm", "gawf"]
DQNFeedbackMode = Literal["none", "qvalues"]

RECURRENT_MODEL_TYPES = ("rnn", "gru", "lstm", "gawf")

AtariRecurrentState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class AtariQNetworkState:
    """Recurrent state plus previous Q-values (used as GaWF gate feedback)."""

    recurrent: AtariRecurrentState
    prev_q: torch.Tensor


class AtariDQNConvFeatures(nn.Module):
    """Nature-DQN convolutional stack returning flattened features (no dense layer)."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 84, 84)
            self.output_size = int(self.conv(dummy).shape[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            obs = obs.float().div(255.0)
        else:
            obs = obs.float()
            if obs.numel() > 0 and obs.detach().amax() > 1.5:
                obs = obs.div(255.0)
        return self.conv(obs)


class AtariQHead(nn.Module):
    """Linear Q-value head for Atari control."""

    def __init__(self, input_size: int, num_actions: int) -> None:
        super().__init__()
        self.q = nn.Linear(input_size, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q(features)


class AtariQNetwork(nn.Module):
    """DRQN-family Atari Q-network; feedforward CNN or a recurrent readout slot."""

    def __init__(
        self,
        num_actions: int,
        input_channels: int = 4,
        model_type: AtariDQNModelType = "cnn",
        hidden_size: int = 512,
        encoder_feature_dim: int = 512,
        core_dropout: float = 0.0,
        feedback_mode: DQNFeedbackMode = "none",
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in {"cnn", *RECURRENT_MODEL_TYPES}:
            raise ValueError(f"Unsupported Atari DQN model_type: {model_type}")
        if feedback_mode not in {"none", "qvalues"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")
        if model_type != "gawf" and feedback_mode != "none":
            raise ValueError(
                f"feedback_mode='qvalues' is only valid for model_type='gawf', not '{model_type}'"
            )
        self.num_actions = int(num_actions)
        self.input_channels = int(input_channels)
        self.model_type = model_type
        self.hidden_size = int(hidden_size)
        self.feedback_mode = feedback_mode
        self.detach_feedback = bool(detach_feedback)

        self.features = AtariDQNConvFeatures(in_channels=self.input_channels)
        conv_out = self.features.output_size

        if self.model_type == "cnn":
            # Feedforward readout slot: Nature-DQN dense layer.
            self.proj = nn.Sequential(nn.Linear(conv_out, encoder_feature_dim), nn.ReLU())
            self.core = None
            head_input_size = encoder_feature_dim
        elif self.model_type == "gawf":
            self.proj = None
            self.core = GaWFCore(
                input_size=conv_out,
                hidden_size=self.hidden_size,
                feedback_dim=max(
                    1, self.feedback_dim_for_mode(self.feedback_mode, self.num_actions)
                ),
                dropout=core_dropout,
            )
            head_input_size = self.core.output_size
        else:
            self.proj = None
            core_cls = {"rnn": RNNCore, "gru": GRUCore, "lstm": LSTMCore}[self.model_type]
            self.core = core_cls(
                input_size=conv_out,
                hidden_size=self.hidden_size,
                dropout=core_dropout,
            )
            head_input_size = self.core.output_size
        self.head = AtariQHead(head_input_size, self.num_actions)

    @property
    def is_recurrent(self) -> bool:
        return self.model_type in RECURRENT_MODEL_TYPES

    @property
    def uses_tuple_state(self) -> bool:
        return self.model_type == "lstm"

    @staticmethod
    def feedback_dim_for_mode(feedback_mode: DQNFeedbackMode, num_actions: int) -> int:
        if feedback_mode == "none":
            return 0
        return int(num_actions)

    @property
    def feedback_dim(self) -> int:
        return self.feedback_dim_for_mode(self.feedback_mode, self.num_actions)

    def _encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps = obs.shape[:2]
        flat_obs = obs.reshape(batch_size * n_steps, *obs.shape[2:])
        encoded = self.features(flat_obs)
        return encoded.view(batch_size, n_steps, -1)

    def _initial_recurrent_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> AtariRecurrentState:
        if self.model_type == "gawf":
            return self.core.initial_state(batch_size, device, dtype)
        shape = (1, batch_size, self.hidden_size)
        if self.uses_tuple_state:
            h = torch.zeros(shape, device=device, dtype=dtype)
            c = torch.zeros(shape, device=device, dtype=dtype)
            return h, c
        return torch.zeros(shape, device=device, dtype=dtype)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> AtariQNetworkState | None:
        if not self.is_recurrent:
            return None
        return AtariQNetworkState(
            recurrent=self._initial_recurrent_state(batch_size, device, dtype),
            prev_q=torch.zeros(batch_size, self.num_actions, device=device, dtype=dtype),
        )

    def detach_state(self, state: AtariQNetworkState | None) -> AtariQNetworkState | None:
        if state is None:
            return None
        recurrent = state.recurrent
        if isinstance(recurrent, tuple):
            recurrent = tuple(part.detach() for part in recurrent)
        else:
            recurrent = recurrent.detach()
        return AtariQNetworkState(recurrent=recurrent, prev_q=state.prev_q.detach())

    def _coerce_state(
        self,
        state: AtariQNetworkState | None,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> AtariQNetworkState:
        if state is None:
            initial = self.initial_state(batch_size, device, dtype)
            assert initial is not None
            return initial
        recurrent = state.recurrent
        if isinstance(recurrent, tuple):
            recurrent = tuple(part.to(device=device, dtype=dtype) for part in recurrent)
        else:
            recurrent = recurrent.to(device=device, dtype=dtype)
        prev_q = state.prev_q.to(device=device, dtype=dtype).view(batch_size, self.num_actions)
        return AtariQNetworkState(recurrent=recurrent, prev_q=prev_q)

    def _mask_recurrent_state(
        self,
        recurrent: AtariRecurrentState,
        done: torch.Tensor,
    ) -> AtariRecurrentState:
        keep = (1.0 - done.float()).to(done.device)
        if isinstance(recurrent, tuple):
            keep_lstm = keep.view(1, -1, 1)
            return recurrent[0] * keep_lstm, recurrent[1] * keep_lstm
        if recurrent.dim() == 3:  # (1, B, H) for RNN/GRU
            return recurrent * keep.view(1, -1, 1)
        return recurrent * keep.view(-1, 1)  # (B, H) for GaWF

    def _mask_output_state(self, prev_q: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        keep = (1.0 - done.float()).to(done.device)
        return prev_q * keep.view(-1, 1)

    def _build_feedback(self, prev_q: torch.Tensor) -> torch.Tensor | None:
        if self.model_type != "gawf" or self.feedback_mode == "none":
            return None
        feedback = prev_q.float()
        if self.detach_feedback:
            feedback = feedback.detach()
        return feedback

    def _gawf_no_feedback_step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        ih = F.linear(x_t, self.core.rnn.weight_ih_l0, self.core.rnn.bias_ih_l0)
        hh = F.linear(h_prev, self.core.rnn.weight_hh_l0, self.core.rnn.bias_hh_l0)
        h_t = torch.tanh(ih + hh)
        h_t = self.core.norm(h_t)
        h_t = F.relu(h_t)
        h_t = F.dropout(h_t, p=self.core.dropout, training=self.training)
        return h_t

    def _core_step(
        self,
        x_t: torch.Tensor,
        recurrent: AtariRecurrentState,
        feedback: torch.Tensor | None,
    ) -> tuple[torch.Tensor, AtariRecurrentState]:
        if self.model_type == "gawf":
            if self.feedback_mode == "none":
                features = self._gawf_no_feedback_step(x_t, recurrent)
                return features, features
            if feedback is None:
                raise ValueError("GaWF qvalues feedback mode requires feedback")
            features = self.core.step(x_t, recurrent, feedback)
            return features, features
        # RNN/GRU/LSTM: run one timestep through the torch recurrent core.
        features, next_recurrent = self.core(x_t.unsqueeze(1), recurrent)
        return features[:, 0, :], next_recurrent

    def forward_sequence(
        self,
        obs: torch.Tensor,
        prev_dones: torch.Tensor,
        state: AtariQNetworkState | None = None,
    ) -> tuple[torch.Tensor, AtariQNetworkState | None]:
        """Run a batch-first observation sequence, returning (B, T, A) Q-values."""
        if obs.ndim != 5:
            raise ValueError(f"obs must have shape (B,T,C,H,W), got {tuple(obs.shape)}")
        encoded = self._encode_sequence(obs)
        if not self.is_recurrent:
            return self.head(self.proj(encoded)), None

        batch_size, n_steps = encoded.shape[:2]
        device = encoded.device
        dtype = encoded.dtype
        model_state = self._coerce_state(state, batch_size, device, dtype)
        recurrent = model_state.recurrent
        prev_q = model_state.prev_q

        q_steps = []
        for t in range(n_steps):
            done_t = prev_dones[:, t].to(device=device, dtype=dtype)
            recurrent = self._mask_recurrent_state(recurrent, done_t)
            prev_q = self._mask_output_state(prev_q, done_t)
            feedback = self._build_feedback(prev_q)
            features, recurrent = self._core_step(encoded[:, t, :], recurrent, feedback)
            q_t = self.head(features)
            q_steps.append(q_t)
            prev_q = q_t

        next_state = AtariQNetworkState(recurrent, prev_q)
        return torch.stack(q_steps, dim=1), next_state

    def step(
        self,
        obs: torch.Tensor,
        prev_done: torch.Tensor,
        state: AtariQNetworkState | None = None,
    ) -> tuple[torch.Tensor, AtariQNetworkState | None]:
        q_values, next_state = self.forward_sequence(
            obs.unsqueeze(1),
            prev_done.view(-1, 1),
            state=state,
        )
        return q_values[:, 0, :], next_state
