"""Atari Q-networks: classic feedforward DQN and a GaWF-gated recurrent variant.

Inputs are preprocessed Atari observations shaped ``(B, C, 84, 84)`` or sequences
``(B, T, C, 84, 84)``. Outputs are Q-values over environment actions. Unlike the
A2C models in ``atari_task_models``, the recurrent input carries encoder features
only (no previous action/reward), so both variants see exactly the classic DQN
input information; the GaWF variant additionally gates its recurrence with the
detached previous-step Q-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .atari_task_models import AtariNatureEncoder
from .recurrent_cores.gawf import GaWFCore

AtariDQNModelType = Literal["cnn", "gawf"]
DQNFeedbackMode = Literal["none", "qvalues"]


@dataclass(frozen=True)
class AtariQNetworkState:
    """Runtime recurrent state plus previous Q-value outputs (GaWF only)."""

    recurrent: torch.Tensor
    prev_q: torch.Tensor


class AtariQHead(nn.Module):
    """Linear Q-value head for Atari control."""

    def __init__(self, input_size: int, num_actions: int) -> None:
        super().__init__()
        self.q = nn.Linear(input_size, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q(features)


class AtariQNetwork(nn.Module):
    """Atari Q-network with optional GaWF recurrence gated by previous Q-values."""

    def __init__(
        self,
        num_actions: int,
        input_channels: int = 4,
        model_type: AtariDQNModelType = "cnn",
        hidden_size: int = 256,
        encoder_feature_dim: int = 512,
        core_dropout: float = 0.0,
        feedback_mode: DQNFeedbackMode = "none",
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in {"cnn", "gawf"}:
            raise ValueError(f"Unsupported Atari DQN model_type: {model_type}")
        if feedback_mode not in {"none", "qvalues"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")
        if model_type == "cnn" and feedback_mode != "none":
            raise ValueError("CNN DQN baseline only supports feedback_mode='none'")
        self.num_actions = int(num_actions)
        self.input_channels = int(input_channels)
        self.model_type = model_type
        self.hidden_size = int(hidden_size)
        self.feedback_mode = feedback_mode
        self.detach_feedback = bool(detach_feedback)
        self.encoder = AtariNatureEncoder(
            in_channels=self.input_channels,
            feature_dim=encoder_feature_dim,
        )
        # Recurrent input carries encoder features only, keeping the observable
        # information identical to the classic DQN baseline.
        self.recurrent_input_size = self.encoder.output_size

        if self.model_type == "cnn":
            self.core = None
            head_input_size = self.encoder.output_size
        else:
            self.core = GaWFCore(
                input_size=self.recurrent_input_size,
                hidden_size=self.hidden_size,
                feedback_dim=max(
                    1,
                    self.feedback_dim_for_mode(self.feedback_mode, self.num_actions),
                ),
                dropout=core_dropout,
            )
            head_input_size = self.core.output_size
        self.head = AtariQHead(head_input_size, self.num_actions)

    @property
    def is_recurrent(self) -> bool:
        return self.model_type == "gawf"

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
        encoded = self.encoder(flat_obs)
        return encoded.view(batch_size, n_steps, -1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> AtariQNetworkState | None:
        if not self.is_recurrent:
            return None
        return AtariQNetworkState(
            recurrent=self.core.initial_state(batch_size, device, dtype),
            prev_q=torch.zeros(batch_size, self.num_actions, device=device, dtype=dtype),
        )

    def detach_state(self, state: AtariQNetworkState | None) -> AtariQNetworkState | None:
        if state is None:
            return None
        return AtariQNetworkState(
            recurrent=state.recurrent.detach(),
            prev_q=state.prev_q.detach(),
        )

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
        return AtariQNetworkState(
            recurrent=state.recurrent.to(device=device, dtype=dtype),
            prev_q=state.prev_q.to(device=device, dtype=dtype).view(
                batch_size, self.num_actions
            ),
        )

    def _mask_recurrent_state(
        self,
        recurrent: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        keep = (1.0 - done.float()).to(done.device)
        return recurrent * keep.view(-1, 1)

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

    def _gawf_no_feedback_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
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
        recurrent: torch.Tensor,
        feedback: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feedback_mode == "none":
            features = self._gawf_no_feedback_step(x_t, recurrent)
            return features, features
        if feedback is None:
            raise ValueError("GaWF qvalues feedback mode requires feedback")
        features = self.core.step(x_t, recurrent, feedback)
        return features, features

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
            return self.head(encoded), None

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
