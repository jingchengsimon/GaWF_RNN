"""Atari recurrent actor-critic models built from a CNN encoder and RL heads.

Inputs are preprocessed Atari observations shaped ``(B, C, 84, 84)`` or sequences
``(B, T, C, 84, 84)``. Outputs are policy logits over environment actions and a
scalar value estimate. LSTM and GaWF variants both receive previous action and
reward as recurrent inputs. GaWF can optionally use previous policy/value outputs
as detached gate feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.rnn import LSTMCore

AtariModelType = Literal["lstm", "gawf"]
FeedbackMode = Literal["none", "output"]


@dataclass(frozen=True)
class AtariActorCriticState:
    """Runtime recurrent state plus previous actor-critic outputs."""

    recurrent: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    prev_policy_logits: torch.Tensor
    prev_value: torch.Tensor


AtariRecurrentState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
AtariStateLike = AtariActorCriticState | AtariRecurrentState | None


class AtariNatureEncoder(nn.Module):
    """Nature-DQN style CNN encoder for 84x84 Atari frames."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 512) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
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
            conv_out = self.conv(dummy).shape[-1]
        self.fc = nn.Sequential(nn.Linear(conv_out, self.feature_dim), nn.ReLU())
        self.output_size = self.feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            obs = obs.float().div(255.0)
        else:
            obs = obs.float()
            if obs.numel() > 0 and obs.detach().amax() > 1.5:
                obs = obs.div(255.0)
        return self.fc(self.conv(obs))


class AtariActorCriticHead(nn.Module):
    """Policy and value heads for Atari control."""

    def __init__(self, input_size: int, num_actions: int) -> None:
        super().__init__()
        self.policy = nn.Linear(input_size, num_actions)
        self.value = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(features), self.value(features).squeeze(-1)


class AtariActorCritic(nn.Module):
    """Atari recurrent policy/value model with optional GaWF output feedback."""

    def __init__(
        self,
        num_actions: int,
        input_channels: int = 4,
        model_type: AtariModelType = "gawf",
        hidden_size: int = 256,
        encoder_feature_dim: int = 512,
        core_dropout: float = 0.0,
        feedback_mode: FeedbackMode = "none",
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in {"lstm", "gawf"}:
            raise ValueError(f"Unsupported Atari model_type: {model_type}")
        if feedback_mode not in {"none", "output"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")
        if model_type == "lstm" and feedback_mode != "none":
            raise ValueError("Atari LSTM baseline only supports feedback_mode='none'")
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
        self.recurrent_input_size = self.encoder.output_size + self.num_actions + 1

        if self.model_type == "lstm":
            self.core = LSTMCore(
                input_size=self.recurrent_input_size,
                hidden_size=self.hidden_size,
                dropout=core_dropout,
            )
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
        core_output_size = self.core.output_size
        self.head = AtariActorCriticHead(core_output_size, self.num_actions)

    @property
    def is_gawf_model(self) -> bool:
        return self.model_type == "gawf"

    @staticmethod
    def feedback_dim_for_mode(feedback_mode: FeedbackMode, num_actions: int) -> int:
        if feedback_mode == "none":
            return 0
        return int(num_actions) + 1

    @property
    def feedback_dim(self) -> int:
        return self.feedback_dim_for_mode(self.feedback_mode, self.num_actions)

    def _encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps = obs.shape[:2]
        flat_obs = obs.reshape(batch_size * n_steps, *obs.shape[2:])
        encoded = self.encoder(flat_obs)
        return encoded.view(batch_size, n_steps, -1)

    def _initial_recurrent_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> AtariRecurrentState:
        if self.model_type == "lstm":
            shape = (1, batch_size, self.hidden_size)
            h = torch.zeros(shape, device=device, dtype=dtype)
            c = torch.zeros(shape, device=device, dtype=dtype)
            return h, c
        return self.core.initial_state(batch_size, device, dtype)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> AtariActorCriticState:
        return AtariActorCriticState(
            recurrent=self._initial_recurrent_state(batch_size, device, dtype),
            prev_policy_logits=torch.zeros(
                batch_size,
                self.num_actions,
                device=device,
                dtype=dtype,
            ),
            prev_value=torch.zeros(batch_size, device=device, dtype=dtype),
        )

    def detach_state(self, state: AtariActorCriticState | None) -> AtariActorCriticState | None:
        if state is None:
            return None
        recurrent = state.recurrent
        if isinstance(recurrent, tuple):
            recurrent = tuple(part.detach() for part in recurrent)
        else:
            recurrent = recurrent.detach()
        return AtariActorCriticState(
            recurrent=recurrent,
            prev_policy_logits=state.prev_policy_logits.detach(),
            prev_value=state.prev_value.detach(),
        )

    def _coerce_state(
        self,
        state: AtariStateLike,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> AtariActorCriticState:
        if state is None:
            return self.initial_state(batch_size, device, dtype)
        if isinstance(state, AtariActorCriticState):
            recurrent = state.recurrent
            prev_logits = state.prev_policy_logits.to(device=device, dtype=dtype)
            prev_value = state.prev_value.to(device=device, dtype=dtype).view(batch_size)
        else:
            recurrent = state
            prev_logits = torch.zeros(batch_size, self.num_actions, device=device, dtype=dtype)
            prev_value = torch.zeros(batch_size, device=device, dtype=dtype)
        if isinstance(recurrent, tuple):
            recurrent = tuple(part.to(device=device, dtype=dtype) for part in recurrent)
        else:
            recurrent = recurrent.to(device=device, dtype=dtype)
        return AtariActorCriticState(recurrent, prev_logits, prev_value)

    def _mask_recurrent_state(
        self,
        state: AtariRecurrentState,
        done: torch.Tensor,
    ) -> AtariRecurrentState:
        keep = (1.0 - done.float()).to(done.device)
        if isinstance(state, tuple):
            keep_lstm = keep.view(1, -1, 1)
            return state[0] * keep_lstm, state[1] * keep_lstm
        return state * keep.view(-1, 1)

    def _mask_output_state(
        self,
        prev_logits: torch.Tensor,
        prev_value: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = (1.0 - done.float()).to(done.device)
        return prev_logits * keep.view(-1, 1), prev_value * keep

    def _build_recurrent_input(
        self,
        encoded_t: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ) -> torch.Tensor:
        prev_action_oh = F.one_hot(prev_action.long(), num_classes=self.num_actions).to(
            device=encoded_t.device,
            dtype=encoded_t.dtype,
        )
        clipped_reward = (
            prev_reward.to(device=encoded_t.device, dtype=encoded_t.dtype)
            .view(-1, 1)
            .clamp(-1.0, 1.0)
        )
        return torch.cat([encoded_t, prev_action_oh, clipped_reward], dim=-1)

    def _build_feedback(
        self,
        prev_policy_logits: torch.Tensor,
        prev_value: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.model_type != "gawf" or self.feedback_mode == "none":
            return None
        feedback = torch.cat(
            [prev_policy_logits.float(), prev_value.view(-1, 1).float()],
            dim=-1,
        )
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
        recurrent: AtariRecurrentState,
        feedback: torch.Tensor | None,
    ) -> tuple[torch.Tensor, AtariRecurrentState]:
        if self.model_type == "lstm":
            features, next_recurrent = self.core(x_t.unsqueeze(1), recurrent)
            return features[:, 0, :], next_recurrent
        if self.feedback_mode == "none":
            if not isinstance(recurrent, torch.Tensor):
                raise TypeError("GaWF recurrent state must be a tensor")
            features = self._gawf_no_feedback_step(x_t, recurrent)
            return features, features
        if feedback is None:
            raise ValueError("GaWF output feedback mode requires feedback")
        if not isinstance(recurrent, torch.Tensor):
            raise TypeError("GaWF recurrent state must be a tensor")
        features = self.core.step(x_t, recurrent, feedback)
        return features, features

    def forward_sequence(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        state: AtariStateLike = None,
    ) -> tuple[torch.Tensor, torch.Tensor, AtariActorCriticState]:
        """Run a batch-first observation sequence through the Atari model."""
        if obs.ndim != 5:
            raise ValueError(f"obs must have shape (B,T,C,H,W), got {tuple(obs.shape)}")
        encoded = self._encode_sequence(obs)
        batch_size, n_steps = encoded.shape[:2]
        device = encoded.device
        dtype = encoded.dtype
        model_state = self._coerce_state(state, batch_size, device, dtype)
        recurrent = model_state.recurrent
        prev_logits = model_state.prev_policy_logits
        prev_value = model_state.prev_value

        logits_steps = []
        value_steps = []
        for t in range(n_steps):
            done_t = prev_dones[:, t].to(device=device, dtype=dtype)
            recurrent = self._mask_recurrent_state(recurrent, done_t)
            prev_logits, prev_value = self._mask_output_state(prev_logits, prev_value, done_t)
            x_t = self._build_recurrent_input(
                encoded[:, t, :],
                prev_actions[:, t].to(device),
                prev_rewards[:, t].to(device),
            )
            feedback = self._build_feedback(prev_logits, prev_value)
            features, recurrent = self._core_step(x_t, recurrent, feedback)
            logits_t, value_t = self.head(features)
            logits_steps.append(logits_t)
            value_steps.append(value_t)
            prev_logits, prev_value = logits_t, value_t

        next_state = AtariActorCriticState(recurrent, prev_logits, prev_value)
        return torch.stack(logits_steps, dim=1), torch.stack(value_steps, dim=1), next_state

    def step(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        state: AtariStateLike = None,
    ) -> tuple[torch.Tensor, torch.Tensor, AtariActorCriticState]:
        obs_seq = obs.unsqueeze(1)
        logits, values, next_state = self.forward_sequence(
            obs_seq,
            prev_action.view(-1, 1),
            prev_reward.view(-1, 1),
            prev_done.view(-1, 1),
            state=state,
        )
        return logits[:, 0, :], values[:, 0], next_state

    def act(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        state: AtariStateLike = None,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        AtariActorCriticState,
    ]:
        logits, value, next_state = self.step(obs, prev_action, prev_reward, prev_done, state)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value, logits, next_state

    def evaluate_actions_sequence(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        actions: torch.Tensor,
        state: AtariStateLike = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _state = self.forward_sequence(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            state=state,
        )
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return logprobs, entropy, values, logits


def get_atari_model_classes() -> dict[str, type[AtariActorCritic]]:
    """Return Atari model registry names accepted by ``train_atari.py``."""
    return {"lstm": AtariActorCritic, "gawf": AtariActorCritic}
