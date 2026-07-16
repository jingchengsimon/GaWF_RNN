"""Paper-aligned MiniGrid PPO actor-critic models and core comparisons.

The observation encoder follows Toro Icarte et al. (2020): a flattened one-hot
MiniGrid view passes through five 128-unit tanh layers. ``paper_lstm`` then uses
a plain LSTM, while the comparison variants replace only that recurrent layer
with a shared project core. GaWF receives detached previous policy logits as its
seven-dimensional feedback vector on RedBlueDoors and MemoryS7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .minigrid_models import MINIGRID_VOCAB
from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.paper_lstm import PaperLSTMCore
from .recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

PaperMiniGridModelType = Literal[
    "paper_lstm", "rnn", "gru", "lstm_core", "gawf", "s5", "mamba"
]
PAPER_MINIGRID_MODEL_TYPES = (
    "paper_lstm",
    "rnn",
    "gru",
    "lstm_core",
    "gawf",
    "s5",
    "mamba",
)
SEQUENCE_CORES = ("s5", "mamba")

TensorState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]


@dataclass(frozen=True)
class GaWFPolicyState:
    """GaWF recurrent state plus detached previous policy-logit feedback."""

    recurrent: torch.Tensor | list[torch.Tensor]
    prev_policy_logits: torch.Tensor


PaperMiniGridState = TensorState | GaWFPolicyState | None


class PaperMiniGridEncoder(nn.Module):
    """Five-layer tanh MLP over the flattened one-hot symbolic observation."""

    def __init__(
        self,
        grid_size: int,
        hidden_size: int = 128,
        num_layers: int = 5,
        vocab: tuple[int, int, int] = MINIGRID_VOCAB,
    ) -> None:
        super().__init__()
        if num_layers != 5:
            raise ValueError("Paper protocol requires exactly five encoder layers")
        self.grid_size = int(grid_size)
        self.hidden_size = int(hidden_size)
        self.output_size = self.hidden_size
        self.vocab = tuple(int(value) for value in vocab)
        flat_size = sum(self.vocab) * self.grid_size * self.grid_size
        sizes = [flat_size] + [self.hidden_size] * num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(sizes, sizes[1:])]
        )
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=2.0**0.5)
            nn.init.zeros_(layer.bias)

    def _one_hot(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.long()
        planes = []
        for channel, n_values in enumerate(self.vocab):
            indices = obs[:, channel].clamp(0, n_values - 1)
            plane = F.one_hot(indices, num_classes=n_values).permute(0, 3, 1, 2).float()
            planes.append(plane)
        return torch.cat(planes, dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self._one_hot(obs).flatten(1)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return hidden


class PaperMiniGridActorCritic(nn.Module):
    """Paper encoder plus a plain LSTM or one parameter-matched project core."""

    def __init__(
        self,
        num_actions: int,
        grid_size: int,
        model_type: PaperMiniGridModelType = "paper_lstm",
        hidden_size: int = 128,
        encoder_hidden_size: int = 128,
        core_dropout: float = 0.0,
        ssm_d_model: int = 128,
        ssm_state_size: int = 64,
        ssm_num_layers: int = 1,
        ssm_context_len: int = 128,
        num_layers: int = 1,
        detach_feedback: bool = True,
    ) -> None:
        super().__init__()
        if model_type not in PAPER_MINIGRID_MODEL_TYPES:
            raise ValueError(f"Unsupported paper MiniGrid model type: {model_type}")
        if num_layers != 1:
            raise ValueError("Paper-aligned MiniGrid comparison currently requires num_layers=1")
        self.num_actions = int(num_actions)
        self.model_type = model_type
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.detach_feedback = bool(detach_feedback)
        self.ssm_context_len = int(ssm_context_len)
        self.encoder = PaperMiniGridEncoder(grid_size, hidden_size=encoder_hidden_size)
        encoder_out = self.encoder.output_size

        if model_type == "paper_lstm":
            self.core = PaperLSTMCore(encoder_out, self.hidden_size)
        elif model_type == "lstm_core":
            self.core = LSTMCore(encoder_out, self.hidden_size, dropout=core_dropout)
        elif model_type in ("rnn", "gru"):
            core_class = {"rnn": RNNCore, "gru": GRUCore}[model_type]
            self.core = core_class(encoder_out, self.hidden_size, dropout=core_dropout)
        elif model_type == "gawf":
            self.core = GaWFCore(
                input_size=encoder_out,
                hidden_size=self.hidden_size,
                feedback_dim=self.num_actions,
                dropout=core_dropout,
                num_layers=1,
            )
        elif model_type == "s5":
            from .recurrent_cores.s5 import S5Core

            self.core = S5Core(
                input_size=encoder_out,
                d_model=int(ssm_d_model),
                state_size=int(ssm_state_size),
                num_layers=int(ssm_num_layers),
                dropout=core_dropout,
            )
        else:
            from .recurrent_cores.mamba import MambaCore

            self.core = MambaCore(
                input_size=encoder_out,
                d_model=int(ssm_d_model),
                num_layers=int(ssm_num_layers),
                dropout=core_dropout,
                d_state=int(ssm_state_size),
            )

        core_out = int(self.core.output_size)
        self.policy = nn.Linear(core_out, self.num_actions)
        self.value = nn.Linear(core_out, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    @property
    def uses_sequence_core(self) -> bool:
        return self.model_type in SEQUENCE_CORES

    @property
    def uses_tuple_state(self) -> bool:
        return self.model_type in ("paper_lstm", "lstm_core")

    @property
    def feedback_mode(self) -> str:
        return "policy_logits" if self.model_type == "gawf" else "none"

    @property
    def feedback_dim(self) -> int:
        return self.num_actions if self.model_type == "gawf" else 0

    def _initial_recurrent_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> TensorState:
        if self.model_type == "gawf":
            return self.core.initial_state(batch_size, device, dtype)
        if self.uses_sequence_core:
            return torch.zeros(
                batch_size,
                self.ssm_context_len,
                self.core.input_size,
                device=device,
                dtype=dtype,
            )
        shape = (1, batch_size, self.hidden_size)
        if self.uses_tuple_state:
            return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(
                shape, device=device, dtype=dtype
            )
        return torch.zeros(shape, device=device, dtype=dtype)

    def initial_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> PaperMiniGridState:
        recurrent = self._initial_recurrent_state(batch_size, device, dtype)
        if self.model_type != "gawf":
            return recurrent
        if isinstance(recurrent, tuple):
            raise TypeError("GaWF recurrent state must not be an LSTM tuple")
        return GaWFPolicyState(
            recurrent=recurrent,
            prev_policy_logits=torch.zeros(
                batch_size, self.num_actions, device=device, dtype=dtype
            ),
        )

    def detach_state(self, state: PaperMiniGridState) -> PaperMiniGridState:
        if state is None:
            return None
        if isinstance(state, GaWFPolicyState):
            recurrent = state.recurrent
            detached = (
                [part.detach() for part in recurrent]
                if isinstance(recurrent, list)
                else recurrent.detach()
            )
            return GaWFPolicyState(detached, state.prev_policy_logits.detach())
        if isinstance(state, tuple):
            return tuple(part.detach() for part in state)
        if isinstance(state, list):
            return [part.detach() for part in state]
        return state.detach()

    def select_state(
        self, state: PaperMiniGridState, env_indices: torch.Tensor
    ) -> PaperMiniGridState:
        """Select environment rows while retaining the recurrent sequence dimension."""
        if state is None:
            return None
        if isinstance(state, GaWFPolicyState):
            recurrent = state.recurrent
            selected = (
                [part.index_select(0, env_indices) for part in recurrent]
                if isinstance(recurrent, list)
                else recurrent.index_select(0, env_indices)
            )
            return GaWFPolicyState(
                selected, state.prev_policy_logits.index_select(0, env_indices)
            )
        if isinstance(state, tuple):
            return tuple(part.index_select(1, env_indices) for part in state)
        if isinstance(state, list):
            return [part.index_select(0, env_indices) for part in state]
        dimension = 0 if self.uses_sequence_core or self.model_type == "gawf" else 1
        return state.index_select(dimension, env_indices)

    @staticmethod
    def _mask_tensor_state(state: TensorState, done: torch.Tensor) -> TensorState:
        keep = 1.0 - done.float()
        if isinstance(state, tuple):
            shaped = keep.view(1, -1, 1)
            return state[0] * shaped, state[1] * shaped
        if isinstance(state, list):
            return [part * keep.view(-1, 1) for part in state]
        if state.dim() == 3:
            return state * keep.view(1, -1, 1)
        return state * keep.view(-1, 1)

    def _encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size, steps = obs.shape[:2]
        flat = obs.reshape(batch_size * steps, *obs.shape[2:])
        return self.encoder(flat).view(batch_size, steps, -1)

    def _forward_stepwise(
        self,
        encoded: torch.Tensor,
        prev_dones: torch.Tensor,
        state: PaperMiniGridState,
    ) -> tuple[torch.Tensor, torch.Tensor, PaperMiniGridState]:
        batch_size, steps = encoded.shape[:2]
        dtype, device = encoded.dtype, encoded.device
        if state is None:
            state = self.initial_state(batch_size, device, dtype)

        logits_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        if self.model_type == "gawf":
            if not isinstance(state, GaWFPolicyState):
                raise TypeError("GaWF requires GaWFPolicyState")
            recurrent = state.recurrent
            prev_logits = state.prev_policy_logits
            for step in range(steps):
                done = prev_dones[:, step].to(device=device, dtype=dtype)
                recurrent = self._mask_tensor_state(recurrent, done)
                prev_logits = prev_logits * (1.0 - done).view(-1, 1)
                feedback = prev_logits.detach() if self.detach_feedback else prev_logits
                recurrent = self.core.step(encoded[:, step], recurrent, feedback)
                if isinstance(recurrent, tuple):
                    features, recurrent = recurrent
                else:
                    features = recurrent
                logits = self.policy(features)
                logits_steps.append(logits)
                value_steps.append(self.value(features).squeeze(-1))
                prev_logits = logits
            next_state: PaperMiniGridState = GaWFPolicyState(recurrent, prev_logits)
        else:
            if isinstance(state, GaWFPolicyState):
                raise TypeError("Non-GaWF model received GaWFPolicyState")
            recurrent = state
            for step in range(steps):
                done = prev_dones[:, step].to(device=device, dtype=dtype)
                recurrent = self._mask_tensor_state(recurrent, done)
                output, recurrent = self.core(encoded[:, step : step + 1], recurrent)
                features = output[:, 0]
                logits_steps.append(self.policy(features))
                value_steps.append(self.value(features).squeeze(-1))
            next_state = recurrent
        return torch.stack(logits_steps, 1), torch.stack(value_steps, 1), next_state

    def forward_sequence(
        self,
        obs: torch.Tensor,
        prev_dones: torch.Tensor,
        state: PaperMiniGridState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, PaperMiniGridState]:
        """Run an environment-major observation sequence through the selected core."""
        if obs.ndim != 5:
            raise ValueError(f"obs must have shape (B,T,C,H,W), got {tuple(obs.shape)}")
        encoded = self._encode_sequence(obs)
        if self.uses_sequence_core:
            core_output, _ = self.core(encoded)
            return (
                self.policy(core_output),
                self.value(core_output).squeeze(-1),
                None,
            )
        return self._forward_stepwise(encoded, prev_dones, state)

    def _step_sequence_core(
        self, obs: torch.Tensor, prev_done: torch.Tensor, state: PaperMiniGridState
    ) -> tuple[torch.Tensor, torch.Tensor, PaperMiniGridState]:
        encoded = self.encoder(obs)
        batch_size, feature_size = encoded.shape
        if state is None:
            buffer = torch.zeros(
                batch_size,
                self.ssm_context_len,
                feature_size,
                device=encoded.device,
                dtype=encoded.dtype,
            )
        elif not isinstance(state, torch.Tensor):
            raise TypeError("Sequence cores require a tensor context buffer")
        else:
            buffer = state
        buffer = buffer * (1.0 - prev_done.float()).view(batch_size, 1, 1)
        buffer = torch.roll(buffer, shifts=-1, dims=1)
        buffer[:, -1] = encoded
        core_output, _ = self.core(buffer)
        features = core_output[:, -1]
        return self.policy(features), self.value(features).squeeze(-1), buffer

    def step(
        self, obs: torch.Tensor, prev_done: torch.Tensor, state: PaperMiniGridState = None
    ) -> tuple[torch.Tensor, torch.Tensor, PaperMiniGridState]:
        if self.uses_sequence_core:
            return self._step_sequence_core(obs, prev_done, state)
        logits, values, next_state = self.forward_sequence(
            obs.unsqueeze(1), prev_done.view(-1, 1), state
        )
        return logits[:, 0], values[:, 0], next_state

    def act(
        self,
        obs: torch.Tensor,
        prev_done: torch.Tensor,
        state: PaperMiniGridState = None,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        PaperMiniGridState,
    ]:
        logits, value, next_state = self.step(obs, prev_done, state)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, -1) if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            value,
            next_state,
        )

    def evaluate_actions_sequence(
        self,
        obs: torch.Tensor,
        prev_dones: torch.Tensor,
        actions: torch.Tensor,
        state: PaperMiniGridState = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _ = self.forward_sequence(obs, prev_dones, state)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions.long()), distribution.entropy(), values
