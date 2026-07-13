"""Recurrent actor-critic models for MiniGrid PPO (BabyAI-style).

Following BabyAI (Chevalier-Boisvert et al., 2019): a small observation encoder
feeds a recurrent memory core, whose output drives a policy head (action logits)
and a value head. We drop the instruction GRU + FiLM (our memory tasks have no
language / a constant mission) and keep just obs -> encoder -> memory -> heads.

Reuses the shared recurrent cores (rnn/gru/lstm/gawf stepwise; s5/mamba as
whole-sequence scan cores) and ``MiniGridEncoder``. This mirrors the DRQN model's
core handling but emits (policy_logits, value) instead of Q-values, and is trained
on-policy by ``train_minigrid_ppo.py`` (PPO) rather than DQN. All six cores run
uniformly with no output feedback (a clean first comparison).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

MiniGridPPOModelType = Literal["rnn", "gru", "lstm", "gawf", "s5", "mamba"]

STEPWISE_CORES = ("rnn", "gru", "lstm", "gawf")
SEQUENCE_CORES = ("s5", "mamba")
RECURRENT_CORES = (*STEPWISE_CORES, *SEQUENCE_CORES)

MiniGridRecurrentState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]


class MiniGridActorCritic(nn.Module):
    """Encoder -> recurrent memory core -> (policy logits, value)."""

    def __init__(
        self,
        num_actions: int,
        encoder,
        model_type: MiniGridPPOModelType = "lstm",
        hidden_size: int = 128,
        core_dropout: float = 0.0,
        ssm_d_model: int = 128,
        ssm_state_size: int = 64,
        ssm_num_layers: int = 1,
        ssm_context_len: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if model_type not in RECURRENT_CORES:
            raise ValueError(f"Unsupported MiniGrid PPO model_type: {model_type}")
        self.num_actions = int(num_actions)
        self.model_type = model_type
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.ssm_context_len = int(ssm_context_len)

        self.encoder = encoder
        enc_out = int(encoder.output_size)

        if model_type == "gawf":
            # No output feedback in this first version: feedback_dim is a dummy 1
            # and we use the no-feedback recurrence path (still gated internally).
            self.core = GaWFCore(
                input_size=enc_out,
                hidden_size=self.hidden_size,
                feedback_dim=1,
                dropout=core_dropout,
                num_layers=self.num_layers,
                layer_feedback_dims=[1] * self.num_layers,
            )
        elif model_type in ("rnn", "gru", "lstm"):
            cls = {"rnn": RNNCore, "gru": GRUCore, "lstm": LSTMCore}[model_type]
            self.core = cls(
                input_size=enc_out,
                hidden_size=self.hidden_size,
                dropout=core_dropout,
                num_layers=self.num_layers,
            )
        elif model_type == "s5":
            from .recurrent_cores.s5 import S5Core

            self.core = S5Core(
                input_size=enc_out,
                d_model=int(ssm_d_model),
                state_size=int(ssm_state_size),
                num_layers=int(ssm_num_layers),
                dropout=core_dropout,
            )
        else:  # mamba
            from .recurrent_cores.mamba import MambaCore

            self.core = MambaCore(
                input_size=enc_out,
                d_model=int(ssm_d_model),
                num_layers=int(ssm_num_layers),
                dropout=core_dropout,
                d_state=int(ssm_state_size),
            )

        core_out = int(self.core.output_size)
        self.policy = nn.Linear(core_out, self.num_actions)
        self.value = nn.Linear(core_out, 1)

    @property
    def uses_sequence_core(self) -> bool:
        return self.model_type in SEQUENCE_CORES

    @property
    def uses_tuple_state(self) -> bool:
        return self.model_type == "lstm"

    # ---- encoding -----------------------------------------------------------
    def _encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        b, t = obs.shape[:2]
        flat = obs.reshape(b * t, *obs.shape[2:])
        return self.encoder(flat).view(b, t, -1)

    # ---- recurrent state ----------------------------------------------------
    def initial_state(self, batch_size, device, dtype=torch.float32) -> MiniGridRecurrentState:
        if self.model_type == "gawf":
            return self.core.initial_state(batch_size, device, dtype)
        if self.uses_sequence_core:
            return torch.zeros(
                batch_size, self.ssm_context_len, self.core.input_size, device=device, dtype=dtype
            )
        shape = (self.num_layers, batch_size, self.hidden_size)
        if self.uses_tuple_state:
            return (
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
        return torch.zeros(shape, device=device, dtype=dtype)

    def detach_state(self, state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(p.detach() for p in state)
        if isinstance(state, list):
            return [p.detach() for p in state]
        return state.detach()

    def _mask_state(self, state, done):
        keep = (1.0 - done.float()).to(done.device)
        if isinstance(state, tuple):
            k = keep.view(1, -1, 1)
            return state[0] * k, state[1] * k
        if isinstance(state, list):
            return [part * keep.view(-1, 1) for part in state]
        if state.dim() == 3:  # (L,B,H) rnn/gru
            return state * keep.view(1, -1, 1)
        return state * keep.view(-1, 1)  # (B,H) gawf

    def _core_step(self, x_t, recurrent):
        if self.model_type == "gawf":
            stepped = self.core.step_no_feedback(x_t, recurrent)
            if self.num_layers == 1:
                return stepped, stepped
            return stepped
        feat, nxt = self.core(x_t.unsqueeze(1), recurrent)
        return feat[:, 0, :], nxt

    # ---- forward ------------------------------------------------------------
    def forward_sequence(self, obs, prev_dones, state=None):
        """(B,T,C,H,W) -> logits (B,T,A), values (B,T), next_state."""
        if obs.ndim != 5:
            raise ValueError(f"obs must be (B,T,C,H,W), got {tuple(obs.shape)}")
        encoded = self._encode_sequence(obs)
        b, t = encoded.shape[:2]
        device, dtype = encoded.device, encoded.dtype

        if self.uses_sequence_core:
            # One-shot scan; mid-sequence resets not applied (standard SSM-in-RL).
            core_out, _ = self.core(encoded)
            logits = self.policy(core_out)
            values = self.value(core_out).squeeze(-1)
            return logits, values, None

        recurrent = self.initial_state(b, device, dtype) if state is None else state
        logits_steps, value_steps = [], []
        for step in range(t):
            recurrent = self._mask_state(
                recurrent, prev_dones[:, step].to(device=device, dtype=dtype)
            )
            feat, recurrent = self._core_step(encoded[:, step, :], recurrent)
            logits_steps.append(self.policy(feat))
            value_steps.append(self.value(feat).squeeze(-1))
        return torch.stack(logits_steps, 1), torch.stack(value_steps, 1), recurrent

    def _step_sequence_core(self, obs, prev_done, state):
        """Online single-frame stepping for s5/mamba via rolling-window re-encode."""
        encoded = self.encoder(obs)  # (B, F)
        b, f = encoded.shape
        device, dtype = encoded.device, encoded.dtype
        buf = (
            self.initial_state(b, device, dtype)
            if state is None
            else state.to(device=device, dtype=dtype)
        )
        keep = (1.0 - prev_done.float()).to(device).view(b, 1, 1)
        buf = torch.roll(buf * keep, shifts=-1, dims=1)
        buf[:, -1, :] = encoded
        core_out, _ = self.core(buf)
        feat = core_out[:, -1, :]
        return self.policy(feat), self.value(feat).squeeze(-1), buf

    def step(self, obs, prev_done, state=None):
        if self.uses_sequence_core:
            return self._step_sequence_core(obs, prev_done, state)
        logits, values, nxt = self.forward_sequence(obs.unsqueeze(1), prev_done.view(-1, 1), state)
        return logits[:, 0, :], values[:, 0], nxt

    # ---- PPO interface ------------------------------------------------------
    def act(self, obs, prev_done, state=None, deterministic=False):
        logits, value, nxt = self.step(obs, prev_done, state)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, -1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, nxt

    def evaluate_actions_sequence(self, obs, prev_dones, actions, state=None):
        logits, values, _ = self.forward_sequence(obs, prev_dones, state)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()), dist.entropy(), values
