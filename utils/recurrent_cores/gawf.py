"""Task-agnostic GaWF recurrent cores and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_gawf_transforms(
    U: torch.Tensor,
    fb_t: torch.Tensor,
    V: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GaWF input/hidden transforms as ``(U * feedback) @ V``."""
    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    trans_ih = torch.matmul(scaled_u, V[:, :input_size])
    trans_hh = torch.matmul(scaled_u, V[:, input_size:])
    return trans_ih, trans_hh


class GaWFDiagnosticsMixin:
    """Small opt-in hooks for collecting GaWF gate and feedback diagnostics."""

    def _init_gawf_diagnostics_state(self) -> None:
        self._gawf_diag_state = None
        self._gawf_diag_gate_eps = 0.01

    def begin_gawf_diagnostics(self, gate_saturation_eps: float = 0.01) -> None:
        self._gawf_diag_gate_eps = float(gate_saturation_eps)
        self._gawf_diag_state = {
            "gate_logit_min": float("inf"),
            "gate_logit_max": float("-inf"),
            "gate_saturation_count": 0,
            "gate_count": 0,
            "feedback_norm_sum": 0.0,
            "feedback_norm_max": 0.0,
            "feedback_count": 0,
            "layers": {},
        }

    def pop_gawf_diagnostics(self):
        state = self._gawf_diag_state
        self._gawf_diag_state = None
        if not state:
            return {}

        out = {
            "gate_logit_min": (
                state["gate_logit_min"] if state["gate_logit_min"] != float("inf") else None
            ),
            "gate_logit_max": (
                state["gate_logit_max"] if state["gate_logit_max"] != float("-inf") else None
            ),
            "gate_saturation_frac": (
                state["gate_saturation_count"] / state["gate_count"]
                if state["gate_count"]
                else None
            ),
            "feedback_norm_mean": (
                state["feedback_norm_sum"] / state["feedback_count"]
                if state["feedback_count"]
                else None
            ),
            "feedback_norm_max": state["feedback_norm_max"],
        }
        for layer_name, layer_state in state["layers"].items():
            prefix = f"{layer_name}_"
            out[prefix + "gate_logit_min"] = (
                layer_state["gate_logit_min"]
                if layer_state["gate_logit_min"] != float("inf")
                else None
            )
            out[prefix + "gate_logit_max"] = (
                layer_state["gate_logit_max"]
                if layer_state["gate_logit_max"] != float("-inf")
                else None
            )
            out[prefix + "gate_saturation_frac"] = (
                layer_state["gate_saturation_count"] / layer_state["gate_count"]
                if layer_state["gate_count"]
                else None
            )
            out[prefix + "feedback_norm_mean"] = (
                layer_state["feedback_norm_sum"] / layer_state["feedback_count"]
                if layer_state["feedback_count"]
                else None
            )
            out[prefix + "feedback_norm_max"] = layer_state["feedback_norm_max"]
        return out

    def _layer_diag_state(self, layer_idx: int):
        state = self._gawf_diag_state
        layer_name = f"layer{layer_idx}"
        layers = state["layers"]
        if layer_name not in layers:
            layers[layer_name] = {
                "gate_logit_min": float("inf"),
                "gate_logit_max": float("-inf"),
                "gate_saturation_count": 0,
                "gate_count": 0,
                "feedback_norm_sum": 0.0,
                "feedback_norm_max": 0.0,
                "feedback_count": 0,
            }
        return layers[layer_name]

    def _record_gawf_feedback(self, layer_idx: int, fb: torch.Tensor) -> None:
        if self._gawf_diag_state is None:
            return
        with torch.no_grad():
            norms = fb.detach().float().norm(dim=-1)
            mean_norm = float(norms.mean().item())
            max_norm = float(norms.max().item())

        for state in (self._gawf_diag_state, self._layer_diag_state(layer_idx)):
            state["feedback_norm_sum"] += mean_norm
            state["feedback_norm_max"] = max(state["feedback_norm_max"], max_norm)
            state["feedback_count"] += 1

    def _record_gawf_gate(
        self,
        layer_idx: int,
        gate_logits_ih: torch.Tensor,
        gate_logits_hh: torch.Tensor,
        gate_ih: torch.Tensor,
        gate_hh: torch.Tensor,
    ) -> None:
        if self._gawf_diag_state is None:
            return
        eps = self._gawf_diag_gate_eps
        with torch.no_grad():
            logit_min = min(
                float(gate_logits_ih.detach().float().amin().item()),
                float(gate_logits_hh.detach().float().amin().item()),
            )
            logit_max = max(
                float(gate_logits_ih.detach().float().amax().item()),
                float(gate_logits_hh.detach().float().amax().item()),
            )
            sat_count = int(
                ((gate_ih <= eps) | (gate_ih >= 1.0 - eps)).sum().item()
                + ((gate_hh <= eps) | (gate_hh >= 1.0 - eps)).sum().item()
            )
            gate_count = int(gate_ih.numel() + gate_hh.numel())

        for state in (self._gawf_diag_state, self._layer_diag_state(layer_idx)):
            state["gate_logit_min"] = min(state["gate_logit_min"], logit_min)
            state["gate_logit_max"] = max(state["gate_logit_max"], logit_max)
            state["gate_saturation_count"] += sat_count
            state["gate_count"] += gate_count


class GaWFCore(GaWFDiagnosticsMixin, nn.Module):
    """Single-layer GaWF core over already-encoded timestep features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int,
        dropout: float = 0.0,
        gate_tau: float = 0.5,
    ) -> None:
        super().__init__()
        if feedback_dim <= 0:
            raise ValueError(f"feedback_dim must be > 0, got {feedback_dim}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.feedback_dim = int(feedback_dim)
        self.dropout = float(dropout)
        self.gate_tau = float(gate_tau)
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.U = nn.Parameter(torch.randn(self.hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(
            torch.randn(self.feedback_dim, self.input_size + self.hidden_size) * 0.01
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self._init_gawf_diagnostics_state()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def set_feedback_frozen(self, freeze: bool) -> None:
        for param in (self.U, self.V):
            param.requires_grad = not freeze

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        feedback: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        input_size = x_t.size(-1)
        fb = feedback.to(device=x_t.device, dtype=torch.float32)
        self._record_gawf_feedback(layer_idx, fb)
        fb_t = fb.clamp(-10, 10).unsqueeze(2)
        trans_ih, trans_hh = _compute_gawf_transforms(self.U, fb_t, self.V, input_size)
        gate_logits_ih = trans_ih / self.gate_tau
        gate_logits_hh = trans_hh / self.gate_tau
        gate_ih = torch.sigmoid(gate_logits_ih)
        gate_hh = torch.sigmoid(gate_logits_hh)
        self._record_gawf_gate(layer_idx, gate_logits_ih, gate_logits_hh, gate_ih, gate_hh)

        ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, self.rnn.weight_ih_l0)
        hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, self.rnn.weight_hh_l0)
        if self.rnn.bias_ih_l0 is not None:
            ih = ih + self.rnn.bias_ih_l0.unsqueeze(0)
        if self.rnn.bias_hh_l0 is not None:
            hh = hh + self.rnn.bias_hh_l0.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        h_t = self.norm(h_t)
        h_t = F.relu(h_t)
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        return h_t

    def forward_no_feedback(self, x: torch.Tensor):
        out, state = self.rnn(x)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out, state


class MultiLayerGaWFCore(GaWFDiagnosticsMixin, nn.Module):
    """Multi-layer GaWF core whose feedback vectors are supplied by a task wrapper."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_feedback_dims: Sequence[int],
        dropout: float = 0.0,
        gate_tau: float = 0.5,
    ) -> None:
        super().__init__()
        if len(layer_feedback_dims) < 2:
            raise ValueError("MultiLayerGaWFCore requires at least two layers")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.layer_feedback_dims = [int(dim) for dim in layer_feedback_dims]
        if any(dim <= 0 for dim in self.layer_feedback_dims):
            raise ValueError(f"all feedback dims must be > 0: {self.layer_feedback_dims}")
        self.num_layers = len(self.layer_feedback_dims)
        self.dropout = float(dropout)
        self.gate_tau = float(gate_tau)

        layer_input_sizes = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
        self.rnns = nn.ModuleList(
            [
                nn.RNN(
                    input_size=layer_input_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                for layer_input_size in layer_input_sizes
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)])
        self.U_layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.hidden_size, feedback_dim) * 0.01)
                for feedback_dim in self.layer_feedback_dims
            ]
        )
        self.V_layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(feedback_dim, layer_input_size + self.hidden_size) * 0.01)
                for feedback_dim, layer_input_size in zip(
                    self.layer_feedback_dims, layer_input_sizes
                )
            ]
        )
        self._init_gawf_diagnostics_state()

    def initial_states(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

    def set_feedback_frozen(self, freeze: bool) -> None:
        for param in list(self.U_layers) + list(self.V_layers):
            param.requires_grad = not freeze

    def _step_layer(
        self,
        layer_idx: int,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x_t.size(-1)
        rnn = self.rnns[layer_idx]
        U = self.U_layers[layer_idx]
        V = self.V_layers[layer_idx]
        fb = feedback.to(device=x_t.device, dtype=torch.float32)
        self._record_gawf_feedback(layer_idx, fb)
        fb_t = fb.clamp(-10, 10).unsqueeze(2)
        trans_ih, trans_hh = _compute_gawf_transforms(U, fb_t, V, input_size)
        gate_logits_ih = trans_ih / self.gate_tau
        gate_logits_hh = trans_hh / self.gate_tau
        gate_ih = torch.sigmoid(gate_logits_ih)
        gate_hh = torch.sigmoid(gate_logits_hh)
        self._record_gawf_gate(
            layer_idx,
            gate_logits_ih,
            gate_logits_hh,
            gate_ih,
            gate_hh,
        )
        gated_weight_ih = gate_ih * rnn.weight_ih_l0.unsqueeze(0)
        gated_weight_hh = gate_hh * rnn.weight_hh_l0.unsqueeze(0)
        ih = torch.bmm(x_t.unsqueeze(1), gated_weight_ih.transpose(1, 2)).squeeze(1)
        hh = torch.bmm(h_prev.unsqueeze(1), gated_weight_hh.transpose(1, 2)).squeeze(1)
        if rnn.bias_ih_l0 is not None:
            ih = ih + rnn.bias_ih_l0.unsqueeze(0)
        if rnn.bias_hh_l0 is not None:
            hh = hh + rnn.bias_hh_l0.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        h_t = self.norms[layer_idx](h_t)
        h_t = F.relu(h_t)
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        return h_t

    def step(
        self,
        x_t: torch.Tensor,
        h_states: Sequence[torch.Tensor],
        feedbacks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(h_states) != self.num_layers or len(feedbacks) != self.num_layers:
            raise ValueError("h_states and feedbacks must match num_layers")
        layer_input = x_t
        next_h_states = []
        for layer_idx in range(self.num_layers):
            h_t = self._step_layer(
                layer_idx,
                layer_input,
                h_states[layer_idx],
                feedbacks[layer_idx],
            )
            next_h_states.append(h_t)
            layer_input = h_t
        return layer_input, next_h_states

    def forward_no_feedback(self, x: torch.Tensor):
        layer_output = x
        final_states = []
        for layer_idx, rnn in enumerate(self.rnns):
            layer_output, state = rnn(layer_output)
            layer_output = self.norms[layer_idx](layer_output)
            layer_output = F.relu(layer_output)
            layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            final_states.append(state.squeeze(0))
        return layer_output, final_states
