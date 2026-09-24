"""GaWF recurrence with feedback-gated weights and ReLU activity inside the loop.

Inputs are one encoded timestep, the previous activity, and the previous detached task
feedback. The returned activity is both the layer output and the next recurrent state:

    z_t = (G_ih * W_ih) x_t + (G_hh * W_hh) h_(t-1) + b_ih + b_hh
    h_t = dropout(relu(layer_norm(z_t)))

There is no bounded inner activation and no feedback-off execution path.
"""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import ElmanAffine


def _compute_gawf_transforms(
    U: torch.Tensor,
    fb_t: torch.Tensor,
    V: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feedback-conditioned input and recurrent gate logits before temperature."""

    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    return (
        torch.matmul(scaled_u, V[:, :input_size]),
        torch.matmul(scaled_u, V[:, input_size:]),
    )


def _gawf_preactivation(
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    feedback: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    gate_tau: float,
) -> torch.Tensor:
    """Pure tensor GaWF preactivation used by the optional compiled fast path."""

    input_size = x_t.size(-1)
    fb_t = feedback.clamp(-10, 10).unsqueeze(2)
    transform_ih, transform_hh = _compute_gawf_transforms(U, fb_t, V, input_size)
    gate_ih = torch.sigmoid(transform_ih / gate_tau)
    gate_hh = torch.sigmoid(transform_hh / gate_tau)
    input_current = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
    recurrent_current = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
    return input_current + recurrent_current + bias_ih + bias_hh


class GaWFCore(nn.Module):
    """Single- or multi-layer GaWF core with one fixed recurrence definition."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int,
        dropout: float = 0.0,
        gate_tau: float = 0.5,
        num_layers: int = 1,
        layer_feedback_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if feedback_dim <= 0:
            raise ValueError(f"feedback_dim must be > 0, got {feedback_dim}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.feedback_dim = int(feedback_dim)
        self.dropout = float(dropout)
        self.gate_tau = float(gate_tau)
        self.num_layers = int(num_layers)
        self.rnn_activation = "identity"
        self.output_wrap = "in_loop_ln_relu_dropout"
        self.wrap_recurrent_state = True

        dims = (
            [int(value) for value in layer_feedback_dims]
            if layer_feedback_dims is not None
            else [self.feedback_dim] * self.num_layers
        )
        if len(dims) != self.num_layers or any(value <= 0 for value in dims):
            raise ValueError("layer_feedback_dims must contain one positive value per layer")
        self.layer_feedback_dims = dims

        if self.num_layers == 1:
            self.rnn = ElmanAffine(self.input_size, self.hidden_size)
            self.U = nn.Parameter(torch.randn(self.hidden_size, dims[0]) * 0.01)
            self.V = nn.Parameter(torch.randn(dims[0], self.input_size + self.hidden_size) * 0.01)
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            layer_inputs = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            self.rnns = nn.ModuleList(
                [ElmanAffine(size, self.hidden_size) for size in layer_inputs]
            )
            self.U_layers = nn.ParameterList(
                [nn.Parameter(torch.randn(self.hidden_size, dim) * 0.01) for dim in dims]
            )
            self.V_layers = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(dim, size + self.hidden_size) * 0.01)
                    for dim, size in zip(dims, layer_inputs)
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers)])

        self._compiled_preactivation = None
        self._diagnostics = None
        self._diagnostic_gate_eps = 0.01

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Return zero recurrent activity for a new independent sequence."""

        if self.num_layers == 1:
            return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

    def configure_feedback_acceleration(
        self,
        compile_feedback: bool,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        """Compile only the pure feedback-conditioned preactivation when requested."""

        self._compiled_preactivation = None
        if not compile_feedback:
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled GaWF feedback requires torch.compile")
        try:
            self._compiled_preactivation = torch.compile(
                _gawf_preactivation,
                mode=compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        except RuntimeError as exc:
            warnings.warn(
                f"Compiled GaWF feedback is unavailable ({exc}); using eager feedback",
                RuntimeWarning,
                stacklevel=2,
            )

    def begin_gawf_diagnostics(self, gate_saturation_eps: float = 0.01) -> None:
        """Start lightweight feedback and gate diagnostics for subsequent steps."""

        self._diagnostic_gate_eps = float(gate_saturation_eps)
        self._diagnostics = {
            "gate_logit_min": float("inf"),
            "gate_logit_max": float("-inf"),
            "gate_saturation_count": 0,
            "gate_count": 0,
            "feedback_norm_sum": 0.0,
            "feedback_norm_max": 0.0,
            "feedback_count": 0,
        }

    def pop_gawf_diagnostics(self) -> dict[str, float | None]:
        """Return and clear the currently accumulated diagnostic summary."""

        state = self._diagnostics
        self._diagnostics = None
        if state is None:
            return {}
        return {
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

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | Sequence[torch.Tensor],
        feedback: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance one timestep; wrapped activity is both output and next state."""

        if self.num_layers == 1:
            if not isinstance(h_prev, torch.Tensor) or not isinstance(feedback, torch.Tensor):
                raise TypeError("single-layer GaWF expects tensor state and feedback")
            states = [h_prev]
            feedbacks = [feedback]
        else:
            if isinstance(h_prev, torch.Tensor) or isinstance(feedback, torch.Tensor):
                raise TypeError("multi-layer GaWF expects state and feedback sequences")
            if len(h_prev) != self.num_layers or len(feedback) != self.num_layers:
                raise ValueError("state and feedback sequences must match num_layers")
            states = list(h_prev)
            feedbacks = list(feedback)

        layer_input = x_t
        next_states: list[torch.Tensor] = []
        for layer_idx, (state, layer_feedback) in enumerate(zip(states, feedbacks)):
            if self.num_layers == 1:
                rnn, U, V, norm = self.rnn, self.U, self.V, self.norm
            else:
                rnn = self.rnns[layer_idx]
                U = self.U_layers[layer_idx]
                V = self.V_layers[layer_idx]
                norm = self.norms[layer_idx]

            fb = layer_feedback.to(device=layer_input.device, dtype=torch.float32)
            if self._diagnostics is not None:
                with torch.no_grad():
                    norms = fb.detach().float().norm(dim=-1)
                    self._diagnostics["feedback_norm_sum"] += float(norms.mean().item())
                    self._diagnostics["feedback_norm_max"] = max(
                        self._diagnostics["feedback_norm_max"], float(norms.max().item())
                    )
                    self._diagnostics["feedback_count"] += 1

            if self._compiled_preactivation is not None and self._diagnostics is None:
                preactivation = self._compiled_preactivation(
                    layer_input,
                    state,
                    fb,
                    U,
                    V,
                    rnn.weight_ih_l0,
                    rnn.weight_hh_l0,
                    rnn.bias_ih_l0,
                    rnn.bias_hh_l0,
                    self.gate_tau,
                )
            else:
                input_size = layer_input.size(-1)
                transform_ih, transform_hh = _compute_gawf_transforms(
                    U, fb.clamp(-10, 10).unsqueeze(2), V, input_size
                )
                logits_ih, logits_hh = transform_ih / self.gate_tau, transform_hh / self.gate_tau
                gate_ih, gate_hh = torch.sigmoid(logits_ih), torch.sigmoid(logits_hh)
                if self._diagnostics is not None:
                    eps = self._diagnostic_gate_eps
                    with torch.no_grad():
                        self._diagnostics["gate_logit_min"] = min(
                            self._diagnostics["gate_logit_min"],
                            float(logits_ih.amin().item()),
                            float(logits_hh.amin().item()),
                        )
                        self._diagnostics["gate_logit_max"] = max(
                            self._diagnostics["gate_logit_max"],
                            float(logits_ih.amax().item()),
                            float(logits_hh.amax().item()),
                        )
                        self._diagnostics["gate_saturation_count"] += int(
                            ((gate_ih <= eps) | (gate_ih >= 1.0 - eps)).sum().item()
                            + ((gate_hh <= eps) | (gate_hh >= 1.0 - eps)).sum().item()
                        )
                        self._diagnostics["gate_count"] += gate_ih.numel() + gate_hh.numel()

                input_current = torch.einsum(
                    "bi,bhi,hi->bh", layer_input, gate_ih, rnn.weight_ih_l0
                )
                recurrent_current = torch.einsum("bi,bhi,hi->bh", state, gate_hh, rnn.weight_hh_l0)
                preactivation = input_current + recurrent_current + rnn.bias_ih_l0 + rnn.bias_hh_l0

            layer_input = F.dropout(F.relu(norm(preactivation)), self.dropout, self.training)
            next_states.append(layer_input)

        if self.num_layers == 1:
            return next_states[0]
        return layer_input, next_states

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | Sequence[torch.Tensor],
        feedback: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Alias the module call to the explicit one-timestep recurrence."""

        return self.step(x_t, h_prev, feedback)


def configure_gawf_feedback_acceleration(
    module: nn.Module,
    enabled: bool,
    compile_mode: str = "reduce-overhead",
) -> int:
    """Configure every nested GaWF core and return the number compiled."""

    compiled = 0
    for child in module.modules():
        if isinstance(child, GaWFCore):
            child.configure_feedback_acceleration(enabled, compile_mode)
            compiled += int(child._compiled_preactivation is not None)
    return compiled
