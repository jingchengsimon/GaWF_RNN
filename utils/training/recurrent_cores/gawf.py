"""Task-agnostic GaWF recurrent core aligned with :class:`torch.nn.RNN`.

GaWF is ``nn.RNN`` plus an element-wise multiplicative gate on the input and hidden weight
matrices. Everything else follows ``nn.RNN`` exactly: the recurrence, the activation inside the
recurrence (``rnn_activation``: the built-in ``tanh`` or ``relu``), both built-in bias vectors, and
the state convention. The state that is fed back at the next step is the raw
``activation(preactivation)``; the outer ``LayerNorm -> ReLU -> dropout`` contract is applied
*outside* the recurrence to the readout only, exactly as in
``utils.training.recurrent_cores.rnn.TorchRecurrentCore``.

The only GaWF-specific code is the per-element weight modulation ``gate * W``, because
``nn.RNN.forward`` cannot express input-dependent weights; the module, parameters, biases, and
activation still come from the ``nn.RNN`` instance held in ``self.rnn``.
``rnn_activation="identity"`` removes the activation entirely (a linear recurrence); ``nn.RNN`` has
no such mode, so that single branch is the one deviation from the built-in op.

The pre-2026-09 implementation, which fed the wrapped value back into the recurrence, is frozen in
``gawf_legacy.GaWFCoreLegacy`` and is retained only to reproduce artifacts trained with it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# The gate tensors are identical in both core generations, so the pure-tensor helpers stay shared
# with the analysis code that recomputes gates from stored U/V parameters.
from .gawf_legacy import (  # noqa: F401
    GaWFDiagnosticsMixin,
    _compute_gawf_transform,
    _compute_gawf_transforms,
)


class GaWFCore(GaWFDiagnosticsMixin, nn.Module):
    """Single-layer ``nn.RNN`` core whose input/hidden weights are gated by feedback."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int | None = None,
        dropout: float = 0.0,
        gate_tau: float = 0.5,
        num_layers: int = 1,
        layer_feedback_dims: Sequence[int] | None = None,
        output_wrap: str = "ln_relu_dropout",
        rnn_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if num_layers != 1:
            raise ValueError(
                "The RNN-aligned GaWFCore supports num_layers == 1; historical multi-layer "
                "checkpoints load through gawf_legacy.GaWFCoreLegacy"
            )
        if feedback_dim is None or feedback_dim <= 0:
            raise ValueError(f"feedback_dim must be > 0, got {feedback_dim}")
        if layer_feedback_dims is not None and list(layer_feedback_dims) != [feedback_dim]:
            raise ValueError("single-layer layer_feedback_dims must equal [feedback_dim]")
        if output_wrap not in ("ln_relu_dropout", "none"):
            raise ValueError(f"Unsupported output_wrap: {output_wrap!r}")
        if rnn_activation not in ("tanh", "relu", "identity"):
            raise ValueError(f"Unsupported rnn_activation: {rnn_activation!r}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.num_layers = 1
        self.feedback_dim = int(feedback_dim)
        self.layer_feedback_dims = [self.feedback_dim]
        self.dropout = float(dropout)
        self.gate_tau = float(gate_tau)
        self.output_wrap = str(output_wrap)
        self.rnn_activation = str(rnn_activation)

        # Weights, biases, and the in-recurrence activation are the built-in ones.
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity="relu" if self.rnn_activation == "relu" else "tanh",
        )
        self.U = nn.Parameter(torch.randn(self.hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(
            torch.randn(self.feedback_dim, self.input_size + self.hidden_size) * 0.01
        )
        self.norm = nn.LayerNorm(self.hidden_size) if self.output_wrap == "ln_relu_dropout" else None
        self._init_gawf_diagnostics_state()
        self._compiled_feedback_preactivation = None

    # --- RNN-aligned interface -----------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the zero recurrent state for one independent sequence batch."""

        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _apply_activation(self, preactivation: torch.Tensor) -> torch.Tensor:
        """Apply the configured in-recurrence activation.

        ``tanh`` and ``relu`` mirror the built-in ``nn.RNN`` modes; ``identity`` is the only
        non-built-in branch and yields a linear recurrence.
        """

        if self.rnn_activation == "relu":
            return F.relu(preactivation)
        if self.rnn_activation == "identity":
            return preactivation
        return torch.tanh(preactivation)

    def _gate(
        self, feedback: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the input/hidden gate values and their logits for one batch of feedback."""

        fb = feedback.to(dtype=torch.float32).clamp(-10, 10)
        self._record_gawf_feedback(0, fb)
        scaled = self.U.unsqueeze(0) * fb.unsqueeze(2).transpose(1, 2)
        logits = torch.matmul(scaled, self.V) / self.gate_tau
        gate = torch.sigmoid(logits)
        self._record_gawf_gate(
            0,
            logits[..., : self.input_size],
            logits[..., self.input_size :],
            gate[..., : self.input_size],
            gate[..., self.input_size :],
        )
        return (
            gate[..., : self.input_size],
            gate[..., self.input_size :],
            logits[..., : self.input_size],
            logits[..., self.input_size :],
        )

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one timestep and return ``activation(preactivation)`` as the next state.

        This is one ``nn.RNN`` step with the input and hidden weights modulated element-wise; with a
        unit gate it reproduces ``nn.RNN`` exactly.
        """

        gate_ih, gate_hh, _logits_ih, _logits_hh = self._gate(feedback)
        # nn.RNN's step with every weight element modulated: (G * W) x + b_ih + (G * W) h + b_hh.
        preactivation = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, self.rnn.weight_ih_l0)
        preactivation = preactivation + torch.einsum(
            "bi,bhi,hi->bh", h_prev, gate_hh, self.rnn.weight_hh_l0
        )
        preactivation = preactivation + self.rnn.bias_ih_l0 + self.rnn.bias_hh_l0
        return self._apply_activation(preactivation)

    def project_readout(self, h_t: torch.Tensor) -> torch.Tensor:
        """Apply the external LayerNorm, ReLU, and dropout contract to a state."""

        if self.norm is None:
            return h_t
        h_t = self.norm(h_t)
        h_t = F.relu(h_t)
        return F.dropout(h_t, p=self.dropout, training=self.training)

    def step_no_feedback(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Advance one timestep using the underlying ungated ``nn.RNN`` weights."""

        preactivation = F.linear(x_t, self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
        preactivation = preactivation + F.linear(
            state, self.rnn.weight_hh_l0, self.rnn.bias_hh_l0
        )
        return self._apply_activation(preactivation)

    def forward_no_feedback(self, x: torch.Tensor):
        """Run the ungated recurrence and return the wrapped readout sequence.

        For the built-in activations this calls ``nn.RNN`` itself, so the ungated path is exactly
        the ``rnn`` baseline core.
        """

        if self.rnn_activation in ("tanh", "relu"):
            out, state = self.rnn(x)
            return self.project_readout(out), state
        batch_size, frame_num = x.shape[:2]
        state = self.initial_state(batch_size, x.device, x.dtype)
        outputs = []
        for step in range(frame_num):
            state = self.step_no_feedback(x[:, step, :], state)
            outputs.append(state)
        return self.project_readout(torch.stack(outputs, dim=1)), state.unsqueeze(0)

    def set_feedback_frozen(self, freeze: bool) -> None:
        """Freeze or unfreeze the gate parameters."""

        for param in (self.U, self.V):
            param.requires_grad = not freeze

    def configure_feedback_acceleration(
        self,
        compile_feedback: bool,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        """Accept the shared acceleration flag; the gated step stays eager.

        ``nn.RNN.forward`` cannot take input-dependent weights, so the gated step runs eagerly per
        timestep. Data-pipeline acceleration and AMP are unaffected.
        """

        self._compiled_feedback_preactivation = None


def configure_gawf_feedback_acceleration(
    module: nn.Module,
    enabled: bool,
    compile_mode: str = "reduce-overhead",
) -> int:
    """Configure every nested :class:`GaWFCore`; the RNN-aligned core always counts as eager."""

    configured = 0
    for child in module.modules():
        if not isinstance(child, GaWFCore):
            continue
        child.configure_feedback_acceleration(enabled, compile_mode)
        configured += int(child._compiled_feedback_preactivation is not None)
    return configured
