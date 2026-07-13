"""Task-agnostic GaWF recurrent cores and diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_gawf_transform(
    U: torch.Tensor,
    fb_t: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Compute the complete GaWF transform as ``(U * feedback) @ V`` once."""

    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    return torch.matmul(scaled_u, V)


def _compute_gawf_transforms(
    U: torch.Tensor,
    fb_t: torch.Tensor,
    V: torch.Tensor,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the historical eager input/hidden transforms without changing numerics."""

    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    return (
        torch.matmul(scaled_u, V[:, :input_size]),
        torch.matmul(scaled_u, V[:, input_size:]),
    )


def _gawf_layer_preactivation(
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
    """Pure tensor GaWF gate and recurrent preactivation computation."""

    input_size = x_t.size(-1)
    fb_t = feedback.clamp(-10, 10).unsqueeze(2)
    gate = torch.sigmoid(_compute_gawf_transform(U, fb_t, V) / gate_tau)
    gate_ih = gate[..., :input_size]
    gate_hh = gate[..., input_size:]
    ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
    hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
    return ih + hh + bias_ih.unsqueeze(0) + bias_hh.unsqueeze(0)


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
    """Unified single- or multi-layer GaWF core over encoded timestep features."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feedback_dim: int | None = None,
        dropout: float = 0.0,
        gate_tau: float = 0.5,
        num_layers: int = 1,
        layer_feedback_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.gate_tau = float(gate_tau)

        if self.num_layers == 1:
            if feedback_dim is None or feedback_dim <= 0:
                raise ValueError(f"feedback_dim must be > 0, got {feedback_dim}")
            if layer_feedback_dims is not None and list(layer_feedback_dims) != [feedback_dim]:
                raise ValueError("single-layer layer_feedback_dims must equal [feedback_dim]")
            self.feedback_dim = int(feedback_dim)
            self.layer_feedback_dims = [self.feedback_dim]
            # Preserve all legacy single-layer parameter names.
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
        else:
            dims = list(layer_feedback_dims) if layer_feedback_dims is not None else None
            if dims is None:
                if feedback_dim is None or feedback_dim <= 0:
                    raise ValueError(
                        "multi-layer GaWF requires layer_feedback_dims or feedback_dim"
                    )
                dims = [int(feedback_dim)] * self.num_layers
            if len(dims) != self.num_layers or any(dim <= 0 for dim in dims):
                raise ValueError(
                    "layer_feedback_dims must contain one positive dimension per layer"
                )
            self.layer_feedback_dims = [int(dim) for dim in dims]
            self.feedback_dim = self.layer_feedback_dims[-1]
            layer_input_sizes = [self.input_size] + [self.hidden_size] * (self.num_layers - 1)
            # Preserve the former MultiLayerGaWFCore key layout for old checkpoints.
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
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)]
            )
            self.U_layers = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.hidden_size, dim) * 0.01)
                    for dim in self.layer_feedback_dims
                ]
            )
            self.V_layers = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(dim, layer_input_size + self.hidden_size) * 0.01)
                    for dim, layer_input_size in zip(self.layer_feedback_dims, layer_input_sizes)
                ]
            )
        self._init_gawf_diagnostics_state()
        self._compiled_feedback_preactivation: Callable[..., torch.Tensor] | None = None

    def configure_feedback_acceleration(
        self,
        compile_feedback: bool,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        """Optionally compile only the pure-tensor feedback/gate computation.

        This avoids compiling task wrappers or runtime state containers. GaWF
        diagnostics automatically use the eager equivalent so their intermediate
        gate tensors remain available.
        """

        if not compile_feedback:
            self._compiled_feedback_preactivation = None
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled GaWF feedback requires torch.compile")
        self._compiled_feedback_preactivation = torch.compile(
            _gawf_layer_preactivation,
            mode=compile_mode,
            fullgraph=True,
            dynamic=False,
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | list[torch.Tensor]:
        if self.num_layers == 1:
            return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

    def set_feedback_frozen(self, freeze: bool) -> None:
        params = (
            (self.U, self.V)
            if self.num_layers == 1
            else tuple(self.U_layers) + tuple(self.V_layers)
        )
        for param in params:
            param.requires_grad = not freeze

    def _step_layer(
        self,
        layer_idx: int,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x_t.size(-1)
        rnn = self.rnn if self.num_layers == 1 else self.rnns[layer_idx]
        norm = self.norm if self.num_layers == 1 else self.norms[layer_idx]
        U = self.U if self.num_layers == 1 else self.U_layers[layer_idx]
        V = self.V if self.num_layers == 1 else self.V_layers[layer_idx]
        fb = feedback.to(device=x_t.device, dtype=torch.float32)
        self._record_gawf_feedback(layer_idx, fb)
        if self._compiled_feedback_preactivation is not None and self._gawf_diag_state is None:
            if rnn.bias_ih_l0 is None or rnn.bias_hh_l0 is None:
                raise RuntimeError("Compiled GaWF feedback requires recurrent biases")
            preactivation = self._compiled_feedback_preactivation(
                x_t,
                h_prev,
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

            ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, rnn.weight_ih_l0)
            hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, rnn.weight_hh_l0)
            preactivation = ih + hh
            if rnn.bias_ih_l0 is not None:
                preactivation = preactivation + rnn.bias_ih_l0.unsqueeze(0)
            if rnn.bias_hh_l0 is not None:
                preactivation = preactivation + rnn.bias_hh_l0.unsqueeze(0)
        h_t = torch.tanh(preactivation)
        h_t = norm(h_t)
        h_t = F.relu(h_t)
        return F.dropout(h_t, p=self.dropout, training=self.training)

    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | Sequence[torch.Tensor],
        feedback: torch.Tensor | Sequence[torch.Tensor],
        layer_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.num_layers == 1:
            if not isinstance(h_prev, torch.Tensor) or not isinstance(feedback, torch.Tensor):
                raise TypeError("single-layer GaWF expects tensor state and feedback")
            return self._step_layer(layer_idx, x_t, h_prev, feedback)
        if isinstance(h_prev, torch.Tensor) or isinstance(feedback, torch.Tensor):
            raise TypeError("multi-layer GaWF expects state and feedback sequences")
        if len(h_prev) != self.num_layers or len(feedback) != self.num_layers:
            raise ValueError("state and feedback sequences must match num_layers")
        layer_input = x_t
        next_states: list[torch.Tensor] = []
        for idx in range(self.num_layers):
            layer_input = self._step_layer(idx, layer_input, h_prev[idx], feedback[idx])
            next_states.append(layer_input)
        return layer_input, next_states

    def step_no_feedback(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance one timestep using the underlying ungated RNN weights."""
        if self.num_layers == 1:
            if not isinstance(state, torch.Tensor):
                raise TypeError("single-layer GaWF expects a tensor state")
            ih = F.linear(x_t, self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
            hh = F.linear(state, self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)
            h_t = F.relu(self.norm(torch.tanh(ih + hh)))
            return F.dropout(h_t, p=self.dropout, training=self.training)
        if isinstance(state, torch.Tensor) or len(state) != self.num_layers:
            raise TypeError("multi-layer GaWF expects one state tensor per layer")
        layer_input = x_t
        next_states: list[torch.Tensor] = []
        for idx, (rnn, norm) in enumerate(zip(self.rnns, self.norms)):
            ih = F.linear(layer_input, rnn.weight_ih_l0, rnn.bias_ih_l0)
            hh = F.linear(state[idx], rnn.weight_hh_l0, rnn.bias_hh_l0)
            layer_input = F.relu(norm(torch.tanh(ih + hh)))
            layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)
            next_states.append(layer_input)
        return layer_input, next_states

    def forward_no_feedback(self, x: torch.Tensor):
        if self.num_layers == 1:
            out, state = self.rnn(x)
            out = self.norm(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out, state
        layer_output = x
        final_states: list[torch.Tensor] = []
        for rnn, norm in zip(self.rnns, self.norms):
            layer_output, state = rnn(layer_output)
            layer_output = F.relu(norm(layer_output))
            layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            final_states.append(state.squeeze(0))
        return layer_output, final_states


def configure_gawf_feedback_acceleration(
    module: nn.Module,
    enabled: bool,
    compile_mode: str = "reduce-overhead",
) -> int:
    """Configure every nested :class:`GaWFCore` feedback subgraph.

    The helper is task-agnostic so Atari, clutter, text, and future task wrappers
    all opt into the same implementation. Compilation is enabled only for CUDA
    cores; calling it on CPU/MPS models safely leaves their eager path active.

    Returns the number of GaWF cores configured for compiled CUDA execution.
    """

    configured = 0
    for child in module.modules():
        if not isinstance(child, GaWFCore):
            continue
        device_type = next(child.parameters()).device.type
        compile_feedback = bool(enabled and device_type == "cuda")
        child.configure_feedback_acceleration(compile_feedback, compile_mode)
        configured += int(compile_feedback)
    return configured
