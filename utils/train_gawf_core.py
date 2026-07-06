import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_rnn_core import BaseConvSequenceModel


def _compute_gawf_transforms(U, fb_t, V, input_size):
    """Compute GaWF input/hidden transforms as ``(U * fb) @ V``.

    Scaling U first materializes a ``(batch, hidden, feedback)`` intermediate instead of
    broadcasting ``fb * V_ih`` to ``(batch, feedback, input)``. The same scaled U is reused
    for the input-to-hidden and hidden-to-hidden transforms.
    """
    scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
    trans_ih = torch.matmul(scaled_u, V[:, :input_size])
    trans_hh = torch.matmul(scaled_u, V[:, input_size:])
    return trans_ih, trans_hh


class GaWFDiagnosticsMixin:
    """Small opt-in hooks for collecting GaWF gate/feedback diagnostics."""

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
                state["gate_logit_min"]
                if state["gate_logit_min"] != float("inf")
                else None
            ),
            "gate_logit_max": (
                state["gate_logit_max"]
                if state["gate_logit_max"] != float("-inf")
                else None
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

    def _record_gawf_feedback(self, layer_idx: int, fb) -> None:
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
        gate_logits_ih,
        gate_logits_hh,
        gate_ih,
        gate_hh,
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


class GaWFRNNConv(GaWFDiagnosticsMixin, BaseConvSequenceModel):
    """
    GaWF (Gated with Feedback) RNN Model.
    Encoder and classifier from BaseConvSequenceModel. Forward overridden for feedback.
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        feedback_dim=None,
    ):
        super(GaWFRNNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=hidden_size,
            max_chars=15,
            predict_all_chars=False,
        )
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.hidden_size = hidden_size
        self.output_feedback_dim = num_classes + num_pos
        self.feedback_dim = (
            self.output_feedback_dim if feedback_dim is None else int(feedback_dim)
        )
        if self.feedback_dim <= 0:
            raise ValueError(f"feedback_dim must be > 0, got {self.feedback_dim}")
        input_size = self.encoder_flatten_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self._init_recurrent_module(self.rnn)

        combined_weight_size = input_size + hidden_size
        self.U = nn.Parameter(torch.randn(hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(self.feedback_dim, combined_weight_size) * 0.01)
        self.proj_out = None
        if feedback_dim is not None:
            self.proj_out = nn.Linear(self.output_feedback_dim, self.feedback_dim)
        self.gate_tau = 0.5
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.register_buffer("prev_feedback", None)

        self._init_gawf_params()
        self._init_gawf_diagnostics_state()

    def _init_gawf_params(self) -> None:
        """
        Explicit initialization hook for GaWF-specific parameters (U and V).
        Currently preserves the initialization defined in __init__ to keep behavior unchanged.
        """
        return

    def set_feedback_frozen(self, freeze: bool):
        params = [self.U, self.V]
        if self.proj_out is not None:
            params.extend(self.proj_out.parameters())
        for p in params:
            p.requires_grad = not freeze

    def _compute_feedback(self, char_t, pos_t):
        y_t = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is None:
            return y_t
        return self.proj_out(y_t)

    def middle_gawf(self, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        trans_ih, trans_hh = _compute_gawf_transforms(
            self.U,
            fb_t,
            self.V,
            input_size,
        )
        gate_logits_ih = trans_ih / self.gate_tau
        gate_logits_hh = trans_hh / self.gate_tau
        gate_ih = torch.sigmoid(gate_logits_ih)
        gate_hh = torch.sigmoid(gate_logits_hh)
        self._record_gawf_gate(0, gate_logits_ih, gate_logits_hh, gate_ih, gate_hh)
        ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
        hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        gated_output = self.LNormRNN(h_t)
        gated_output = F.relu(gated_output)
        return gated_output

    def forward(self, x, use_feedback=True, reset_feedback=False):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)

        if use_feedback:
            if reset_feedback or self.prev_feedback is None:
                fb = torch.zeros(batch_size, self.feedback_dim, device=x.device, dtype=torch.float32)
            else:
                fb = self.prev_feedback.to(device=x.device, dtype=torch.float32)

            hidden_size = self.rnn.hidden_size
            char_out = torch.empty(batch_size, frame_num, self.num_classes, device=x.device, dtype=x.dtype)
            pos_out = torch.empty(batch_size, frame_num, self.num_pos, device=x.device, dtype=x.dtype)
            h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)

            for t in range(frame_num):
                x_t = x[:, t, :]
                self._record_gawf_feedback(0, fb)
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                gated_output = self.middle_gawf(x_t, h, fb_t)
                gated_output = F.dropout(gated_output, p=self.rnn_dropout, training=self.training)
                char_t, pos_t = self.classifier(gated_output)
                if self.proj_out is None:
                    with torch.no_grad():
                        fb = self._compute_feedback(char_t, pos_t)
                else:
                    fb = self._compute_feedback(char_t, pos_t)
                h = gated_output
                char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

            self.prev_feedback = fb.detach().to(dtype=torch.float32)
        else:
            self.prev_feedback = None
            x, _ = self.rnn(x)
            x = self.LNormRNN(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.rnn_dropout, training=self.training)
            char_out, pos_out = self.classifier(x)

        return char_out, pos_out


class MultiLayerGaWFRNNConv(GaWFDiagnosticsMixin, BaseConvSequenceModel):
    """
    Multi-layer GaWF RNN model with explicit projected feedback.

    This class is intentionally separate from GaWFRNNConv so the existing
    single-layer legacy/projected GaWF behavior remains unchanged.
    """

    def __init__(
        self,
        num_classes,
        num_pos,
        kernel_size=3,
        device="cuda",
        cnn_dropout=0.0,
        rnn_dropout=0.5,
        hidden_size=256,
        max_chars=15,
        predict_all_chars=False,
        feedback_dim=None,
        num_layers=2,
    ):
        if predict_all_chars:
            raise ValueError("MultiLayerGaWFRNNConv currently supports single-character heads only.")
        super(MultiLayerGaWFRNNConv, self).__init__(
            num_classes,
            num_pos,
            kernel_size=kernel_size,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            hidden_size=hidden_size,
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
        )
        self.num_classes = num_classes
        self.num_pos = num_pos
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError(
                f"MultiLayerGaWFRNNConv requires num_layers >= 2, got {self.num_layers}"
            )

        self.output_feedback_dim = num_classes + num_pos
        requested_feedback_dim = None if feedback_dim is None else int(feedback_dim)
        if requested_feedback_dim is not None and requested_feedback_dim < 0:
            raise ValueError(
                f"feedback_dim must be >= 0 for multi-layer GaWF, got {requested_feedback_dim}"
            )
        self.use_feedback_projector = (
            requested_feedback_dim is not None and requested_feedback_dim > 0
        )
        self.feedback_dim = requested_feedback_dim if self.use_feedback_projector else 0

        input_size = self.encoder_flatten_size
        layer_input_sizes = [input_size] + [hidden_size] * (self.num_layers - 1)
        self.layer_feedback_dims = (
            [self.feedback_dim] * self.num_layers
            if self.use_feedback_projector
            else [hidden_size] * (self.num_layers - 1) + [self.output_feedback_dim]
        )
        self.top_feedback_dim = self.layer_feedback_dims[-1]
        self.rnns = nn.ModuleList(
            [
                nn.RNN(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                for layer_input_size in layer_input_sizes
            ]
        )
        self.LNormRNN = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(self.num_layers)]
        )

        self.U_layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, layer_feedback_dim) * 0.01)
                for layer_feedback_dim in self.layer_feedback_dims
            ]
        )
        self.V_layers = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(layer_feedback_dim, layer_input_size + hidden_size) * 0.01
                )
                for layer_feedback_dim, layer_input_size in zip(
                    self.layer_feedback_dims, layer_input_sizes
                )
            ]
        )

        if self.use_feedback_projector:
            self.hidden_projectors = nn.ModuleList(
                [
                    nn.Linear(hidden_size, self.feedback_dim)
                    for _ in range(self.num_layers - 1)
                ]
            )
            self.proj_out = nn.Linear(self.output_feedback_dim, self.feedback_dim)
        else:
            self.hidden_projectors = nn.ModuleList()
            self.proj_out = None
        self.gate_tau = 0.5
        self.register_buffer("prev_feedback", None)
        self._init_gawf_diagnostics_state()

    def set_feedback_frozen(self, freeze: bool):
        params = list(self.U_layers) + list(self.V_layers)
        if self.use_feedback_projector:
            params.extend(self.hidden_projectors.parameters())
            params.extend(self.proj_out.parameters())
        for p in params:
            p.requires_grad = not freeze

    def _compute_output_feedback(self, char_t, pos_t):
        y_t = torch.cat([char_t, pos_t], dim=-1)
        if self.proj_out is None:
            return y_t
        return self.proj_out(y_t)

    def _compute_hidden_feedback(self, h_t):
        return h_t.detach()

    def middle_gawf_layer(self, layer_idx, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        rnn = self.rnns[layer_idx]
        weight_ih = rnn.weight_ih_l0
        weight_hh = rnn.weight_hh_l0
        bias_ih = rnn.bias_ih_l0
        bias_hh = rnn.bias_hh_l0
        U = self.U_layers[layer_idx]
        V = self.V_layers[layer_idx]
        trans_ih, trans_hh = _compute_gawf_transforms(
            U,
            fb_t,
            V,
            input_size,
        )
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
        gated_weight_ih = gate_ih * weight_ih.unsqueeze(0)
        gated_weight_hh = gate_hh * weight_hh.unsqueeze(0)
        ih = torch.bmm(x_t.unsqueeze(1), gated_weight_ih.transpose(1, 2)).squeeze(1)
        hh = torch.bmm(h_prev.unsqueeze(1), gated_weight_hh.transpose(1, 2)).squeeze(1)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        gated_output = self.LNormRNN[layer_idx](h_t)
        gated_output = F.relu(gated_output)
        return gated_output

    def forward(self, x, use_feedback=True, reset_feedback=False):
        x = x.to(self.device)
        batch_size, frame_num, channels, height, width = x.size()
        x = x.view(batch_size * frame_num, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, frame_num, -1)

        if use_feedback:
            if reset_feedback or self.prev_feedback is None:
                fb_top = torch.zeros(
                    batch_size, self.top_feedback_dim, device=x.device, dtype=torch.float32
                )
            else:
                fb_top = self.prev_feedback.to(device=x.device, dtype=torch.float32)

            char_out = torch.empty(
                batch_size, frame_num, self.num_classes, device=x.device, dtype=x.dtype
            )
            pos_out = torch.empty(
                batch_size, frame_num, self.num_pos, device=x.device, dtype=x.dtype
            )
            h_states = [
                torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]

            for t in range(frame_num):
                layer_input = x[:, t, :]
                next_h_states = []
                for layer_idx in range(self.num_layers):
                    if layer_idx == self.num_layers - 1:
                        fb = fb_top
                    elif self.use_feedback_projector:
                        fb = self.hidden_projectors[layer_idx](h_states[layer_idx + 1])
                    else:
                        with torch.no_grad():
                            fb = self._compute_hidden_feedback(h_states[layer_idx + 1])
                    self._record_gawf_feedback(layer_idx, fb)
                    fb_t = fb.clamp(-10, 10).unsqueeze(2)
                    h_t = self.middle_gawf_layer(
                        layer_idx,
                        layer_input,
                        h_states[layer_idx],
                        fb_t,
                    )
                    h_t = F.dropout(h_t, p=self.rnn_dropout, training=self.training)
                    next_h_states.append(h_t)
                    layer_input = h_t

                char_t, pos_t = self.classifier(layer_input)
                if self.proj_out is None:
                    with torch.no_grad():
                        fb_top = self._compute_output_feedback(char_t, pos_t)
                else:
                    fb_top = self._compute_output_feedback(char_t, pos_t)
                h_states = next_h_states
                char_out[:, t, :], pos_out[:, t, :] = char_t, pos_t

            self.prev_feedback = fb_top.detach().to(dtype=torch.float32)
        else:
            self.prev_feedback = None
            layer_output = x
            for layer_idx, rnn in enumerate(self.rnns):
                layer_output, _ = rnn(layer_output)
                layer_output = self.LNormRNN[layer_idx](layer_output)
                layer_output = F.relu(layer_output)
                layer_output = F.dropout(
                    layer_output, p=self.rnn_dropout, training=self.training
                )
            char_out, pos_out = self.classifier(layer_output)

        return char_out, pos_out
