"""Reusable text sequence-classification models for language benchmarks.

A small parallel subsystem to the vision ``*Conv`` models. The vision pipeline is
left untouched; here the **embedding layer is the encoder** (token id ->
``embed_dim`` vector per timestep) and a **single classification head is applied
once** per sequence after pooling. IMDB uses review tokens directly; SentiHood
uses flattened query-pair tokens (sentence + <sep> + location/aspect query).

Models share an identical ``nn.Embedding`` + ``SentimentHead`` so that any
parameter matching isolates the recurrent block.

- ``TextLSTM``  : thin wrapper over ``nn.LSTM``.
- ``TextGaWF``  : gated-weight-on-feedback RNN. Feedback is the **previous hidden
  state ``h_{t-1}`` directly** (not output logits), so ``feedback_dim = hidden_size``
  and there is **no ``proj_out`` / fixed-size ``dz`` projection** (on the vision task
  ``dz=8`` did worse and larger ``dz`` overfit). Reuses ``_compute_gawf_transforms``
  and ``gate_tau`` from the vision GaWF.
- ``TextGaWFLogits`` : vision-style GaWF for IMDB. Each timestep produces sentiment
  logits with the shared head, and the previous timestep logits provide the next
  feedback vector (``feedback_dim = num_classes``).
- ``TextGaWFMulti`` : multi-layer GaWF. By default it mirrors the vision
  ``gawf_multi`` direct-feedback path: lower layers receive the detached adjacent
  higher layer's previous hidden state, and the final layer receives detached
  previous sentiment logits. Passing ``feedback_dim > 0`` enables per-layer
  projectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .train_gawf_core import _compute_gawf_transforms

PAD_ID = 0


class TextSequenceClassifier(nn.Module):
    """Shared embed -> recurrence -> pool -> head pipeline.

    Subclasses implement ``middle(x) -> (B, T, hidden)`` returning the per-timestep
    hidden representation (already LayerNorm'd + activated). ``forward(ids, lengths)``
    applies dropout, pools once, and produces ``(B, num_classes)`` logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_classes: int = 2,
        embed_dropout: float = 0.0,
        rnn_dropout: float = 0.5,
        pooling: str = "last",
        padding_idx: int = PAD_ID,
        device: str = "cuda",
    ):
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")
        self.device = device
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embed_dropout = embed_dropout
        self.rnn_dropout = rnn_dropout
        self.pooling = pooling

        # Embedding = encoder. This + the head are the only params shared across models.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.to(self.device)

    def middle(self, x):  # pragma: no cover - interface
        raise NotImplementedError("Subclass must implement middle(x) -> (B, T, hidden)")

    def _pool(self, seq, lengths):
        batch_size, frame_num, hidden = seq.shape
        if self.pooling == "last":
            idx = (lengths - 1).clamp(min=0, max=frame_num - 1)
            idx = idx.view(batch_size, 1, 1).expand(batch_size, 1, hidden)
            return seq.gather(1, idx).squeeze(1)
        # mean over valid (non-pad) timesteps
        ar = torch.arange(frame_num, device=seq.device).unsqueeze(0)
        mask = (ar < lengths.clamp(max=frame_num).unsqueeze(1)).to(seq.dtype)
        summed = (seq * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return summed / denom

    def forward(self, ids, lengths):
        ids = ids.to(self.device)
        lengths = lengths.to(self.device).long()
        x = self.embedding(ids)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        seq = self.middle(x)
        seq = F.dropout(seq, p=self.rnn_dropout, training=self.training)
        pooled = self._pool(seq, lengths)
        return self.fc(pooled)


class TextLSTM(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.LNormRNN = nn.LayerNorm(self.hidden_size)
        self.to(self.device)

    def middle(self, x):
        out = self.lstm(x)[0]
        out = self.LNormRNN(out)
        out = F.relu(out)
        return out


class TextRNN(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.rnn = nn.RNN(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.LNormRNN = nn.LayerNorm(self.hidden_size)
        self.to(self.device)

    def middle(self, x):
        out = self.rnn(x)[0]
        out = self.LNormRNN(out)
        out = F.relu(out)
        return out


class TextGRU(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.LNormRNN = nn.LayerNorm(self.hidden_size)
        self.to(self.device)

    def middle(self, x):
        out = self.gru(x)[0]
        out = self.LNormRNN(out)
        out = F.relu(out)
        return out


class TextGaWF(TextSequenceClassifier):
    """GaWF with hidden-state feedback for single-label sequence classification."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        input_size = self.embed_dim
        hidden_size = self.hidden_size
        # Feedback = previous hidden state directly -> feedback_dim == hidden_size.
        self.feedback_dim = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        combined_weight_size = input_size + hidden_size
        self.U = nn.Parameter(torch.randn(hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(self.feedback_dim, combined_weight_size) * 0.01)
        self.gate_tau = 0.5
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.to(self.device)

    def _step(self, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        trans_ih, trans_hh = _compute_gawf_transforms(self.U, fb_t, self.V, input_size)
        gate_ih = torch.sigmoid(trans_ih / self.gate_tau)
        gate_hh = torch.sigmoid(trans_hh / self.gate_tau)
        ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
        hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        h_t = self.LNormRNN(h_t)
        h_t = F.relu(h_t)
        return h_t

    def middle(self, x):
        batch_size, frame_num, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = torch.empty(
            batch_size, frame_num, self.hidden_size, device=x.device, dtype=x.dtype
        )
        for t in range(frame_num):
            # Feedback = previous hidden state h_{t-1} (zeros at t=0).
            fb_t = h.clamp(-10, 10).unsqueeze(2)
            h = self._step(x[:, t, :], h, fb_t)
            outputs[:, t, :] = h
        return outputs


class TextGaWFLogits(TextSequenceClassifier):
    """GaWF with previous sentiment logits as the feedback vector.

    This keeps the legacy ``TextGaWF`` untouched while matching the vision GaWF
    design more closely: every recurrent step emits logits through ``fc`` and the
    next step gates its input/hidden weights with those logits. The final IMDB
    prediction still follows the text pipeline, pooling the hidden sequence once.
    """

    include_fc_in_core_params = True

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        input_size = self.embed_dim
        hidden_size = self.hidden_size
        self.feedback_dim = self.num_classes
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        combined_weight_size = input_size + hidden_size
        self.U = nn.Parameter(torch.randn(hidden_size, self.feedback_dim) * 0.01)
        self.V = nn.Parameter(torch.randn(self.feedback_dim, combined_weight_size) * 0.01)
        self.gate_tau = 0.5
        self.LNormRNN = nn.LayerNorm(hidden_size)
        self.to(self.device)

    def _step(self, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        weight_ih = self.rnn.weight_ih_l0
        weight_hh = self.rnn.weight_hh_l0
        bias_ih = self.rnn.bias_ih_l0
        bias_hh = self.rnn.bias_hh_l0
        trans_ih, trans_hh = _compute_gawf_transforms(self.U, fb_t, self.V, input_size)
        gate_ih = torch.sigmoid(trans_ih / self.gate_tau)
        gate_hh = torch.sigmoid(trans_hh / self.gate_tau)
        ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
        hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        h_t = self.LNormRNN(h_t)
        h_t = F.relu(h_t)
        return h_t

    def _middle_with_logit_feedback(self, x, apply_recurrent_dropout: bool):
        batch_size, frame_num, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fb = torch.zeros(batch_size, self.feedback_dim, device=x.device, dtype=x.dtype)
        outputs = torch.empty(
            batch_size, frame_num, self.hidden_size, device=x.device, dtype=x.dtype
        )
        for t in range(frame_num):
            fb_t = fb.clamp(-10, 10).unsqueeze(2)
            h = self._step(x[:, t, :], h, fb_t)
            if apply_recurrent_dropout:
                h = F.dropout(h, p=self.rnn_dropout, training=self.training)
            logits_t = self.fc(h)
            fb = logits_t.detach()
            outputs[:, t, :] = h
        return outputs

    def middle(self, x):
        return self._middle_with_logit_feedback(x, apply_recurrent_dropout=False)

    def forward(self, ids, lengths):
        ids = ids.to(self.device)
        lengths = lengths.to(self.device).long()
        x = self.embedding(ids)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        seq = self._middle_with_logit_feedback(x, apply_recurrent_dropout=True)
        pooled = self._pool(seq, lengths)
        return self.fc(pooled)


class TextGaWFMulti(TextSequenceClassifier):
    """Multi-layer GaWF for text sequence classification.

    The default direct-feedback mode follows ``MultiLayerGaWFRNNConv``: non-final
    recurrent layers are gated by the adjacent higher layer's previous hidden
    state, while the final recurrent layer is gated by previous output logits.
    """

    include_fc_in_core_params = True

    def __init__(
        self,
        vocab_size: int,
        feedback_dim: int | None = None,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__(vocab_size, **kwargs)
        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError(f"TextGaWFMulti requires num_layers >= 2, got {self.num_layers}")

        input_size = self.embed_dim
        hidden_size = self.hidden_size
        self.output_feedback_dim = self.num_classes
        requested_feedback_dim = None if feedback_dim is None else int(feedback_dim)
        if requested_feedback_dim is not None and requested_feedback_dim < 0:
            raise ValueError(
                f"feedback_dim must be >= 0 for multi-layer GaWF, got {requested_feedback_dim}"
            )
        self.use_feedback_projector = (
            requested_feedback_dim is not None and requested_feedback_dim > 0
        )
        self.feedback_dim = requested_feedback_dim if self.use_feedback_projector else 0

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
        self.to(self.device)

    def set_feedback_frozen(self, freeze: bool):
        params = list(self.U_layers) + list(self.V_layers)
        if self.use_feedback_projector:
            params.extend(self.hidden_projectors.parameters())
            params.extend(self.proj_out.parameters())
        for p in params:
            p.requires_grad = not freeze

    def _step(self, layer_idx, x_t, h_prev, fb_t):
        input_size = x_t.size(-1)
        rnn = self.rnns[layer_idx]
        weight_ih = rnn.weight_ih_l0
        weight_hh = rnn.weight_hh_l0
        bias_ih = rnn.bias_ih_l0
        bias_hh = rnn.bias_hh_l0
        U = self.U_layers[layer_idx]
        V = self.V_layers[layer_idx]
        trans_ih, trans_hh = _compute_gawf_transforms(U, fb_t, V, input_size)
        gate_ih = torch.sigmoid(trans_ih / self.gate_tau)
        gate_hh = torch.sigmoid(trans_hh / self.gate_tau)
        ih = torch.einsum("bi,bhi,hi->bh", x_t, gate_ih, weight_ih)
        hh = torch.einsum("bi,bhi,hi->bh", h_prev, gate_hh, weight_hh)
        if bias_ih is not None:
            ih = ih + bias_ih.unsqueeze(0)
        if bias_hh is not None:
            hh = hh + bias_hh.unsqueeze(0)
        h_t = torch.tanh(ih + hh)
        h_t = self.LNormRNN[layer_idx](h_t)
        return F.relu(h_t)

    def _hidden_feedback(self, layer_idx, h_states):
        if self.use_feedback_projector:
            return self.hidden_projectors[layer_idx](h_states[layer_idx + 1])
        with torch.no_grad():
            return h_states[layer_idx + 1].detach()

    def _output_feedback(self, logits_t):
        if self.proj_out is not None:
            return self.proj_out(logits_t)
        with torch.no_grad():
            return logits_t.detach()

    def _middle_with_feedback(self, x, apply_recurrent_dropout: bool):
        batch_size, frame_num, _ = x.shape
        h_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        fb_top = torch.zeros(batch_size, self.top_feedback_dim, device=x.device, dtype=x.dtype)
        outputs = torch.empty(
            batch_size, frame_num, self.hidden_size, device=x.device, dtype=x.dtype
        )

        for t in range(frame_num):
            layer_input = x[:, t, :]
            next_h_states = []
            for layer_idx in range(self.num_layers):
                if layer_idx == self.num_layers - 1:
                    fb = fb_top
                else:
                    fb = self._hidden_feedback(layer_idx, h_states)
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                h_t = self._step(layer_idx, layer_input, h_states[layer_idx], fb_t)
                if apply_recurrent_dropout:
                    h_t = F.dropout(h_t, p=self.rnn_dropout, training=self.training)
                next_h_states.append(h_t)
                layer_input = h_t

            logits_t = self.fc(layer_input)
            fb_top = self._output_feedback(logits_t)
            outputs[:, t, :] = layer_input
            h_states = next_h_states
        return outputs

    def middle(self, x):
        return self._middle_with_feedback(x, apply_recurrent_dropout=False)

    def forward(self, ids, lengths):
        ids = ids.to(self.device)
        lengths = lengths.to(self.device).long()
        x = self.embedding(ids)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        seq = self._middle_with_feedback(x, apply_recurrent_dropout=True)
        pooled = self._pool(seq, lengths)
        return self.fc(pooled)


def get_text_model_classes():
    """Factory mapping model-type name -> class (extensible to mamba/s5)."""
    return {
        "rnn": TextRNN,
        "lstm": TextLSTM,
        "gru": TextGRU,
        "gawf": TextGaWF,
        "gawf_logits": TextGaWFLogits,
        "gawf_multi": TextGaWFMulti,
    }
