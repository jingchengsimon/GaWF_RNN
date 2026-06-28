"""Text sequence-classification models for the IMDB benchmark.

A small parallel subsystem to the vision ``*Conv`` models. The vision pipeline is
left untouched; here the **embedding layer is the encoder** (token id ->
``embed_dim`` vector per timestep) and a **single sentiment head is applied once**
per sequence after pooling. Intermediate timesteps only update hidden state; no
per-step logits (IMDB is sequence *classification*, not the per-frame labeling of
the vision task).

Models share an identical ``nn.Embedding`` + ``SentimentHead`` so that any
parameter matching isolates the recurrent block.

- ``TextLSTM``  : thin wrapper over ``nn.LSTM``.
- ``TextGaWF``  : gated-weight-on-feedback RNN. Feedback is the **previous hidden
  state ``h_{t-1}`` directly** (not output logits), so ``feedback_dim = hidden_size``
  and there is **no ``proj_out`` / fixed-size ``dz`` projection** (on the vision task
  ``dz=8`` did worse and larger ``dz`` overfit). Reuses ``_compute_gawf_transforms``
  and ``gate_tau`` from the vision GaWF.
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


def get_text_model_classes():
    """Factory mapping model-type name -> class (extensible to mamba/s5)."""
    return {
        "rnn": TextRNN,
        "lstm": TextLSTM,
        "gru": TextGRU,
        "gawf": TextGaWF,
    }
