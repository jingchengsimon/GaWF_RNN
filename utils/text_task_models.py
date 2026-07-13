"""Text task models built from embedding/head wrappers and shared recurrent cores."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrent_cores.gawf import GaWFCore
from .recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore

PAD_ID = 0


class TextSequenceClassifier(nn.Module):
    """Embedding encoder -> recurrent core -> pooling -> classifier head."""

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
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")
        self.device = device
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.embed_dropout = float(embed_dropout)
        self.rnn_dropout = float(rnn_dropout)
        self.pooling = pooling
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.to(self.device)

    def middle(self, x: torch.Tensor) -> torch.Tensor:
        out, _state = self.core(x)
        return out

    def _pool(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, frame_num, hidden = seq.shape
        if self.pooling == "last":
            idx = (lengths - 1).clamp(min=0, max=frame_num - 1)
            idx = idx.view(batch_size, 1, 1).expand(batch_size, 1, hidden)
            return seq.gather(1, idx).squeeze(1)
        ar = torch.arange(frame_num, device=seq.device).unsqueeze(0)
        mask = (ar < lengths.clamp(max=frame_num).unsqueeze(1)).to(seq.dtype)
        summed = (seq * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return summed / denom

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.device)
        lengths = lengths.to(self.device).long()
        x = self.embedding(ids)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        seq = self.middle(x)
        seq = F.dropout(seq, p=self.rnn_dropout, training=self.training)
        pooled = self._pool(seq, lengths)
        return self.fc(pooled)


class TextRNN(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.core = RNNCore(
            self.embed_dim, self.hidden_size, dropout=0.0, num_layers=self.num_layers
        )
        self.to(self.device)


class TextGRU(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.core = GRUCore(
            self.embed_dim, self.hidden_size, dropout=0.0, num_layers=self.num_layers
        )
        self.to(self.device)


class TextLSTM(TextSequenceClassifier):
    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.core = LSTMCore(
            self.embed_dim, self.hidden_size, dropout=0.0, num_layers=self.num_layers
        )
        self.to(self.device)


class TextGaWF(TextSequenceClassifier):
    """GaWF text model using previous hidden state as feedback."""

    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.feedback_dim = self.hidden_size
        self.core = GaWFCore(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            feedback_dim=self.feedback_dim,
            dropout=0.0,
            num_layers=self.num_layers,
            layer_feedback_dims=([self.hidden_size] * self.num_layers),
        )
        self.to(self.device)

    @property
    def U(self):
        return self.core.U

    @property
    def V(self):
        return self.core.V

    def middle(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, frame_num, _ = x.shape
        h = self.core.initial_state(batch_size, x.device, x.dtype)
        outputs = torch.empty(
            batch_size,
            frame_num,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        for t in range(frame_num):
            if self.num_layers == 1:
                h = self.core.step(x[:, t, :], h, h)
                feature = h
            else:
                feedbacks = [part.detach() for part in h[1:]] + [h[-1]]
                feature, h = self.core.step(x[:, t, :], h, feedbacks)
            outputs[:, t, :] = feature
        return outputs


class TextGaWFLogits(TextSequenceClassifier):
    """GaWF text model using previous classifier logits as feedback."""

    include_fc_in_core_params = True

    def __init__(self, vocab_size: int, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.feedback_dim = self.num_classes
        self.core = GaWFCore(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            feedback_dim=self.feedback_dim,
            dropout=0.0,
            num_layers=self.num_layers,
            layer_feedback_dims=([self.hidden_size] * (self.num_layers - 1) + [self.feedback_dim]),
        )
        self.to(self.device)

    @property
    def U(self):
        return self.core.U

    @property
    def V(self):
        return self.core.V

    def _middle_with_logit_feedback(
        self,
        x: torch.Tensor,
        apply_recurrent_dropout: bool,
    ) -> torch.Tensor:
        batch_size, frame_num, _ = x.shape
        h = self.core.initial_state(batch_size, x.device, x.dtype)
        fb = torch.zeros(batch_size, self.feedback_dim, device=x.device, dtype=x.dtype)
        outputs = torch.empty(
            batch_size,
            frame_num,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        for t in range(frame_num):
            if self.num_layers == 1:
                h = self.core.step(x[:, t, :], h, fb)
                feature = h
            else:
                feedbacks = [part.detach() for part in h[1:]] + [fb]
                feature, h = self.core.step(x[:, t, :], h, feedbacks)
            if apply_recurrent_dropout:
                feature = F.dropout(feature, p=self.rnn_dropout, training=self.training)
            logits_t = self.fc(feature)
            fb = logits_t.detach()
            outputs[:, t, :] = feature
        return outputs

    def middle(self, x: torch.Tensor) -> torch.Tensor:
        return self._middle_with_logit_feedback(x, apply_recurrent_dropout=False)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.device)
        lengths = lengths.to(self.device).long()
        x = self.embedding(ids)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        seq = self._middle_with_logit_feedback(x, apply_recurrent_dropout=True)
        pooled = self._pool(seq, lengths)
        return self.fc(pooled)


def get_text_model_classes():
    """Factory mapping model type name to text model class."""
    return {
        "rnn": TextRNN,
        "lstm": TextLSTM,
        "gru": TextGRU,
        "gawf": TextGaWF,
        "gawf_logits": TextGaWFLogits,
    }
