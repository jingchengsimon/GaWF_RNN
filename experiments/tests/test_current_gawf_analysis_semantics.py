"""Regression checks for the current no-tanh GaWF analysis rollout."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.analysis.clutter.fig3_gate_distribution import _trajectory
from utils.training.recurrent_cores.gawf import GaWFCore


class _AnalysisModel(nn.Module):
    """Minimal shell exposing the interfaces used by the trajectory collector."""

    def __init__(self) -> None:
        super().__init__()
        self.core = GaWFCore(5, 4, 3, dropout=0.0)
        self.head = nn.Linear(4, 3)

    @property
    def rnn(self) -> nn.RNN:
        return self.core.rnn

    @property
    def U(self) -> torch.Tensor:
        return self.core.U

    @property
    def V(self) -> torch.Tensor:
        return self.core.V

    @property
    def LNormRNN(self) -> nn.LayerNorm:
        return self.core.norm

    @property
    def feedback_dim(self) -> int:
        return self.core.feedback_dim

    @property
    def gate_tau(self) -> float:
        return self.core.gate_tau

    def classifier(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(hidden)
        return logits[:, :2], logits[:, 2:]


def test_gate_trajectory_matches_current_core_step() -> None:
    """The analysis rollout must use LayerNorm-ReLU directly, with no hidden tanh."""

    torch.manual_seed(7)
    model = _AnalysisModel().eval()
    encoded = torch.randn(2, 6, 5)

    char, sector, hidden, feedback = _trajectory(encoded, model, 1.0, True)

    state = model.core.initial_state(2, encoded.device, encoded.dtype)
    current_feedback = torch.zeros(2, 3)
    expected_hidden = []
    expected_feedback = []
    expected_char = []
    expected_sector = []
    for time_idx in range(encoded.shape[1]):
        expected_feedback.append(current_feedback)
        state = model.core.step(encoded[:, time_idx], state, current_feedback)
        char_t, sector_t = model.classifier(state)
        current_feedback = torch.cat((char_t, sector_t), dim=-1)
        expected_hidden.append(state)
        expected_char.append(char_t)
        expected_sector.append(sector_t)

    torch.testing.assert_close(hidden, torch.stack(expected_hidden, dim=1))
    torch.testing.assert_close(char, torch.stack(expected_char, dim=1))
    torch.testing.assert_close(sector, torch.stack(expected_sector, dim=1))
    assert feedback is not None
    torch.testing.assert_close(feedback, torch.stack(expected_feedback, dim=1))
