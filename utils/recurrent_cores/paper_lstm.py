"""Plain LSTM core matching the OpenAI Baselines recurrent policy path.

This module intentionally omits the LayerNorm, ReLU, and output dropout used by
the project's regular ``LSTMCore``. It exists only for paper-reproduction
baselines that require a plain LSTM output before actor and critic heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaperLSTMCore(nn.Module):
    """Single-layer plain LSTM with Baselines-style recurrent initialization."""

    def __init__(self, input_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = self.hidden_size
        self.num_layers = 1
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use orthogonal recurrent weights, zero bias, and unit forget bias."""
        for name, parameter in self.rnn.named_parameters():
            if "weight" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(parameter)
        # PyTorch adds both bias tensors. Put the full +1 forget bias in bias_ih.
        forget = slice(self.hidden_size, 2 * self.hidden_size)
        with torch.no_grad():
            self.rnn.bias_ih_l0[forget].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run a batch-first sequence without post-LSTM normalization or activation."""
        if state is None:
            return self.rnn(x)
        return self.rnn(x, state)
