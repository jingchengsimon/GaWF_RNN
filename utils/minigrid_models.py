"""MiniGrid observation encoders for the DRQN family.

MiniGrid's default observation is a 7x7x3 *symbolic* egocentric view: each cell
holds (object_idx, color_idx, state_idx) integer codes, not pixels. This module
provides a small, swappable encoder that one-hot encodes those categorical codes
and compresses them to a fixed feature vector, which feeds the shared recurrent
cores (RNN/GRU/LSTM/GaWF/S5/Mamba) unchanged via ``AtariQNetwork(encoder_factory=...)``.

Two encoder forms (perception is trivial here, so both are small and kept
identical across models so differences reflect memory, not encoder capacity):
  - ``mlp`` (default): flatten one-hot -> Linear -> ReLU -> feature.
  - ``cnn``           : small 2x2 convs over the 7x7 grid -> flatten -> Linear.

Sizing defaults follow standard MiniGrid baselines (torch-ac / BabyAI): a compact
~128-d feature. Channel-first ``(B, 3, H, W)`` input matches the ``(B, C, H, W)``
convention the DRQN encoder slot already expects.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# MiniGrid constants (minigrid.core.constants): 11 object types, 6 colors,
# 3 door states. Kept here so this module has no hard minigrid import.
MINIGRID_VOCAB = (11, 6, 3)

MiniGridEncoderType = Literal["mlp", "cnn"]


class MiniGridEncoder(nn.Module):
    """One-hot symbolic MiniGrid encoder returning a fixed feature vector."""

    def __init__(
        self,
        output_size: int = 128,
        encoder_type: MiniGridEncoderType = "mlp",
        grid_size: int = 7,
        vocab: tuple[int, int, int] = MINIGRID_VOCAB,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if encoder_type not in ("mlp", "cnn"):
            raise ValueError(f"Unsupported MiniGrid encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.output_size = int(output_size)
        self.grid_size = int(grid_size)
        self.vocab = tuple(int(v) for v in vocab)
        self.in_channels = sum(self.vocab)  # one-hot channels (11+6+3 = 20)

        if encoder_type == "mlp":
            flat = self.in_channels * self.grid_size * self.grid_size
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.output_size),
                nn.ReLU(),
            )
        else:  # cnn: small kernels suited to the 7x7 grid (Nature-DQN conv is too big)
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, self.in_channels, self.grid_size, self.grid_size)
                conv_out = int(self.conv(dummy).shape[-1])
            self.proj = nn.Sequential(nn.Linear(conv_out, self.output_size), nn.ReLU())

    def _one_hot(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) int codes -> (B, sum(vocab), H, W) float one-hot."""
        obs = obs.long()
        planes = []
        for ch, n in enumerate(self.vocab):
            idx = obs[:, ch, :, :].clamp(0, n - 1)  # guard out-of-range codes
            oh = F.one_hot(idx, num_classes=n).permute(0, 3, 1, 2).float()
            planes.append(oh)
        return torch.cat(planes, dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._one_hot(obs)
        if self.encoder_type == "mlp":
            return self.net(x)
        return self.proj(self.conv(x))
