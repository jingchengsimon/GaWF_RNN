"""Task-agnostic recurrent cores shared by clutter, text, and future RL models."""

from __future__ import annotations

from .gawf import (
    GaWFCore,
    GaWFDiagnosticsMixin,
    _compute_gawf_transforms,
    configure_gawf_feedback_acceleration,
)
from .rnn import GRUCore, LSTMCore, RNNCore, TorchRecurrentCore

__all__ = [
    "GaWFCore",
    "GaWFDiagnosticsMixin",
    "GRUCore",
    "LSTMCore",
    "RNNCore",
    "TorchRecurrentCore",
    "_compute_gawf_transforms",
    "configure_gawf_feedback_acceleration",
]
