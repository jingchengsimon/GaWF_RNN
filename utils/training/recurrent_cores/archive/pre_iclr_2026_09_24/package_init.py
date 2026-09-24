"""Task-agnostic recurrent cores shared by clutter, text, and future RL models."""

from __future__ import annotations

from .gawf import (
    GaWFCore,
    configure_gawf_feedback_acceleration,
)
from .gawf_legacy import GaWFCoreLegacy, GaWFDiagnosticsMixin, _compute_gawf_transforms
from .brims import BRIMsCore
from .hyper_lstm import HyperLSTMCore
from .mlstm import MLSTMCore
from .rnn import GRUCore, LSTMCore, RNNCore, TorchRecurrentCore

__all__ = [
    "GaWFCore",
    "GaWFCoreLegacy",
    "GaWFDiagnosticsMixin",
    "BRIMsCore",
    "GRUCore",
    "HyperLSTMCore",
    "LSTMCore",
    "MLSTMCore",
    "RNNCore",
    "TorchRecurrentCore",
    "_compute_gawf_transforms",
    "configure_gawf_feedback_acceleration",
]
