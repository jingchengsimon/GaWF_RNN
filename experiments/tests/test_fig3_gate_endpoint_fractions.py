"""Tests for strict GaWF gate endpoint counting."""

from __future__ import annotations

import numpy as np
import torch

from utils.analysis.clutter.fig3_gate_endpoint_fractions import count_gate_endpoints
from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors


def test_count_gate_endpoints_matches_direct_float32_comparison() -> None:
    """Streaming strict counts must match a direct reconstruction after reset exclusion."""

    feedback = np.asarray([[[0.0, 0.0], [0.2, -0.4], [1.0, 0.5]]], dtype=np.float32)
    u = np.asarray([[1.0, -0.5], [-1.5, 0.25]], dtype=np.float32)
    v = np.asarray(
        [[1.0, -0.5, 0.25, -1.0], [-0.75, 0.5, 1.5, 0.25]], dtype=np.float32
    )
    expected_feedback = torch.from_numpy(feedback.reshape(-1, 2)[1:])
    expected = _gate_tensors(
        expected_feedback, torch.from_numpy(u), torch.from_numpy(v), input_size=2, tau=0.5
    )

    actual = count_gate_endpoints(
        feedback, u, v, input_size=2, tau=0.5, chunk_size=1, device=torch.device("cpu")
    )
    for kind, gate in zip(("input", "recurrent"), expected):
        assert actual[kind]["total"] == gate.numel()
        assert actual[kind]["below_0_1"] == int(torch.count_nonzero(gate < 0.1))
        assert actual[kind]["above_0_9"] == int(torch.count_nonzero(gate > 0.9))
        assert actual[kind]["equal_0_1"] == int(torch.count_nonzero(gate == 0.1))
        assert actual[kind]["equal_0_9"] == int(torch.count_nonzero(gate == 0.9))
