"""Verify the input-path intervention and fixed-point stability on known small systems."""
import numpy as np
import torch
from utils.analysis.clutter.rnn_fixed_points import step, search, characterize


def test_autonomous_removes_input_bias_but_retains_recurrent_bias() -> None:
    rnn = torch.nn.RNN(1, 1, batch_first=True)
    with torch.no_grad():
        rnn.weight_ih_l0.fill_(2)
        rnn.weight_hh_l0.fill_(0.5)
        rnn.bias_ih_l0.fill_(0.8)
        rnn.bias_hh_l0.fill_(0.1)
    h = torch.tensor([[0.2]])
    x = torch.tensor([[0.4]])
    torch.testing.assert_close(step(rnn, h, x, True), torch.tanh(0.5 * h + 0.1))
    torch.testing.assert_close(step(rnn, h, x), torch.tanh(2 * x + 0.8 + 0.5 * h + 0.1))
    assert not torch.allclose(step(rnn, h, x, True), step(rnn, h, torch.zeros_like(x)))
    assert float(rnn.bias_ih_l0[0]) > 0.79


def test_search_refinement_dedup_and_eigenvalues() -> None:
    rnn = torch.nn.RNN(1, 1, batch_first=True)
    with torch.no_grad():
        rnn.weight_hh_l0.fill_(0.5)
        rnn.bias_hh_l0.fill_(0.1)
    for p in rnn.parameters():
        p.requires_grad_(False)
    h = torch.tensor([[-0.8], [0.0], [0.8]])
    inputs = torch.zeros(3, 1)
    candidates, q = search(rnn, h, inputs, True, 100, 0.03)
    result = characterize(rnn, candidates, inputs, True, 1e-8, 1e-4)
    assert result['roots'].shape == (1, 1)
    root = float(result['roots'][0, 0])
    assert abs(np.tanh(0.5 * root + 0.1) - root) < 1e-7
    np.testing.assert_allclose(result['spectral_radius'], [(1-root**2)*0.5], atol=1e-6)
    assert result['linearization_relative_error'][0] < 1e-3
