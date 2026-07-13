"""Unit tests for unified single- and multi-layer recurrent cores."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from utils.recurrent_cores.gawf import (
    GaWFCore,
    _compute_gawf_transforms,
    _gawf_layer_preactivation,
)
from utils.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore


class UnifiedRecurrentCoreTest(unittest.TestCase):
    def test_combined_gawf_transform_matches_split_reference_and_gradients(self) -> None:
        torch.manual_seed(3)
        batch_size, hidden_size, feedback_dim, input_size = 2, 5, 3, 7
        tensors = [
            torch.randn(hidden_size, feedback_dim, dtype=torch.float64, requires_grad=True),
            torch.randn(batch_size, feedback_dim, 1, dtype=torch.float64, requires_grad=True),
            torch.randn(
                feedback_dim,
                input_size + hidden_size,
                dtype=torch.float64,
                requires_grad=True,
            ),
        ]
        U, fb_t, V = tensors
        scaled_u = U.unsqueeze(0) * fb_t.transpose(1, 2)
        reference = (
            torch.matmul(scaled_u, V[:, :input_size]),
            torch.matmul(scaled_u, V[:, input_size:]),
        )
        actual = _compute_gawf_transforms(U, fb_t, V, input_size)
        for actual_part, reference_part in zip(actual, reference):
            self.assertTrue(torch.allclose(actual_part, reference_part, atol=1e-12, rtol=1e-12))

        weights = [torch.randn_like(part) for part in actual]
        actual_loss = sum((part * weight).sum() for part, weight in zip(actual, weights))
        actual_grads = torch.autograd.grad(actual_loss, tensors, retain_graph=True)
        reference_loss = sum(
            (part * weight).sum() for part, weight in zip(reference, weights)
        )
        reference_grads = torch.autograd.grad(reference_loss, tensors)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            self.assertTrue(
                torch.allclose(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)
            )

    def test_pure_tensor_gawf_preactivation_matches_legacy_formula(self) -> None:
        torch.manual_seed(5)
        batch_size, input_size, hidden_size, feedback_dim = 3, 7, 5, 4
        tensors = [
            torch.randn(*shape, dtype=torch.float64, requires_grad=True)
            for shape in (
                (batch_size, input_size),
                (batch_size, hidden_size),
                (batch_size, feedback_dim),
                (hidden_size, feedback_dim),
                (feedback_dim, input_size + hidden_size),
                (hidden_size, input_size),
                (hidden_size, hidden_size),
                (hidden_size,),
                (hidden_size,),
            )
        ]
        x_t, h_prev, feedback, U, V, weight_ih, weight_hh, bias_ih, bias_hh = tensors
        gate_tau = 0.5

        fb_t = feedback.clamp(-10, 10).unsqueeze(2)
        trans_ih, trans_hh = (
            torch.matmul((U.unsqueeze(0) * fb_t.transpose(1, 2)), V[:, :input_size]),
            torch.matmul((U.unsqueeze(0) * fb_t.transpose(1, 2)), V[:, input_size:]),
        )
        reference = (
            torch.einsum(
                "bi,bhi,hi->bh", x_t, torch.sigmoid(trans_ih / gate_tau), weight_ih
            )
            + torch.einsum(
                "bi,bhi,hi->bh",
                h_prev,
                torch.sigmoid(trans_hh / gate_tau),
                weight_hh,
            )
            + bias_ih.unsqueeze(0)
            + bias_hh.unsqueeze(0)
        )
        actual = _gawf_layer_preactivation(
            x_t,
            h_prev,
            feedback,
            U,
            V,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            gate_tau,
        )
        self.assertTrue(torch.allclose(actual, reference, atol=1e-12, rtol=1e-12))
        reference_grads = torch.autograd.grad(reference.sum(), tensors, retain_graph=True)
        actual_grads = torch.autograd.grad(actual.sum(), tensors)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            self.assertTrue(
                torch.allclose(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)
            )

    def test_compiled_feedback_path_and_diagnostics_fallback_match_eager(self) -> None:
        torch.manual_seed(7)
        eager = GaWFCore(7, 5, feedback_dim=3)
        accelerated = GaWFCore(7, 5, feedback_dim=3)
        accelerated.load_state_dict(eager.state_dict())
        x_t = torch.randn(3, 7)
        h_prev = torch.randn(3, 5)
        feedback = torch.randn(3, 3)
        expected = eager.step(x_t, h_prev, feedback)

        with mock.patch("torch.compile", side_effect=lambda fn, **_: fn):
            accelerated.configure_feedback_acceleration(True)
        actual = accelerated.step(x_t, h_prev, feedback)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertEqual(set(accelerated.state_dict()), set(eager.state_dict()))

        accelerated.begin_gawf_diagnostics()
        diagnostic_output = accelerated.step(x_t, h_prev, feedback)
        diagnostics = accelerated.pop_gawf_diagnostics()
        self.assertTrue(torch.allclose(diagnostic_output, expected, atol=1e-6, rtol=1e-6))
        self.assertGreater(diagnostics["gate_saturation_frac"], -1.0)

    def test_single_layer_parameter_names_remain_legacy(self) -> None:
        expected_torch_keys = {
            "rnn.weight_ih_l0",
            "rnn.weight_hh_l0",
            "rnn.bias_ih_l0",
            "rnn.bias_hh_l0",
            "norm.weight",
            "norm.bias",
        }
        for core_cls in (RNNCore, GRUCore, LSTMCore):
            with self.subTest(core=core_cls.__name__):
                self.assertEqual(set(core_cls(7, 5).state_dict()), expected_torch_keys)
        gawf = GaWFCore(7, 5, feedback_dim=3)
        self.assertEqual(set(gawf.state_dict()), expected_torch_keys | {"U", "V"})

    def test_multilayer_torch_core_shapes_and_gradients(self) -> None:
        x = torch.randn(3, 4, 7)
        for core_cls in (RNNCore, GRUCore, LSTMCore):
            with self.subTest(core=core_cls.__name__):
                core = core_cls(7, 5, num_layers=3)
                out, state = core(x)
                self.assertEqual(out.shape, (3, 4, 5))
                if core_cls is LSTMCore:
                    self.assertEqual(state[0].shape, (3, 3, 5))
                    self.assertEqual(state[1].shape, (3, 3, 5))
                else:
                    self.assertEqual(state.shape, (3, 3, 5))
                out.sum().backward()
                self.assertTrue(all(p.grad is not None for p in core.parameters()))

    def test_multilayer_gawf_feedback_and_no_feedback(self) -> None:
        core = GaWFCore(
            7,
            5,
            feedback_dim=3,
            num_layers=2,
            layer_feedback_dims=[5, 3],
        )
        state = core.initial_state(3, "cpu", torch.float32)
        x_t = torch.randn(3, 7)
        output, next_state = core.step(
            x_t,
            state,
            [torch.randn(3, 5), torch.randn(3, 3)],
        )
        self.assertEqual(output.shape, (3, 5))
        self.assertEqual(len(next_state), 2)
        nofb_output, nofb_state = core.step_no_feedback(x_t, state)
        self.assertEqual(nofb_output.shape, (3, 5))
        self.assertEqual(len(nofb_state), 2)
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in core.parameters()))
        self.assertTrue(any(key.startswith("rnns.0") for key in core.state_dict()))
        self.assertTrue(any(key.startswith("U_layers.0") for key in core.state_dict()))

    def test_invalid_depth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RNNCore(4, 4, num_layers=0)
        with self.assertRaises(ValueError):
            GaWFCore(4, 4, feedback_dim=2, num_layers=0)


if __name__ == "__main__":
    unittest.main()
