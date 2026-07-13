"""Unit tests for unified single- and multi-layer recurrent cores."""

from __future__ import annotations

import unittest

import torch

from utils.recurrent_cores.gawf import GaWFCore
from utils.recurrent_cores.rnn import GRUCore, LSTMCore, RNNCore


class UnifiedRecurrentCoreTest(unittest.TestCase):
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
