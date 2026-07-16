"""Smoke tests for Atari recurrent actor-critic models without Gymnasium/ALE."""

from __future__ import annotations

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_task_models import AtariActorCritic


class AtariModelSmokeTest(unittest.TestCase):
    def test_multilayer_lstm_and_gawf_shapes(self) -> None:
        obs, prev_actions, prev_rewards, prev_dones, _actions = self._inputs(num_actions=4)
        for model_type, feedback_mode in (("lstm", "none"), ("gawf", "output")):
            with self.subTest(model_type=model_type):
                model = AtariActorCritic(
                    num_actions=4,
                    input_channels=4,
                    model_type=model_type,
                    hidden_size=16,
                    encoder_feature_dim=32,
                    feedback_mode=feedback_mode,
                    num_layers=2,
                )
                logits, values, state = model.forward_sequence(
                    obs, prev_actions, prev_rewards, prev_dones
                )
                self.assertEqual(logits.shape, (2, 4, 4))
                self.assertEqual(values.shape, (2, 4))
                if model_type == "lstm":
                    self.assertEqual(state.recurrent[0].shape, (2, 2, 16))
                else:
                    self.assertEqual(len(state.recurrent), 2)

    def _inputs(self, batch_size: int = 2, n_steps: int = 4, num_actions: int = 7):
        obs = torch.randint(0, 256, (batch_size, n_steps, 4, 84, 84), dtype=torch.uint8)
        prev_actions = torch.zeros(batch_size, n_steps, dtype=torch.long)
        prev_rewards = torch.zeros(batch_size, n_steps)
        prev_dones = torch.ones(batch_size, n_steps)
        actions = torch.zeros(batch_size, n_steps, dtype=torch.long)
        return obs, prev_actions, prev_rewards, prev_dones, actions

    def _assert_sequence_shapes(self, model: AtariActorCritic, num_actions: int) -> None:
        obs, prev_actions, prev_rewards, prev_dones, actions = self._inputs(num_actions=num_actions)
        logprobs, entropy, values, logits = model.evaluate_actions_sequence(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            actions,
        )
        self.assertEqual(logprobs.shape, (2, 4))
        self.assertEqual(entropy.shape, (2, 4))
        self.assertEqual(values.shape, (2, 4))
        self.assertEqual(logits.shape, (2, 4, num_actions))

    def test_lstm_shapes(self) -> None:
        num_actions = 7
        model = AtariActorCritic(
            num_actions=num_actions,
            input_channels=4,
            model_type="lstm",
            hidden_size=32,
            encoder_feature_dim=64,
            core_dropout=0.0,
            feedback_mode="none",
        )
        self.assertEqual(model.feedback_dim, 0)
        self._assert_sequence_shapes(model, num_actions)

    def test_gawf_none_feedback_shapes(self) -> None:
        num_actions = 7
        model = AtariActorCritic(
            num_actions=num_actions,
            input_channels=4,
            model_type="gawf",
            hidden_size=32,
            encoder_feature_dim=64,
            core_dropout=0.0,
            feedback_mode="none",
        )
        self.assertEqual(model.feedback_dim, 0)
        self._assert_sequence_shapes(model, num_actions)

    def test_gawf_output_feedback_shapes(self) -> None:
        num_actions = 7
        model = AtariActorCritic(
            num_actions=num_actions,
            input_channels=4,
            model_type="gawf",
            hidden_size=32,
            encoder_feature_dim=64,
            core_dropout=0.0,
            feedback_mode="output",
        )
        self.assertEqual(model.feedback_dim, num_actions + 1)
        self._assert_sequence_shapes(model, num_actions)

    def test_done_masks_recurrent_state(self) -> None:
        num_actions = 4
        model = AtariActorCritic(
            num_actions=num_actions,
            input_channels=4,
            model_type="lstm",
            hidden_size=16,
            encoder_feature_dim=32,
            core_dropout=0.0,
            feedback_mode="none",
        )
        model.eval()
        obs, prev_actions, prev_rewards, prev_dones, _actions = self._inputs(
            batch_size=1,
            n_steps=1,
            num_actions=num_actions,
        )
        state = model.initial_state(batch_size=1, device="cpu")
        recurrent = (torch.ones_like(state.recurrent[0]), torch.ones_like(state.recurrent[1]))
        dirty_state = type(state)(
            recurrent=recurrent,
            prev_policy_logits=torch.ones_like(state.prev_policy_logits),
            prev_value=torch.ones_like(state.prev_value),
        )
        logits_dirty, values_dirty, _state_dirty = model.forward_sequence(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            state=dirty_state,
        )
        logits_clean, values_clean, _state_clean = model.forward_sequence(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            state=None,
        )
        self.assertTrue(torch.allclose(logits_dirty, logits_clean))
        self.assertTrue(torch.allclose(values_dirty, values_clean))

    def test_non_gawf_model_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtariActorCritic(num_actions=4, model_type="dqn")

    def test_lstm_rejects_output_feedback(self) -> None:
        with self.assertRaises(ValueError):
            AtariActorCritic(num_actions=4, model_type="lstm", feedback_mode="output")


if __name__ == "__main__":
    unittest.main()
