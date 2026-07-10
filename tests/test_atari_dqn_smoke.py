"""Smoke tests for Atari DQN models without Gymnasium/ALE."""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_dqn_models import AtariQNetwork, AtariQNetworkState


def _build_model(model_type: str, feedback_mode: str, num_actions: int = 6) -> AtariQNetwork:
    return AtariQNetwork(
        num_actions=num_actions,
        input_channels=4,
        model_type=model_type,
        hidden_size=16,
        encoder_feature_dim=32,
        core_dropout=0.0,
        feedback_mode=feedback_mode,
    )


def _inputs(batch_size: int = 2, n_steps: int = 4):
    obs = torch.randint(0, 256, (batch_size, n_steps, 4, 84, 84), dtype=torch.uint8)
    prev_dones = torch.ones(batch_size, n_steps)
    return obs, prev_dones


class AtariDQNModelSmokeTest(unittest.TestCase):
    def test_cnn_shapes(self) -> None:
        model = _build_model("cnn", "none")
        self.assertEqual(model.feedback_dim, 0)
        obs, prev_dones = _inputs()
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, 6))
        self.assertIsNone(next_state)
        q_step, state = model.step(obs[:, 0], prev_dones[:, 0])
        self.assertEqual(q_step.shape, (2, 6))
        self.assertIsNone(state)

    def test_gawf_qvalues_shapes(self) -> None:
        num_actions = 6
        model = _build_model("gawf", "qvalues", num_actions)
        self.assertEqual(model.feedback_dim, num_actions)
        obs, prev_dones = _inputs()
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, num_actions))
        self.assertEqual(next_state.recurrent.shape, (2, 16))
        self.assertEqual(next_state.prev_q.shape, (2, num_actions))
        q_step, state = model.step(obs[:, 0], prev_dones[:, 0], state=next_state)
        self.assertEqual(q_step.shape, (2, num_actions))
        self.assertEqual(state.recurrent.shape, (2, 16))

    def test_gawf_none_shapes(self) -> None:
        model = _build_model("gawf", "none")
        self.assertEqual(model.feedback_dim, 0)
        obs, prev_dones = _inputs()
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, 6))
        self.assertIsNotNone(next_state)

    def test_cnn_rejects_qvalues_feedback(self) -> None:
        with self.assertRaises(ValueError):
            AtariQNetwork(num_actions=4, model_type="cnn", feedback_mode="qvalues")

    def test_invalid_model_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtariQNetwork(num_actions=4, model_type="lstm")

    def test_done_masks_state(self) -> None:
        model = _build_model("gawf", "qvalues", num_actions=4)
        model.eval()
        obs, prev_dones = _inputs(batch_size=1, n_steps=1)
        dirty_state = AtariQNetworkState(
            recurrent=torch.ones(1, 16),
            prev_q=torch.ones(1, 4),
        )
        q_dirty, _ = model.forward_sequence(obs, prev_dones, state=dirty_state)
        q_clean, _ = model.forward_sequence(obs, prev_dones, state=None)
        self.assertTrue(torch.allclose(q_dirty, q_clean))

    def test_gradient_step_cnn_branch(self) -> None:
        model = _build_model("cnn", "none", num_actions=4)
        target = _build_model("cnn", "none", num_actions=4)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = 4
        obs = torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8)
        next_obs = torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8)
        actions = torch.randint(0, 4, (batch,))
        rewards = torch.randn(batch)
        dones = torch.tensor([0.0, 1.0, 0.0, 0.0])

        q_all, _ = model.step(obs, torch.zeros(batch))
        q_taken = q_all.gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next, _ = target.step(next_obs, torch.zeros(batch))
            td_target = rewards.clamp(-1, 1) + 0.99 * (1 - dones) * q_next.max(dim=1).values
        loss = F.smooth_l1_loss(q_taken, td_target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
        optimizer.step()

    def test_gradient_step_gawf_branch(self) -> None:
        num_actions = 4
        model = _build_model("gawf", "qvalues", num_actions)
        target = _build_model("gawf", "qvalues", num_actions)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch, seq_len = 2, 4
        obs = torch.randint(0, 256, (batch, seq_len + 1, 4, 84, 84), dtype=torch.uint8)
        actions = torch.randint(0, num_actions, (batch, seq_len + 1))
        rewards = torch.randn(batch, seq_len + 1)
        dones = torch.zeros(batch, seq_len + 1)
        dones[0, 2] = 1.0
        prev_dones = torch.zeros(batch, seq_len + 1)
        prev_dones[:, 0] = 1.0
        prev_dones[0, 3] = 1.0
        loss_mask = torch.ones(batch, seq_len + 1)
        loss_mask[0, 3] = 0.0

        q_online, _ = model.forward_sequence(obs, prev_dones)
        q_taken = q_online[:, :seq_len].gather(
            -1, actions[:, :seq_len].unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            q_target, _ = target.forward_sequence(obs, prev_dones)
            q_next = q_target[:, 1:].max(-1).values
            td_target = (
                rewards[:, :seq_len].clamp(-1, 1)
                + 0.99 * (1 - dones[:, :seq_len]) * q_next
            )
        mask = loss_mask[:, :seq_len]
        loss = (
            F.smooth_l1_loss(q_taken, td_target, reduction="none") * mask
        ).sum() / mask.sum().clamp(min=1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
