"""Smoke tests for Atari DRQN-family Q-networks without Gymnasium/ALE."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.atari_dqn_models import (
    AtariQNetwork,
    AtariQNetworkState,
    normalize_atari_dqn_model_type,
)
from utils.atari_train_acceleration import AtariAcceleration

try:  # optional deps: only present on the GPU boxes (Amarel), not locally
    import s5  # noqa: F401

    HAS_S5 = True
except ImportError:
    HAS_S5 = False
try:
    import mamba_ssm  # noqa: F401

    # The installed mamba-ssm kernels require CUDA even when import succeeds on
    # a CPU-only login node.
    HAS_MAMBA = torch.cuda.is_available()
except ImportError:
    HAS_MAMBA = False


def _build_model(
    model_type: str,
    feedback_mode: str = "none",
    num_actions: int = 6,
    num_layers: int = 1,
) -> AtariQNetwork:
    return AtariQNetwork(
        num_actions=num_actions,
        input_channels=4,
        model_type=model_type,
        hidden_size=16,
        encoder_feature_dim=32,
        core_dropout=0.0,
        feedback_mode=feedback_mode,
        num_layers=num_layers,
    )


def _build_sequence_model(
    model_type: str, num_actions: int = 6, context_len: int = 4
) -> AtariQNetwork:
    return AtariQNetwork(
        num_actions=num_actions,
        input_channels=1,
        model_type=model_type,
        ssm_d_model=16,
        ssm_state_size=8,
        ssm_num_layers=1,
        ssm_context_len=context_len,
    )


def _inputs(batch_size: int = 2, n_steps: int = 4):
    obs = torch.randint(0, 256, (batch_size, n_steps, 4, 84, 84), dtype=torch.uint8)
    prev_dones = torch.ones(batch_size, n_steps)
    return obs, prev_dones


class AtariDQNModelSmokeTest(unittest.TestCase):
    def test_pong_dqn_defaults_to_one_ale_frame_per_step(self) -> None:
        from train_atari_dqn import build_arg_parser

        args = build_arg_parser().parse_args([])
        self.assertEqual(args.frame_skip, 1)
        self.assertEqual(args.frame_stack, 1)
        self.assertEqual(args.amp_dtype, "none")
        self.assertFalse(args.allow_tf32)
        self.assertFalse(args.cudnn_benchmark)
        self.assertFalse(args.fused_optimizer)
        self.assertFalse(args.compile_model)
        self.assertIsNone(args.env_ids)
        self.assertEqual(args.action_space_mode, "auto")
        self.assertEqual(args.replay_sampling, "task_balanced")

    def test_global_uniform_replay_remains_selectable(self) -> None:
        from train_atari_dqn import build_arg_parser

        args = build_arg_parser().parse_args(["--replay_sampling", "global_uniform"])
        self.assertEqual(args.replay_sampling, "global_uniform")

    def test_task_balanced_loss_gives_each_task_equal_weight(self) -> None:
        from train_atari_dqn import _aggregate_td_loss
        from utils.atari_replay import AtariReplayBuffer

        buffer = AtariReplayBuffer(
            buffer_size=10,
            num_envs=1,
            obs_shape=(1, 1, 1),
            device="cpu",
            seed=0,
            num_tasks=2,
        )
        loss = _aggregate_td_loss(
            elementwise_loss=torch.tensor([1.0, 1.0, 9.0]),
            task_ids=torch.tensor([0, 0, 1]),
            loss_mask=torch.ones(3),
            buffer=buffer,
        )
        self.assertAlmostEqual(float(loss), 5.0)

    def test_next_step_autoreset_resets_state_twice(self) -> None:
        from train_atari_dqn import _next_state_reset_flags

        terminal_step = _next_state_reset_flags(
            dones=np.array([1], dtype=np.uint8),
            autoreset_rows=np.array([0], dtype=np.uint8),
        )
        first_new_episode_step = _next_state_reset_flags(
            dones=np.array([0], dtype=np.uint8),
            autoreset_rows=np.array([1], dtype=np.uint8),
        )
        np.testing.assert_array_equal(terminal_step, np.array([1], dtype=np.uint8))
        np.testing.assert_array_equal(
            first_new_episode_step,
            np.array([1], dtype=np.uint8),
        )

    def test_multitask_phase0_resolves_to_full_action_space(self) -> None:
        from train_atari_dqn import _resolve_task_config, build_arg_parser

        args = build_arg_parser().parse_args(
            ["--env_ids", "ALE/Pong-v5", "ALE/Breakout-v5"]
        )
        env_ids, action_space_mode = _resolve_task_config(args)
        self.assertEqual(env_ids, ("ALE/Pong-v5", "ALE/Breakout-v5"))
        self.assertEqual(action_space_mode, "full18")

    def test_single_pong_task_config_preserves_minimal_actions(self) -> None:
        from train_atari_dqn import _resolve_task_config, build_arg_parser

        args = build_arg_parser().parse_args([])
        self.assertEqual(_resolve_task_config(args), (("ALE/Pong-v5",), "minimal"))

    def test_multitask_rejects_minimal_action_space(self) -> None:
        from train_atari_dqn import _resolve_task_config, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--env_ids",
                "ALE/Pong-v5",
                "ALE/Breakout-v5",
                "--action_space_mode",
                "minimal",
            ]
        )
        with self.assertRaisesRegex(ValueError, "full18"):
            _resolve_task_config(args)

    def test_ann_supports_phase0_eighteen_action_head(self) -> None:
        model = _build_model("ann", num_actions=18)
        obs, prev_dones = _inputs()
        q_values, state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, 18))
        self.assertIsNone(state)

    def test_ann_shapes(self) -> None:
        model = _build_model("ann")
        self.assertFalse(model.is_recurrent)
        self.assertEqual(model.feedback_dim, 0)
        obs, prev_dones = _inputs()
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, 6))
        self.assertIsNone(next_state)
        q_step, state = model.step(obs[:, 0], prev_dones[:, 0])
        self.assertEqual(q_step.shape, (2, 6))
        self.assertIsNone(state)

    def test_recurrent_shapes(self) -> None:
        for model_type in ("rnn", "gru", "lstm"):
            with self.subTest(model_type=model_type):
                model = _build_model(model_type)
                self.assertTrue(model.is_recurrent)
                self.assertEqual(model.feedback_dim, 0)
                obs, prev_dones = _inputs()
                q_values, next_state = model.forward_sequence(obs, prev_dones)
                self.assertEqual(q_values.shape, (2, 4, 6))
                self.assertIsNotNone(next_state)
                q_step, state = model.step(obs[:, 0], prev_dones[:, 0], state=next_state)
                self.assertEqual(q_step.shape, (2, 6))

    def test_gawf_qvalues_shapes(self) -> None:
        num_actions = 6
        model = _build_model("gawf", "qvalues", num_actions)
        self.assertEqual(model.feedback_dim, num_actions)
        obs, prev_dones = _inputs()
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, num_actions))
        self.assertEqual(next_state.recurrent.shape, (2, 16))
        self.assertEqual(next_state.prev_q.shape, (2, num_actions))

    def test_gawf_none_shapes(self) -> None:
        model = _build_model("gawf", "none")
        self.assertEqual(model.feedback_dim, 0)
        obs, prev_dones = _inputs()
        q_values, _ = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (2, 4, 6))

    def test_non_gawf_rejects_qvalues_feedback(self) -> None:
        for model_type in ("ann", "rnn", "gru", "lstm"):
            with self.subTest(model_type=model_type):
                with self.assertRaises(ValueError):
                    AtariQNetwork(num_actions=4, model_type=model_type, feedback_mode="qvalues")

    def test_invalid_model_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AtariQNetwork(num_actions=4, model_type="transformer")

    def test_historical_cnn_metadata_normalizes_to_ann(self) -> None:
        self.assertEqual(normalize_atari_dqn_model_type("cnn"), "ann")
        self.assertEqual(normalize_atari_dqn_model_type("lstm"), "lstm")

    def test_done_masks_state(self) -> None:
        for model_type in ("rnn", "gru", "lstm", "gawf"):
            with self.subTest(model_type=model_type):
                fb = "qvalues" if model_type == "gawf" else "none"
                model = _build_model(model_type, fb, num_actions=4)
                model.eval()
                obs, prev_dones = _inputs(batch_size=1, n_steps=1)
                clean = model.initial_state(batch_size=1, device="cpu")
                if isinstance(clean.recurrent, tuple):
                    recurrent = tuple(torch.ones_like(p) for p in clean.recurrent)
                else:
                    recurrent = torch.ones_like(clean.recurrent)
                dirty = AtariQNetworkState(
                    recurrent=recurrent, prev_q=torch.ones_like(clean.prev_q)
                )
                q_dirty, _ = model.forward_sequence(obs, prev_dones, state=dirty)
                q_clean, _ = model.forward_sequence(obs, prev_dones, state=None)
                self.assertTrue(torch.allclose(q_dirty, q_clean, atol=1e-5))

    def test_gradient_step_ann_branch(self) -> None:
        model = _build_model("ann", num_actions=4)
        target = _build_model("ann", num_actions=4)
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

    def test_multilayer_stepwise_shapes_and_reset(self) -> None:
        obs, prev_dones = _inputs()
        for model_type in ("ann", "rnn", "gru", "lstm", "gawf"):
            with self.subTest(model_type=model_type):
                feedback_mode = "qvalues" if model_type == "gawf" else "none"
                model = _build_model(model_type, feedback_mode, num_layers=2)
                q_values, state = model.forward_sequence(obs, prev_dones)
                self.assertEqual(q_values.shape, (2, 4, 6))
                if model_type == "ann":
                    self.assertIsNone(state)
                elif model_type == "lstm":
                    self.assertEqual(state.recurrent[0].shape, (2, 2, 16))
                    self.assertEqual(state.recurrent[1].shape, (2, 2, 16))
                elif model_type == "gawf":
                    self.assertEqual(len(state.recurrent), 2)
                    self.assertEqual(state.recurrent[0].shape, (2, 16))
                else:
                    self.assertEqual(state.recurrent.shape, (2, 2, 16))

    def test_fused_standard_recurrent_scan_matches_online_steps(self) -> None:
        batch, n_steps = 2, 5
        obs = torch.randint(0, 256, (batch, n_steps, 4, 84, 84), dtype=torch.uint8)
        prev_dones = torch.zeros(batch, n_steps)
        prev_dones[:, 0] = 1.0
        for num_layers in (1, 2):
            for model_type in ("rnn", "gru", "lstm"):
                with self.subTest(model_type=model_type, num_layers=num_layers):
                    model = _build_model(model_type, num_layers=num_layers)
                    model.eval()
                    q_fused, state_fused = model.forward_sequence(obs, prev_dones)
                    q_with_cpu_hint, _ = model.forward_sequence(
                        obs,
                        prev_dones,
                        has_internal_reset=False,
                    )
                    self.assertTrue(torch.allclose(q_fused, q_with_cpu_hint, atol=1e-5))
                    state_step = None
                    q_steps = []
                    for time_idx in range(n_steps):
                        q_step, state_step = model.step(
                            obs[:, time_idx], prev_dones[:, time_idx], state_step
                        )
                        q_steps.append(q_step)
                    q_stepwise = torch.stack(q_steps, dim=1)
                    self.assertTrue(torch.allclose(q_fused, q_stepwise, atol=1e-5))
                    if model_type == "lstm":
                        self.assertTrue(
                            torch.allclose(
                                state_fused.recurrent[0], state_step.recurrent[0], atol=1e-5
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                state_fused.recurrent[1], state_step.recurrent[1], atol=1e-5
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.allclose(
                                state_fused.recurrent, state_step.recurrent, atol=1e-5
                            )
                        )

    def test_cpu_reset_hint_preserves_stepwise_sequence(self) -> None:
        obs, prev_dones = _inputs(batch_size=2, n_steps=5)
        prev_dones.zero_()
        prev_dones[:, 0] = 1.0
        prev_dones[0, 3] = 1.0
        for model_type in ("rnn", "gru", "lstm", "gawf"):
            with self.subTest(model_type=model_type):
                feedback_mode = "qvalues" if model_type == "gawf" else "none"
                model = _build_model(model_type, feedback_mode)
                model.eval()
                q_default, _ = model.forward_sequence(obs, prev_dones)
                q_with_cpu_hint, _ = model.forward_sequence(
                    obs,
                    prev_dones,
                    has_internal_reset=True,
                )
                self.assertTrue(torch.allclose(q_default, q_with_cpu_hint, atol=1e-5))

    def test_cpu_acceleration_policy_disables_cuda_only_features(self) -> None:
        acceleration = AtariAcceleration(
            device=torch.device("cpu"),
            amp_dtype_name="bfloat16",
            allow_tf32=True,
            compile_model=True,
        )
        self.assertIsNone(acceleration.amp_dtype)
        fn = lambda value: value + 1
        self.assertIs(acceleration.compile_callable(fn), fn)

    def test_gradient_step_recurrent_branch(self) -> None:
        for model_type in ("rnn", "lstm", "gawf"):
            with self.subTest(model_type=model_type):
                num_actions = 4
                fb = "qvalues" if model_type == "gawf" else "none"
                model = _build_model(model_type, fb, num_actions)
                target = _build_model(model_type, fb, num_actions)
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
                q_taken = (
                    q_online[:, :seq_len].gather(-1, actions[:, :seq_len].unsqueeze(-1)).squeeze(-1)
                )
                with torch.no_grad():
                    q_target, _ = target.forward_sequence(obs, prev_dones)
                    q_next = q_target[:, 1:].max(-1).values
                    td_target = (
                        rewards[:, :seq_len].clamp(-1, 1) + 0.99 * (1 - dones[:, :seq_len]) * q_next
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

    def _check_sequence_core(self, model_type: str) -> None:
        num_actions = 6
        model = _build_sequence_model(model_type, num_actions, context_len=4)
        self.assertTrue(model.is_recurrent)
        self.assertTrue(model.uses_sequence_core)
        model.eval()

        # One-shot training path over a (B, T, C, H, W) window.
        batch, n_steps = 2, 5
        obs = torch.randint(0, 256, (batch, n_steps, 1, 84, 84), dtype=torch.uint8)
        prev_dones = torch.zeros(batch, n_steps)
        q_values, next_state = model.forward_sequence(obs, prev_dones)
        self.assertEqual(q_values.shape, (batch, n_steps, num_actions))
        self.assertIsNone(next_state)

        # Online rolling-window step carries a (B, context_len, F) buffer.
        q_step, state = model.step(obs[:, 0], torch.ones(batch))
        self.assertEqual(q_step.shape, (batch, num_actions))
        self.assertEqual(
            state.recurrent.shape, (batch, model.ssm_context_len, model.core.input_size)
        )

        # A prev_done=1 must clear the window: same frame with/without history match.
        _, state2 = model.step(obs[:, 1], torch.zeros(batch), state=state)
        q_reset, _ = model.step(obs[:, 2], torch.ones(batch), state=state2)
        q_fresh, _ = model.step(obs[:, 2], torch.ones(batch), state=None)
        self.assertTrue(torch.allclose(q_reset, q_fresh, atol=1e-5))

    @unittest.skipUnless(HAS_S5, "s5-pytorch not installed")
    def test_s5_sequence_core(self) -> None:
        self._check_sequence_core("s5")

    @unittest.skipUnless(HAS_MAMBA, "mamba-ssm not installed")
    def test_mamba_sequence_core(self) -> None:
        self._check_sequence_core("mamba")


class FlickerWrapperTest(unittest.TestCase):
    """Test the Flickering-Atari observation logic without a real Gymnasium env.

    ``_flicker`` takes ``gym`` as an argument, so we inject a fake module whose
    ``ObservationWrapper`` is a trivial base class and exercise the blanking logic
    directly (no ROMs / gymnasium required).
    """

    @staticmethod
    def _fake_gym():
        class _ObservationWrapper:
            def __init__(self, env):
                self.env = env

        class _FakeGym:
            ObservationWrapper = _ObservationWrapper

        return _FakeGym

    def test_disabled_returns_env_unchanged(self) -> None:
        from utils.atari_envs import _flicker

        sentinel = object()
        self.assertIs(_flicker(sentinel, self._fake_gym(), 0.0, seed=0), sentinel)

    def test_always_blank(self) -> None:
        import numpy as np

        from utils.atari_envs import _flicker

        wrapper = _flicker(object(), self._fake_gym(), 1.0, seed=0)
        frame = np.ones((84, 84), dtype=np.uint8)
        out = wrapper.observation(frame)
        self.assertTrue((np.asarray(out) == 0).all())
        self.assertEqual(np.asarray(out).shape, frame.shape)

    def test_never_blank(self) -> None:
        import numpy as np

        from utils.atari_envs import _flicker

        wrapper = _flicker(object(), self._fake_gym(), 1e-9, seed=0)
        frame = np.ones((84, 84), dtype=np.uint8)
        # prob ~ 0: overwhelmingly passes the true frame through.
        kept = sum(int((np.asarray(wrapper.observation(frame)) == 1).all()) for _ in range(50))
        self.assertGreaterEqual(kept, 49)

    def test_half_blank_rate(self) -> None:
        import numpy as np

        from utils.atari_envs import _flicker

        wrapper = _flicker(object(), self._fake_gym(), 0.5, seed=123)
        frame = np.ones((84, 84), dtype=np.uint8)
        blanks = sum(int((np.asarray(wrapper.observation(frame)) == 0).all()) for _ in range(2000))
        # ~50% blanked; wide tolerance to stay deterministic-ish across seeds.
        self.assertGreater(blanks, 850)
        self.assertLess(blanks, 1150)


if __name__ == "__main__":
    unittest.main()
