"""Smoke tests for MiniGrid PPO acceleration and vector-backend configuration."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from train_minigrid_ppo import _materialize_ppo_stats, build_arg_parser
from utils.minigrid_envs import make_vector_minigrid_env
from utils.minigrid_models import MiniGridEncoder
from utils.minigrid_ppo_models import MiniGridActorCritic
from utils.minigrid_train_acceleration import MiniGridAcceleration


class MiniGridPPOAccelerationSmokeTest(unittest.TestCase):
    """Check acceleration defaults without requiring MiniGrid or CUDA."""

    def test_parser_preserves_compatible_defaults(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.env_backend, "sync")
        self.assertEqual(args.amp_dtype, "none")
        self.assertFalse(args.allow_tf32)
        self.assertFalse(args.cudnn_benchmark)
        self.assertFalse(args.fused_optimizer)
        self.assertFalse(args.compile_model)

    def test_cpu_policy_disables_cuda_only_features(self) -> None:
        acceleration = MiniGridAcceleration(
            device=torch.device("cpu"),
            amp_dtype_name="bfloat16",
            allow_tf32=True,
            cudnn_benchmark=True,
            compile_model=True,
        )
        self.assertIsNone(acceleration.amp_dtype)
        fn = lambda value: value + 1
        self.assertIs(acceleration.compile_callable(fn), fn)

    def test_logging_stats_materialize_only_on_request(self) -> None:
        values = _materialize_ppo_stats(
            torch.tensor(1.25),
            torch.tensor(2.5),
            torch.tensor(0.75),
        )
        self.assertEqual(values, (1.25, 2.5, 0.75))

    def test_invalid_vector_backend_fails_before_optional_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MiniGrid vector backend"):
            make_vector_minigrid_env(
                "MiniGrid-MemoryS7-v0",
                seed=42,
                num_envs=1,
                vector_backend="threads",  # type: ignore[arg-type]
            )

    def test_recurrent_ppo_gradient_step(self) -> None:
        batch_size, sequence_length, num_actions = 2, 4, 7
        obs = torch.randint(
            0,
            6,
            (batch_size, sequence_length, 3, 3, 3),
            dtype=torch.uint8,
        )
        prev_dones = torch.zeros(batch_size, sequence_length)
        prev_dones[:, 0] = 1.0
        prev_dones[0, 2] = 1.0
        actions = torch.randint(0, num_actions, (batch_size, sequence_length))
        advantages = torch.randn(batch_size, sequence_length)
        returns = torch.randn(batch_size, sequence_length)

        for model_type in ("lstm", "gawf"):
            with self.subTest(model_type=model_type):
                model = MiniGridActorCritic(
                    num_actions=num_actions,
                    encoder=MiniGridEncoder(
                        output_size=16,
                        encoder_type="mlp",
                        grid_size=3,
                        hidden_size=16,
                    ),
                    model_type=model_type,
                    hidden_size=16,
                    num_layers=1,
                )
                self.assertEqual(model.num_layers, 1)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                with torch.no_grad():
                    old_logp, _, _ = model.evaluate_actions_sequence(
                        obs,
                        prev_dones,
                        actions,
                    )
                new_logp, entropy, values = model.evaluate_actions_sequence(
                    obs,
                    prev_dones,
                    actions,
                )
                ratio = (new_logp - old_logp).exp()
                policy_loss = -(ratio * advantages).mean()
                value_loss = 0.5 * F.mse_loss(values, returns)
                loss = policy_loss - 0.01 * entropy.mean() + 0.5 * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
