"""Check the Seaquest sweep commands and reject incomplete/non-finite smoke results.

Run with python -B -m unittest experiments.tests.test_seaquest_diagnostics.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.remote.run_sjc_seaquest_diagnostics import (
    HIDDEN, VARIANTS, accept_smoke, command, run_unit, validate,
)


class SeaquestDiagnosticTests(unittest.TestCase):
    def test_matrix(self) -> None:
        from utils.training.train_scripts.atari_dqn import build_arg_parser, _linear_epsilon

        parser = build_arg_parser()
        self.assertEqual(list(VARIANTS), ["a", "b", "c", "f995", "f999"])
        for model in HIDDEN:
            for variant, (epsilon, exploration, decay, gamma) in VARIANTS.items():
                args = parser.parse_args(command(model, variant, Path("/tmp/example"))[4:])
                self.assertEqual(args.env_id, "ALE/Seaquest-v5")
                self.assertEqual(args.hidden_size, HIDDEN[model])
                self.assertEqual(args.action_space_mode, "full18")
                self.assertEqual(args.total_timesteps, 3_000_000)
                self.assertEqual(args.gamma, gamma)
                self.assertEqual(args.learning_rate_decay_step, decay)
                self.assertEqual(args.exploration_steps, exploration)
                self.assertAlmostEqual(_linear_epsilon(args, exploration), epsilon)
                self.assertAlmostEqual(_linear_epsilon(args, exploration // 2), (1 + epsilon) / 2)
                self.assertFalse(args.double_dqn)
        self.assertEqual(VARIANTS["c"][2], 2_000_000)
        self.assertEqual(VARIANTS["f995"][:3], VARIANTS["b"][:3])
        self.assertEqual(VARIANTS["f999"][:3], VARIANTS["b"][:3])

    def test_partial_result_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "lstm_a_seed2"
            result.mkdir()
            (result / "metrics_history.jsonl").write_text("protected\n")
            with patch("subprocess.run") as launch, self.assertRaises(RuntimeError):
                run_unit("lstm", "a", result, root / "artifacts", False)
            launch.assert_not_called()
            self.assertEqual((result / "metrics_history.jsonl").read_text(), "protected\n")

    def test_cleanup_rejects_formal_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "lstm_a_seed2"
            formal.mkdir()
            with self.assertRaises(RuntimeError):
                accept_smoke(formal, root / "artifact", root / "accepted.json",
                             {"model_type": "lstm"})
            self.assertTrue(formal.is_dir())

    def test_missing_metrics_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                validate(Path(temporary), "gawf", "a", True)


if __name__ == "__main__":
    unittest.main()
