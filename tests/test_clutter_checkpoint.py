import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_model import _load_clutter_checkpoint, _save_clutter_checkpoint
from utils.clutter_train_helpers import build_arg_parser


class _Loader:
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(11)
        self.sampler = SimpleNamespace(generator=torch.Generator().manual_seed(12))


def _components(model: torch.nn.Module):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.ones(2, 3)
    model(inputs).sum().backward()
    optimizer.step()
    return {
        "optim": optimizer,
        "scaler": None,
        "train_dl": _Loader(),
        "train_acc_char": np.arange(10, dtype=np.float64),
        "val_acc_char": np.arange(10, dtype=np.float64) + 0.5,
        "train_loss_pos": None,
    }


class ClutterCheckpointTest(unittest.TestCase):
    def test_round_trip_restores_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = torch.nn.Linear(3, 2)
            components = _components(model)
            metadata = {"model_type": "rnn", "seed": 4, "num_epochs": 10}
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            expected_model = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            expected_metric = components["train_acc_char"].copy()
            checkpoint_path = Path(temporary_dir) / "state.pth"

            _save_clutter_checkpoint(
                str(checkpoint_path),
                mdl=model,
                components=components,
                metadata=metadata,
                completed_epochs=5,
                best_val_acc=82.5,
                best_epoch_idx=3,
                best_state=best_state,
                epochs_without_improvement=1,
                stopped_by_patience=False,
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            components["train_acc_char"].fill(-1)
            components["train_dl"].generator.manual_seed(99)
            components["train_dl"].sampler.generator.manual_seed(98)

            checkpoint = _load_clutter_checkpoint(
                str(checkpoint_path),
                mdl=model,
                components=components,
                expected_metadata=metadata,
            )

            self.assertEqual(checkpoint["completed_epochs"], 5)
            self.assertEqual(checkpoint["best_epoch_idx"], 3)
            self.assertEqual(checkpoint["epochs_without_improvement"], 1)
            self.assertTrue(np.array_equal(components["train_acc_char"], expected_metric))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]))
            self.assertTrue(
                torch.equal(
                    components["train_dl"].generator.get_state(),
                    checkpoint["loader_rng_state"]["loader_generator"],
                )
            )
            self.assertTrue(
                torch.equal(
                    components["train_dl"].sampler.generator.get_state(),
                    checkpoint["loader_rng_state"]["sampler_generator"],
                )
            )

    def test_rejects_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = torch.nn.Linear(3, 2)
            components = _components(model)
            checkpoint_path = Path(temporary_dir) / "state.pth"
            metadata = {"model_type": "gru", "seed": 1}
            _save_clutter_checkpoint(
                str(checkpoint_path),
                mdl=model,
                components=components,
                metadata=metadata,
                completed_epochs=5,
                best_val_acc=1.0,
                best_epoch_idx=0,
                best_state=None,
                epochs_without_improvement=0,
                stopped_by_patience=False,
            )

            with self.assertRaisesRegex(ValueError, "protocol mismatch.*seed"):
                _load_clutter_checkpoint(
                    str(checkpoint_path),
                    mdl=model,
                    components=components,
                    expected_metadata={"model_type": "gru", "seed": 2},
                )

    def test_checkpoint_cli_is_opt_in(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.checkpoint_interval_epochs, 0)
        self.assertFalse(args.auto_resume)
        self.assertEqual(args.resume_from, "")


if __name__ == "__main__":
    unittest.main()
