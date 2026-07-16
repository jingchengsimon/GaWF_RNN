"""Smoke tests for Atari learning-curve JSONL parsing without training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_viz.atari_learning_curves import _discover_env_ids, _load_curve


class AtariLearningCurvesSmokeTest(unittest.TestCase):
    def test_nested_per_env_metric_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "metrics_history.jsonl"
            records = [
                {
                    "global_step": 2000,
                    "episodic_return_100": 2.0,
                    "per_env": {
                        "ALE/Pong-v5": {"episodic_return_100": 3.0},
                        "ALE/Breakout-v5": {"episodic_return_100": 1.0},
                    },
                },
                {
                    "global_step": 1000,
                    "episodic_return_100": 1.0,
                    "per_env": {
                        "ALE/Pong-v5": {"episodic_return_100": 1.5},
                        "ALE/Breakout-v5": {"episodic_return_100": 0.5},
                    },
                },
            ]
            history_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            curve = _load_curve(
                history_path,
                "per_env.ALE/Pong-v5.episodic_return_100",
            )
            self.assertIsNotNone(curve)
            steps, values = curve
            np.testing.assert_array_equal(steps, np.array([1000, 2000]))
            np.testing.assert_array_equal(values, np.array([1.5, 3.0]))
            self.assertEqual(
                _discover_env_ids(history_path),
                ["ALE/Breakout-v5", "ALE/Pong-v5"],
            )


if __name__ == "__main__":
    unittest.main()
