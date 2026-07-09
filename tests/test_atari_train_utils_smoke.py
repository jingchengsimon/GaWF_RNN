"""Smoke tests for Atari training metric extraction."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from train_atari import _extract_episode_returns


class AtariTrainMetricsSmokeTest(unittest.TestCase):
    def test_extracts_gymnasium_episode_returns(self) -> None:
        infos = {
            "episode": {
                "r": np.asarray([-21.0, -18.0]),
                "l": np.asarray([3296, 3100]),
            },
            "_episode": np.asarray([True, False]),
        }
        self.assertEqual(_extract_episode_returns(infos), [-21.0])

    def test_extracts_final_info_episode_returns(self) -> None:
        infos = {
            "final_info": [
                {"episode": {"r": np.asarray([-20.0])}},
                None,
                {"episode": {"r": -19.0}},
            ]
        }
        self.assertEqual(_extract_episode_returns(infos), [-20.0, -19.0])


if __name__ == "__main__":
    unittest.main()
