"""Tests for strict digit-by-sector joint-switch stimulus scheduling."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from source.GenerateMovies import StimulusConfig
from source.GenerateMovies_joint_balanced import (
    NUM_DIGITS,
    NUM_SECTORS,
    build_balanced_condition_schedule,
    generate_balanced_joint_test,
    resolve_repeats_per_condition,
    sample_rendered_center_for_sector,
    sample_switch_frames,
    sector_from_center,
    valid_integer_centers_for_axis,
)


class JointBalancedGeneratorTests(unittest.TestCase):
    """Validate exact condition balance and sector-safe center sampling."""

    def test_condition_schedule_is_strictly_balanced(self) -> None:
        repeats = 3
        schedule = build_balanced_condition_schedule(repeats, np.random.default_rng(7))
        self.assertEqual(schedule.dtype, np.int64)
        self.assertEqual(schedule.shape, (NUM_DIGITS * NUM_SECTORS * repeats, 2))
        counts = np.zeros((NUM_DIGITS, NUM_SECTORS), dtype=np.int64)
        np.add.at(counts, (schedule[:, 0], schedule[:, 1]), 1)
        np.testing.assert_array_equal(counts, np.full_like(counts, repeats))

    def test_default_repeats_matches_2400_second_protocol(self) -> None:
        self.assertEqual(resolve_repeats_per_condition(2400, 1.0, None), 27)
        self.assertEqual(resolve_repeats_per_condition(2400, 1.0, 4), 4)

    def test_switch_frames_are_unique_sorted_and_in_bounds(self) -> None:
        frames = sample_switch_frames(1000, 90, np.random.default_rng(3))
        self.assertEqual(frames.dtype, np.int64)
        self.assertEqual(frames.size, 90)
        self.assertTrue(np.all(np.diff(frames) > 0))
        self.assertGreaterEqual(int(frames.min()), 2)
        self.assertLess(int(frames.max()), 1000)

    def test_all_valid_axis_centers_map_to_requested_bin(self) -> None:
        for axis_index in range(3):
            centers = valid_integer_centers_for_axis(96, 28, axis_index)
            mapped = np.clip((centers / 95.0) * 3, 0, 2).astype(np.int64)
            np.testing.assert_array_equal(mapped, np.full_like(mapped, axis_index))
            self.assertGreaterEqual(int(centers.min()), 14)
            self.assertLessEqual(int(centers.max()), 82)

    def test_sampled_rendered_centers_match_all_sectors(self) -> None:
        rng = np.random.default_rng(11)
        for sector in range(NUM_SECTORS):
            for _ in range(20):
                center = sample_rendered_center_for_sector(sector, 96, 96, 28, 28, rng)
                actual = sector_from_center(center[0], center[1], 96, 96)
                self.assertEqual(actual, sector)

    def test_small_generation_validates_observed_condition_counts(self) -> None:
        np.random.seed(5)
        rng = np.random.default_rng(5)
        schedule = build_balanced_condition_schedule(1, rng)
        switch_frames = sample_switch_frames(96, schedule.shape[0], rng)
        mnist_data = {
            digit: [np.full((28, 28), digit, dtype=np.uint8)]
            for digit in range(NUM_DIGITS)
        }
        with tempfile.TemporaryDirectory() as output_dir:
            config = StimulusConfig(
                width=96,
                height=96,
                duration_seconds=4,
                fps=24,
                fg_speeds=[0.0, 1.0, 2.0],
                bg_char_counts=[1, 2],
                bg_mean_speeds=[1.0, 2.0],
                mean_switch_interval_seconds=1.0,
                switch_mode="joint",
                output_dir=output_dir,
                suffix="tiny-balanced",
                output_mode="simple",
            )
            metadata = generate_balanced_joint_test(
                config,
                mnist_data,
                schedule,
                switch_frames,
                rng,
                seed=5,
                mean_switch_interval_seconds=1.0,
            )
            counts = np.asarray(metadata["balance"]["condition_counts_digit_by_sector"])
            np.testing.assert_array_equal(counts, np.ones((10, 9), dtype=np.int64))
            stimulus = np.load(f"{output_dir}/stimulus_tiny-balanced.npy", mmap_mode="r")
            self.assertEqual(stimulus.shape, (96, 96, 96))
            self.assertEqual(stimulus.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
