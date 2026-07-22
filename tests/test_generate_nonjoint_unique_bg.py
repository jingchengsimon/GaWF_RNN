"""Tests for paired non-joint unique-digit background-switch stimulus generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from source.clutter.generate_movies import StimulusConfig
from source.clutter.generate_nonjoint_unique_bg import (
    NUM_DIGITS,
    NUM_SECTORS,
    build_balanced_event_schedule,
    generate_nonjoint_unique_dataset,
)


def _synthetic_digits() -> dict[int, list[np.ndarray]]:
    return {
        digit: [np.full((8, 8), 20 + digit, dtype=np.uint8)]
        for digit in range(NUM_DIGITS)
    }


def test_schedule_is_strictly_balanced() -> None:
    rng = np.random.default_rng(42)
    trials, conditions, duplicates = build_balanced_event_schedule(180, rng, 2)
    counts = np.zeros((NUM_DIGITS, NUM_SECTORS), dtype=np.int64)
    for digit, sector in conditions:
        counts[int(digit), int(sector)] += 1
    assert trials.size == 180
    assert np.all(counts == 2)
    assert np.all(np.bincount(duplicates, minlength=NUM_SECTORS) == 20)


def test_small_paired_generation_validates_interventions(tmp_path: Path) -> None:
    frame_num = 24
    chan_num = 2
    event_output_t = 12
    num_trials = 90
    total_frames = chan_num + num_trials * frame_num + 5
    schedule_rng = np.random.default_rng(7)
    trials, conditions, duplicates = build_balanced_event_schedule(
        num_trials,
        schedule_rng,
        1,
    )
    config = StimulusConfig(
        width=48,
        height=48,
        duration_seconds=total_frames,
        fps=1,
        fg_speeds=[0.0, 1.0],
        bg_char_counts=[9],
        bg_mean_speeds=[1.0],
        output_dir=str(tmp_path),
        output_mode="simple",
        mnist_sample_start=0,
        mnist_sample_end=10,
    )

    generated: dict[str, np.ndarray] = {}
    metadata_by_mode: dict[str, dict[str, object]] = {}
    for mode in ("full_reset_spatial", "causal_continuous"):
        config.suffix = mode
        metadata = generate_nonjoint_unique_dataset(
            config,
            _synthetic_digits(),
            mode=mode,
            total_frames=total_frames,
            frame_num=frame_num,
            chan_num=chan_num,
            event_output_t=event_output_t,
            event_trials=trials,
            event_conditions=conditions,
            duplicate_sectors=duplicates,
            seed=7,
        )
        generated[mode] = np.load(tmp_path / f"stimulus_{mode}.npy", mmap_mode="r")
        metadata_by_mode[mode] = metadata
        labels = pd.read_csv(tmp_path / f"stimulus_{mode}.tsv", sep="\t")
        assert int(labels["fg_switch"].sum()) == 0
        assert int(labels["bg_switch"].sum()) == 90
        assert labels.shape[0] == total_frames
        assert metadata["balance"]["strict_foreground_digit_sector_event_balance"]
        if mode == "full_reset_spatial":
            assert metadata["clutter_composition"][
                "switch_frames_with_valid_sector_coverage"
            ] == 90
        else:
            control = metadata["nonjoint_control"]
            assert control["causal_identity_derangements"] == 90
            assert control["causal_max_bg_position_jump_pixels"] == 0.0
        with (tmp_path / f"stimulus_{mode}_meta.json").open() as meta_file:
            assert json.load(meta_file)["intervention"] == mode

    full = generated["full_reset_spatial"]
    causal = generated["causal_continuous"]
    event_frames = metadata_by_mode["full_reset_spatial"]["event_frames"]
    for switch_frame in event_frames:
        sequence_start = int(switch_frame) - event_output_t
        first_input_frame = sequence_start - (chan_num - 1)
        assert np.array_equal(
            full[first_input_frame:int(switch_frame)],
            causal[first_input_frame:int(switch_frame)],
        )
