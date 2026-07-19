"""Generate paired non-joint unique-digit datasets for background-switch analysis.

Each evaluation trial contains one straight-moving foreground digit and the nine remaining
digits as background characters, so every rendered frame contains digits 0--9 exactly once.
Only ``bg_switch`` is marked. Foreground digit x sector at the marked frame is strictly
balanced across trials. Two background interventions are supported:

``full_reset_spatial``
    Replace all background exemplars, positions, velocities, and mean speed. At post1 the
    foreground plus nine backgrounds occupy all nine sectors, with one balanced duplicate.
``causal_continuous``
    Apply a derangement of background digit identities/exemplars across the existing tracks.
    Positions, velocities, and mean speed are unchanged on the switch frame.

Outputs (in ``--output-dir``):
- ``stimulus_<suffix>.npy`` (T, H, W), float32 -- rendered grayscale stimulus.
- ``stimulus_<suffix>.tsv`` -- foreground labels and fg/bg switch indicators.
- ``stimulus_<suffix>_meta.json`` -- schedule, balance, and intervention validation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import numpy.lib.format as npfmt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.clutter.generate_movies import MovingCharacter, StimulusConfig, paste_character


LOGGER = logging.getLogger(__name__)
NUM_DIGITS = 10
NUM_SECTORS = 9
GRID_SIZE = 3
DEFAULT_FULL_SUFFIX = (
    "reg-test-40h-float32-nonjoint-10digit-unique-bg-full-reset-spatial"
)
DEFAULT_CAUSAL_SUFFIX = (
    "reg-test-40h-float32-nonjoint-10digit-unique-bg-causal-continuous"
)
LABEL_COLUMNS = [
    "frame",
    "fg_char_id",
    "fg_char_x",
    "fg_char_y",
    "bg_char_ids",
    "fg_speed",
    "bg_mean_speed",
    "fg_switch",
    "bg_switch",
    "trial_id",
    "trial_start",
]


def sector_from_center(
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
) -> int:
    """Map a rendered center to the training pipeline's 3 x 3 sector label."""

    col = int(np.clip((center_x / max(frame_width - 1, 1)) * GRID_SIZE, 0, GRID_SIZE - 1))
    row = int(
        np.clip((center_y / max(frame_height - 1, 1)) * GRID_SIZE, 0, GRID_SIZE - 1)
    )
    return row * GRID_SIZE + col


def rendered_center(
    character: MovingCharacter,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    """Return the center produced by ``paste_character`` after pixel rounding/clipping."""

    x = int(round(character.pos[0] - character.width / 2.0))
    y = int(round(character.pos[1] - character.height / 2.0))
    x_start, x_end = max(0, x), min(frame_width, x + character.width)
    y_start, y_end = max(0, y), min(frame_height, y + character.height)
    return (x_start + x_end) / 2.0, (y_start + y_end) / 2.0


def _valid_centers_for_axis(
    frame_size: int,
    image_size: int,
    sector_axis_index: int,
) -> np.ndarray:
    """Return integer centers that fit in-frame and map to one sector axis cell."""

    half_size = image_size / 2.0
    minimum = int(math.ceil(half_size))
    maximum = int(math.floor(frame_size - half_size))
    centers = np.arange(minimum, maximum + 1, dtype=np.int64)
    mapped = np.clip(
        (centers.astype(np.float64) / max(frame_size - 1, 1)) * GRID_SIZE,
        0,
        GRID_SIZE - 1,
    ).astype(np.int64)
    valid = centers[mapped == sector_axis_index]
    if valid.size == 0:
        raise ValueError(
            f"No valid centers for frame_size={frame_size}, image_size={image_size}, "
            f"sector_axis_index={sector_axis_index}"
        )
    return valid


def sample_center_for_sector(
    sector: int,
    frame_width: int,
    frame_height: int,
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an in-frame integer center guaranteed to map to ``sector``."""

    if not 0 <= sector < NUM_SECTORS:
        raise ValueError(f"sector must be in [0, {NUM_SECTORS - 1}]")
    row, col = divmod(sector, GRID_SIZE)
    x_values = _valid_centers_for_axis(frame_width, image_width, col)
    y_values = _valid_centers_for_axis(frame_height, image_height, row)
    return np.asarray([rng.choice(x_values), rng.choice(y_values)], dtype=np.float64)


def _read_idx(path: Path, header_bytes: int, dtype: str = "uint8") -> np.ndarray:
    """Read an uncompressed or gzip-compressed MNIST IDX payload."""

    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rb") as handle:
        payload = handle.read()
    return np.frombuffer(payload, dtype=dtype, offset=header_bytes)


def load_mnist_idx_data(
    sample_start: int,
    sample_end: int,
    raw_dir: Path,
) -> dict[int, list[np.ndarray]]:
    """Load held-out MNIST images from local IDX files, including ``.gz`` files."""

    image_plain = raw_dir / "train-images-idx3-ubyte"
    label_plain = raw_dir / "train-labels-idx1-ubyte"
    image_path = image_plain if image_plain.is_file() else Path(f"{image_plain}.gz")
    label_path = label_plain if label_plain.is_file() else Path(f"{label_plain}.gz")
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"MNIST IDX image/label files were not found under {raw_dir}")

    image_values = _read_idx(image_path, 16)
    label_values = _read_idx(label_path, 8)
    if image_values.size % (28 * 28) != 0:
        raise ValueError(f"Invalid MNIST image payload size in {image_path}")
    images = image_values.reshape(-1, 28, 28)
    if images.shape[0] != label_values.shape[0]:
        raise ValueError("MNIST image/label count mismatch")
    stop = min(int(sample_end), int(images.shape[0]))
    start = int(sample_start)
    if not 0 <= start < stop:
        raise ValueError(f"Invalid MNIST sample range [{start}, {stop})")

    by_digit: dict[int, list[np.ndarray]] = {digit: [] for digit in range(NUM_DIGITS)}
    for image, label in zip(images[start:stop], label_values[start:stop]):
        by_digit[int(label)].append(np.array(image, copy=True))
    missing = [digit for digit, values in by_digit.items() if not values]
    if missing:
        raise RuntimeError(f"Selected MNIST range has no examples for digits {missing}")
    LOGGER.info("Loaded MNIST samples [%d, %d) from %s", start, stop, raw_dir)
    return by_digit


def build_balanced_event_schedule(
    num_trials: int,
    rng: np.random.Generator,
    repeats_per_condition: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event trial ids, balanced foreground conditions, and duplicate sectors."""

    max_repeats = num_trials // (NUM_DIGITS * NUM_SECTORS)
    repeats = max_repeats if repeats_per_condition is None else int(repeats_per_condition)
    if repeats <= 0 or repeats > max_repeats:
        raise ValueError(f"repeats_per_condition must be in [1, {max_repeats}], got {repeats}")

    base = np.asarray(
        [(digit, sector) for digit in range(NUM_DIGITS) for sector in range(NUM_SECTORS)],
        dtype=np.int64,
    )
    conditions = np.tile(base, (repeats, 1))
    rng.shuffle(conditions, axis=0)
    event_trials = np.sort(
        rng.choice(np.arange(num_trials, dtype=np.int64), conditions.shape[0], replace=False)
    )
    duplicate_sectors = np.tile(
        np.arange(NUM_SECTORS, dtype=np.int64),
        conditions.shape[0] // NUM_SECTORS,
    )
    rng.shuffle(duplicate_sectors)
    return event_trials, conditions, duplicate_sectors


def _sample_image(
    mnist_data: dict[int, Sequence[np.ndarray]],
    digit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one image exemplar for a specified digit."""

    choices = mnist_data[digit]
    return np.asarray(choices[int(rng.integers(0, len(choices)))])


def _sample_safe_foreground(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    digit: int,
    target_sector: int,
    updates_until_event: int,
    updates_in_trial: int,
    rng: np.random.Generator,
) -> MovingCharacter:
    """Initialize a straight trajectory that reaches the target sector without bouncing."""

    for _ in range(10_000):
        image = _sample_image(mnist_data, digit, rng)
        speed = float(rng.choice(config.fg_speeds))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        velocity = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64) * speed
        target = sample_center_for_sector(
            target_sector,
            config.width,
            config.height,
            int(image.shape[1]),
            int(image.shape[0]),
            rng,
        )
        initial = target - velocity * updates_until_event
        final = initial + velocity * updates_in_trial
        lower = np.asarray([image.shape[1] / 2.0, image.shape[0] / 2.0])
        upper = np.asarray(
            [config.width - image.shape[1] / 2.0, config.height - image.shape[0] / 2.0]
        )
        if np.all(initial >= lower) and np.all(initial <= upper):
            if np.all(final >= lower) and np.all(final <= upper):
                return MovingCharacter(digit, image, initial, velocity)
    raise RuntimeError(
        f"Could not sample a safe foreground path for digit={digit}, sector={target_sector}"
    )


def _build_sector_covered_background(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    foreground_digit: int,
    foreground_sector: int,
    duplicate_sector: int,
    rng: np.random.Generator,
) -> tuple[float, list[MovingCharacter]]:
    """Build nine unique backgrounds completing sector coverage at the rendered frame."""

    digits = np.asarray(
        [digit for digit in range(NUM_DIGITS) if digit != foreground_digit],
        dtype=np.int64,
    )
    rng.shuffle(digits)
    sectors = list(range(NUM_SECTORS)) + [int(duplicate_sector)]
    sectors.remove(int(foreground_sector))
    rng.shuffle(sectors)
    mean_speed = float(rng.choice(config.bg_mean_speeds))
    characters: list[MovingCharacter] = []
    for digit_value, sector in zip(digits, sectors):
        digit = int(digit_value)
        image = _sample_image(mnist_data, digit, rng)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        velocity = np.asarray([np.cos(angle), np.sin(angle)]) * mean_speed
        center = sample_center_for_sector(
            int(sector),
            config.width,
            config.height,
            int(image.shape[1]),
            int(image.shape[0]),
            rng,
        )
        characters.append(MovingCharacter(digit, image, center, velocity))
    return mean_speed, characters


def _update_background_random_walk(
    characters: list[MovingCharacter],
    mean_speed: float,
    frame_dims: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Apply the canonical per-frame random-direction background motion deterministically."""

    for character in characters:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        character.vel = np.asarray([np.cos(angle), np.sin(angle)]) * mean_speed
        character.update_position(frame_dims)


def _derange_background_identities(
    characters: list[MovingCharacter],
    mnist_data: dict[int, Sequence[np.ndarray]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Change every BG identity/exemplar while retaining each physical track's state."""

    old_digits = np.asarray([character.label for character in characters], dtype=np.int64)
    for _ in range(1_000):
        new_digits = rng.permutation(old_digits)
        if np.all(new_digits != old_digits):
            break
    else:
        raise RuntimeError("Could not sample a derangement for background identities")

    mapping: list[tuple[int, int]] = []
    for character, digit_value in zip(characters, new_digits):
        old_digit = int(character.label)
        new_digit = int(digit_value)
        character.label = new_digit
        character.image = _sample_image(mnist_data, new_digit, rng)
        character.height, character.width = character.image.shape
        mapping.append((old_digit, new_digit))
    return mapping


def _validate_unique_digits(
    frame_idx: int,
    foreground: MovingCharacter,
    backgrounds: list[MovingCharacter],
) -> None:
    digits = [foreground.label] + [character.label for character in backgrounds]
    if sorted(digits) != list(range(NUM_DIGITS)):
        raise RuntimeError(f"Frame {frame_idx}: expected digits 0-9 once, got {digits}")


def _validate_sector_coverage(
    frame_idx: int,
    foreground: MovingCharacter,
    backgrounds: list[MovingCharacter],
    duplicate_sector: int,
    config: StimulusConfig,
) -> None:
    centers = [(foreground.center_x, foreground.center_y)]
    centers.extend((character.center_x, character.center_y) for character in backgrounds)
    sectors = np.asarray(
        [
            sector_from_center(x, y, config.width, config.height)
            for x, y in centers
        ],
        dtype=np.int64,
    )
    counts = np.bincount(sectors, minlength=NUM_SECTORS)
    expected = np.ones(NUM_SECTORS, dtype=np.int64)
    expected[int(duplicate_sector)] += 1
    if not np.array_equal(counts, expected):
        raise RuntimeError(
            f"Frame {frame_idx}: sector counts {counts.tolist()} != {expected.tolist()}"
        )


def _mode_suffix(mode: str) -> str:
    if mode == "full_reset_spatial":
        return DEFAULT_FULL_SUFFIX
    if mode == "causal_continuous":
        return DEFAULT_CAUSAL_SUFFIX
    raise ValueError(f"Unsupported mode: {mode}")


def generate_nonjoint_unique_dataset(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    *,
    mode: str,
    total_frames: int,
    frame_num: int,
    chan_num: int,
    event_output_t: int,
    event_trials: np.ndarray,
    event_conditions: np.ndarray,
    duplicate_sectors: np.ndarray,
    seed: int,
) -> dict[str, object]:
    """Generate and validate one non-joint unique-digit BG-switch dataset."""

    if mode not in ("full_reset_spatial", "causal_continuous"):
        raise ValueError(f"Unsupported mode: {mode}")
    if not 10 <= event_output_t <= frame_num - 10:
        raise ValueError("event_output_t must leave at least 10 pre and 10 post frames")
    if event_conditions.shape != (event_trials.size, 2):
        raise ValueError("event_conditions must have shape (num_events, 2)")
    if duplicate_sectors.shape != (event_trials.size,):
        raise ValueError("duplicate_sectors must have shape (num_events,)")

    num_trials = (total_frames - chan_num) // frame_num
    if event_trials.size and int(event_trials.max()) >= num_trials:
        raise ValueError("event trial index exceeds available model sequences")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stimulus_{config.suffix}"
    npy_path = output_dir / f"{stem}.npy"
    tsv_path = output_dir / f"{stem}.tsv"
    meta_path = output_dir / f"{stem}_meta.json"
    npy_data = npfmt.open_memmap(
        npy_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, config.height, config.width),
    )

    schedule_by_trial = {
        int(trial): (
            int(condition[0]),
            int(condition[1]),
            int(duplicate),
        )
        for trial, condition, duplicate in zip(
            event_trials, event_conditions, duplicate_sectors
        )
    }
    condition_counts = np.zeros((NUM_DIGITS, NUM_SECTORS), dtype=np.int64)
    duplicate_counts = np.zeros(NUM_SECTORS, dtype=np.int64)
    event_frames: list[int] = []
    causal_max_position_jump = 0.0
    causal_derangement_count = 0
    trial_onset_coverage_count = 0
    event_coverage_count = 0
    boundary_guard_count = 0
    frame_dims = (config.height, config.width)

    foreground: MovingCharacter | None = None
    backgrounds: list[MovingCharacter] = []
    bg_mean_speed = 0.0
    trial_rng: np.random.Generator | None = None
    current_trial = -1
    current_duplicate_sector = 0
    trial_start_frame = 0

    with tsv_path.open("w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(LABEL_COLUMNS)
        for frame_idx in tqdm(
            range(total_frames),
            desc=f"Generating {mode}",
        ):
            output_position = frame_idx - chan_num
            if output_position < 0:
                trial_id = 0
                output_t = output_position
            elif output_position >= num_trials * frame_num:
                trial_id = num_trials - 1
                output_t = -1
            else:
                trial_id = output_position // frame_num
                output_t = output_position % frame_num
            new_trial = foreground is None or trial_id != current_trial

            if new_trial:
                current_trial = int(trial_id)
                trial_start_frame = frame_idx
                trial_rng = np.random.default_rng(np.random.SeedSequence([seed, current_trial]))
                scheduled = schedule_by_trial.get(current_trial)
                if scheduled is None:
                    target_digit = int(trial_rng.integers(0, NUM_DIGITS))
                    target_sector = int(trial_rng.integers(0, NUM_SECTORS))
                else:
                    target_digit, target_sector, _ = scheduled
                updates_until_event = (
                    event_output_t + 1 + (chan_num if frame_idx == 0 else 0)
                )
                updates_in_trial = frame_num + (chan_num if frame_idx == 0 else 0)
                foreground = _sample_safe_foreground(
                    config,
                    mnist_data,
                    target_digit,
                    target_sector,
                    updates_until_event,
                    updates_in_trial,
                    trial_rng,
                )
                foreground.update_position(frame_dims)
                onset_center = rendered_center(foreground, config.width, config.height)
                onset_sector = sector_from_center(
                    onset_center[0], onset_center[1], config.width, config.height
                )
                current_duplicate_sector = int(trial_rng.integers(0, NUM_SECTORS))
                bg_mean_speed, backgrounds = _build_sector_covered_background(
                    config,
                    mnist_data,
                    foreground.label,
                    onset_sector,
                    current_duplicate_sector,
                    trial_rng,
                )
                bg_moved = False
            else:
                if trial_rng is None or foreground is None:
                    raise RuntimeError("Missing initialized trial state")
                foreground.update_position(frame_dims)
                bg_moved = True

            scheduled = schedule_by_trial.get(current_trial)
            is_event = scheduled is not None and output_t == event_output_t
            bg_switch_flag = int(is_event)
            event_duplicate_sector: int | None = None
            positions_before: np.ndarray | None = None
            if is_event:
                target_digit, target_sector, event_duplicate_sector = scheduled
                actual_center = rendered_center(foreground, config.width, config.height)
                actual_sector = sector_from_center(
                    actual_center[0], actual_center[1], config.width, config.height
                )
                if foreground.label != target_digit or actual_sector != target_sector:
                    raise RuntimeError(
                        f"Frame {frame_idx}: expected FG {(target_digit, target_sector)}, "
                        f"got {(foreground.label, actual_sector)}"
                    )
                if mode == "full_reset_spatial":
                    bg_mean_speed, backgrounds = _build_sector_covered_background(
                        config,
                        mnist_data,
                        foreground.label,
                        actual_sector,
                        event_duplicate_sector,
                        trial_rng,
                    )
                else:
                    positions_before = np.stack(
                        [character.pos.copy() for character in backgrounds]
                    )
                    mapping = _derange_background_identities(
                        backgrounds,
                        mnist_data,
                        trial_rng,
                    )
                    if any(old == new for old, new in mapping):
                        raise RuntimeError("Causal switch retained at least one BG identity")
                    causal_derangement_count += 1
                bg_moved = False
                event_frames.append(frame_idx)
                condition_counts[target_digit, target_sector] += 1
                duplicate_counts[event_duplicate_sector] += 1

            if bg_moved:
                _update_background_random_walk(
                    backgrounds,
                    bg_mean_speed,
                    frame_dims,
                    trial_rng,
                )

            is_boundary_guard = (
                output_t == frame_num - 1
                and 0 <= output_position < num_trials * frame_num
            )
            if is_boundary_guard:
                guard_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, current_trial, 1_000_003])
                )
                guard_digit = int(guard_rng.integers(0, NUM_DIGITS))
                guard_sector = int(guard_rng.integers(0, NUM_SECTORS))
                foreground = _sample_safe_foreground(
                    config,
                    mnist_data,
                    guard_digit,
                    guard_sector,
                    updates_until_event=1,
                    updates_in_trial=1,
                    rng=guard_rng,
                )
                foreground.update_position(frame_dims)
                guard_center = rendered_center(foreground, config.width, config.height)
                rendered_guard_sector = sector_from_center(
                    guard_center[0], guard_center[1], config.width, config.height
                )
                current_duplicate_sector = int(guard_rng.integers(0, NUM_SECTORS))
                bg_mean_speed, backgrounds = _build_sector_covered_background(
                    config,
                    mnist_data,
                    foreground.label,
                    rendered_guard_sector,
                    current_duplicate_sector,
                    guard_rng,
                )
                boundary_guard_count += 1

            frame = np.zeros(frame_dims, dtype=np.float32)
            for character in backgrounds:
                paste_character(frame, character)
            paste_character(frame, foreground)
            npy_data[frame_idx] = frame
            _validate_unique_digits(frame_idx, foreground, backgrounds)

            if new_trial:
                _validate_sector_coverage(
                    frame_idx,
                    foreground,
                    backgrounds,
                    current_duplicate_sector,
                    config,
                )
                trial_onset_coverage_count += 1
            if is_event and mode == "full_reset_spatial":
                _validate_sector_coverage(
                    frame_idx,
                    foreground,
                    backgrounds,
                    int(event_duplicate_sector),
                    config,
                )
                event_coverage_count += 1
            if positions_before is not None:
                positions_after = np.stack([character.pos for character in backgrounds])
                position_jump = float(
                    np.linalg.norm(positions_after - positions_before, axis=1).max()
                )
                causal_max_position_jump = max(causal_max_position_jump, position_jump)

            bg_ids = ",".join(str(character.label) for character in backgrounds)
            writer.writerow(
                [
                    frame_idx,
                    foreground.label,
                    f"{foreground.center_x:.2f}",
                    f"{foreground.center_y:.2f}",
                    bg_ids,
                    f"{np.linalg.norm(foreground.vel):.2f}",
                    f"{bg_mean_speed:.2f}",
                    0,
                    bg_switch_flag,
                    current_trial,
                    int(new_trial),
                ]
            )

    npy_data.flush()
    expected_repeats = event_trials.size // (NUM_DIGITS * NUM_SECTORS)
    if not np.all(condition_counts == expected_repeats):
        raise RuntimeError(
            "Foreground digit x sector event balance failed: "
            f"min={condition_counts.min()}, max={condition_counts.max()}"
        )
    expected_duplicate_count = event_trials.size // NUM_SECTORS
    if not np.all(duplicate_counts == expected_duplicate_count):
        raise RuntimeError(
            f"Duplicate-sector balance failed: {duplicate_counts.tolist()}"
        )
    if mode == "causal_continuous" and causal_max_position_jump != 0.0:
        raise RuntimeError(f"Causal intervention moved BG tracks by {causal_max_position_jump}")

    metadata: dict[str, object] = {
        "protocol": "nonjoint_unique_digit_bg_switch_balanced_trials",
        "intervention": mode,
        "suffix": config.suffix,
        "seed": seed,
        "files": {
            "stimulus": str(npy_path.resolve()),
            "labels": str(tsv_path.resolve()),
        },
        "stimulus": {
            "shape": [total_frames, config.height, config.width],
            "dtype": "float32",
            "fps": config.fps,
            "nominal_duration_seconds": config.duration_seconds,
            "mnist_sample_range": [config.mnist_sample_start, config.mnist_sample_end],
        },
        "model_sequence_layout": {
            "chan_num": chan_num,
            "frame_num": frame_num,
            "num_model_sequences": num_trials,
            "event_output_t_zero_based": event_output_t,
            "event_windows_do_not_cross_sequence_boundaries": True,
            "boundary_guard_output_t_zero_based": frame_num - 1,
            "boundary_guard_count": boundary_guard_count,
            "boundary_guards_are_identical_across_interventions": True,
            "boundary_guards_are_outside_pre10_post10": True,
        },
        "balance": {
            "unit": "bg-switch events",
            "num_events": int(event_trials.size),
            "events_per_foreground_digit_sector": int(expected_repeats),
            "condition_counts_digit_by_sector": condition_counts.tolist(),
            "strict_foreground_digit_sector_event_balance": True,
            "duplicate_sector_counts": duplicate_counts.tolist(),
            "strict_duplicate_sector_balance": True,
        },
        "clutter_composition": {
            "all_digits_unique_per_frame": True,
            "digit_set_per_frame": list(range(NUM_DIGITS)),
            "num_characters_per_frame": NUM_DIGITS,
            "num_background_characters": NUM_DIGITS - 1,
            "trial_onsets_with_valid_sector_coverage": trial_onset_coverage_count,
            "switch_frames_with_valid_sector_coverage": (
                event_coverage_count if mode == "full_reset_spatial" else None
            ),
        },
        "nonjoint_control": {
            "fg_switch_count": 0,
            "bg_switch_count": int(event_trials.size),
            "foreground_is_unchanged_at_bg_switch": True,
            "causal_identity_derangements": (
                causal_derangement_count if mode == "causal_continuous" else None
            ),
            "causal_max_bg_position_jump_pixels": (
                causal_max_position_jump if mode == "causal_continuous" else None
            ),
            "causal_preserves_bg_velocity_and_mean_speed": (
                True if mode == "causal_continuous" else None
            ),
            "full_reset_post1_sector_rule": (
                "one character per sector plus one balanced duplicate sector"
                if mode == "full_reset_spatial"
                else None
            ),
        },
        "event_frames": event_frames,
    }
    with meta_path.open("w") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    LOGGER.info("Wrote %s, %s, and %s", npy_path, tsv_path, meta_path)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse paired non-joint BG-switch dataset generation options."""

    parser = argparse.ArgumentParser(
        description="Generate full-reset and/or causal non-joint unique-digit BG switches."
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "stimuli"))
    parser.add_argument(
        "--mode",
        choices=("both", "full_reset_spatial", "causal_continuous"),
        default="both",
    )
    parser.add_argument("--full-suffix", type=str, default=DEFAULT_FULL_SUFFIX)
    parser.add_argument("--causal-suffix", type=str, default=DEFAULT_CAUSAL_SUFFIX)
    parser.add_argument("--duration-seconds", type=int, default=2400)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-num", type=int, default=32)
    parser.add_argument("--chan-num", type=int, default=2)
    parser.add_argument("--event-output-t", type=int, default=20)
    parser.add_argument("--repeats-per-condition", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist-sample-start", type=int, default=50000)
    parser.add_argument("--mnist-sample-end", type=int, default=60000)
    parser.add_argument(
        "--mnist-raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "mnist_data_pytorch" / "MNIST" / "raw"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the requested paired datasets with a shared balanced event schedule."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    total_frames = int(args.duration_seconds * args.fps)
    num_trials = (total_frames - args.chan_num) // args.frame_num
    schedule_rng = np.random.default_rng(args.seed)
    event_trials, conditions, duplicate_sectors = build_balanced_event_schedule(
        num_trials,
        schedule_rng,
        args.repeats_per_condition,
    )
    mnist_data = load_mnist_idx_data(
        args.mnist_sample_start,
        args.mnist_sample_end,
        Path(args.mnist_raw_dir),
    )
    modes = (
        ("full_reset_spatial", "causal_continuous")
        if args.mode == "both"
        else (args.mode,)
    )
    for mode in modes:
        suffix = args.full_suffix if mode == "full_reset_spatial" else args.causal_suffix
        config = StimulusConfig(
            width=96,
            height=96,
            duration_seconds=args.duration_seconds,
            fps=args.fps,
            fg_speeds=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
            bg_char_counts=[9],
            bg_mean_speeds=[1.0, 2.0, 4.0, 6.0, 8.0],
            output_dir=args.output_dir,
            output_mode="simple",
            mnist_sample_start=args.mnist_sample_start,
            mnist_sample_end=args.mnist_sample_end,
            suffix=suffix,
        )
        generate_nonjoint_unique_dataset(
            config,
            mnist_data,
            mode=mode,
            total_frames=total_frames,
            frame_num=args.frame_num,
            chan_num=args.chan_num,
            event_output_t=args.event_output_t,
            event_trials=event_trials,
            event_conditions=conditions,
            duplicate_sectors=duplicate_sectors,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
