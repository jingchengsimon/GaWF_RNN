"""Generate a joint-switch test stimulus with strict digit-by-sector event balance.

At every scheduled event, foreground and background clutter switch together. The foreground
``(digit, sector)`` pair follows a shuffled schedule containing every one of the 90 conditions
the same number of times. Switch times remain random; therefore strict balance applies to switch
events/episodes, not to total frame occupancy when episode durations differ.

Inputs are held-out MNIST samples (default indices 50000:60000). Outputs preserve the standard
``stimulus_<suffix>.npy`` and ``stimulus_<suffix>.tsv`` schema and add a JSON metadata sidecar.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import numpy.lib.format as npfmt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.clutter.generate_movies import (
    MovingCharacter,
    StimulusConfig,
    load_mnist_data,
    paste_character,
)


LOGGER = logging.getLogger(__name__)
NUM_DIGITS = 10
NUM_SECTORS = 9
GRID_SIZE = 3
DEFAULT_SUFFIX = "reg-test-40h-float32-jointswitch-balanced"
DEFAULT_UNIQUE_SUFFIX = f"{DEFAULT_SUFFIX}-10digit-unique"
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
]


def load_mnist_idx_data(
    config: StimulusConfig,
    raw_dir: Path | None = None,
) -> dict[int, list[np.ndarray]]:
    """Load held-out MNIST samples directly from raw IDX files without torchvision."""
    source_dir = raw_dir or PROJECT_ROOT / "mnist_data_pytorch" / "MNIST" / "raw"
    images_path = source_dir / "train-images-idx3-ubyte"
    labels_path = source_dir / "train-labels-idx1-ubyte"
    if not images_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            "torchvision is unavailable and raw MNIST IDX files were not found at "
            f"{source_dir}"
        )

    with images_path.open("rb") as image_file:
        image_header = np.frombuffer(image_file.read(16), dtype=">i4")
    with labels_path.open("rb") as label_file:
        label_header = np.frombuffer(label_file.read(8), dtype=">i4")
    if image_header.tolist()[:1] != [2051] or label_header.tolist()[:1] != [2049]:
        raise ValueError("Invalid MNIST IDX magic number")
    num_images, rows, columns = (int(value) for value in image_header[1:])
    num_labels = int(label_header[1])
    if num_images != num_labels:
        raise ValueError(f"MNIST image/label count mismatch: {num_images} != {num_labels}")

    start = int(config.mnist_sample_start)
    stop = min(int(config.mnist_sample_end), num_images)
    if not 0 <= start < stop:
        raise ValueError(f"Invalid MNIST sample range [{start}, {stop}) for {num_images} rows")
    images = np.memmap(
        images_path,
        mode="r",
        dtype=np.uint8,
        offset=16,
        shape=(num_images, rows, columns),
    )
    labels = np.memmap(
        labels_path,
        mode="r",
        dtype=np.uint8,
        offset=8,
        shape=(num_labels,),
    )
    mnist_digits: dict[int, list[np.ndarray]] = {digit: [] for digit in range(NUM_DIGITS)}
    selected_images: list[np.ndarray] = []
    selected_labels: list[int] = []
    for index in range(start, stop):
        digit = int(labels[index])
        image = np.array(images[index], copy=True)
        mnist_digits[digit].append(image)
        if config.output_mode == "full":
            selected_images.append(image)
            selected_labels.append(digit)

    if config.output_mode == "full":
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"mnist_images_{config.suffix}.npy", np.stack(selected_images))
        np.save(
            output_dir / f"mnist_labels_{config.suffix}.npy",
            np.asarray(selected_labels, dtype=np.int64),
        )
    LOGGER.info("Loaded MNIST IDX samples [%s, %s) from %s", start, stop, source_dir)
    return mnist_digits


def load_balanced_mnist_data(config: StimulusConfig) -> dict[int, Sequence[np.ndarray]]:
    """Use the canonical torchvision loader, falling back to local raw IDX files."""
    try:
        return load_mnist_data(config)
    except ModuleNotFoundError as error:
        if error.name != "torchvision":
            raise
        LOGGER.warning("torchvision is unavailable; using the local MNIST IDX fallback")
        return load_mnist_idx_data(config)


def build_balanced_condition_schedule(
    repeats_per_condition: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a shuffled ``(digit, sector)`` schedule with exact 90-condition balance."""
    if repeats_per_condition <= 0:
        raise ValueError("repeats_per_condition must be positive")
    base = np.asarray(
        [(digit, sector) for digit in range(NUM_DIGITS) for sector in range(NUM_SECTORS)],
        dtype=np.int64,
    )
    schedule = np.tile(base, (repeats_per_condition, 1))
    rng.shuffle(schedule, axis=0)
    return schedule


def resolve_repeats_per_condition(
    duration_seconds: int,
    mean_switch_interval_seconds: float,
    requested_repeats: int | None,
) -> int:
    """Resolve a balanced event count close to the requested mean switch rate."""
    if requested_repeats is not None:
        if requested_repeats <= 0:
            raise ValueError("--repeats-per-condition must be positive")
        return requested_repeats
    if duration_seconds <= 0 or mean_switch_interval_seconds <= 0:
        raise ValueError("duration and mean switch interval must be positive")
    expected_events = duration_seconds / mean_switch_interval_seconds
    return max(1, int(math.floor(expected_events / (NUM_DIGITS * NUM_SECTORS) + 0.5)))


def sample_switch_frames(
    total_frames: int,
    num_switches: int,
    rng: np.random.Generator,
    *,
    minimum_frame: int = 2,
) -> np.ndarray:
    """Sample sorted unique random switch frames, conditional on the exact event count."""
    if total_frames <= minimum_frame:
        raise ValueError("total_frames must exceed minimum_frame")
    available = total_frames - minimum_frame
    if num_switches <= 0 or num_switches > available:
        raise ValueError(
            f"num_switches must be in [1, {available}], got {num_switches}"
        )
    candidates = np.arange(minimum_frame, total_frames, dtype=np.int64)
    return np.sort(rng.choice(candidates, size=num_switches, replace=False)).astype(
        np.int64,
        copy=False,
    )


def sector_from_center(
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
) -> int:
    """Map a rendered foreground center to the same 3x3 sector used by training."""
    col = int(np.clip((center_x / max(frame_width - 1, 1)) * GRID_SIZE, 0, GRID_SIZE - 1))
    row = int(
        np.clip((center_y / max(frame_height - 1, 1)) * GRID_SIZE, 0, GRID_SIZE - 1)
    )
    return row * GRID_SIZE + col


def valid_integer_centers_for_axis(
    frame_size: int,
    image_size: int,
    sector_axis_index: int,
) -> np.ndarray:
    """Return rendered integer centers that fit in-frame and map to one sector row/column."""
    if not 0 <= sector_axis_index < GRID_SIZE:
        raise ValueError(f"sector_axis_index must be in [0, {GRID_SIZE - 1}]")
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


def sample_rendered_center_for_sector(
    sector: int,
    frame_width: int,
    frame_height: int,
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an integer rendered center guaranteed to map to ``sector``."""
    if not 0 <= sector < NUM_SECTORS:
        raise ValueError(f"sector must be in [0, {NUM_SECTORS - 1}]")
    row, col = divmod(sector, GRID_SIZE)
    x_values = valid_integer_centers_for_axis(frame_width, image_width, col)
    y_values = valid_integer_centers_for_axis(frame_height, image_height, row)
    return np.asarray([rng.choice(x_values), rng.choice(y_values)], dtype=np.float64)


def _sample_digit_image(
    mnist_data: dict[int, Sequence[np.ndarray]],
    digit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    images = mnist_data[digit]
    return np.asarray(images[int(rng.integers(0, len(images)))])


def _build_background(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    rng: np.random.Generator,
) -> tuple[int, float, list[MovingCharacter]]:
    bg_char_count = int(rng.choice(config.bg_char_counts))
    bg_mean_speed = float(rng.choice(config.bg_mean_speeds))
    background_chars: list[MovingCharacter] = []
    for _ in range(bg_char_count):
        digit = int(rng.integers(0, NUM_DIGITS))
        image = _sample_digit_image(mnist_data, digit, rng)
        position = rng.random(2) * [config.width - image.shape[1], config.height - image.shape[0]]
        position += [image.shape[1] / 2.0, image.shape[0] / 2.0]
        angle = float(rng.uniform(0, 2 * np.pi))
        velocity = np.asarray([np.cos(angle), np.sin(angle)]) * bg_mean_speed
        background_chars.append(MovingCharacter(digit, image, position, velocity))
    return bg_char_count, bg_mean_speed, background_chars


def _build_unique_digit_background(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    rng: np.random.Generator,
    foreground_digit: int,
    foreground_sector: int,
) -> tuple[int, float, list[MovingCharacter], int]:
    """Build nine unique backgrounds whose onset slots cover all sectors."""

    if not 0 <= foreground_digit < NUM_DIGITS:
        raise ValueError(f"foreground_digit must be in [0, {NUM_DIGITS - 1}]")
    if not 0 <= foreground_sector < NUM_SECTORS:
        raise ValueError(f"foreground_sector must be in [0, {NUM_SECTORS - 1}]")
    background_digits = np.asarray(
        [digit for digit in range(NUM_DIGITS) if digit != foreground_digit],
        dtype=np.int64,
    )
    rng.shuffle(background_digits)
    duplicate_sector = int(rng.integers(0, NUM_SECTORS))
    background_sectors = list(range(NUM_SECTORS)) + [duplicate_sector]
    background_sectors.remove(foreground_sector)
    rng.shuffle(background_sectors)
    bg_mean_speed = float(rng.choice(config.bg_mean_speeds))
    background_chars: list[MovingCharacter] = []
    for digit_value, sector in zip(background_digits, background_sectors):
        digit = int(digit_value)
        image = _sample_digit_image(mnist_data, digit, rng)
        angle = float(rng.uniform(0, 2 * np.pi))
        velocity = np.asarray([np.cos(angle), np.sin(angle)]) * bg_mean_speed
        target_center = sample_rendered_center_for_sector(
            sector,
            config.width,
            config.height,
            int(image.shape[1]),
            int(image.shape[0]),
            rng,
        )
        position = target_center - velocity
        background_chars.append(MovingCharacter(digit, image, position, velocity))
    return len(background_chars), bg_mean_speed, background_chars, duplicate_sector


def _prepare_scheduled_foreground(
    fg_char: MovingCharacter,
    digit: int,
    sector: int,
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Prepare state so the normal position update lands exactly on the scheduled center."""
    speed = float(rng.choice(config.fg_speeds))
    current_norm = float(np.linalg.norm(fg_char.vel))
    if current_norm == 0:
        direction = np.asarray([1.0, 0.0])
    else:
        direction = fg_char.vel / current_norm
    velocity = direction * speed
    image = _sample_digit_image(mnist_data, digit, rng)
    target_center = sample_rendered_center_for_sector(
        sector,
        config.width,
        config.height,
        int(image.shape[1]),
        int(image.shape[0]),
        rng,
    )
    fg_char.vel = velocity
    fg_char.pos = target_center - velocity
    fg_char.label = digit
    fg_char.image = image
    fg_char.height, fg_char.width = image.shape
    return target_center


def _initialize_foreground(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    rng: np.random.Generator,
) -> MovingCharacter:
    digit = int(rng.integers(0, NUM_DIGITS))
    image = _sample_digit_image(mnist_data, digit, rng)
    speed = float(rng.choice(config.fg_speeds))
    angle = float(rng.uniform(0, 2 * np.pi))
    velocity = np.asarray([np.cos(angle), np.sin(angle)]) * speed
    position = rng.random(2) * [config.width - image.shape[1], config.height - image.shape[0]]
    position += [image.shape[1] / 2.0, image.shape[0] / 2.0]
    return MovingCharacter(digit, image, position, velocity)


def generate_balanced_joint_test(
    config: StimulusConfig,
    mnist_data: dict[int, Sequence[np.ndarray]],
    schedule: np.ndarray,
    switch_frames: np.ndarray,
    rng: np.random.Generator,
    *,
    seed: int,
    mean_switch_interval_seconds: float,
    all_digits_unique: bool = False,
) -> dict[str, object]:
    """Generate one balanced joint-switch test and return validated metadata."""
    if schedule.shape != (switch_frames.size, 2):
        raise ValueError(
            f"schedule shape {schedule.shape} does not match {switch_frames.size} switch frames"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stimulus_{config.suffix}"
    npy_path = output_dir / f"{stem}.npy"
    tsv_path = output_dir / f"{stem}.tsv"
    meta_path = output_dir / f"{stem}_meta.json"
    mp4_path = output_dir / f"{stem}.mp4"
    total_frames = int(config.duration_seconds * config.fps)
    frame_dims = (config.height, config.width)

    video_writer = None
    if config.output_mode == "full":
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(mp4_path), fourcc, config.fps, (config.width, config.height)
        )

    npy_data = npfmt.open_memmap(
        npy_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_frames, config.height, config.width),
    )
    fg_char = _initialize_foreground(config, mnist_data, rng)
    if all_digits_unique:
        initial_sector = int(rng.integers(0, NUM_SECTORS))
        initial_target_center = _prepare_scheduled_foreground(
            fg_char,
            fg_char.label,
            initial_sector,
            config,
            mnist_data,
            rng,
        )
        (
            _,
            bg_mean_speed,
            background_chars,
            onset_duplicate_sector,
        ) = _build_unique_digit_background(
            config,
            mnist_data,
            rng,
            fg_char.label,
            initial_sector,
        )
    else:
        initial_target_center = None
        onset_duplicate_sector = None
        _, bg_mean_speed, background_chars = _build_background(config, mnist_data, rng)
    switch_lookup = {
        int(frame): (int(condition[0]), int(condition[1]))
        for frame, condition in zip(switch_frames, schedule)
    }
    observed_counts = np.zeros((NUM_DIGITS, NUM_SECTORS), dtype=np.int64)
    onset_duplicate_sector_counts = np.zeros(NUM_SECTORS, dtype=np.int64)
    onset_coverage_count = 0

    with tsv_path.open("w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(LABEL_COLUMNS)
        for frame_idx in tqdm(range(total_frames), desc="Generating balanced joint test"):
            fg_switch_flag = 0
            bg_switch_flag = 0
            target_center = initial_target_center if frame_idx == 0 else None
            scheduled = switch_lookup.get(frame_idx)
            if scheduled is not None:
                digit, sector = scheduled
                fg_switch_flag = 1
                bg_switch_flag = 1
                target_center = _prepare_scheduled_foreground(
                    fg_char,
                    digit,
                    sector,
                    config,
                    mnist_data,
                    rng,
                )
                if all_digits_unique:
                    (
                        _,
                        bg_mean_speed,
                        background_chars,
                        onset_duplicate_sector,
                    ) = _build_unique_digit_background(
                        config,
                        mnist_data,
                        rng,
                        fg_char.label,
                        sector,
                    )
                else:
                    _, bg_mean_speed, background_chars = _build_background(
                        config, mnist_data, rng
                    )
            clutter_onset = frame_idx == 0 or scheduled is not None
            fg_char.update_position(frame_dims)
            for character in background_chars:
                if all_digits_unique and clutter_onset:
                    character.update_position(frame_dims)
                else:
                    character.update_random_walk(frame_dims, bg_mean_speed)

            if target_center is not None and not np.allclose(fg_char.pos, target_center):
                raise RuntimeError(
                    f"Switch frame {frame_idx}: updated center {fg_char.pos} != target "
                    f"{target_center}"
                )

            frame = np.zeros(frame_dims, dtype=np.float32)
            for character in background_chars:
                paste_character(frame, character)
            paste_character(frame, fg_char)
            npy_data[frame_idx] = frame

            if all_digits_unique:
                rendered_digits = [fg_char.label]
                rendered_digits.extend(character.label for character in background_chars)
                if sorted(rendered_digits) != list(range(NUM_DIGITS)):
                    raise RuntimeError(
                        f"Frame {frame_idx}: expected digits 0-9 exactly once, got "
                        f"{rendered_digits}"
                    )
                if clutter_onset:
                    rendered_sectors = [
                        sector_from_center(
                            fg_char.center_x,
                            fg_char.center_y,
                            config.width,
                            config.height,
                        )
                    ]
                    rendered_sectors.extend(
                        sector_from_center(
                            character.center_x,
                            character.center_y,
                            config.width,
                            config.height,
                        )
                        for character in background_chars
                    )
                    sector_counts = np.bincount(
                        np.asarray(rendered_sectors, dtype=np.int64),
                        minlength=NUM_SECTORS,
                    )
                    expected_sector_counts = np.ones(NUM_SECTORS, dtype=np.int64)
                    if onset_duplicate_sector is None:
                        raise RuntimeError("Missing duplicate sector for unique clutter onset")
                    expected_sector_counts[onset_duplicate_sector] += 1
                    if not np.array_equal(sector_counts, expected_sector_counts):
                        raise RuntimeError(
                            f"Frame {frame_idx}: onset sector counts {sector_counts.tolist()} "
                            f"!= expected {expected_sector_counts.tolist()}"
                        )
                    onset_duplicate_sector_counts[onset_duplicate_sector] += 1
                    onset_coverage_count += 1

            if scheduled is not None:
                digit, sector = scheduled
                actual_sector = sector_from_center(
                    fg_char.center_x,
                    fg_char.center_y,
                    config.width,
                    config.height,
                )
                if fg_char.label != digit or actual_sector != sector:
                    raise RuntimeError(
                        f"Switch frame {frame_idx}: scheduled {(digit, sector)}, "
                        f"observed {(fg_char.label, actual_sector)}"
                    )
                observed_counts[digit, sector] += 1

            bg_ids = ",".join(str(character.label) for character in background_chars)
            writer.writerow(
                [
                    frame_idx,
                    fg_char.label,
                    f"{fg_char.center_x:.2f}",
                    f"{fg_char.center_y:.2f}",
                    bg_ids,
                    f"{np.linalg.norm(fg_char.vel):.2f}",
                    f"{bg_mean_speed:.2f}",
                    fg_switch_flag,
                    bg_switch_flag,
                ]
            )

            if video_writer is not None:
                import cv2

                video_writer.write(
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                )

    npy_data.flush()
    if video_writer is not None:
        video_writer.release()

    expected_per_condition = int(schedule.shape[0] // (NUM_DIGITS * NUM_SECTORS))
    if not np.all(observed_counts == expected_per_condition):
        raise RuntimeError(
            "Generated switch conditions are not strictly balanced: "
            f"min={observed_counts.min()}, max={observed_counts.max()}, "
            f"expected={expected_per_condition}"
        )

    intervals = np.diff(switch_frames)
    metadata: dict[str, object] = {
        "protocol": "joint_switch_balanced_digit_sector",
        "suffix": config.suffix,
        "seed": seed,
        "files": {"stimulus": str(npy_path), "labels": str(tsv_path)},
        "stimulus": {
            "shape": [total_frames, config.height, config.width],
            "dtype": "float32",
            "fps": config.fps,
            "duration_seconds": config.duration_seconds,
            "mnist_sample_range": [config.mnist_sample_start, config.mnist_sample_end],
        },
        "balance": {
            "unit": "joint-switch events",
            "num_digits": NUM_DIGITS,
            "num_sectors": NUM_SECTORS,
            "num_conditions": NUM_DIGITS * NUM_SECTORS,
            "events_per_condition": expected_per_condition,
            "num_switches": int(switch_frames.size),
            "condition_counts_digit_by_sector": observed_counts.tolist(),
            "strict_event_balance": True,
            "strict_frame_occupancy_balance": False,
        },
        "clutter_composition": {
            "all_digits_unique_per_frame": all_digits_unique,
            "num_characters_per_frame": NUM_DIGITS if all_digits_unique else None,
            "num_background_characters": NUM_DIGITS - 1 if all_digits_unique else None,
            "digit_set_per_frame": list(range(NUM_DIGITS)) if all_digits_unique else None,
            "all_sectors_occupied_at_clutter_onset": all_digits_unique,
            "onset_sector_slot_rule": (
                "one slot per sector plus one independently uniform duplicate sector"
                if all_digits_unique
                else None
            ),
            "num_validated_clutter_onsets": (
                int(onset_coverage_count) if all_digits_unique else None
            ),
            "duplicate_sector_counts_at_onset": (
                onset_duplicate_sector_counts.tolist() if all_digits_unique else None
            ),
        },
        "switch_timing": {
            "sampling": "sorted unique uniform frames (Poisson event times conditional on count)",
            "requested_mean_interval_seconds": mean_switch_interval_seconds,
            "minimum_frame": int(switch_frames.min()),
            "maximum_frame": int(switch_frames.max()),
            "mean_observed_interval_frames": (
                float(intervals.mean()) if intervals.size else None
            ),
            "minimum_observed_interval_frames": (
                int(intervals.min()) if intervals.size else None
            ),
            "maximum_observed_interval_frames": (
                int(intervals.max()) if intervals.size else None
            ),
        },
        "uncontrolled_variables": [
            "foreground speed",
            "background count",
            "background speed",
            "background identities",
            "switch interval",
            "MNIST exemplar within scheduled digit",
        ],
    }
    with meta_path.open("w") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    LOGGER.info("Wrote %s, %s, and %s", npy_path, tsv_path, meta_path)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for balanced joint-switch test generation."""
    parser = argparse.ArgumentParser(
        description="Generate a strict digit×sector-balanced joint-switch test stimulus."
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "stimuli"))
    parser.add_argument("--suffix", type=str, default=DEFAULT_SUFFIX)
    parser.add_argument("--duration-seconds", type=int, default=2400)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--mean-switch-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--repeats-per-condition",
        type=int,
        default=None,
        help=(
            "Exact number of events for each of 90 digit×sector conditions. Default chooses "
            "the nearest balanced count to duration/mean-switch-interval (27 for 2400 s/1 s)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist-sample-start", type=int, default=50000)
    parser.add_argument("--mnist-sample-end", type=int, default=60000)
    parser.add_argument("--output-mode", choices=("simple", "full"), default="simple")
    parser.add_argument(
        "--all-digits-unique",
        action="store_true",
        help=(
            "Render one foreground plus nine backgrounds so digits 0-9 occur exactly once "
            "in every frame. At each clutter onset, all nine sectors are occupied and one "
            "independently uniform sector contains the extra digit. With the default suffix, "
            "writes to a distinct '-10digit-unique' dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the balanced schedule, generate the test stimulus, and validate outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    repeats = resolve_repeats_per_condition(
        args.duration_seconds,
        args.mean_switch_interval_seconds,
        args.repeats_per_condition,
    )
    schedule = build_balanced_condition_schedule(repeats, rng)
    total_frames = args.duration_seconds * args.fps
    switch_frames = sample_switch_frames(total_frames, schedule.shape[0], rng)
    suffix = args.suffix
    if args.all_digits_unique and suffix == DEFAULT_SUFFIX:
        suffix = DEFAULT_UNIQUE_SUFFIX

    config = StimulusConfig(
        width=96,
        height=96,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        fg_speeds=[1.0, 0.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        bg_char_counts=[9] if args.all_digits_unique else [1, 2, 4, 8, 12],
        bg_mean_speeds=[1.0, 2.0, 4.0, 6.0, 8.0],
        mean_switch_interval_seconds=args.mean_switch_interval_seconds,
        switch_mode="joint",
        output_dir=args.output_dir,
        mnist_sample_start=args.mnist_sample_start,
        mnist_sample_end=args.mnist_sample_end,
        suffix=suffix,
        output_mode=args.output_mode,
    )
    mnist_data = load_balanced_mnist_data(config)
    generate_balanced_joint_test(
        config,
        mnist_data,
        schedule,
        switch_frames,
        rng,
        seed=args.seed,
        mean_switch_interval_seconds=args.mean_switch_interval_seconds,
        all_digits_unique=args.all_digits_unique,
    )


if __name__ == "__main__":
    main()
