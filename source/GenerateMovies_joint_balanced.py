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

from source.GenerateMovies import MovingCharacter, StimulusConfig, load_mnist_data, paste_character


LOGGER = logging.getLogger(__name__)
NUM_DIGITS = 10
NUM_SECTORS = 9
GRID_SIZE = 3
DEFAULT_SUFFIX = "reg-test-40h-float32-jointswitch-balanced"
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
    _, bg_mean_speed, background_chars = _build_background(config, mnist_data, rng)
    switch_lookup = {
        int(frame): (int(condition[0]), int(condition[1]))
        for frame, condition in zip(switch_frames, schedule)
    }
    observed_counts = np.zeros((NUM_DIGITS, NUM_SECTORS), dtype=np.int64)

    with tsv_path.open("w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(LABEL_COLUMNS)
        for frame_idx in tqdm(range(total_frames), desc="Generating balanced joint test"):
            fg_switch_flag = 0
            bg_switch_flag = 0
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
                _, bg_mean_speed, background_chars = _build_background(
                    config, mnist_data, rng
                )
            else:
                target_center = None

            fg_char.update_position(frame_dims)
            for character in background_chars:
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

    config = StimulusConfig(
        width=96,
        height=96,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        fg_speeds=[1.0, 0.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        bg_char_counts=[1, 2, 4, 8, 12],
        bg_mean_speeds=[1.0, 2.0, 4.0, 6.0, 8.0],
        mean_switch_interval_seconds=args.mean_switch_interval_seconds,
        switch_mode="joint",
        output_dir=args.output_dir,
        mnist_sample_start=args.mnist_sample_start,
        mnist_sample_end=args.mnist_sample_end,
        suffix=args.suffix,
        output_mode=args.output_mode,
    )
    mnist_data = load_mnist_data(config)
    generate_balanced_joint_test(
        config,
        mnist_data,
        schedule,
        switch_frames,
        rng,
        seed=args.seed,
        mean_switch_interval_seconds=args.mean_switch_interval_seconds,
    )


if __name__ == "__main__":
    main()
