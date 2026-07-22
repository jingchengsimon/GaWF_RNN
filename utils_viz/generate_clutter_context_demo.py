"""Generate a presentation demo of history-dependent target selection in cluttered MNIST.

The script creates two deterministic ten-frame sequences. Sequence A establishes one digit as
the coherently moving foreground target, while Sequence B establishes another. Both histories
end in the exact same rendered frame, so the final sensory input is pixel-identical even though
the correct target identity and sector differ. The generated assets are intended for a slide
that contrasts the running video with a decomposition into its ten constituent frames.

Outputs (in --output_dir):
- context_pair_side_by_side.mp4  — simultaneous slow playback of histories A and B.
- context_pair_reveal.mp4  — paired playback followed by target/sector revelation.
- context_pair_alternating.mp4  — ten frames of A followed by ten frames of B.
- sequence_a.mp4 / sequence_b.mp4  — individual slow sequence videos.
- sequence_a_frames/ and sequence_b_frames/  — ten raw grayscale PNG frames per sequence.
- sequence_a_10_frames.png / sequence_b_10_frames.png  — raw 5x2 frame decompositions.
- sequence_a_10_frames_annotated.png / sequence_b_10_frames_annotated.png  — target reveal.
- sequence_a_switch_5_frames_annotated.png  — two pre-switch frames plus A frames 1-3.
- sequence_a_labels.tsv / sequence_b_labels.tsv  — per-frame target identity and sector.
- context_demo_manifest.json  — settings, labels, output names, and final-frame identity check.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from source.clutter.generate_movies import MovingCharacter, paste_character
from utils_anal.anal_paths import output_dir


CANVAS_SIZE = 96
IMAGE_SIZE = 28
GRID_SIZE = 3
DEFAULT_OUTPUT_DIR = str(output_dir("H_controls", "generate_clutter_context_demo", "figs"))
DEFAULT_MNIST_ROOT = os.path.join(PROJECT_ROOT, "mnist_data_pytorch")
TARGET_A_COLOR = (80, 210, 90)  # BGR
TARGET_B_COLOR = (80, 170, 255)  # BGR
SECTOR_GRID_COLOR = (190, 125, 45)  # BGR: muted blue on black stimulus background
MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"


def parse_args() -> argparse.Namespace:
    """Parse presentation-demo generation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate two slow cluttered-MNIST histories with a pixel-identical final frame."
        )
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mnist_root", type=str, default=DEFAULT_MNIST_ROOT)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--final_hold_seconds", type=float, default=2.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_a", type=int, default=3)
    parser.add_argument("--target_b", type=int, default=7)
    parser.add_argument(
        "--distractor_digits",
        type=int,
        nargs="+",
        default=[1, 6, 9],
        help="Additional digits shared by both histories.",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Require an existing local MNIST copy instead of downloading it.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 4:
        raise ValueError("--num_frames must be at least 4")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.final_hold_seconds < 0:
        raise ValueError("--final_hold_seconds cannot be negative")
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")
    all_digits = [args.target_a, args.target_b, *args.distractor_digits]
    if any(digit < 0 or digit > 9 for digit in all_digits):
        raise ValueError("All target and distractor digits must be in [0, 9]")
    if args.target_a == args.target_b:
        raise ValueError("--target_a and --target_b must differ")
    if len(set(all_digits)) != len(all_digits):
        raise ValueError("Target and distractor digits must be unique")


def _load_digit_images(
    digits: Iterable[int], mnist_root: str, seed: int, download: bool
) -> dict[int, np.ndarray]:
    """Load one deterministic MNIST exemplar for every requested digit."""
    images_all, labels_all = _load_raw_mnist(mnist_root, download)
    requested = sorted(set(int(digit) for digit in digits))
    indices: dict[int, list[int]] = {digit: [] for digit in requested}
    for index, label in enumerate(labels_all.tolist()):
        if label in indices:
            indices[label].append(index)

    rng = np.random.default_rng(seed)
    images: dict[int, np.ndarray] = {}
    for digit in requested:
        if not indices[digit]:
            raise RuntimeError(f"No MNIST samples found for digit {digit}")
        sample_index = int(rng.choice(indices[digit]))
        images[digit] = images_all[sample_index].astype(np.float32, copy=False)
    return images


def _download_mnist_file(destination: Path, filename: str) -> None:
    """Download one standard MNIST gzip file into the local raw-data cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MNIST_BASE_URL}/{filename}"
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download MNIST file {url}") from exc


def _read_idx_images(path: Path) -> np.ndarray:
    """Read gzip-compressed IDX image data as uint8 (N, H, W)."""
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise RuntimeError(f"Unexpected image IDX magic {magic} in {path}")
        buffer = handle.read()
    expected = count * rows * cols
    values = np.frombuffer(buffer, dtype=np.uint8)
    if values.size != expected:
        raise RuntimeError(
            f"Image IDX payload has {values.size} bytes; expected {expected} in {path}"
        )
    return values.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read gzip-compressed IDX labels as uint8 (N,)."""
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise RuntimeError(f"Unexpected label IDX magic {magic} in {path}")
        buffer = handle.read()
    values = np.frombuffer(buffer, dtype=np.uint8)
    if values.size != count:
        raise RuntimeError(
            f"Label IDX payload has {values.size} bytes; expected {count} in {path}"
        )
    return values


def _load_raw_mnist(mnist_root: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST directly from standard IDX files without requiring torchvision."""
    raw_dir = Path(mnist_root).expanduser().resolve() / "MNIST" / "raw"
    image_name = "train-images-idx3-ubyte.gz"
    label_name = "train-labels-idx1-ubyte.gz"
    image_path = raw_dir / image_name
    label_path = raw_dir / label_name
    for path, filename in [(image_path, image_name), (label_path, label_name)]:
        if path.exists():
            continue
        if not download:
            raise RuntimeError(
                f"Missing {path}. Remove --no_download to fetch the standard MNIST files."
            )
        _download_mnist_file(path, filename)

    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"MNIST image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}"
        )
    return images, labels


def _normalized_position(x_frac: float, y_frac: float) -> np.ndarray:
    """Map normalized in-frame coordinates to a valid MNIST-image center."""
    margin = IMAGE_SIZE / 2.0
    usable = CANVAS_SIZE - IMAGE_SIZE
    return np.array(
        [margin + x_frac * usable, margin + y_frac * usable], dtype=np.float64
    )


def _linear_path(
    start: tuple[float, float], end: tuple[float, float], num_frames: int
) -> np.ndarray:
    """Return a straight constant-velocity path in normalized coordinates."""
    start_pos = _normalized_position(*start)
    end_pos = _normalized_position(*end)
    return np.linspace(start_pos, end_pos, num_frames, dtype=np.float64)


def _turning_path(
    anchors: list[tuple[float, float]], num_frames: int
) -> np.ndarray:
    """Return a smooth piecewise path with direction changes between anchors."""
    if len(anchors) < 2:
        raise ValueError("A turning path needs at least two anchors")
    anchor_positions = np.stack([_normalized_position(*point) for point in anchors])
    anchor_times = np.linspace(0.0, 1.0, len(anchors))
    frame_times = np.linspace(0.0, 1.0, num_frames)
    x_values = np.interp(frame_times, anchor_times, anchor_positions[:, 0])
    y_values = np.interp(frame_times, anchor_times, anchor_positions[:, 1])
    return np.stack([x_values, y_values], axis=1).astype(np.float64)


def _build_paths(
    target_a: int,
    target_b: int,
    distractor_digits: list[int],
    num_frames: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build two histories that differ before sharing identical final positions."""
    final_a = (0.15, 0.16)
    final_b = (0.85, 0.84)

    paths_a = {
        target_a: _linear_path((0.08, 0.88), final_a, num_frames),
        target_b: _turning_path(
            [(0.70, 0.08), (0.16, 0.34), (0.78, 0.54), final_b], num_frames
        ),
    }
    paths_b = {
        target_a: _turning_path(
            [(0.90, 0.50), (0.58, 0.08), (0.10, 0.66), final_a], num_frames
        ),
        target_b: _linear_path((0.88, 0.08), final_b, num_frames),
    }

    endpoint_slots = [
        (0.84, 0.16),
        (0.16, 0.84),
        (0.50, 0.50),
        (0.50, 0.14),
        (0.86, 0.50),
        (0.50, 0.86),
    ]
    if len(distractor_digits) > len(endpoint_slots):
        raise ValueError(
            f"At most {len(endpoint_slots)} distractors are supported for this demo layout"
        )

    rng = np.random.default_rng(seed + 1)
    for index, digit in enumerate(distractor_digits):
        end = endpoint_slots[index]
        anchors = [
            tuple(rng.uniform(0.08, 0.92, size=2).tolist()),
            tuple(rng.uniform(0.08, 0.92, size=2).tolist()),
            tuple(rng.uniform(0.08, 0.92, size=2).tolist()),
            end,
        ]
        shared_path = _turning_path(anchors, num_frames)
        paths_a[digit] = shared_path.copy()
        paths_b[digit] = shared_path.copy()

    return paths_a, paths_b


def _render_sequence(
    digit_images: dict[int, np.ndarray],
    paths: dict[int, np.ndarray],
    draw_order: list[int],
    num_frames: int,
) -> list[np.ndarray]:
    """Render a deterministic cluttered sequence with the training generator's compositor."""
    frames: list[np.ndarray] = []
    for frame_index in range(num_frames):
        frame = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        for digit in draw_order:
            position = paths[digit][frame_index]
            character = MovingCharacter(
                label=digit,
                image=digit_images[digit],
                pos=position,
                vel=np.zeros(2, dtype=np.float64),
            )
            paste_character(frame, character)
        frames.append(np.clip(frame, 0.0, 255.0).astype(np.uint8))
    return frames


def _sector_from_position(position: np.ndarray) -> int:
    """Map a character center to the same 3x3 sector convention used in training."""
    x_value, y_value = float(position[0]), float(position[1])
    col = int(np.clip((x_value / (CANVAS_SIZE - 1)) * GRID_SIZE, 0, GRID_SIZE - 1))
    row = int(np.clip((y_value / (CANVAS_SIZE - 1)) * GRID_SIZE, 0, GRID_SIZE - 1))
    return row * GRID_SIZE + col


def _upscale_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    """Convert one grayscale stimulus into a nearest-neighbor BGR presentation frame."""
    enlarged = cv2.resize(
        frame,
        (frame.shape[1] * scale, frame.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)


def _put_centered_text(
    image: np.ndarray,
    text_value: str,
    y_value: int,
    font_scale: float,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 2,
) -> None:
    """Draw centered OpenCV text on a presentation panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(text_value, font, font_scale, thickness)
    x_value = max(0, (image.shape[1] - text_width) // 2)
    cv2.putText(
        image,
        text_value,
        (x_value, y_value),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _single_panel(
    frame: np.ndarray,
    title: str,
    scale: int,
    target_position: np.ndarray | None = None,
    target_digit: int | None = None,
    target_sector: int | None = None,
    target_color: tuple[int, int, int] = TARGET_A_COLOR,
) -> np.ndarray:
    """Create one labeled video panel, optionally revealing its target."""
    header_height = 52
    # Reserve the footer even before reveal so every video frame has identical dimensions.
    footer_height = 52
    stimulus = _upscale_frame(frame, scale)
    panel = np.full(
        (header_height + stimulus.shape[0] + footer_height, stimulus.shape[1], 3),
        20,
        dtype=np.uint8,
    )
    _put_centered_text(panel, title, 35, 0.8)
    panel[header_height : header_height + stimulus.shape[0]] = stimulus

    if target_position is not None and target_digit is not None and target_sector is not None:
        center = (
            int(round(target_position[0] * scale)),
            header_height + int(round(target_position[1] * scale)),
        )
        radius = int(round((IMAGE_SIZE / 2.0 + 3) * scale))
        cv2.circle(panel, center, radius, target_color, max(2, scale), cv2.LINE_AA)
        label = f"target {target_digit} | sector {target_sector}"
        _put_centered_text(
            panel,
            label,
            header_height + stimulus.shape[0] + 34,
            0.65,
            target_color,
            2,
        )
    return panel


def _paired_panel(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    scale: int,
    reveal_a: tuple[np.ndarray, int, int] | None = None,
    reveal_b: tuple[np.ndarray, int, int] | None = None,
) -> np.ndarray:
    """Place histories A and B side by side with consistent dimensions."""
    panel_a = _single_panel(
        frame_a,
        "History A",
        scale,
        *(reveal_a or (None, None, None)),
        target_color=TARGET_A_COLOR,
    )
    panel_b = _single_panel(
        frame_b,
        "History B",
        scale,
        *(reveal_b or (None, None, None)),
        target_color=TARGET_B_COLOR,
    )
    target_height = max(panel_a.shape[0], panel_b.shape[0])
    if panel_a.shape[0] < target_height:
        panel_a = cv2.copyMakeBorder(
            panel_a, 0, target_height - panel_a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=20
        )
    if panel_b.shape[0] < target_height:
        panel_b = cv2.copyMakeBorder(
            panel_b, 0, target_height - panel_b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=20
        )
    gap = np.full((target_height, 24, 3), 20, dtype=np.uint8)
    return np.concatenate([panel_a, gap, panel_b], axis=1)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write Keynote-compatible H.264/yuv420p video from equally sized BGR frames."""
    if not frames:
        raise RuntimeError(f"No frames supplied for {path}")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to encode Keynote-compatible H.264 presentation videos"
        )
    height, width = frames[0].shape[:2]
    temporary_path = path.with_name(f".{path.stem}_mjpeg_intermediate.avi")
    writer = cv2.VideoWriter(
        str(temporary_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open intermediate video writer for {temporary_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Video frame shape mismatch in {path}")
            writer.write(frame)
    finally:
        writer.release()

    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-i",
        str(temporary_path),
        "-an",
        "-c:v",
        "libx264",
        "-tag:v",
        "avc1",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        "scale=in_range=pc:out_range=tv,format=yuv420p",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
        "-movflags",
        "+faststart",
        str(path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"H.264 encoding failed for {path}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_frame_directory(path: Path, frames: list[np.ndarray]) -> None:
    """Save raw grayscale frames for direct placement in a slide."""
    path.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames, start=1):
        frame_path = path / f"frame_{frame_index:02d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to save {frame_path}")


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    """Draw a dashed horizontal or vertical line directly on a BGR image."""
    x_start, y_start = start
    x_end, y_end = end
    if x_start != x_end and y_start != y_end:
        raise ValueError("Dashed grid lines must be horizontal or vertical")
    total_length = abs(x_end - x_start) + abs(y_end - y_start)
    if total_length == 0:
        return
    x_direction = int(np.sign(x_end - x_start))
    y_direction = int(np.sign(y_end - y_start))
    cursor = 0
    while cursor < total_length:
        segment_end = min(cursor + dash_length, total_length)
        segment_start_point = (
            x_start + x_direction * cursor,
            y_start + y_direction * cursor,
        )
        segment_end_point = (
            x_start + x_direction * segment_end,
            y_start + y_direction * segment_end,
        )
        cv2.line(
            image,
            segment_start_point,
            segment_end_point,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor += dash_length + gap_length


def _draw_sector_grid(
    image: np.ndarray,
    x_start: int,
    y_start: int,
    tile_size: int,
    scale: int,
) -> None:
    """Draw the task's 3x3 sector boundaries inside one contact-sheet frame."""
    thickness = max(1, scale)
    dash_length = max(4, 4 * scale)
    gap_length = max(3, 3 * scale)
    x_end = x_start + tile_size - 1
    y_end = y_start + tile_size - 1
    for boundary_index in (1, 2):
        x_boundary = x_start + int(round(tile_size * boundary_index / GRID_SIZE))
        y_boundary = y_start + int(round(tile_size * boundary_index / GRID_SIZE))
        _draw_dashed_line(
            image,
            (x_boundary, y_start),
            (x_boundary, y_end),
            SECTOR_GRID_COLOR,
            thickness,
            dash_length,
            gap_length,
        )
        _draw_dashed_line(
            image,
            (x_start, y_boundary),
            (x_end, y_boundary),
            SECTOR_GRID_COLOR,
            thickness,
            dash_length,
            gap_length,
        )


def _make_contact_sheet(
    frames: list[np.ndarray],
    positions: np.ndarray,
    scale: int,
    title: str,
    target_digit: int,
    target_sector: int,
    target_color: tuple[int, int, int],
    annotated: bool,
) -> np.ndarray:
    """Create a chronological 5x2 decomposition of one ten-frame sequence."""
    columns = min(5, len(frames))
    rows = int(np.ceil(len(frames) / columns))
    tile_size = CANVAS_SIZE * scale
    tile_header = 30
    title_height = 48
    gap = 10
    sheet_width = columns * tile_size + (columns - 1) * gap
    sheet_height = title_height + rows * (tile_size + tile_header) + (rows - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 20, dtype=np.uint8)
    title_suffix = f" | target {target_digit}, final sector {target_sector}" if annotated else ""
    _put_centered_text(sheet, title + title_suffix, 34, 0.75)

    for frame_index, frame in enumerate(frames):
        row, col = divmod(frame_index, columns)
        x_start = col * (tile_size + gap)
        y_start = title_height + row * (tile_size + tile_header + gap)
        tile = _upscale_frame(frame, scale)
        sheet[
            y_start + tile_header : y_start + tile_header + tile_size,
            x_start : x_start + tile_size,
        ] = tile
        _draw_sector_grid(
            sheet,
            x_start,
            y_start + tile_header,
            tile_size,
            scale,
        )
        cv2.putText(
            sheet,
            f"t{frame_index + 1}",
            (x_start + 6, y_start + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        if annotated:
            center = (
                x_start + int(round(positions[frame_index, 0] * scale)),
                y_start + tile_header + int(round(positions[frame_index, 1] * scale)),
            )
            radius = int(round((IMAGE_SIZE / 2.0 + 2) * scale))
            cv2.circle(sheet, center, radius, target_color, max(2, scale), cv2.LINE_AA)
    return sheet


def _make_switch_contact_sheet(
    previous_frames: list[np.ndarray],
    previous_positions: np.ndarray,
    current_frames: list[np.ndarray],
    current_positions: np.ndarray,
    previous_target: int,
    current_target: int,
    scale: int,
) -> np.ndarray:
    """Create a five-frame strip spanning an explicit target/context switch."""
    if len(previous_frames) != 2 or len(current_frames) != 3:
        raise ValueError("Switch sheet requires two previous and three current frames")

    frames = [*previous_frames, *current_frames]
    positions = np.concatenate([previous_positions, current_positions], axis=0)
    colors = [TARGET_B_COLOR, TARGET_B_COLOR] + [TARGET_A_COLOR] * 3
    time_labels = ["t-2", "t-1", "t0", "t+1", "t+2"]
    tile_size = CANVAS_SIZE * scale
    tile_header = 34
    title_height = 54
    gap = 10
    switch_gap = 32
    sheet_width = 5 * tile_size + 3 * gap + switch_gap
    sheet_height = title_height + tile_header + tile_size
    sheet = np.full((sheet_height, sheet_width, 3), 20, dtype=np.uint8)
    _put_centered_text(
        sheet,
        f"Target switch: {previous_target} -> {current_target}",
        36,
        0.78,
    )

    x_offsets: list[int] = []
    cursor = 0
    for frame_index in range(5):
        if frame_index > 0:
            cursor += switch_gap if frame_index == 2 else gap
        x_offsets.append(cursor)
        cursor += tile_size

    stimulus_y = title_height + tile_header
    for frame_index, (frame, position, color) in enumerate(
        zip(frames, positions, colors)
    ):
        x_start = x_offsets[frame_index]
        tile = _upscale_frame(frame, scale)
        sheet[
            stimulus_y : stimulus_y + tile_size,
            x_start : x_start + tile_size,
        ] = tile
        _draw_sector_grid(sheet, x_start, stimulus_y, tile_size, scale)
        cv2.putText(
            sheet,
            time_labels[frame_index],
            (x_start + 6, title_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        center = (
            x_start + int(round(float(position[0]) * scale)),
            stimulus_y + int(round(float(position[1]) * scale)),
        )
        radius = int(round((IMAGE_SIZE / 2.0 + 2) * scale))
        cv2.circle(sheet, center, radius, color, max(2, scale), cv2.LINE_AA)

    switch_x = x_offsets[1] + tile_size + switch_gap // 2
    cv2.line(
        sheet,
        (switch_x, title_height + 3),
        (switch_x, sheet_height - 2),
        (205, 205, 205),
        2,
        cv2.LINE_AA,
    )
    return sheet


def _write_labels(
    path: Path, target_digit: int, positions: np.ndarray
) -> list[dict[str, int | float]]:
    """Write one target label row per frame and return the serialized records."""
    records: list[dict[str, int | float]] = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "target_digit", "target_x", "target_y", "target_sector"],
            delimiter="\t",
        )
        writer.writeheader()
        for frame_index, position in enumerate(positions, start=1):
            record: dict[str, int | float] = {
                "frame": frame_index,
                "target_digit": target_digit,
                "target_x": round(float(position[0]), 2),
                "target_y": round(float(position[1]), 2),
                "target_sector": _sector_from_position(position),
            }
            records.append(record)
            writer.writerow(record)
    return records


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save {path}")


def main() -> None:
    """Generate slow videos and ten-frame decomposition assets for the presentation."""
    args = parse_args()
    _validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_order = [*args.distractor_digits, args.target_a, args.target_b]
    digit_images = _load_digit_images(
        draw_order,
        str(Path(args.mnist_root).expanduser().resolve()),
        args.seed,
        download=not args.no_download,
    )
    paths_a, paths_b = _build_paths(
        args.target_a,
        args.target_b,
        args.distractor_digits,
        args.num_frames,
        args.seed,
    )
    frames_a = _render_sequence(digit_images, paths_a, draw_order, args.num_frames)
    frames_b = _render_sequence(digit_images, paths_b, draw_order, args.num_frames)

    final_frames_identical = bool(np.array_equal(frames_a[-1], frames_b[-1]))
    if not final_frames_identical:
        raise RuntimeError("Final frames are not pixel-identical; presentation invariant failed")

    _write_frame_directory(output_dir / "sequence_a_frames", frames_a)
    _write_frame_directory(output_dir / "sequence_b_frames", frames_b)

    final_sector_a = _sector_from_position(paths_a[args.target_a][-1])
    final_sector_b = _sector_from_position(paths_b[args.target_b][-1])
    labels_a = _write_labels(
        output_dir / "sequence_a_labels.tsv", args.target_a, paths_a[args.target_a]
    )
    labels_b = _write_labels(
        output_dir / "sequence_b_labels.tsv", args.target_b, paths_b[args.target_b]
    )

    for name, frames, positions, title, target_digit, target_sector, color in [
        (
            "sequence_a",
            frames_a,
            paths_a[args.target_a],
            "History A",
            args.target_a,
            final_sector_a,
            TARGET_A_COLOR,
        ),
        (
            "sequence_b",
            frames_b,
            paths_b[args.target_b],
            "History B",
            args.target_b,
            final_sector_b,
            TARGET_B_COLOR,
        ),
    ]:
        raw_sheet = _make_contact_sheet(
            frames,
            positions,
            max(1, args.scale // 2),
            title,
            target_digit,
            target_sector,
            color,
            annotated=False,
        )
        annotated_sheet = _make_contact_sheet(
            frames,
            positions,
            max(1, args.scale // 2),
            title,
            target_digit,
            target_sector,
            color,
            annotated=True,
        )
        _save_image(output_dir / f"{name}_10_frames.png", raw_sheet)
        _save_image(output_dir / f"{name}_10_frames_annotated.png", annotated_sheet)

    switch_sheet = _make_switch_contact_sheet(
        previous_frames=frames_b[-2:],
        previous_positions=paths_b[args.target_b][-2:],
        current_frames=frames_a[:3],
        current_positions=paths_a[args.target_a][:3],
        previous_target=args.target_b,
        current_target=args.target_a,
        scale=max(1, args.scale // 2),
    )
    _save_image(
        output_dir / "sequence_a_switch_5_frames_annotated.png", switch_sheet
    )

    hold_count = max(1, int(round(args.final_hold_seconds * args.fps)))
    individual_a = [_single_panel(frame, "History A", args.scale) for frame in frames_a]
    individual_b = [_single_panel(frame, "History B", args.scale) for frame in frames_b]
    _write_video(
        output_dir / "sequence_a.mp4",
        individual_a + [individual_a[-1]] * hold_count,
        args.fps,
    )
    _write_video(
        output_dir / "sequence_b.mp4",
        individual_b + [individual_b[-1]] * hold_count,
        args.fps,
    )

    paired_raw = [
        _paired_panel(frame_a, frame_b, args.scale)
        for frame_a, frame_b in zip(frames_a, frames_b)
    ]
    _write_video(
        output_dir / "context_pair_side_by_side.mp4",
        paired_raw + [paired_raw[-1]] * hold_count,
        args.fps,
    )

    final_reveal = _paired_panel(
        frames_a[-1],
        frames_b[-1],
        args.scale,
        reveal_a=(paths_a[args.target_a][-1], args.target_a, final_sector_a),
        reveal_b=(paths_b[args.target_b][-1], args.target_b, final_sector_b),
    )
    _write_video(
        output_dir / "context_pair_reveal.mp4",
        paired_raw + [paired_raw[-1]] * hold_count + [final_reveal] * hold_count,
        args.fps,
    )

    separator = np.full_like(individual_a[0], 20)
    _put_centered_text(separator, "History B", separator.shape[0] // 2, 1.0)
    alternating_frames = (
        individual_a
        + [individual_a[-1]] * hold_count
        + [separator] * max(1, int(round(args.fps)))
        + individual_b
        + [individual_b[-1]] * hold_count
    )
    _write_video(
        output_dir / "context_pair_alternating.mp4", alternating_frames, args.fps
    )

    final_digest = hashlib.sha256(frames_a[-1].tobytes()).hexdigest()
    manifest = {
        "protocol": "two_histories_one_snapshot",
        "num_unique_frames_per_sequence": args.num_frames,
        "fps": args.fps,
        "final_hold_seconds": args.final_hold_seconds,
        "canvas_shape": [CANVAS_SIZE, CANVAS_SIZE],
        "targets": {
            "sequence_a": {
                "digit": args.target_a,
                "final_sector": final_sector_a,
                "labels": labels_a,
            },
            "sequence_b": {
                "digit": args.target_b,
                "final_sector": final_sector_b,
                "labels": labels_b,
            },
        },
        "distractor_digits": args.distractor_digits,
        "final_frames_pixel_identical": final_frames_identical,
        "final_frame_sha256": final_digest,
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    manifest_path = output_dir / "context_demo_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved presentation context demo to {output_dir}")
    print(
        "Final frames pixel-identical: "
        f"{final_frames_identical} | SHA256: {final_digest[:16]}..."
    )


if __name__ == "__main__":
    main()
