"""Generate Figure 1c with History A's target and new distractor identities.

The target digit prototype and its ten framewise locations are copied from History A. Four
standard MNIST training exemplars with identities absent from History A are moved along new
deterministic paths. Outputs are ten source PNG frames, a TSV label record, a JSON provenance
manifest, and one mixed vector/raster SVG whose labels, grid, and target outline remain scalable.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.analysis.clutter.render_context_demo_svg import render_sequence_svg


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = PROJECT_ROOT / "results/figs/presentation_context_demo"
PROTOTYPE_PATH = DEMO_DIR / "sequence_c_distractor_prototypes.npz"
FRAME_SIZE = 28
CANVAS_SHAPE = (96, 96)
TARGET_CROP = (10, 11)
DISTRACTOR_DIGITS = (0, 2, 4, 8)
DISTRACTOR_SAMPLE_INDICES = {0: 1, 2: 5, 4: 2, 8: 17}

# New center-position trajectories. They stay away from History A's left-side target path.
DISTRACTOR_PATHS = {
    0: ((78.0, 78.0), (76.0, 20.0)),
    2: ((48.0, 18.0), (50.0, 78.0)),
    4: ((80.0, 48.0), (47.0, 48.0)),
    8: ((42.0, 80.0), (80.0, 67.0)),
}


def _crop_prototype(frame: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    """Extract one full MNIST-sized prototype from an original source frame."""
    x, y = top_left
    crop = frame[y : y + FRAME_SIZE, x : x + FRAME_SIZE]
    if crop.shape != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Invalid prototype crop at {(x, y)}: {crop.shape}")
    return crop.copy()


def _load_distractor_prototypes(path: Path) -> dict[int, np.ndarray]:
    """Load and validate the four retained MNIST uint8 exemplars."""
    prototypes = {}
    with np.load(path) as source:
        for digit in DISTRACTOR_DIGITS:
            image = np.asarray(source[f"digit_{digit}"])
            if image.shape != (FRAME_SIZE, FRAME_SIZE) or image.dtype != np.uint8:
                raise ValueError(
                    f"Invalid prototype digit {digit}: shape={image.shape}, dtype={image.dtype}"
                )
            prototypes[digit] = image
    return prototypes


def _paste_saturated(frame: np.ndarray, image: np.ndarray, center: tuple[float, float]) -> None:
    """Paste one digit using the stimulus generator's saturating-overlap rule."""
    center_x, center_y = center
    x = int(round(center_x - image.shape[1] / 2))
    y = int(round(center_y - image.shape[0] / 2))
    x0, x1 = max(0, x), min(frame.shape[1], x + image.shape[1])
    y0, y1 = max(0, y), min(frame.shape[0], y + image.shape[0])
    image_x0, image_y0 = max(0, -x), max(0, -y)
    image_x1 = image.shape[1] - max(0, x + image.shape[1] - frame.shape[1])
    image_y1 = image.shape[0] - max(0, y + image.shape[0] - frame.shape[0])
    source = image[image_y0:image_y1, image_x0:image_x1].astype(np.uint16)
    destination = frame[y0:y1, x0:x1].astype(np.uint16)
    frame[y0:y1, x0:x1] = np.clip(destination + source, 0, 255).astype(np.uint8)


def _interpolate_path(
    endpoints: tuple[tuple[float, float], tuple[float, float]], frame_index: int
) -> tuple[float, float]:
    """Return one of ten evenly spaced centers along a distractor path."""
    fraction = frame_index / 9.0
    start, end = np.asarray(endpoints[0]), np.asarray(endpoints[1])
    position = start + fraction * (end - start)
    return float(position[0]), float(position[1])


def _write_labels(path: Path, labels: list[dict[str, Any]]) -> None:
    """Write the target record in the same TSV shape as Histories A and B."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("frame", "target_digit", "target_x", "target_y", "target_sector"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(labels)


def generate_sequence_c(demo_dir: Path = DEMO_DIR) -> Path:
    """Generate History C frames, provenance, labels, and the requested SVG."""
    main_manifest_path = demo_dir / "context_demo_manifest.json"
    main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
    sequence_a = main_manifest["targets"]["sequence_a"]
    labels = sequence_a["labels"]
    if len(labels) != 10:
        raise ValueError("History A must contain exactly ten target labels")

    source_path = demo_dir / "sequence_a_frames/frame_10.png"
    source = np.asarray(Image.open(source_path).convert("L"), dtype=np.uint8)
    if source.shape != CANVAS_SHAPE:
        raise ValueError(f"Expected a 96-by-96 source frame, got {source.shape}")
    distractor_prototypes = _load_distractor_prototypes(PROTOTYPE_PATH)
    target_prototype = _crop_prototype(source, TARGET_CROP)

    frame_dir = demo_dir / "sequence_c_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_hashes: list[str] = []
    distractor_centers: dict[str, list[list[float]]] = {
        str(digit): [] for digit in DISTRACTOR_DIGITS
    }
    for frame_index, label in enumerate(labels):
        frame = np.zeros(CANVAS_SHAPE, dtype=np.uint8)
        for digit, endpoints in DISTRACTOR_PATHS.items():
            center = _interpolate_path(endpoints, frame_index)
            distractor_centers[str(digit)].append(
                [round(center[0], 4), round(center[1], 4)]
            )
            _paste_saturated(frame, distractor_prototypes[digit], center)
        _paste_saturated(
            frame,
            target_prototype,
            (float(label["target_x"]), float(label["target_y"])),
        )
        frame_path = frame_dir / f"frame_{frame_index + 1:02d}.png"
        Image.fromarray(frame, mode="L").save(frame_path)
        frame_hashes.append(hashlib.sha256(frame_path.read_bytes()).hexdigest())

    sequence_c = {
        "title": "History C | same target as A, different distractor identities",
        "digit": sequence_a["digit"],
        "final_sector": sequence_a["final_sector"],
        "labels": labels,
    }
    _write_labels(demo_dir / "sequence_c_labels.tsv", labels)
    provenance = {
        "protocol": "same_target_different_distractor_identities",
        "target_source": "sequence_a",
        "target_identity_and_framewise_locations_match_source": True,
        "canvas_shape": list(CANVAS_SHAPE),
        "target_prototype_source_frame": str(source_path.relative_to(PROJECT_ROOT)),
        "target_prototype_crop_xy": [*TARGET_CROP, FRAME_SIZE, FRAME_SIZE],
        "distractor_dataset": "MNIST training split",
        "distractor_digits": list(DISTRACTOR_DIGITS),
        "distractor_sample_indices": DISTRACTOR_SAMPLE_INDICES,
        "distractor_prototype_path": str(PROTOTYPE_PATH.relative_to(PROJECT_ROOT)),
        "distractor_prototype_sha256": hashlib.sha256(PROTOTYPE_PATH.read_bytes()).hexdigest(),
        "distractor_paths": distractor_centers,
        "target": sequence_c,
        "frame_sha256": frame_hashes,
    }
    (demo_dir / "sequence_c_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return render_sequence_svg(demo_dir, "sequence_c", sequence_c, "#50df5d")


def main() -> None:
    """Generate Figure 1c and print its path."""
    print(generate_sequence_c())


if __name__ == "__main__":
    main()
