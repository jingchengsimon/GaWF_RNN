"""Render presentation context sequences as mixed vector/raster SVG files.

Inputs are the structured context-demo manifest and the ten original 96-by-96 PNG frames per
sequence. Outputs embed the intrinsic raster frames while keeping titles, time labels, sector
grids, and target circles as scalable SVG elements.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 502
OUTPUT_WIDTH_PT = 480
OUTPUT_HEIGHT_PT = 240.96
FRAME_SIZE = 192
FRAME_GAP = 10
TITLE_HEIGHT = 48
LABEL_HEIGHT = 30
ROW_GAP = 10


def parse_args() -> argparse.Namespace:
    """Parse the context-demo directory containing the manifest and source frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("results/figs/presentation_context_demo"),
        help="Directory containing context_demo_manifest.json and sequence frame folders.",
    )
    return parser.parse_args()


def png_data_uri(path: Path) -> str:
    """Return one PNG as an embedded SVG data URI."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_sequence_svg(
    demo_dir: Path,
    sequence_name: str,
    sequence: dict[str, Any],
    circle_color: str,
) -> Path:
    """Render one ten-frame sequence from original PNG frames and manifest annotations."""
    history_label = sequence_name.rsplit("_", maxsplit=1)[-1].upper()
    title = sequence.get(
        "title",
        (
            f"History {history_label} | target {sequence['digit']}, "
            f"final sector {sequence['final_sector']}"
        ),
    )
    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{OUTPUT_WIDTH_PT}pt" height="{OUTPUT_HEIGHT_PT}pt" '
            f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">'
        ),
        "<title>" + title + "</title>",
        "<desc>Ten CM-MNIST frames with vector time labels, sector grids, and target circles.</desc>",
        '<rect width="100%" height="100%" fill="#151515"/>',
        (
            '<text x="500" y="31" text-anchor="middle" fill="#f2f2f2" '
            'font-family="Arial, Helvetica, sans-serif" font-size="24" '
            'font-weight="700">' + title + "</text>"
        ),
    ]

    labels = sequence["labels"]
    if len(labels) != 10:
        raise ValueError(f"{sequence_name} must contain exactly 10 labels")

    for index, label in enumerate(labels):
        row, column = divmod(index, 5)
        x = column * (FRAME_SIZE + FRAME_GAP)
        label_y = TITLE_HEIGHT + row * (LABEL_HEIGHT + FRAME_SIZE + ROW_GAP)
        frame_y = label_y + LABEL_HEIGHT
        frame_path = demo_dir / f"{sequence_name}_frames" / f"frame_{index + 1:02d}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)

        data_uri = png_data_uri(frame_path)
        svg.extend(
            [
                (
                    f'<text x="{x + 6}" y="{label_y + 22}" fill="#aaaaaa" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="18">'
                    f't{index + 1}</text>'
                ),
                (
                    f'<image x="{x}" y="{frame_y}" width="{FRAME_SIZE}" '
                    f'height="{FRAME_SIZE}" preserveAspectRatio="none" '
                    f'style="image-rendering:pixelated" href="{data_uri}" '
                    f'xlink:href="{data_uri}"/>'
                ),
            ]
        )

        for fraction in (1 / 3, 2 / 3):
            grid_x = x + FRAME_SIZE * fraction
            grid_y = frame_y + FRAME_SIZE * fraction
            svg.append(
                f'<line x1="{grid_x:.1f}" y1="{frame_y}" x2="{grid_x:.1f}" '
                f'y2="{frame_y + FRAME_SIZE}" stroke="#3498db" stroke-width="3" '
                'stroke-dasharray="8 5"/>'
            )
            svg.append(
                f'<line x1="{x}" y1="{grid_y:.1f}" x2="{x + FRAME_SIZE}" '
                f'y2="{grid_y:.1f}" stroke="#3498db" stroke-width="3" '
                'stroke-dasharray="8 5"/>'
            )

        target_x = x + float(label["target_x"]) * FRAME_SIZE / 96.0
        target_y = frame_y + float(label["target_y"]) * FRAME_SIZE / 96.0
        svg.append(
            f'<circle cx="{target_x:.2f}" cy="{target_y:.2f}" r="33" fill="none" '
            f'stroke="{circle_color}" stroke-width="4"/>'
        )

    svg.append("</svg>")
    output_path = demo_dir / f"{sequence_name}_10_frames_annotated.svg"
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """Render both context sequences and report their output paths."""
    args = parse_args()
    manifest_path = args.demo_dir / "context_demo_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = manifest["targets"]
    outputs = (
        render_sequence_svg(args.demo_dir, "sequence_a", targets["sequence_a"], "#50df5d"),
        render_sequence_svg(args.demo_dir, "sequence_b", targets["sequence_b"], "#f39a38"),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
