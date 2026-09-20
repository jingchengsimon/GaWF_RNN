"""Reflow the retained vector Figure 1 master to an ICLR 5.5-inch-wide canvas.

Input is the frozen Illustrator-authored vector PDF. The three schematic blocks and the
timeline are clipped as vector regions and repositioned independently; no rasterization or
numeric-result transformation is performed. Output is a single-page vector PDF.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, RectangleObject


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results/save_data/fig1/layout_source/Fig1_gawf_and_cm_mnist_4p375in_master.pdf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/save/iclr_figs/Fig1_gawf_and_cm_mnist.pdf"
PAGE_WIDTH_PT = 396.0
PAGE_HEIGHT_PT = 230.4
CONTENT_WIDTH_PT = 345.0


def parse_args() -> argparse.Namespace:
    """Parse the frozen vector master and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _merge_region(
    canvas: PageObject,
    source: PageObject,
    crop: tuple[float, float, float, float],
    scale: float,
    destination: tuple[float, float],
) -> None:
    """Place one clipped source region on the canvas while retaining vector operators."""

    left, bottom, right, top = crop
    page = deepcopy(source)
    page.cropbox = RectangleObject((left, bottom, right, top))
    transform = Transformation().scale(scale).translate(
        destination[0] - left * scale,
        destination[1] - bottom * scale,
    )
    canvas.merge_transformed_page(page, transform, expand=False)


def _add_panel_labels(page: PageObject) -> None:
    """Overlay A/B/C/D labels for the cell, task, and two architectures."""

    overlay = PageObject.create_blank_page(
        width=float(page.mediabox.width), height=float(page.mediabox.height)
    )
    overlay[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/PanelLabel"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
                        }
                    )
                }
            )
        }
    )
    commands = ["BT", "/PanelLabel 9 Tf"]
    for label, x, y in (
        ("A", 5.0, 218.0),
        ("B", 30.5, 76.0),
        ("C", 110.0, 218.0),
        ("D", 240.0, 218.0),
    ):
        commands.append(f"1 0 0 1 {x:g} {y:g} Tm ({label}) Tj")
    commands.append("ET")
    content = DecodedStreamObject()
    content.set_data("\n".join(commands).encode("ascii"))
    overlay[NameObject("/Contents")] = content
    page.merge_page(overlay)


def render(source_path: Path, output_path: Path) -> None:
    """Render the compact Figure 1 vector layout."""

    reader = PdfReader(source_path)
    if len(reader.pages) != 1:
        raise RuntimeError(f"Expected one source page in {source_path}, found {len(reader.pages)}")
    source = reader.pages[0]
    if abs(float(source.mediabox.width) - PAGE_WIDTH_PT) > 0.01:
        raise RuntimeError(f"Unexpected source width: {float(source.mediabox.width):.3f} pt")
    if abs(float(source.mediabox.height) - 315.0) > 0.01:
        raise RuntimeError(f"Unexpected source height: {float(source.mediabox.height):.3f} pt")

    canvas = PageObject.create_blank_page(width=PAGE_WIDTH_PT, height=PAGE_HEIGHT_PT)
    top_scale = 0.76
    top_bottom = 79.0
    _merge_region(canvas, source, (0.0, 116.0, 105.0, 315.0), top_scale, (10.0, top_bottom))
    _merge_region(canvas, source, (105.0, 116.0, 275.0, 315.0), top_scale, (115.0, top_bottom))
    _merge_region(canvas, source, (275.0, 116.0, 396.0, 315.0), top_scale, (245.0, top_bottom))

    # The two movie labels extend below the schematic blocks. Add only those narrow text
    # regions so the nearby timeline headings are not duplicated across schematic clips.
    movie_label_bottom = top_bottom - 10.0 * top_scale
    _merge_region(canvas, source, (150.0, 106.0, 210.0, 116.0), top_scale, (149.2, movie_label_bottom))
    _merge_region(canvas, source, (300.0, 106.0, 370.0, 116.0), top_scale, (264.0, movie_label_bottom))

    timeline_scale = 0.69
    # Center the timeline under the full A/C/D group rather than under the page.
    timeline_left = 37.0
    _merge_region(
        canvas,
        source,
        (0.0, 0.0, PAGE_WIDTH_PT, 105.0),
        timeline_scale,
        (timeline_left, 0.0),
    )
    _merge_region(
        canvas,
        source,
        (40.0, 105.0, 100.0, 110.0),
        timeline_scale,
        (timeline_left + 40.0 * timeline_scale, 105.0 * timeline_scale),
    )
    _merge_region(
        canvas,
        source,
        (235.0, 105.0, 305.0, 110.0),
        timeline_scale,
        (timeline_left + 235.0 * timeline_scale, 105.0 * timeline_scale),
    )
    _add_panel_labels(canvas)

    # The compact composition occupies 345 pt. Scale that vector area back to the full
    # 396-pt ICLR text width instead of retaining an unused right margin.
    content_scale = PAGE_WIDTH_PT / CONTENT_WIDTH_PT
    final_page = PageObject.create_blank_page(
        width=PAGE_WIDTH_PT, height=PAGE_HEIGHT_PT * content_scale
    )
    final_page.merge_transformed_page(
        canvas, Transformation().scale(content_scale), expand=False
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(final_page)
    with output_path.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    render(args.source, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
