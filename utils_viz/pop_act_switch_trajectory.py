"""Render switch-aligned PCA trajectories as an interactive two-panel 3D HTML figure.

Input is ``switch_transient_pca.npz`` plus its companion metadata produced by
``utils_anal/pop_act_switch_trajectory.py``. The left subplot excludes any trial whose full
window contains a background switch; the right subplot includes every eligible foreground
switch trial. Start, foreground-switch, and end points are explicitly labeled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from utils_anal.anal_paths import output_dir

def parse_args() -> argparse.Namespace:
    """Parse interactive trajectory plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Run directory containing switch_transient_pca.npz and metadata.",
    )
    parser.add_argument(
        "--save_dir",
        default=str(output_dir("F_timing", "pop_act_switch_trajectory", "figs")),
        help="Figure parent directory; writes one run-tag subdirectory.",
    )
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--out_html", default="switch_transient_3d.html")
    return parser.parse_args()


def _axis_range(*coordinate_sets: np.ndarray) -> list[float]:
    """Return one padded symmetric axis range shared by both subplots."""

    values = np.concatenate([coords.reshape(-1) for coords in coordinate_sets])
    bound = float(np.max(np.abs(values))) if values.size else 1.0
    bound = max(bound * 1.08, 1e-6)
    return [-bound, bound]


def build_trajectory_figure(
    pca_payload: Any,
    meta: dict[str, Any],
    run_tag: str,
) -> Any:
    """Build a Plotly figure with background-filtered and unfiltered 3D trajectories."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    offsets = np.asarray(pca_payload["offsets"], dtype=np.int64)
    coords_filtered = np.asarray(pca_payload["coords_bg_filtered"], dtype=np.float32)
    coords_unfiltered = np.asarray(pca_payload["coords_unfiltered"], dtype=np.float32)
    variance = np.asarray(pca_payload["explained_variance_ratio"], dtype=np.float32)
    if coords_filtered.shape != coords_unfiltered.shape or coords_filtered.shape[1] != 3:
        raise ValueError("Expected matching (time, 3) coordinates for both trajectory versions")
    switch_positions = np.flatnonzero(offsets == 0)
    if switch_positions.size != 1:
        raise ValueError("offsets must contain exactly one foreground-switch offset 0")
    switch_pos = int(switch_positions[0])

    counts = meta["selection"]
    titles = [
        f"No bg_switch in window (n={counts['eligible_bg_filtered']})",
        f"All eligible fg_switch trials (n={counts['eligible_unfiltered']})",
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles,
        horizontal_spacing=0.04,
    )

    marker_color = offsets.astype(np.float32)
    panels = [(coords_filtered, 1), (coords_unfiltered, 2)]
    for coords, col in panels:
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="lines+markers",
                line={"color": "#607D8B", "width": 7},
                marker={
                    "size": 3.5,
                    "color": marker_color,
                    "colorscale": "Viridis",
                    "cmin": float(offsets.min()),
                    "cmax": float(offsets.max()),
                    "showscale": col == 2,
                    "colorbar": {"title": "Frame offset", "x": 1.02},
                },
                customdata=offsets,
                hovertemplate=(
                    "offset=%{customdata}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}"
                    "<br>PC3=%{z:.3f}<extra></extra>"
                ),
                name="Trial-mean trajectory",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        point_specs = [
            (0, "Start", "circle", "#2CA02C"),
            (switch_pos, "fg_switch", "diamond", "#D62728"),
            (len(offsets) - 1, "End", "square", "#1F77B4"),
        ]
        for point_idx, label, symbol, color in point_specs:
            fig.add_trace(
                go.Scatter3d(
                    x=[coords[point_idx, 0]],
                    y=[coords[point_idx, 1]],
                    z=[coords[point_idx, 2]],
                    mode="markers+text",
                    marker={
                        "size": 8,
                        "symbol": symbol,
                        "color": color,
                        "line": {"color": "#FFFFFF", "width": 1.5},
                    },
                    text=[label],
                    textposition="top center",
                    customdata=[int(offsets[point_idx])],
                    hovertemplate=(
                        f"{label}<br>offset=%{{customdata}}<br>PC1=%{{x:.3f}}"
                        "<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>"
                    ),
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )

    axis_range = _axis_range(coords_filtered, coords_unfiltered)
    variance_labels = [f"PC{i + 1} ({variance[i] * 100:.1f}%)" for i in range(3)]
    scene = {
        "xaxis": {"title": variance_labels[0], "range": axis_range},
        "yaxis": {"title": variance_labels[1], "range": axis_range},
        "zaxis": {"title": variance_labels[2], "range": axis_range},
        "aspectmode": "cube",
        "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
    }
    fig.update_layout(
        title={
            "text": (
                f"{run_tag}: fg-switch transient trajectory "
                f"[{offsets.min()}, {offsets.max()}] frames"
            ),
            "x": 0.5,
        },
        scene=scene,
        scene2=scene,
        width=1500,
        height=720,
        margin={"l": 15, "r": 90, "t": 85, "b": 20},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.02},
    )
    return fig


def main() -> None:
    """Load one analysis result and save its self-contained interactive HTML."""

    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    pca_path = os.path.join(data_dir, "switch_transient_pca.npz")
    meta_path = os.path.join(data_dir, "switch_transient_meta.json")
    if not os.path.isfile(pca_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing PCA or metadata under {data_dir}")
    run_tag = args.run_tag.strip() or os.path.basename(os.path.normpath(data_dir))
    with np.load(pca_path) as pca_payload:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        fig = build_trajectory_figure(pca_payload, meta, run_tag)

    out_dir = os.path.join(args.save_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_html)
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
