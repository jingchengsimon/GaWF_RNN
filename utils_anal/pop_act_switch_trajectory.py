"""Build foreground-switch-aligned population-activity trajectories.

Inputs are one ``export_pop_act`` directory containing ``pop_act.npy`` with shape ``(T, H)``
and ``labels.tsv``. Foreground switches are aligned to offset zero, restricted to events with at
least ``post_frames`` before the next foreground switch, and exported in two versions: all
eligible trials and trials whose full window contains no background switch.

Outputs under ``--save_dir/<run_tag>/``:

- ``switch_transient_trials.npz``: both trial tensors ``(trial, time, hidden)`` as float32,
  their event indices, offsets, and trial-mean trajectories.
- ``switch_transient_pca.npz``: shared-PCA coordinates ``(time, 3)``, components, feature mean,
  and explained-variance ratios as float32.
- ``switch_transient_meta.json``: window definition, selection counts, shapes, and source paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEFAULT_PRE_FRAMES = 8
DEFAULT_POST_FRAMES = 20


def parse_args() -> argparse.Namespace:
    """Parse switch-window analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pop_act_dir",
        required=True,
        help="Directory containing pop_act.npy and labels.tsv.",
    )
    parser.add_argument(
        "--save_dir",
        default="results/anal_data/5_pop_act_switch_trajectory",
        help="Analysis-data parent directory; writes one run-tag subdirectory.",
    )
    parser.add_argument(
        "--run_tag",
        default="",
        help="Output subdirectory; defaults to basename of --pop_act_dir.",
    )
    parser.add_argument("--pre_frames", type=int, default=DEFAULT_PRE_FRAMES)
    parser.add_argument("--post_frames", type=int, default=DEFAULT_POST_FRAMES)
    parser.add_argument("--n_components", type=int, default=3)
    return parser.parse_args()


def load_switch_flags(labels_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load foreground/background switch flags from an exported label table."""

    fg_switch: list[int] = []
    bg_switch: list[int] = []
    with open(labels_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"fg_switch", "bg_switch"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"labels.tsv requires {sorted(required)}, got {reader.fieldnames}"
            )
        for row in reader:
            fg_switch.append(int(float(row["fg_switch"])))
            bg_switch.append(int(float(row["bg_switch"])))
    return (
        np.asarray(fg_switch, dtype=np.int64),
        np.asarray(bg_switch, dtype=np.int64),
    )


def select_switch_events(
    fg_switch: np.ndarray,
    bg_switch: np.ndarray,
    *,
    pre_frames: int,
    post_frames: int,
) -> dict[str, np.ndarray]:
    """Select eligible foreground events with and without background-switch filtering.

    The extracted half-open window is ``[-pre_frames, post_frames)``. An event is eligible only
    when the next foreground switch is at least ``post_frames`` frames away. The final observed
    foreground switch is excluded because its next-switch interval is right-censored.
    """

    if fg_switch.ndim != 1 or bg_switch.ndim != 1 or fg_switch.shape != bg_switch.shape:
        raise ValueError("fg_switch and bg_switch must be same-length 1D arrays")
    if pre_frames < 0 or post_frames < 1:
        raise ValueError("pre_frames must be >= 0 and post_frames must be >= 1")

    switch_indices = np.flatnonzero(fg_switch != 0).astype(np.int64)
    eligible: list[int] = []
    filtered: list[int] = []
    rejected_short = 0
    rejected_boundary = 0
    rejected_bg = 0
    n_frames = fg_switch.size

    for event_pos, event_idx in enumerate(switch_indices[:-1]):
        next_distance = int(switch_indices[event_pos + 1] - event_idx)
        start = int(event_idx - pre_frames)
        stop = int(event_idx + post_frames)
        if start < 0 or stop > n_frames:
            rejected_boundary += 1
            continue
        if next_distance < post_frames:
            rejected_short += 1
            continue
        eligible.append(int(event_idx))
        if np.any(bg_switch[start:stop] != 0):
            rejected_bg += 1
        else:
            filtered.append(int(event_idx))

    return {
        "all_fg_switch_indices": switch_indices,
        "eligible_unfiltered": np.asarray(eligible, dtype=np.int64),
        "eligible_bg_filtered": np.asarray(filtered, dtype=np.int64),
        "rejected_short_next_fg": np.asarray([rejected_short], dtype=np.int64),
        "rejected_boundary": np.asarray([rejected_boundary], dtype=np.int64),
        "rejected_bg_in_window": np.asarray([rejected_bg], dtype=np.int64),
    }


def extract_trials(
    pop_act: np.ndarray,
    event_indices: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Extract an ``(event, time, hidden)`` float32 activation tensor."""

    if pop_act.ndim != 2:
        raise ValueError(f"pop_act must have shape (T, H), got {pop_act.shape}")
    if event_indices.size == 0:
        raise RuntimeError("No foreground-switch events satisfy the requested filters.")
    window_indices = event_indices[:, np.newaxis] + offsets[np.newaxis, :]
    return np.asarray(pop_act[window_indices], dtype=np.float32)


def fit_shared_pca(
    mean_bg_filtered: np.ndarray,
    mean_unfiltered: np.ndarray,
    n_components: int,
) -> dict[str, np.ndarray]:
    """Fit one PCA basis to both trial-mean trajectories and transform each version."""

    if mean_bg_filtered.shape != mean_unfiltered.shape or mean_unfiltered.ndim != 2:
        raise ValueError("Both mean trajectories must have the same (time, hidden) shape")
    combined = np.concatenate([mean_bg_filtered, mean_unfiltered], axis=0).astype(
        np.float64,
        copy=False,
    )
    feature_mean = combined.mean(axis=0, dtype=np.float64)
    centered = combined - feature_mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(centered.shape[0], centered.shape[1])
    if n_components < 1 or n_components > max_components:
        raise ValueError(f"n_components must be in [1, {max_components}]")

    components = vt[:n_components].copy()
    for component_idx in range(n_components):
        pivot = int(np.argmax(np.abs(components[component_idx])))
        if components[component_idx, pivot] < 0:
            components[component_idx] *= -1.0

    coords_filtered = (mean_bg_filtered.astype(np.float64) - feature_mean) @ components.T
    coords_unfiltered = (mean_unfiltered.astype(np.float64) - feature_mean) @ components.T
    variance = singular_values**2
    variance_ratio = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    return {
        "coords_bg_filtered": coords_filtered.astype(np.float32),
        "coords_unfiltered": coords_unfiltered.astype(np.float32),
        "components": components.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "explained_variance_ratio": variance_ratio[:n_components].astype(np.float32),
    }


def analyze_switch_trajectory(
    pop_act_path: str,
    labels_path: str,
    *,
    pre_frames: int,
    post_frames: int,
    n_components: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Run event selection, trial averaging, and shared PCA for one model."""

    pop_act = np.load(pop_act_path, mmap_mode="r")
    fg_switch, bg_switch = load_switch_flags(labels_path)
    if pop_act.shape[0] != fg_switch.size:
        raise ValueError(
            f"pop_act rows ({pop_act.shape[0]}) != label rows ({fg_switch.size})"
        )
    selected = select_switch_events(
        fg_switch,
        bg_switch,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )
    offsets = np.arange(-pre_frames, post_frames, dtype=np.int64)
    trials_unfiltered = extract_trials(
        pop_act,
        selected["eligible_unfiltered"],
        offsets,
    )
    trials_filtered = extract_trials(
        pop_act,
        selected["eligible_bg_filtered"],
        offsets,
    )
    mean_unfiltered = trials_unfiltered.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_filtered = trials_filtered.mean(axis=0, dtype=np.float64).astype(np.float32)

    trial_payload = {
        "offsets": offsets,
        "trials_bg_filtered": trials_filtered,
        "trials_unfiltered": trials_unfiltered,
        "event_indices_bg_filtered": selected["eligible_bg_filtered"],
        "event_indices_unfiltered": selected["eligible_unfiltered"],
        "mean_bg_filtered": mean_filtered,
        "mean_unfiltered": mean_unfiltered,
    }
    pca_payload = fit_shared_pca(mean_filtered, mean_unfiltered, n_components)
    pca_payload["offsets"] = offsets

    meta: dict[str, Any] = {
        "source": {
            "pop_act": os.path.abspath(pop_act_path),
            "labels": os.path.abspath(labels_path),
        },
        "input_shape": list(pop_act.shape),
        "window": {
            "pre_frames": int(pre_frames),
            "post_frames": int(post_frames),
            "offset_semantics": "half-open [-pre_frames, post_frames)",
            "offsets": offsets.tolist(),
            "switch_offset": 0,
            "minimum_next_fg_switch_distance": int(post_frames),
        },
        "selection": {
            "total_fg_switch_events": int(selected["all_fg_switch_indices"].size),
            "eligible_unfiltered": int(selected["eligible_unfiltered"].size),
            "eligible_bg_filtered": int(selected["eligible_bg_filtered"].size),
            "rejected_short_next_fg": int(selected["rejected_short_next_fg"][0]),
            "rejected_boundary": int(selected["rejected_boundary"][0]),
            "rejected_bg_in_window": int(selected["rejected_bg_in_window"][0]),
            "bg_filter_scope": "entire transient window",
        },
        "trial_shapes": {
            "bg_filtered": list(trials_filtered.shape),
            "unfiltered": list(trials_unfiltered.shape),
        },
        "mean_shape": list(mean_filtered.shape),
        "pca": {
            "fit_input": "concatenated bg-filtered and unfiltered trial-mean trajectories",
            "n_components": int(n_components),
            "coordinate_shape": list(pca_payload["coords_bg_filtered"].shape),
            "explained_variance_ratio": pca_payload["explained_variance_ratio"].tolist(),
        },
        "dtypes": {
            "activations": "float32",
            "coordinates": "float32",
            "indices": "int64",
        },
    }
    return trial_payload, pca_payload, meta


def save_analysis(
    out_dir: str,
    trial_payload: dict[str, np.ndarray],
    pca_payload: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> tuple[str, str, str]:
    """Save trial tensors, PCA results, and metadata for downstream plotting."""

    os.makedirs(out_dir, exist_ok=True)
    trial_path = os.path.join(out_dir, "switch_transient_trials.npz")
    pca_path = os.path.join(out_dir, "switch_transient_pca.npz")
    meta_path = os.path.join(out_dir, "switch_transient_meta.json")
    np.savez_compressed(trial_path, **trial_payload)
    np.savez_compressed(pca_path, **pca_payload)
    Path(meta_path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return trial_path, pca_path, meta_path


def main() -> None:
    """Analyze one exported population-activity stream."""

    args = parse_args()
    if args.n_components != 3:
        raise ValueError("This visualization protocol currently requires exactly 3 PCs.")
    pop_act_dir = os.path.abspath(args.pop_act_dir)
    pop_act_path = os.path.join(pop_act_dir, "pop_act.npy")
    labels_path = os.path.join(pop_act_dir, "labels.tsv")
    if not os.path.isfile(pop_act_path) or not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Expected pop_act.npy and labels.tsv under {pop_act_dir}")

    run_tag = args.run_tag.strip() or os.path.basename(os.path.normpath(pop_act_dir))
    out_dir = os.path.join(args.save_dir, run_tag)
    trial_payload, pca_payload, meta = analyze_switch_trajectory(
        pop_act_path,
        labels_path,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
        n_components=args.n_components,
    )
    trial_path, pca_path, meta_path = save_analysis(
        out_dir,
        trial_payload,
        pca_payload,
        meta,
    )
    print(
        "Saved switch trajectories: "
        f"filtered={meta['selection']['eligible_bg_filtered']}, "
        f"unfiltered={meta['selection']['eligible_unfiltered']}"
    )
    print(trial_path)
    print(pca_path)
    print(meta_path)


if __name__ == "__main__":
    main()
