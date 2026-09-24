"""Collect continuous CM-MNIST rollouts and plot event-aligned PCA trajectories.

``collect`` runs one single-layer RNN or GaWF checkpoint over the complete held-out movie in
ordered chunks without resetting state at chunk or switch boundaries.  It saves full-stream
raw recurrent state, readout activation, outputs, labels, frame ids, and (for GaWF) feedback,
plus clean joint-switch windows of shape ``(events, 2 * radius, units)``.  ``plot`` fits an
unscaled, centred PCA separately to each saved seed and writes 3D time-coloured trajectories
and variance diagnostics.  Numeric arrays are float32 or int64.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.anal_paths import output_dir


SCRIPT_NAME = "continuous_switch_pca"
CATEGORY = "F_timing"


def parse_args() -> argparse.Namespace:
    """Parse collection and PCA plotting commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--ckpt", required=True, type=Path)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--model", required=True, choices=("rnn", "gawf"))
    collect.add_argument("--data_dir", default="./source/clutter/stimuli")
    collect.add_argument(
        "--data_suffix", default="40h-float32-jointswitch-balanced-10digit-unique"
    )
    collect.add_argument("--save_dir", "--output_dir", dest="save_dir", type=Path, default=None)
    collect.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    collect.add_argument("--chunk_size", type=int, default=32)
    collect.add_argument("--radius", type=int, default=50)
    collect.add_argument("--chan_num", type=int, default=2)
    collect.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)

    plot = commands.add_parser("plot")
    plot.add_argument("--input_dirs", required=True, type=Path, nargs="+")
    plot.add_argument("--figure_dir", type=Path, default=None)
    plot.add_argument("--components", type=int, default=3)
    plot.add_argument("--variance_only", action="store_true")
    plot.add_argument("--interactive_only", action="store_true")
    return parser.parse_args()


def _stacked_frames(
    data: np.ndarray,
    start: int,
    stop: int,
    chan_num: int,
) -> np.ndarray:
    """Return model inputs for raw output frames ``[start, stop)`` without copying the movie."""

    if not (0 < chan_num <= start <= stop <= len(data)):
        raise ValueError("Invalid raw-frame range or channel count for stacked input construction")
    channels = [data[start - chan_num + 1 + channel : stop - chan_num + 1 + channel]
                for channel in range(chan_num)]
    return np.stack(channels, axis=1)


def clean_joint_switch_frames(
    fg_switch: np.ndarray,
    bg_switch: np.ndarray,
    *,
    radius: int,
    first_frame: int,
    stop_frame: int,
) -> np.ndarray:
    """Select joint switches with one FG-or-BG event in ``[s-radius, s+radius)``.

    ``first_frame`` is normally ``chan_num`` because the earliest input stack is aligned to
    raw frame ``chan_num``.  The half-open convention yields exactly ``2 * radius`` records,
    with relative frame ids ``-radius .. radius-1`` and zero at the switch frame.
    """

    if radius <= 0 or first_frame < 0 or stop_frame > len(fg_switch):
        raise ValueError("Invalid clean-window bounds")
    events = (np.asarray(fg_switch) != 0) | (np.asarray(bg_switch) != 0)
    joint = (np.asarray(fg_switch) != 0) & (np.asarray(bg_switch) != 0)
    selected: list[int] = []
    for frame in np.flatnonzero(joint):
        start, stop = int(frame) - radius, int(frame) + radius
        if start < first_frame or stop > stop_frame:
            continue
        if int(events[start:stop].sum()) == 1:
            selected.append(int(frame))
    return np.asarray(selected, dtype=np.int64)


def _rnn_chunk(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a canonical one-layer RNN chunk while preserving its in-loop state."""

    if state is None:
        state = model.core.initial_state(encoded.size(0), encoded.device, encoded.dtype)
    activation_steps: list[torch.Tensor] = []
    for index in range(encoded.shape[1]):
        activation, state = model.core.step(encoded[:, index], state)
        activation_steps.append(activation)
    sequence = torch.stack(activation_steps, dim=1)
    char_logits, sector_logits = model.classifier(sequence)
    return sequence, sequence, char_logits, sector_logits, state


def _gawf_chunk(
    model: torch.nn.Module,
    encoded: torch.Tensor,
    state: torch.Tensor,
    feedback: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Advance one GaWF chunk through its canonical ``core.step``/feedback update."""

    hidden, feedback_before, feedback_after = [], [], []
    char_steps, sector_steps = [], []
    for index in range(encoded.shape[1]):
        feedback_before.append(feedback)
        state = model.core.step(encoded[:, index], state, feedback)
        # The state carried into the next step is the raw recurrence value; the classifier reads the
        char_logits, sector_logits = model.classifier(state)
        feedback = model._compute_feedback(char_logits, sector_logits).to(torch.float32)
        hidden.append(state)
        char_steps.append(char_logits)
        sector_steps.append(sector_logits)
        feedback_after.append(feedback)
    return (
        torch.stack(hidden, dim=1),
        torch.stack(char_steps, dim=1),
        torch.stack(sector_steps, dim=1),
        state,
        feedback,
        torch.stack(feedback_before, dim=1),
        torch.stack(feedback_after, dim=1),
    )


def _initial_gawf_state(model: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    state = model.core.initial_state(1, device, torch.float32)
    feedback = torch.zeros(1, model.feedback_dim, device=device, dtype=torch.float32)
    return state, feedback


def continuous_rollout(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    model_name: str,
    chan_num: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the whole raw movie once, preserving canonical state across every chunk boundary."""

    if chunk_size <= 0 or len(data) <= chan_num:
        raise ValueError("chunk_size must be positive and movie must exceed chan_num frames")
    if model_name not in {"rnn", "gawf"}:
        raise ValueError(f"Unsupported continuous rollout model {model_name!r}")
    first_frame, stop_frame = chan_num, len(data)
    raw_states: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    chars: list[np.ndarray] = []
    sectors: list[np.ndarray] = []
    feedback_before: list[np.ndarray] = []
    feedback_after: list[np.ndarray] = []
    rnn_state: torch.Tensor | None = None
    if model_name == "gawf":
        gawf_state, gawf_feedback = _initial_gawf_state(model, device)

    with torch.no_grad():
        for chunk_index, start in enumerate(range(first_frame, stop_frame, chunk_size), start=1):
            end = min(start + chunk_size, stop_frame)
            frames = torch.as_tensor(
                _stacked_frames(data, start, end, chan_num), device=device, dtype=torch.float32
            ).unsqueeze(0)
            encoded = model.encode_frames(frames)
            if model_name == "rnn":
                raw, activation, char, sector, rnn_state = _rnn_chunk(model, encoded, rnn_state)
                raw_states.append(raw.squeeze(0).cpu().numpy().astype(np.float32))
                activations.append(activation.squeeze(0).cpu().numpy().astype(np.float32))
            else:
                hidden, char, sector, gawf_state, gawf_feedback, fb_before, fb_after = _gawf_chunk(
                    model, encoded, gawf_state, gawf_feedback
                )
                raw_states.append(hidden.squeeze(0).cpu().numpy().astype(np.float32))
                activations.append(hidden.squeeze(0).cpu().numpy().astype(np.float32))
                feedback_before.append(fb_before.squeeze(0).cpu().numpy().astype(np.float32))
                feedback_after.append(fb_after.squeeze(0).cpu().numpy().astype(np.float32))
            chars.append(char.squeeze(0).cpu().numpy().astype(np.float32))
            sectors.append(sector.squeeze(0).cpu().numpy().astype(np.float32))
            if chunk_index % 200 == 0:
                print(f"processed {end - first_frame}/{stop_frame - first_frame} continuous frames")

    result = {
        "frame_ids": np.arange(first_frame, stop_frame, dtype=np.int64),
        "raw_hidden": np.concatenate(raw_states, axis=0),
        "readout_activation": np.concatenate(activations, axis=0),
        "char_logits": np.concatenate(chars, axis=0),
        "sector_logits": np.concatenate(sectors, axis=0),
    }
    if feedback_before:
        result["feedback_before"] = np.concatenate(feedback_before, axis=0)
        result["feedback_after"] = np.concatenate(feedback_after, axis=0)
    for name, values in result.items():
        if name != "frame_ids" and not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values in continuous rollout: {name}")
    return result


def _event_windows(values: np.ndarray, frames: np.ndarray, events: np.ndarray, radius: int) -> np.ndarray:
    index = {int(frame): row for row, frame in enumerate(frames)}
    return np.stack(
        [values[index[int(event) - radius] : index[int(event) - radius] + 2 * radius]
        for event in events
        ]
    )


def _save_collection(
    save_dir: Path,
    rollout: dict[str, np.ndarray],
    labels: np.ndarray,
    events: np.ndarray,
    radius: int,
    metadata: dict[str, Any],
) -> None:
    """Write full and event-aligned state records atomically enough for per-seed completion."""

    save_dir.mkdir(parents=True, exist_ok=True)
    frames = rollout["frame_ids"]
    stream_labels = labels[frames]
    np.savez_compressed(save_dir / "stream_trajectory.npz", **rollout, labels=stream_labels)
    event_data = {
        "event_frames": events,
        "relative_frames": np.arange(-radius, radius, dtype=np.int64),
        "frame_ids": np.stack([np.arange(event - radius, event + radius) for event in events]),
        "labels": _event_windows(stream_labels, frames, events, radius),
        "raw_hidden": _event_windows(rollout["raw_hidden"], frames, events, radius),
        "readout_activation": _event_windows(rollout["readout_activation"], frames, events, radius),
        "char_logits": _event_windows(rollout["char_logits"], frames, events, radius),
        "sector_logits": _event_windows(rollout["sector_logits"], frames, events, radius),
    }
    for key in ("feedback_before", "feedback_after"):
        if key in rollout:
            event_data[key] = _event_windows(rollout[key], frames, events, radius)
    np.savez_compressed(save_dir / "event_trajectories.npz", **event_data)
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (save_dir / "manifest.json").write_text(
        json.dumps(
            {
                "script_path": "utils/analysis/clutter/continuous_switch_pca.py",
                "category": CATEGORY,
                "data_root": str(save_dir),
                "figure_root": f"results/figs/{CATEGORY}",
                "files_written": [
                    "stream_trajectory.npz", "event_trajectories.npz", "metadata.json",
                ],
                "key_numerical_results": {
                    "stream_frames": int(metadata["stream_frames"]),
                    "clean_joint_events": int(metadata["clean_joint_events"]),
                    "radius": int(metadata["radius"]),
                    "stream_digit_accuracy": float(metadata["stream_digit_accuracy"]),
                    "stream_sector_accuracy": float(metadata["stream_sector_accuracy"]),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (save_dir / ".complete").touch()


def collect(args: argparse.Namespace) -> None:
    """Collect one checkpoint's continuous rollout and clean joint-switch event bank."""

    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset, num_pos = build_test_dataset(args)
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    is_gawf = bool(getattr(model, "is_gawf_model", False))
    expected = "gawf" if is_gawf else "rnn"
    if expected != args.model or getattr(model.core, "num_layers", 1) != 1:
        raise RuntimeError("Checkpoint/model argument must be a matching single-layer RNN or GaWF checkpoint")
    save_dir = args.save_dir or output_dir(CATEGORY, SCRIPT_NAME, "data") / f"{args.model}-seed{args.seed:02d}"
    if Path(save_dir).exists() and any(Path(save_dir).iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing analysis output: {save_dir}")
    rollout = continuous_rollout(
        model, dataset.data, model_name=args.model, chan_num=args.chan_num,
        chunk_size=args.chunk_size, device=device
    )
    labels = np.asarray(dataset.labels_sector, dtype=np.int64)
    events = clean_joint_switch_frames(
        dataset.fg_switch, dataset.bg_switch, radius=args.radius, first_frame=args.chan_num,
        stop_frame=len(dataset.data)
    )
    if not len(events):
        raise RuntimeError("No clean joint switches remain for the requested radius")
    predicted_char = rollout["char_logits"].argmax(axis=1)
    predicted_sector = rollout["sector_logits"].argmax(axis=1)
    stream_labels = labels[rollout["frame_ids"]]
    metadata = {
        "checkpoint": str(args.ckpt.resolve()), "model": args.model, "seed": args.seed,
        "data_suffix": args.data_suffix, "chunk_size": args.chunk_size, "radius": args.radius,
        "chan_num": args.chan_num, "first_output_frame": args.chan_num,
        "stream_frames": int(len(rollout["frame_ids"])), "clean_joint_events": int(len(events)),
        "stream_digit_accuracy": float((predicted_char == stream_labels[:, 0]).mean()),
        "stream_sector_accuracy": float((predicted_sector == stream_labels[:, 1]).mean()),
        "event_window": "[switch-radius, switch+radius); zero is the switch frame",
        "rnn_raw_hidden": "pre-LayerNorm recurrent output" if args.model == "rnn" else None,
        "gawf_raw_hidden": "canonical core.step state" if args.model == "gawf" else None,
        "state_reset": "only once at raw stream start; never at chunks or switches",
    }
    _save_collection(Path(save_dir), rollout, labels, events, args.radius, metadata)
    print(f"Saved {len(events)} clean joint-switch trajectories to {save_dir}")


def _pca(values: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    _left, singular, right = np.linalg.svd(flat - mean, full_matrices=False)
    variance = np.square(singular) / max(flat.shape[0] - 1, 1)
    explained = variance / variance.sum()
    return mean.astype(np.float32), right[:components].astype(np.float32), explained.astype(np.float32), singular


def _heldout_capture(
    values: np.ndarray,
    frame_ids: np.ndarray,
    components: int,
) -> tuple[float, dict[str, Any]]:
    """Score the final 20% after removing training windows sharing their raw frames."""

    events = values.shape[0]
    split = max(1, int(np.floor(events * 0.8)))
    if split >= events:
        return float("nan"), {"reason": "fewer_than_two_events"}
    heldout_frame_set = set(frame_ids[split:].reshape(-1).tolist())
    train_keep = np.asarray(
        [not bool(heldout_frame_set.intersection(frames.tolist())) for frames in frame_ids[:split]],
        dtype=bool,
    )
    if not train_keep.any():
        return float("nan"), {
            "reason": "all_training_events_overlap_heldout_frames",
            "requested_train_event_indices": list(range(split)),
            "heldout_event_indices": list(range(split, events)),
        }
    train = values[:split][train_keep].reshape(-1, values.shape[-1]).astype(np.float64)
    test = values[split:].reshape(-1, values.shape[-1]).astype(np.float64)
    mean = train.mean(axis=0)
    _left, _singular, right = np.linalg.svd(train - mean, full_matrices=False)
    centered = test - mean
    captured = float(np.square(centered @ right[:components].T).sum() / np.square(centered).sum())
    kept = np.flatnonzero(train_keep).astype(np.int64)
    removed = np.flatnonzero(~train_keep).astype(np.int64)
    return captured, {
        "requested_train_event_indices": list(range(split)),
        "train_event_indices_after_frame_disjoint_filter": kept.tolist(),
        "removed_overlapping_train_event_indices": removed.tolist(),
        "heldout_event_indices": list(range(split, events)),
        "train_events_after_filter": int(kept.size),
        "heldout_events": int(events - split),
        "shared_raw_frames_after_filter": 0,
    }


def _plot_seed(path: Path, figure_dir: Path, components: int) -> None:
    event_path = path / "event_trajectories.npz"
    with np.load(event_path) as stored:
        values = stored["raw_hidden"]
        relative = stored["relative_frames"]
        event_frame_ids = stored["frame_ids"]
    if values.shape[0] < 2:
        raise RuntimeError(f"Need at least two clean events for PCA: {path}")
    mean, basis, explained, _singular = _pca(values, components)
    scores = ((values - mean) @ basis.T).astype(np.float32)
    capture, split_provenance = _heldout_capture(values, event_frame_ids, components)
    np.savez_compressed(
        path / "pca_raw_hidden.npz", mean=mean, components=basis, explained_variance_ratio=explained,
        scores=scores, relative_frames=relative, heldout_capture=np.float32(capture),
        heldout_split_provenance=json.dumps(split_provenance, sort_keys=True),
    )
    meta_path = path / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["pca"] = {
        "representation": "raw_hidden", "centered": True, "standardized": False,
        "fit": "all clean events for saved scores and figures", "components": components,
        "heldout_protocol": "first deterministic 80% events fit; final 20% projected",
        "heldout_captured_variance": capture,
        "heldout_split_provenance": split_provenance,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    label = f"{metadata['model']} seed {metadata['seed']:02d}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(relative.min(), relative.max())
    fig = plt.figure(figsize=(7.2, 6.2))
    axis = fig.add_subplot(111, projection="3d")
    for trajectory in scores:
        axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#6E6E6E", alpha=0.22,
                  linewidth=0.45)
        axis.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=relative, cmap=cmap,
                     norm=norm, s=4, alpha=0.8, linewidths=0, depthshade=False)
        # Arrows follow actual successive states; no temporal smoothing or rescaling.
        for index in (len(relative) // 4, 3 * len(relative) // 4):
            start = trajectory[index]
            delta = trajectory[index + 1] - start
            axis.quiver(*start, *delta, color="black", linewidth=0.8,
                        arrow_length_ratio=0.4, alpha=0.8)
    switch_index = int(np.flatnonzero(relative == 0)[0])
    markers = (
        (0, "o", "#7030A0", f"Start: {relative[0]:+d}"),
        (switch_index, "X", "#E31A1C", "Switch: 0"),
        (-1, "s", "#FFD92F", f"End: {relative[-1]:+d} (post record {len(relative) - switch_index})"),
    )
    for index, marker, color, marker_label in markers:
        points = scores[:, index]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], marker=marker,
                     c=color, s=38, edgecolors="black", linewidths=0.5,
                     depthshade=False, label=marker_label)
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1.04), fontsize=8, frameon=False)
    axis.text2D(0.02, 0.02, "Black arrows: increasing time", transform=axis.transAxes, fontsize=8)
    axis.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=f"{label}: clean joint switches")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.1)
    colorbar.set_label("Relative frame (0 = switch)")
    colorbar.set_ticks([relative[0], 0, relative[-1]])
    fig.tight_layout()
    fig.savefig(figure_dir / f"continuous_switch_pca_{metadata['model']}_seed{metadata['seed']:02d}_3d.pdf")
    plt.close(fig)


def _plot_variance(paths: list[Path], figure_dir: Path) -> None:
    """Plot per-model seed means with sample SD; preserve the numeric aggregation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for model, color in (("rnn", "#2C7FB8"), ("gawf", "#D95F0E")):
        selected = [p for p in paths if json.loads((p / "metadata.json").read_text())["model"] == model]
        if not selected:
            continue
        curves = []
        seeds = []
        for path in selected:
            with np.load(path / "pca_raw_hidden.npz") as data:
                curves.append(np.cumsum(data["explained_variance_ratio"]) * 100)
            seeds.append(json.loads((path / "metadata.json").read_text())["seed"])
        if len(curves) < 2:
            raise ValueError("Mean and sample SD require at least two seeds")
        curves = np.stack(curves)
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        np.savez_compressed(selected[0].parent / f"variance_{model}_mean_std.npz",
                            seeds=np.asarray(seeds, dtype=np.int64), cumulative=curves,
                            mean=mean, std=std)
        for axis, top in zip(axes, (20, len(mean))):
            x = np.arange(1, top + 1)
            axis.plot(x, mean[:top], color=color, label=f"{model.upper()} (n={len(seeds)})")
            axis.fill_between(x, mean[:top] - std[:top], mean[:top] + std[:top],
                              color=color, alpha=0.2, linewidth=0)
    axes[0].set_ylabel("Cumulative explained variance (%)")
    for axis, title in zip(axes, ("First 20 PCs", "Full spectrum")):
        axis.set(xlabel="Number of PCs", ylim=(0, 101), title=title)
        axis.legend(frameon=False)
    fig.suptitle("Raw hidden PCA: 10-seed mean ± SD")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "continuous_switch_pca_variance_mean_std.pdf")
    plt.close(fig)


def _interactive_figure(path: Path) -> dict[str, Any]:
    """Build time-coloured trajectories with a shared colorbar and red switch crosses."""
    with np.load(path / "pca_raw_hidden.npz") as data:
        scores, relative = data["scores"], data["relative_frames"]
    with np.load(path / "event_trajectories.npz") as data:
        events, labels = data["event_frames"], data["labels"]
    meta = json.loads((path / "metadata.json").read_text())
    traces = []
    for index, trajectory in enumerate(scores):
        group = f"Event {index + 1} (frame {events[index]})"
        xyz = dict(zip(("x", "y", "z"), trajectory.T.tolist()))
        custom = np.column_stack((relative, events[index] + relative, labels[index])).tolist()
        traces.append({"type": "scatter3d", **xyz, "mode": "lines+markers",
                       "name": group, "showlegend": False,
                       "line": {"color": "#999999", "width": 2},
                       "marker": {"size": 2, "color": relative.tolist(),
                                  "colorscale": "Viridis", "cmin": int(relative[0]),
                                  "cmax": int(relative[-1]), "showscale": index == 0,
                                  "colorbar": {"title": "Relative frame",
                                               "tickvals": [int(relative[0]), 0, int(relative[-1])],
                                               "ticktext": ["−50 (pre50)", "0 (switch)",
                                                            "+49 (post record 50)"]}},
                       "customdata": custom,
                       "hovertemplate": "Relative: %{customdata[0]}<br>Frame: %{customdata[1]}"
                       "<br>Digit: %{customdata[2]}<br>Sector: %{customdata[3]}"
                       "<br>PC: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra>%{fullData.name}</extra>"})
    switch_index = int(np.flatnonzero(relative == 0)[0])
    points = scores[:, switch_index]
    traces.append({"type": "scatter3d",
                   **dict(zip(("x", "y", "z"), points.T.tolist())),
                   "mode": "markers", "name": "Switch (0)", "showlegend": False,
                   "marker": {"symbol": "x", "size": 5, "color": "#E31A1C",
                              "line": {"color": "#E31A1C", "width": 1}},
                   "customdata": np.column_stack((events, labels[:, switch_index])).tolist(),
                   "hovertemplate": "Switch: 0<br>Frame: %{customdata[0]}"
                   "<br>Digit: %{customdata[1]}<br>Sector: %{customdata[2]}<extra></extra>"})
    figure = {"data": traces, "layout": {
        "title": f"{meta['model'].upper()} seed {meta['seed']:02d} — continuous PCA",
        "height": 760, "scene": {"xaxis": {"title": "PC1"}, "yaxis": {"title": "PC2"},
                                  "zaxis": {"title": "PC3"}, "aspectmode": "data"},
        "showlegend": False, "margin": {"l": 0, "r": 0, "b": 0, "t": 50}},
        "config": {"scrollZoom": True, "displaylogo": False}}
    return figure


def _save_interactive(paths: list[Path], figure_dir: Path) -> None:
    """Write one notebook with GaWF/RNN loop cells and pre-rendered seed outputs."""
    banks = {model: {} for model in ("gawf", "rnn")}
    data_root = paths[0].parent.resolve()
    for path in paths:
        if path.parent.resolve() != data_root:
            raise ValueError("Interactive export requires one common data directory")
        meta = json.loads((path / "metadata.json").read_text())
        banks[meta["model"]][str(meta["seed"])] = _interactive_figure(path)
    if any(set(bank) != {str(seed) for seed in range(1, 11)} for bank in banks.values()):
        raise ValueError("Interactive notebook requires seeds 1–10 for both models")
    bank_path = data_root / "interactive_figures.json"
    bank_path.write_text(json.dumps(banks, ensure_ascii=False, allow_nan=False))
    relative_data = data_root.relative_to(Path.cwd().resolve()).as_posix()
    cells = [{"cell_type": "markdown", "id": "instructions", "metadata": {}, "source":
              "拖动旋转；滚轮缩放；悬停查看 frame/digit/sector。仅用 colorbar 标记 −50、"
              "switch 0、+49（第50个post记录）；红色X标记各轨迹的switch位置。"
              "每个 code cell 循环显示10个seed。\n\n"
              "已嵌入交互输出，无需运行 cell。重新运行需从本项目目录打开 Notebook，"
              "并保留 results/data 下的 interactive_figures.json。"}]
    for index, model in enumerate(("gawf", "rnn"), start=1):
        code = ("import json\nfrom pathlib import Path\nfrom IPython.display import display\n\n"
                f"relative_data = Path({relative_data!r})\n"
                "data_dir = next(p / relative_data for p in [Path.cwd(), *Path.cwd().parents]\n"
                "                if (p / relative_data / 'interactive_figures.json').is_file())\n"
                "figures = json.loads((data_dir / 'interactive_figures.json').read_text())\n"
                f"for seed in range(1, 11):\n    figure = figures[{model!r}][str(seed)]\n"
                "    display({'application/vnd.plotly.v1+json': figure}, raw=True)\n")
        outputs = [{"output_type": "display_data", "metadata": {},
                    "data": {"application/vnd.plotly.v1+json": banks[model][str(seed)]}}
                   for seed in range(1, 11)]
        cells.append({"cell_type": "code", "id": model, "metadata": {}, "source": code,
                      "execution_count": index, "outputs": outputs})
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                            "language": "python"}}, "cells": cells}
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / "continuous_switch_pca_interactive.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, allow_nan=False))


def _complete_input_dirs(input_dirs: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for value in input_dirs:
        if (value / "event_trajectories.npz").is_file() and (value / ".complete").is_file():
            result.append(value)
        else:
            result.extend(path.parent for path in value.rglob("event_trajectories.npz")
                          if (path.parent / ".complete").is_file())
    return sorted(set(result))


def plot(args: argparse.Namespace) -> None:
    """Fit and render independent per-seed raw-hidden PCAs from completed collections."""

    if args.components != 3:
        raise ValueError("This first visualization protocol is fixed to --components 3")
    paths = _complete_input_dirs(args.input_dirs)
    if not paths:
        raise RuntimeError("No completed collection directories contain event_trajectories.npz")
    figure_dir = args.figure_dir or output_dir(CATEGORY, SCRIPT_NAME, "figs")
    if args.interactive_only:
        _save_interactive(paths, Path(figure_dir))
        print(f"Saved one interactive PCA notebook with {len(paths)} seed outputs in {figure_dir}")
        return
    if not args.variance_only:
        for path in paths:
            _plot_seed(path, Path(figure_dir), args.components)
    _plot_variance(paths, Path(figure_dir))
    print(f"Rendered PCA figures for {len(paths)} completed seed collections in {figure_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
