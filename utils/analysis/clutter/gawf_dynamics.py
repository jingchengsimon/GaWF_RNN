"""Describe GaWF recurrent dynamics along balanced CM-MNIST target-switch trajectories.

``collect`` loads one single-layer direct-feedback GaWF checkpoint, selects an equal number of
events from every new digit-by-sector cell, and saves matrix descriptors for the static recurrent
weight, realized effective weight, and closed-loop Jacobian. ``plot`` combines completed seed
outputs into descriptive figures. Full event-by-matrix tensors are never written; only scalar
descriptors and landmark eigenvalue samples are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.anal_paths import output_dir


OBJECTS = ("static_weight", "effective_weight", "closed_loop_jacobian")
METRICS = ("spectral_radius", "sigma_max", "frobenius_norm", "expansive_fraction")
WINDOW_CANDIDATES = (10, 20, 32, 50)
PLOT_COLORS = {
    "static_weight": "#6E6E6E",
    "effective_weight": "#2C7FB8",
    "closed_loop_jacobian": "#D95F0E",
}
FIGURE_BASENAMES = (
    "gawf_static_recurrent_spectrum",
    "gawf_dynamic_landmark_spectra",
    "gawf_dynamics_switch_timecourses",
    "gawf_dynamics_digit_sector_interaction",
    "gawf_finite_time_gain",
)


def parse_args() -> argparse.Namespace:
    """Parse the collect/plot command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--ckpt", required=True)
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--data_dir", default="./source/clutter/stimuli")
    collect.add_argument(
        "--data_suffix", default="40h-float32-jointswitch-balanced-10digit-unique"
    )
    collect.add_argument("--save_dir", type=Path, default=None)
    collect.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    collect.add_argument("--sequence_length", type=int, default=512)
    collect.add_argument("--chan_num", type=int, default=2)
    collect.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    collect.add_argument("--events_per_cell", type=int, default=10)
    collect.add_argument("--minimum_events_per_cell", type=int, default=10)
    collect.add_argument("--spectrum_events_per_cell", type=int, default=1)
    collect.add_argument("--window_candidates", nargs="+", type=int, default=WINDOW_CANDIDATES)
    collect.add_argument("--selection_seed", type=int, default=260902)
    collect.add_argument("--spectrum_dtype", choices=("float32", "float64"), default="float32")

    plot = subparsers.add_parser("plot")
    plot.add_argument("--input_root", type=Path, required=True)
    plot.add_argument("--figure_dir", type=Path, default=None)
    plot.add_argument("--expected_seeds", type=int, default=10)
    return parser.parse_args()


def _offsets(radius: int) -> np.ndarray:
    return np.concatenate((np.arange(-radius, 0), np.arange(1, radius + 1))).astype(np.int64)


def _offset_to_time(center: int, offset: int) -> int:
    return center + offset if offset < 0 else center + offset - 1


def _event_candidates(dataset: Any, radius: int) -> list[dict[str, int]]:
    """Return target-switch events whose symmetric window stays clean and in one rollout."""

    fg_switch = np.asarray(dataset.fg_switch, dtype=np.int64)
    bg_switch = np.asarray(dataset.bg_switch, dtype=np.int64)
    labels = np.asarray(dataset.labels_sector, dtype=np.int64)
    frame_num = int(dataset.frame_num)
    chan_num = int(dataset.chan_num)
    complete_output_frames = ((fg_switch.size - chan_num) // frame_num) * frame_num
    candidates: list[dict[str, int]] = []
    for raw_frame in np.flatnonzero(fg_switch != 0):
        if bg_switch[int(raw_frame)] == 0:
            continue
        output_position = int(raw_frame) - chan_num
        if output_position < 0 or output_position >= complete_output_frames:
            continue
        sequence_id, center = divmod(output_position, frame_num)
        if center - radius < 1 or center + radius - 1 >= frame_num:
            continue
        start = int(raw_frame) - radius
        stop = int(raw_frame) + radius
        other_switches = (fg_switch[start:stop] != 0) | (bg_switch[start:stop] != 0)
        if int(other_switches.sum()) != 1:
            continue
        digit, sector = labels[int(raw_frame)]
        candidates.append(
            {
                "raw_frame": int(raw_frame),
                "sequence_id": int(sequence_id),
                "center": int(center),
                "digit": int(digit),
                "sector": int(sector),
            }
        )
    return candidates


def select_balanced_events(
    dataset: Any,
    window_candidates: list[int],
    events_per_cell: int,
    minimum_events_per_cell: int,
    selection_seed: int,
) -> tuple[int, list[dict[str, int]], dict[str, Any]]:
    """Choose the widest window retaining an equal event count in all 90 condition cells."""

    if events_per_cell <= 0 or minimum_events_per_cell <= 0:
        raise ValueError("event counts must be positive")
    selected_radius = -1
    grouped: dict[tuple[int, int], list[dict[str, int]]] = {}
    audit: dict[str, Any] = {"candidate_counts": {}}
    for radius in sorted(set(window_candidates), reverse=True):
        if radius < 10:
            raise ValueError("every window candidate must be at least 10")
        current: dict[tuple[int, int], list[dict[str, int]]] = {
            (digit, sector): [] for digit in range(10) for sector in range(9)
        }
        for event in _event_candidates(dataset, radius):
            current[(event["digit"], event["sector"])].append(event)
        counts = np.asarray([len(value) for value in current.values()], dtype=np.int64)
        audit["candidate_counts"][str(radius)] = {
            "minimum": int(counts.min()),
            "maximum": int(counts.max()),
            "total": int(counts.sum()),
        }
        if counts.min() >= minimum_events_per_cell:
            selected_radius, grouped = radius, current
            break
    if selected_radius < 0:
        raise RuntimeError(
            "No candidate window retains the requested minimum events in every digit-sector cell"
        )

    count = min(events_per_cell, min(len(value) for value in grouped.values()))
    rng = np.random.default_rng(selection_seed)
    selected: list[dict[str, int]] = []
    for condition in sorted(grouped):
        values = grouped[condition]
        indices = np.sort(rng.choice(len(values), count, replace=False))
        for cell_rank, index in enumerate(indices):
            event = dict(values[int(index)])
            event["cell_rank"] = cell_rank
            selected.append(event)
    selected.sort(key=lambda value: value["raw_frame"])
    for event_id, event in enumerate(selected):
        event["event_id"] = event_id
    audit.update(
        {
            "selected_radius": selected_radius,
            "events_per_cell": count,
            "selected_events": len(selected),
            "selection_seed": selection_seed,
        }
    )
    return selected_radius, selected, audit


def _readout_derivative(model: torch.nn.Module) -> torch.Tensor:
    """Return d(feedback)/d(hidden) for the affine pre-softmax readout."""

    head = model.head
    if not hasattr(head, "fcchar") or head.fcpos is None:
        raise RuntimeError("Dynamics analysis requires digit and sector linear readout heads")
    derivative = torch.cat((head.fcchar.weight, head.fcpos.weight), dim=0)
    projector = getattr(model, "proj_out", None)
    if projector is not None:
        derivative = projector.weight @ derivative
    return derivative


def _layernorm_jacobian(
    values: torch.Tensor,
    norm: torch.nn.LayerNorm,
) -> torch.Tensor:
    """Return the exact per-sample LayerNorm Jacobian for vectors shaped ``(B, H)``."""

    hidden_size = values.shape[-1]
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    scale = torch.sqrt(variance + norm.eps)
    eye = torch.eye(hidden_size, device=values.device, dtype=values.dtype).unsqueeze(0)
    ones = torch.ones_like(eye) / hidden_size
    covariance = centered.unsqueeze(2) * centered.unsqueeze(1)
    core = eye - ones - covariance / (hidden_size * scale.square().unsqueeze(2))
    gamma = norm.weight.to(dtype=values.dtype).view(1, hidden_size, 1)
    return gamma * core / scale.unsqueeze(2)


def gawf_jacobian_objects(
    model: torch.nn.Module,
    encoded_t: torch.Tensor,
    hidden_prev: torch.Tensor,
    feedback_prev: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the exact next state and GaWF matrix objects for one or more samples."""

    if getattr(model.core, "num_layers", 1) != 1:
        raise RuntimeError("Dynamics analysis currently supports single-layer GaWF only")
    if getattr(model, "gawf_core", "legacy") != "legacy":
        raise RuntimeError(
            "The dynamics Jacobian implements the historical in-loop-wrap GaWF map "
            "(hidden_next = relu(LN(tanh(preactivation)))). Pass gawf_core='legacy' to analyze the "
            "checkpoints this analysis was derived for; the RNN-aligned core needs a new derivation."
        )
    input_size = encoded_t.shape[-1]
    hidden_size = hidden_prev.shape[-1]
    feedback = feedback_prev.to(dtype=torch.float32)
    clipped = feedback.clamp(-10, 10)
    scaled_u = model.U.unsqueeze(0) * clipped.unsqueeze(1)
    transform = torch.matmul(scaled_u, model.V) / model.gate_tau
    gate = torch.sigmoid(transform)
    gate_input, gate_hidden = gate[..., :input_size], gate[..., input_size:]

    weight_input = model.rnn.weight_ih_l0
    weight_hidden = model.rnn.weight_hh_l0
    input_term = torch.einsum("bi,bhi,hi->bh", encoded_t, gate_input, weight_input)
    hidden_term = torch.einsum("bi,bhi,hi->bh", hidden_prev, gate_hidden, weight_hidden)
    preactivation = input_term + hidden_term
    if model.rnn.bias_ih_l0 is not None:
        preactivation = preactivation + model.rnn.bias_ih_l0
    if model.rnn.bias_hh_l0 is not None:
        preactivation = preactivation + model.rnn.bias_hh_l0

    tanh_value = torch.tanh(preactivation)
    normalized = model.LNormRNN(tanh_value)
    hidden_next = torch.relu(normalized)
    tanh_derivative = 1.0 - tanh_value.square()
    relu_derivative = (normalized > 0).to(dtype=tanh_value.dtype)
    dphi = _layernorm_jacobian(tanh_value, model.LNormRNN)
    dphi = relu_derivative.unsqueeze(2) * dphi * tanh_derivative.unsqueeze(1)

    effective = gate_hidden * weight_hidden.unsqueeze(0)
    realized = torch.matmul(dphi, effective)

    sigmoid_derivative_input = gate_input * (1.0 - gate_input)
    sigmoid_derivative_hidden = gate_hidden * (1.0 - gate_hidden)
    base_input = (
        encoded_t.unsqueeze(1)
        * weight_input.unsqueeze(0)
        * sigmoid_derivative_input
    )
    base_hidden = (
        hidden_prev.unsqueeze(1)
        * weight_hidden.unsqueeze(0)
        * sigmoid_derivative_hidden
    )
    v_input, v_hidden = model.V[:, :input_size], model.V[:, input_size:]
    q_input = torch.einsum("bhi,hk,ki->bhk", base_input, model.U, v_input)
    q_hidden = torch.einsum("bhi,hk,ki->bhk", base_hidden, model.U, v_hidden)
    clamp_derivative = ((feedback > -10) & (feedback < 10)).to(dtype=feedback.dtype)
    q_matrix = (q_input + q_hidden) * clamp_derivative.unsqueeze(1) / model.gate_tau
    readout = _readout_derivative(model).to(dtype=q_matrix.dtype)
    feedback_term = torch.matmul(dphi, torch.matmul(q_matrix, readout))
    closed = realized + feedback_term
    return {
        "hidden_next": hidden_next,
        "static_weight": weight_hidden.unsqueeze(0).expand_as(effective),
        "effective_weight": effective,
        "closed_loop_jacobian": closed,
    }


def _matrix_descriptors(matrix: torch.Tensor) -> tuple[dict[str, float], np.ndarray]:
    """Return exact spectral descriptors and complex eigenvalues for one square matrix."""

    eigenvalues = torch.linalg.eigvals(matrix)
    singular_values = torch.linalg.svdvals(matrix)
    abs_eigenvalues = eigenvalues.abs()
    values = {
        "spectral_radius": float(abs_eigenvalues.max().item()),
        "sigma_max": float(singular_values.max().item()),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        "expansive_fraction": float((abs_eigenvalues > 1.0).to(torch.float64).mean().item()),
    }
    return values, eigenvalues.detach().cpu().numpy().astype(np.complex64)


def _landmark_offsets(radius: int) -> dict[str, list[int]]:
    return {
        "pre_baseline": list(range(-10, 0)),
        "post1": [1],
        "post3": [3],
        "post4": [4],
        "post10": [10],
        "post_extended": [radius],
    }


def _event_measurements(
    dataset: Any,
    model: torch.nn.Module,
    device: torch.device,
    event: dict[str, int],
    radius: int,
    spectrum_dtype: torch.dtype,
    spectrum_events_per_cell: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run one complete rollout and measure one selected event window."""

    frames, _labels = dataset[event["sequence_id"]][:2]
    frames = torch.tensor(frames, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encode_frames(frames)
    hidden = model.core.initial_state(1, device, encoded.dtype)
    feedback = torch.zeros(1, model.feedback_dim, device=device, dtype=torch.float32)
    offsets = _offsets(radius)
    time_to_offset = {
        _offset_to_time(event["center"], int(offset)): int(offset) for offset in offsets
    }
    rows: list[dict[str, Any]] = []
    matrices: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for time_idx in range(encoded.shape[1]):
            if time_idx in time_to_offset:
                objects = gawf_jacobian_objects(
                    model, encoded[:, time_idx], hidden, feedback
                )
                offset = time_to_offset[time_idx]
                matrices[offset] = {
                    name: value[0].to(dtype=spectrum_dtype) for name, value in objects.items()
                    if name != "hidden_next"
                }
                hidden = objects["hidden_next"]
            else:
                hidden = model.core.step(encoded[:, time_idx], hidden, feedback)
            char_logits, sector_logits = model.classifier(hidden)
            feedback = model._compute_feedback(char_logits, sector_logits).to(torch.float32)

    eigen_rows: list[dict[str, Any]] = []
    landmarks = _landmark_offsets(radius)
    for offset in offsets:
        current = matrices[int(offset)]
        for object_name in OBJECTS:
            descriptors, eigenvalues = _matrix_descriptors(current[object_name])
            rows.append(
                {
                    **event,
                    "offset": int(offset),
                    "object": object_name,
                    **descriptors,
                }
            )
            for landmark, landmark_offsets in landmarks.items():
                if int(offset) not in landmark_offsets or landmark == "pre_baseline":
                    continue
                if event["cell_rank"] >= spectrum_events_per_cell:
                    continue
                for eigenvalue in eigenvalues:
                    eigen_rows.append(
                        {
                            "event_id": event["event_id"],
                            "digit": event["digit"],
                            "sector": event["sector"],
                            "landmark": landmark,
                            "object": object_name,
                            "real": float(eigenvalue.real),
                            "imag": float(eigenvalue.imag),
                        }
                    )
    propagator_rows: list[dict[str, Any]] = []
    windows = {
        "post1_to_post3": list(range(2, 4)),
        "post1_to_post4": list(range(2, 5)),
        "post1_to_post10": list(range(2, 11)),
        "post1_to_post_extended": list(range(2, radius + 1)),
    }
    hidden_size = int(model.rnn.hidden_size)
    for object_name in ("closed_loop_jacobian",):
        for window_name, window_offsets in windows.items():
            propagator = torch.eye(hidden_size, device=device, dtype=spectrum_dtype)
            for offset in window_offsets:
                propagator = matrices[offset][object_name] @ propagator
            sigma_max = float(torch.linalg.svdvals(propagator).max().item())
            propagator_rows.append(
                {
                    **event,
                    "window": window_name,
                    "steps": len(window_offsets),
                    "object": object_name,
                    "maximum_log_gain": math.log(sigma_max + 1e-12) / len(window_offsets),
                }
            )
    return rows, eigen_rows, propagator_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(args: argparse.Namespace) -> None:
    """Collect one checkpoint's event-aligned dynamics descriptors."""

    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset, num_pos = build_test_dataset(args)
    model = build_model_from_ckpt(args.ckpt, num_pos, device, chan_num=args.chan_num)
    if getattr(model, "proj_out", None) is not None or int(model.feedback_dim) != 19:
        raise RuntimeError("Formal dynamics protocol requires single-layer direct 19-d feedback")
    radius, events, audit = select_balanced_events(
        dataset,
        args.window_candidates,
        args.events_per_cell,
        args.minimum_events_per_cell,
        args.selection_seed,
    )
    save_dir = args.save_dir or output_dir("F_timing", "gawf_dynamics", "data")
    save_dir = Path(save_dir)
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    spectrum_dtype = torch.float64 if args.spectrum_dtype == "float64" else torch.float32

    rows: list[dict[str, Any]] = []
    eigen_rows: list[dict[str, Any]] = []
    propagator_rows: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        event_rows, event_eigen, event_propagator = _event_measurements(
            dataset,
            model,
            device,
            event,
            radius,
            spectrum_dtype,
            args.spectrum_events_per_cell,
        )
        rows.extend(event_rows)
        eigen_rows.extend(event_eigen)
        propagator_rows.extend(event_propagator)
        if index % 25 == 0 or index == len(events):
            print(f"processed {index}/{len(events)} events", flush=True)

    for row in rows:
        row["seed"] = args.seed
    for row in eigen_rows:
        row["seed"] = args.seed
    for row in propagator_rows:
        row["seed"] = args.seed
    _write_csv(save_dir / "event_matrix_metrics.csv", rows)
    _write_csv(save_dir / "landmark_eigenvalues.csv", eigen_rows)
    _write_csv(save_dir / "finite_time_gain.csv", propagator_rows)

    static_matrix = model.rnn.weight_hh_l0.detach().to(dtype=spectrum_dtype)
    static_descriptors, static_eigenvalues = _matrix_descriptors(static_matrix)
    np.savez_compressed(
        save_dir / "static_recurrent_spectrum.npz",
        eigenvalues=static_eigenvalues,
        **{key: np.float32(value) for key, value in static_descriptors.items()},
    )
    metadata = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "data_suffix": args.data_suffix,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "chan_num": args.chan_num,
        "hidden_size": int(model.rnn.hidden_size),
        "feedback_dim": int(model.feedback_dim),
        "gate_tau": float(model.gate_tau),
        "matrix_objects": list(OBJECTS),
        "matrix_metrics": list(METRICS),
        "event_selection": audit,
        "spectrum_events_per_cell": args.spectrum_events_per_cell,
        "old_to_new_transition_balance": "not_performed",
        "dropout": "disabled_eval_mode",
        "softmax_in_feedback": False,
        "closed_loop_derivative": "analysis_only_feedback_path_included_despite_training_detach",
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (save_dir / ".complete").touch()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_seed_rows(input_root: Path, filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed_dir in sorted(input_root.glob("gawf-seed*")):
        if not (seed_dir / ".complete").is_file():
            continue
        rows.extend(_read_csv(seed_dir / filename))
    return rows


def _seed_time_summary(
    rows: list[dict[str, str]],
    object_name: str | None,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filtered = [row for row in rows if object_name is None or row.get("object") == object_name]
    offsets = np.asarray(sorted({int(row["offset"]) for row in filtered}), dtype=np.int64)
    seeds = sorted({int(row["seed"]) for row in filtered})
    values = np.full((len(seeds), len(offsets)), np.nan, dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for offset_index, offset in enumerate(offsets):
            selected = [
                float(row[metric])
                for row in filtered
                if int(row["seed"]) == seed and int(row["offset"]) == offset
            ]
            values[seed_index, offset_index] = np.mean(selected)
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0, ddof=1) / math.sqrt(len(seeds))
    return offsets, values, mean, sem


def _ci95(sem: np.ndarray, seed_count: int) -> np.ndarray:
    """Return a two-sided 95% t interval half-width for the formal ten-seed protocol."""

    if seed_count != 10:
        raise RuntimeError("The formal dynamics plot currently requires exactly ten seeds")
    return 2.262157 * sem


def _plot_timecourses(rows: list[dict[str, str]], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True)
    for column, object_name in enumerate(OBJECTS):
        for row_index, metric in enumerate(("spectral_radius", "sigma_max")):
            ax = axes[row_index, column]
            offsets, seed_values, mean, sem = _seed_time_summary(rows, object_name, metric)
            ci = _ci95(sem, seed_values.shape[0])
            for values in seed_values:
                ax.plot(offsets, values, color=PLOT_COLORS[object_name], alpha=0.16, lw=0.7)
            ax.plot(offsets, mean, color=PLOT_COLORS[object_name], lw=2.0)
            ax.fill_between(
                offsets,
                mean - ci,
                mean + ci,
                color=PLOT_COLORS[object_name],
                alpha=0.22,
                linewidth=0,
            )
            for marker in (1, 3, 4):
                ax.axvline(marker, color="#999999", lw=0.7, ls="--")
            ax.axhline(1.0, color="#333333", lw=0.7, ls=":")
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(object_name.replace("_", " ").title())
            if column == 0:
                ax.set_ylabel("Spectral radius" if row_index == 0 else "Largest singular value")
            if row_index == 1:
                ax.set_xlabel("Frames relative to joint switch")
    fig.tight_layout()
    fig.savefig(figure_dir / "gawf_dynamics_switch_timecourses.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "gawf_dynamics_switch_timecourses.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_static_spectrum(input_root: Path, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for seed_dir in sorted(input_root.glob("gawf-seed*")):
        if not (seed_dir / ".complete").is_file():
            continue
        path = seed_dir / "static_recurrent_spectrum.npz"
        if not path.is_file():
            continue
        with np.load(path) as data:
            eigenvalues = data["eigenvalues"]
        ax.scatter(eigenvalues.real, eigenvalues.imag, s=4, alpha=0.18)
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#333333", lw=0.8, ls=":")
    ax.axhline(0, color="#AAAAAA", lw=0.5)
    ax.axvline(0, color="#AAAAAA", lw=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Static recurrent-weight eigenvalues")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "gawf_static_recurrent_spectrum.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "gawf_static_recurrent_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_landmark_spectra(rows: list[dict[str, str]], figure_dir: Path) -> None:
    landmarks = ("post1", "post3", "post4", "post10")
    fig, axes = plt.subplots(
        3,
        len(landmarks),
        figsize=(10.5, 7.3),
        sharex="row",
        sharey="row",
    )
    theta = np.linspace(0, 2 * np.pi, 400)
    for row_index, object_name in enumerate(OBJECTS):
        for column, landmark in enumerate(landmarks):
            ax = axes[row_index, column]
            selected = [
                row for row in rows
                if row["object"] == object_name and row["landmark"] == landmark
            ]
            real = np.asarray([float(row["real"]) for row in selected])
            imag = np.asarray([float(row["imag"]) for row in selected])
            ax.hexbin(real, imag, gridsize=45, bins="log", cmap="magma", mincnt=1)
            ax.plot(np.cos(theta), np.sin(theta), color="white", lw=0.7, ls=":")
            ax.axhline(0, color="white", lw=0.35, alpha=0.7)
            ax.axvline(0, color="white", lw=0.35, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            if row_index == 0:
                ax.set_title(landmark.replace("_", " ").title())
            if column == 0:
                ax.set_ylabel(object_name.replace("_", " ").title())
            if row_index == len(OBJECTS) - 1:
                ax.set_xlabel("Real")
    fig.tight_layout()
    fig.savefig(figure_dir / "gawf_dynamic_landmark_spectra.pdf", bbox_inches="tight")
    fig.savefig(
        figure_dir / "gawf_dynamic_landmark_spectra.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_condition_heatmaps(rows: list[dict[str, str]], figure_dir: Path) -> None:
    offsets = (-1, 3, 4, 10)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.1), sharex=True, sharey=True)
    for ax, offset in zip(axes, offsets):
        cell = np.full((10, 9), np.nan, dtype=np.float64)
        for digit in range(10):
            for sector in range(9):
                values = [
                    float(row["sigma_max"])
                    for row in rows
                    if row["object"] == "closed_loop_jacobian"
                    and int(row["offset"]) == offset
                    and int(row["digit"]) == digit
                    and int(row["sector"]) == sector
                ]
                cell[digit, sector] = np.mean(values)
        row_mean = np.nanmean(cell, axis=1, keepdims=True)
        column_mean = np.nanmean(cell, axis=0, keepdims=True)
        interaction = cell - row_mean - column_mean + np.nanmean(cell)
        image = ax.imshow(interaction, aspect="auto", cmap="coolwarm")
        ax.set_title("pre1" if offset == -1 else f"post{offset}")
        ax.set_xlabel("Sector")
        ax.set_xticks(range(9))
    axes[0].set_ylabel("Digit")
    axes[0].set_yticks(range(10))
    fig.colorbar(image, ax=axes, label="Digit × sector interaction residual")
    fig.savefig(figure_dir / "gawf_dynamics_digit_sector_interaction.pdf", bbox_inches="tight")
    fig.savefig(
        figure_dir / "gawf_dynamics_digit_sector_interaction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_finite_time_gain(rows: list[dict[str, str]], figure_dir: Path) -> None:
    windows = (
        "post1_to_post3",
        "post1_to_post4",
        "post1_to_post10",
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 10:
        raise RuntimeError(f"Expected ten seeds for finite-time plot, found {len(seeds)}")
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x_values = np.arange(len(windows), dtype=np.float64)
    object_name = "closed_loop_jacobian"
    seed_values = np.full((len(seeds), len(windows)), np.nan, dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for window_index, window in enumerate(windows):
            selected = [
                float(row["maximum_log_gain"])
                for row in rows
                if int(row["seed"]) == seed
                and row["object"] == object_name
                and row["window"] == window
            ]
            seed_values[seed_index, window_index] = np.mean(selected)
    mean = np.mean(seed_values, axis=0)
    sem = np.std(seed_values, axis=0, ddof=1) / math.sqrt(len(seeds))
    ax.bar(x_values, mean, color=PLOT_COLORS[object_name], alpha=0.8)
    ax.errorbar(x_values, mean, yerr=_ci95(sem, len(seeds)), fmt="none", color="black")
    for seed_index in range(len(seeds)):
        ax.scatter(x_values, seed_values[seed_index], color="black", s=8, alpha=0.45)
    ax.axhline(0, color="#333333", lw=0.8, ls=":")
    ax.set_xticks(x_values, [value.replace("post1_to_", "to ") for value in windows])
    ax.set_ylabel("Maximum finite-time log gain per step")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "gawf_finite_time_gain.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "gawf_finite_time_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(args: argparse.Namespace) -> None:
    """Aggregate complete seed outputs and render the descriptive figures."""

    complete = sorted(args.input_root.glob("gawf-seed*/.complete"))
    if len(complete) != args.expected_seeds:
        raise RuntimeError(f"Expected {args.expected_seeds} complete seeds, found {len(complete)}")
    figure_dir = args.figure_dir or output_dir("F_timing", "gawf_dynamics", "figs")
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        figure_dir / f"{basename}.{suffix}"
        for basename in FIGURE_BASENAMES
        for suffix in ("pdf", "png")
        if (figure_dir / f"{basename}.{suffix}").exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing figures: {existing}")
    matrix_rows = _load_seed_rows(args.input_root, "event_matrix_metrics.csv")
    eigen_rows = _load_seed_rows(args.input_root, "landmark_eigenvalues.csv")
    propagator_rows = _load_seed_rows(args.input_root, "finite_time_gain.csv")
    _plot_static_spectrum(args.input_root, figure_dir)
    _plot_landmark_spectra(eigen_rows, figure_dir)
    _plot_timecourses(matrix_rows, figure_dir)
    _plot_condition_heatmaps(matrix_rows, figure_dir)
    _plot_finite_time_gain(propagator_rows, figure_dir)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "script": "utils/analysis/clutter/gawf_dynamics.py",
        "git_commit": commit or "unknown",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_root": str(args.input_root.resolve()),
        "figure_dir": str(figure_dir.resolve()),
        "complete_seeds": len(complete),
        "scope": "descriptive_task_driven_dynamics_no_intervention",
    }
    (args.input_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
