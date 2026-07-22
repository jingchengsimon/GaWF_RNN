"""Audit empirical GaWF gate distributions on a held-out Clutter test set.

Inputs are a trained single-layer GaWF checkpoint and a test stimulus/label pair. The script
replays every sequence with reset feedback, records the exact per-frame feedback sufficient to
reconstruct both gate tensors, and streams statistics over all input and recurrent gate entries.
Outputs are a compact trajectory ``.npz``, a related statistics ``.npz``, and JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset


@dataclass
class MomentAccumulator:
    """Streaming raw moments and threshold counts for one scalar population."""

    count: int = 0
    total: float = 0.0
    total2: float = 0.0
    total3: float = 0.0
    above_half: int = 0

    def update(self, values: np.ndarray, block_size: int = 1_000_000) -> None:
        """Accumulate float64 moments from a potentially large float32 array."""

        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        for start in range(0, flat.size, block_size):
            block = flat[start : start + block_size].astype(np.float64)
            self.count += int(block.size)
            self.total += float(block.sum())
            self.total2 += float(np.square(block).sum())
            self.total3 += float((np.square(block) * block).sum())
            self.above_half += int(np.count_nonzero(block > 0.5))

    def summary(self) -> dict[str, float | int]:
        """Return mean, population standard deviation, skew, and fraction above 0.5."""

        if self.count == 0:
            raise RuntimeError("Cannot summarize an empty accumulator")
        mean = self.total / self.count
        second = self.total2 / self.count
        variance = max(0.0, second - mean * mean)
        std = variance**0.5
        third_central = self.total3 / self.count - 3.0 * mean * second + 2.0 * mean**3
        skew = third_central / std**3 if std > 0.0 else 0.0
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "skew": skew,
            "fraction_above_0_5": self.above_half / self.count,
        }


def parse_args() -> argparse.Namespace:
    """Parse analysis arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--data_suffix", required=True)
    parser.add_argument(
        "--save_dir",
        default=str(output_dir("A_raw_gate", "gawf_gate_distribution", "data")),
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gate_chunk_size", type=int, default=32)
    parser.add_argument("--hist_bins", type=int, default=400)
    parser.add_argument("--effective_hist_bins", type=int, default=500)
    parser.add_argument("--chan_num", type=int, default=2)
    parser.add_argument("--use_mmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse_trajectory", action="store_true")
    return parser.parse_args()


def _gate_tensors(
    feedback: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    input_size: int,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the model's eager float32 gate computation exactly."""

    fb_t = feedback.to(dtype=torch.float32).clamp(-10, 10).unsqueeze(2)
    scaled_u = u.unsqueeze(0) * fb_t.transpose(1, 2)
    transform = torch.matmul(scaled_u, v)
    gate = torch.sigmoid(transform / tau)
    return gate[..., :input_size], gate[..., input_size:]


def _trajectory(
    encoded: torch.Tensor,
    model: torch.nn.Module,
    gate_scale: float,
    record_feedback: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run one reset-feedback trajectory, optionally retaining pre-step feedback."""

    batch_size, frame_num, input_size = encoded.shape
    hidden_size = model.rnn.hidden_size
    feedback_dim = model.feedback_dim
    hidden = torch.zeros(batch_size, hidden_size, dtype=encoded.dtype, device=encoded.device)
    feedback = torch.zeros(batch_size, feedback_dim, dtype=torch.float32, device=encoded.device)
    char_steps: list[torch.Tensor] = []
    sector_steps: list[torch.Tensor] = []
    hidden_steps: list[torch.Tensor] = []
    feedback_steps: list[torch.Tensor] = []

    for time_idx in range(frame_num):
        if record_feedback:
            feedback_steps.append(feedback.detach().clone())
        gate_ih, gate_hh = _gate_tensors(
            feedback,
            model.U,
            model.V,
            input_size,
            model.gate_tau,
        )
        gate_ih = gate_ih * gate_scale
        gate_hh = gate_hh * gate_scale
        input_term = torch.einsum(
            "bi,bhi,hi->bh", encoded[:, time_idx], gate_ih, model.rnn.weight_ih_l0
        )
        hidden_term = torch.einsum(
            "bi,bhi,hi->bh", hidden, gate_hh, model.rnn.weight_hh_l0
        )
        preactivation = input_term + hidden_term
        if model.rnn.bias_ih_l0 is not None:
            preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
        if model.rnn.bias_hh_l0 is not None:
            preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
        hidden = torch.relu(model.LNormRNN(torch.tanh(preactivation)))
        char_t, sector_t = model.classifier(hidden)
        feedback = torch.cat([char_t, sector_t], dim=-1).to(dtype=torch.float32)
        char_steps.append(char_t)
        sector_steps.append(sector_t)
        hidden_steps.append(hidden)

    stored_feedback = torch.stack(feedback_steps, dim=1) if record_feedback else None
    return (
        torch.stack(char_steps, dim=1),
        torch.stack(sector_steps, dim=1),
        torch.stack(hidden_steps, dim=1),
        stored_feedback,
    )


def collect_trajectory(
    test_ds: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run baseline and gate-scale interventions over every test sequence."""

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feedback_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    output_max = {"0.5": 0.0, "2.0": 0.0}
    hidden_max = {"0.5": 0.0, "2.0": 0.0}
    accuracy_counts = {
        key: {"char_correct": 0, "sector_correct": 0, "count": 0}
        for key in ("1.0", "0.5", "2.0")
    }
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.int64)
            encoded = model.encode_frames(frames)

            baseline = _trajectory(encoded, model, 1.0, True)
            half = _trajectory(encoded, model, 0.5, False)
            double = _trajectory(encoded, model, 2.0, False)
            if baseline[3] is None:
                raise RuntimeError("Baseline feedback was not recorded")
            feedback_batches.append(baseline[3].cpu().numpy().astype(np.float32))
            label_batches.append(labels.cpu().numpy().astype(np.int64))

            for key, result in (("1.0", baseline), ("0.5", half), ("2.0", double)):
                char_pred = result[0].argmax(dim=-1)
                sector_pred = result[1].argmax(dim=-1)
                accuracy_counts[key]["char_correct"] += int((char_pred == labels[..., 0]).sum())
                accuracy_counts[key]["sector_correct"] += int(
                    (sector_pred == labels[..., 1]).sum()
                )
                accuracy_counts[key]["count"] += int(labels[..., 0].numel())

            for key, result in (("0.5", half), ("2.0", double)):
                char_diff = float((result[0] - baseline[0]).abs().max())
                sector_diff = float((result[1] - baseline[1]).abs().max())
                output_max[key] = max(output_max[key], char_diff, sector_diff)
                hidden_max[key] = max(
                    hidden_max[key], float((result[2] - baseline[2]).abs().max())
                )

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                elapsed = time.perf_counter() - start_time
                print(
                    f"trajectory batches {batch_idx + 1}/{len(loader)} | elapsed={elapsed:.1f}s",
                    flush=True,
                )

    performance: dict[str, object] = {}
    for key, counts in accuracy_counts.items():
        count = int(counts["count"])
        performance[key] = {
            "char_accuracy": counts["char_correct"] / count,
            "sector_accuracy": counts["sector_correct"] / count,
            "n_frames": count,
        }
    for key in ("0.5", "2.0"):
        performance[key]["max_abs_output_difference"] = output_max[key]
        performance[key]["max_abs_hidden_difference"] = hidden_max[key]

    arrays = {
        "feedback": np.concatenate(feedback_batches, axis=0).astype(np.float32),
        "labels": np.concatenate(label_batches, axis=0).astype(np.int64),
    }
    return arrays, performance


def iter_gate_chunks(
    feedback: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    input_size: int,
    tau: float,
    chunk_size: int,
    *,
    device: str | torch.device = "cpu",
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield exact float32 input and recurrent gates for flattened frame chunks."""

    flat_feedback = feedback.reshape(-1, feedback.shape[-1])
    target_device = torch.device(device)
    u_t = torch.from_numpy(u).to(target_device)
    v_t = torch.from_numpy(v).to(target_device)
    for start in range(0, flat_feedback.shape[0], chunk_size):
        end = min(start + chunk_size, flat_feedback.shape[0])
        feedback_t = torch.from_numpy(flat_feedback[start:end]).to(target_device)
        with torch.no_grad():
            gate_ih, gate_hh = _gate_tensors(feedback_t, u_t, v_t, input_size, tau)
        yield start, end, gate_ih.cpu().numpy(), gate_hh.cpu().numpy()


def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Count uniform histogram bins directly, avoiding NumPy's sorting-based edge path."""

    flat = np.asarray(values).reshape(-1)
    bins = edges.size - 1
    low = float(edges[0])
    high = float(edges[-1])
    scale = bins / (high - low)
    counts = np.zeros(bins, dtype=np.int64)
    block_size = 2_000_000
    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        indices = np.floor((block - low) * scale).astype(np.int32)
        np.clip(indices, 0, bins - 1, out=indices)
        counts += np.bincount(indices, minlength=bins).astype(np.int64)
    return counts


def _group_mean_delta(
    group_sums: np.ndarray,
    group_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return group means, trial-weighted per-synapse grand mean, and group deltas."""

    counts = np.asarray(group_counts, dtype=np.int64)
    if group_sums.ndim < 2 or group_sums.shape[0] != counts.size:
        raise ValueError(
            f"group_sums leading dimension and group_counts must match: "
            f"{group_sums.shape} vs {counts.shape}"
        )
    if np.any(counts <= 0):
        raise ValueError(f"Every group must be nonempty, got counts={counts.tolist()}")
    denominator = counts.reshape((counts.size,) + (1,) * (group_sums.ndim - 1))
    group_means = group_sums / denominator
    grand_mean = group_sums.sum(axis=0) / counts.sum()
    group_delta = group_means - grand_mean[None, ...]
    return group_means, grand_mean, group_delta


def _spatial_sector_indices(input_size: int) -> list[np.ndarray]:
    """Map flattened 32x6x6 encoder features to their 3x3 coarse spatial sectors."""

    if input_size != 32 * 6 * 6:
        raise RuntimeError(f"Task-relevance proxy requires input_size=1152, got {input_size}")
    indices: list[np.ndarray] = []
    layout = np.arange(input_size, dtype=np.int64).reshape(32, 6, 6)
    for sector in range(9):
        row, col = divmod(sector, 3)
        indices.append(layout[:, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2].reshape(-1))
    return indices


def _gini(values: np.ndarray) -> float:
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if flat.size == 0 or float(flat.sum()) == 0.0:
        return 0.0
    ranks = np.arange(1, flat.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(ranks, flat) / flat.sum() - flat.size - 1.0) / flat.size)


def _sparsity(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    ordered = np.sort(flat)[::-1]
    total = float(ordered.sum())
    top5 = max(1, int(np.ceil(0.05 * ordered.size)))
    top10 = max(1, int(np.ceil(0.10 * ordered.size)))
    participation = total * total / float(np.square(ordered).sum())
    return {
        "top_5pct_mass_fraction": float(ordered[:top5].sum() / total),
        "top_10pct_mass_fraction": float(ordered[:top10].sum() / total),
        "gini": _gini(ordered),
        "participation_ratio": participation,
        "normalized_participation_ratio": participation / ordered.size,
    }


def _hist_mode_and_modality(counts: np.ndarray, edges: np.ndarray) -> dict[str, object]:
    modal_idx = int(np.argmax(counts))
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma=2.0)
    prominence = max(1.0, float(smooth.max()) * 0.02)
    internal_peaks, properties = find_peaks(smooth, prominence=prominence, distance=8)
    peaks = list(int(value) for value in internal_peaks)
    prominences = list(float(value) for value in properties.get("prominences", []))
    if smooth[0] > smooth[1] and smooth[0] >= prominence:
        peaks.append(0)
        prominences.append(float(smooth[0] - smooth.min()))
    if smooth[-1] > smooth[-2] and smooth[-1] >= prominence:
        peaks.append(smooth.size - 1)
        prominences.append(float(smooth[-1] - smooth.min()))
    ordered = sorted(zip(peaks, prominences))
    peaks = [item[0] for item in ordered]
    prominences = [item[1] for item in ordered]
    if len(peaks) == 1:
        label = "unimodal"
    elif len(peaks) == 2:
        label = "bimodal"
    else:
        label = "multimodal"
    low_stop = int(np.searchsorted(edges, 0.01, side="left"))
    high_start = int(np.searchsorted(edges, 0.99, side="left"))
    total = int(counts.sum())
    return {
        "histogram_mode": float((edges[modal_idx] + edges[modal_idx + 1]) / 2.0),
        "histogram_mode_bin": [float(edges[modal_idx]), float(edges[modal_idx + 1])],
        "smoothed_peak_count": len(peaks),
        "smoothed_peak_locations": [
            float((edges[index] + edges[index + 1]) / 2.0) for index in peaks
        ],
        "modality": label,
        "peak_prominences": prominences,
        "fraction_below_0_01": int(counts[:low_stop].sum()) / total,
        "fraction_at_or_above_0_99": int(counts[high_start:].sum()) / total,
    }


def _radix_medians(
    feedback: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    input_size: int,
    tau: float,
    chunk_size: int,
) -> dict[str, float]:
    """Find exact even-sample float32 medians with four streaming radix passes."""

    n_frames = int(np.prod(feedback.shape[:-1]))
    hidden_size = int(u.shape[0])
    sizes = {
        "input": n_frames * hidden_size * input_size,
        "recurrent": n_frames * hidden_size * hidden_size,
    }
    states: dict[str, list[dict[str, object]]] = {}
    for kind, size in sizes.items():
        states[kind] = [
            {
                "prefix": 0,
                "prefix_bits": 0,
                "targets": [("lower", size // 2 - 1), ("upper", size // 2)],
            }
        ]

    resolved: dict[str, dict[str, int]] = {"input": {}, "recurrent": {}}
    for byte_position in (3, 2, 1, 0):
        histograms = {
            kind: [np.zeros(256, dtype=np.int64) for _state in kind_states]
            for kind, kind_states in states.items()
        }
        for _start, _end, gate_ih, gate_hh in iter_gate_chunks(
            feedback, u, v, input_size, tau, chunk_size
        ):
            for kind, gate in (("input", gate_ih), ("recurrent", gate_hh)):
                bits = np.ascontiguousarray(gate).view(np.uint32).reshape(-1)
                for state_idx, state in enumerate(states[kind]):
                    prefix_bits = int(state["prefix_bits"])
                    if prefix_bits:
                        mask = (bits >> (32 - prefix_bits)) == int(state["prefix"])
                        selected = bits[mask]
                    else:
                        selected = bits
                    byte = ((selected >> (byte_position * 8)) & 255).astype(np.int64)
                    histograms[kind][state_idx] += np.bincount(byte, minlength=256)

        next_states: dict[str, list[dict[str, object]]] = {"input": [], "recurrent": []}
        for kind, kind_states in states.items():
            grouped: dict[tuple[int, int], list[tuple[str, int]]] = {}
            for state_idx, state in enumerate(kind_states):
                counts = histograms[kind][state_idx]
                cumulative = np.cumsum(counts)
                for target_name, rank_value in state["targets"]:
                    rank = int(rank_value)
                    bucket = int(np.searchsorted(cumulative, rank + 1, side="left"))
                    before = int(cumulative[bucket - 1]) if bucket > 0 else 0
                    prefix = (int(state["prefix"]) << 8) | bucket
                    prefix_bits = int(state["prefix_bits"]) + 8
                    grouped.setdefault((prefix, prefix_bits), []).append(
                        (str(target_name), rank - before)
                    )
            for (prefix, prefix_bits), targets in grouped.items():
                if prefix_bits == 32:
                    for target_name, _rank in targets:
                        resolved[kind][target_name] = prefix
                else:
                    next_states[kind].append(
                        {
                            "prefix": prefix,
                            "prefix_bits": prefix_bits,
                            "targets": targets,
                        }
                    )
        states = next_states
        print(f"median radix pass {4 - byte_position}/4 complete", flush=True)

    medians: dict[str, float] = {}
    for kind, target_bits in resolved.items():
        lower = np.array([target_bits["lower"]], dtype=np.uint32).view(np.float32)[0]
        upper = np.array([target_bits["upper"]], dtype=np.uint32).view(np.float32)[0]
        medians[kind] = float((float(lower) + float(upper)) / 2.0)
    return medians


def compute_distribution_statistics(
    feedback: np.ndarray,
    labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    tau: float,
    chunk_size: int,
    hist_bins: int,
    effective_hist_bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Stream all requested distribution, centering, sign, and effective-weight statistics."""

    hidden_size, input_size = weight_ih.shape
    flat_labels = labels.reshape(-1, labels.shape[-1])
    sectors = flat_labels[:, 1].astype(np.int64)
    edges = np.linspace(0.0, 1.0, hist_bins + 1, dtype=np.float64)
    delta_edges = np.linspace(-1.0, 1.0, hist_bins * 2 + 1, dtype=np.float64)
    spatial_indices = _spatial_sector_indices(input_size)
    all_input_indices = np.arange(input_size, dtype=np.int64)
    irrelevant_indices = [
        np.setdiff1d(all_input_indices, relevant, assume_unique=True)
        for relevant in spatial_indices
    ]

    moments = {"input": MomentAccumulator(), "recurrent": MomentAccumulator()}
    sign_moments = {
        kind: {"positive": MomentAccumulator(), "negative": MomentAccumulator()}
        for kind in ("input", "recurrent")
    }
    relevance_moments = {
        "relevant": MomentAccumulator(),
        "irrelevant": MomentAccumulator(),
    }
    hist_all = {
        "input": np.zeros(hist_bins, dtype=np.int64),
        "recurrent": np.zeros(hist_bins, dtype=np.int64),
    }
    hist_sign = {
        kind: np.zeros((2, hist_bins), dtype=np.int64) for kind in ("input", "recurrent")
    }
    hist_context = {
        kind: np.zeros((9, hist_bins), dtype=np.int64) for kind in ("input", "recurrent")
    }
    hist_relevance = np.zeros((2, hist_bins), dtype=np.int64)
    context_sums_ih = np.zeros((9, hidden_size, input_size), dtype=np.float64)
    context_sums_hh = np.zeros((9, hidden_size, hidden_size), dtype=np.float64)
    context_counts = np.zeros(9, dtype=np.int64)
    frame_norm_ratios = {"input": [], "recurrent": []}
    weight_ih_norm = float(np.linalg.norm(weight_ih))
    weight_hh_norm = float(np.linalg.norm(weight_hh))
    positive_masks = {"input": weight_ih > 0, "recurrent": weight_hh > 0}
    negative_masks = {"input": weight_ih < 0, "recurrent": weight_hh < 0}

    gate_pass_start = time.perf_counter()
    for chunk_idx, (start, end, gate_ih, gate_hh) in enumerate(
        iter_gate_chunks(feedback, u, v, input_size, tau, chunk_size)
    ):
        chunk_sectors = sectors[start:end]
        for kind, gate, weight in (
            ("input", gate_ih, weight_ih),
            ("recurrent", gate_hh, weight_hh),
        ):
            moments[kind].update(gate)
            hist_all[kind] += _hist(gate, edges)
            for sign_idx, sign_name in enumerate(("positive", "negative")):
                sign_mask = positive_masks[kind] if sign_idx == 0 else negative_masks[kind]
                sign_values = gate[:, sign_mask]
                sign_moments[kind][sign_name].update(sign_values)
                hist_sign[kind][sign_idx] += _hist(sign_values, edges)
            weighted = gate * weight[None, ...]
            denominator = weight_ih_norm if kind == "input" else weight_hh_norm
            ratios = np.sqrt(np.square(weighted).sum(axis=(1, 2), dtype=np.float64)) / denominator
            frame_norm_ratios[kind].append(ratios.astype(np.float64))

        for sector in range(9):
            frame_mask = chunk_sectors == sector
            count = int(np.count_nonzero(frame_mask))
            if count == 0:
                continue
            context_counts[sector] += count
            selected_ih = gate_ih[frame_mask]
            selected_hh = gate_hh[frame_mask]
            context_sums_ih[sector] += selected_ih.sum(axis=0, dtype=np.float64)
            context_sums_hh[sector] += selected_hh.sum(axis=0, dtype=np.float64)
            hist_context["input"][sector] += _hist(selected_ih, edges)
            hist_context["recurrent"][sector] += _hist(selected_hh, edges)

            relevant_columns = spatial_indices[sector]
            irrelevant_columns = irrelevant_indices[sector]
            relevant = selected_ih[:, :, relevant_columns]
            irrelevant = selected_ih[:, :, irrelevant_columns]
            relevance_moments["relevant"].update(relevant)
            relevance_moments["irrelevant"].update(irrelevant)
            hist_relevance[0] += _hist(relevant, edges)
            hist_relevance[1] += _hist(irrelevant, edges)

        if (chunk_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - gate_pass_start
            print(f"distribution chunks {chunk_idx + 1} | elapsed={elapsed:.1f}s", flush=True)

    if np.any(context_counts == 0):
        raise RuntimeError(f"At least one sector has no frames: {context_counts.tolist()}")
    context_mean_ih, _grand_context_mean_ih, delta_group_ih = _group_mean_delta(
        context_sums_ih, context_counts
    )
    context_mean_hh, _grand_context_mean_hh, delta_group_hh = _group_mean_delta(
        context_sums_hh, context_counts
    )

    delta_hist = {
        "input": _hist(delta_group_ih, delta_edges),
        "recurrent": _hist(delta_group_hh, delta_edges),
    }
    delta_moments = {"input": MomentAccumulator(), "recurrent": MomentAccumulator()}
    thresholds = (0.01, 0.05, 0.10)
    delta_threshold_counts = {
        kind: {threshold: 0 for threshold in thresholds} for kind in ("input", "recurrent")
    }
    delta_counts = {"input": 0, "recurrent": 0}
    for kind, delta in (("input", delta_group_ih), ("recurrent", delta_group_hh)):
        delta_moments[kind].update(delta.astype(np.float32))
        delta_counts[kind] = int(delta.size)
        for threshold in thresholds:
            delta_threshold_counts[kind][threshold] = int(
                np.count_nonzero(np.abs(delta) > threshold)
            )

    medians = _radix_medians(feedback, u, v, input_size, tau, chunk_size)
    base_summary: dict[str, object] = {}
    for kind in ("input", "recurrent"):
        base_summary[kind] = moments[kind].summary()
        base_summary[kind]["median"] = medians[kind]
        base_summary[kind].update(_hist_mode_and_modality(hist_all[kind], edges))
        base_summary[kind]["centered"] = delta_moments[kind].summary()
        base_summary[kind]["centered"]["fraction_abs_above"] = {
            str(threshold): delta_threshold_counts[kind][threshold] / delta_counts[kind]
            for threshold in thresholds
        }
        base_summary[kind]["sign_split"] = {
            sign: sign_moments[kind][sign].summary()
            for sign in ("positive", "negative")
        }
        ratios = np.concatenate(frame_norm_ratios[kind])
        base_summary[kind]["frame_effective_norm_ratio"] = {
            "mean": float(ratios.mean()),
            "std": float(ratios.std()),
            "min": float(ratios.min()),
            "max": float(ratios.max()),
        }

    relevant_summary = relevance_moments["relevant"].summary()
    irrelevant_summary = relevance_moments["irrelevant"].summary()
    pooled_std = np.sqrt(
        (
            (relevant_summary["count"] - 1) * relevant_summary["std"] ** 2
            + (irrelevant_summary["count"] - 1) * irrelevant_summary["std"] ** 2
        )
        / (relevant_summary["count"] + irrelevant_summary["count"] - 2)
    )
    relevance_summary = {
        "proxy": "CNN 6x6 positions inside the ground-truth 3x3 foreground sector",
        "relevant": relevant_summary,
        "irrelevant": irrelevant_summary,
        "cohens_d_relevant_minus_irrelevant": (
            relevant_summary["mean"] - irrelevant_summary["mean"]
        )
        / pooled_std,
    }

    sparsity: dict[str, list[dict[str, float]]] = {"input": [], "recurrent": []}
    for sector in range(9):
        sparsity["input"].append(_sparsity(context_mean_ih[sector]))
        sparsity["recurrent"].append(_sparsity(context_mean_hh[sector]))

    spectral_radius_w = float(np.max(np.abs(np.linalg.eigvals(weight_hh))))
    spectral_radius_effective = [
        float(np.max(np.abs(np.linalg.eigvals(context_mean_hh[sector] * weight_hh))))
        for sector in range(9)
    ]
    spectral = {
        "weight_hh": spectral_radius_w,
        "effective_by_sector": spectral_radius_effective,
        "effective_mean": float(np.mean(spectral_radius_effective)),
        "effective_std": float(np.std(spectral_radius_effective)),
    }

    max_abs_weight_ih = float(np.max(np.abs(weight_ih)))
    max_abs_weight_hh = float(np.max(np.abs(weight_hh)))
    effective_edges_ih = np.linspace(
        -max_abs_weight_ih, max_abs_weight_ih, effective_hist_bins + 1
    )
    effective_edges_hh = np.linspace(
        -max_abs_weight_hh, max_abs_weight_hh, effective_hist_bins + 1
    )
    weight_hist_ih = _hist(weight_ih, effective_edges_ih)
    weight_hist_hh = _hist(weight_hh, effective_edges_hh)
    effective_hist_ih = np.zeros(effective_hist_bins, dtype=np.int64)
    effective_hist_hh = np.zeros(effective_hist_bins, dtype=np.int64)
    for _start, _end, gate_ih, gate_hh in iter_gate_chunks(
        feedback, u, v, input_size, tau, chunk_size
    ):
        effective_hist_ih += _hist(gate_ih * weight_ih[None, ...], effective_edges_ih)
        effective_hist_hh += _hist(gate_hh * weight_hh[None, ...], effective_edges_hh)

    arrays = {
        "gate_edges": edges.astype(np.float32),
        "delta_edges": delta_edges.astype(np.float32),
        "hist_input_all": hist_all["input"],
        "hist_recurrent_all": hist_all["recurrent"],
        "hist_input_sign": hist_sign["input"],
        "hist_recurrent_sign": hist_sign["recurrent"],
        "hist_input_context": hist_context["input"],
        "hist_recurrent_context": hist_context["recurrent"],
        "hist_input_delta": delta_hist["input"],
        "hist_recurrent_delta": delta_hist["recurrent"],
        "hist_input_relevance": hist_relevance,
        "context_counts": context_counts,
        "context_mean_input": context_mean_ih.astype(np.float32),
        "context_mean_recurrent": context_mean_hh.astype(np.float32),
        "effective_edges_input": effective_edges_ih.astype(np.float32),
        "effective_edges_recurrent": effective_edges_hh.astype(np.float32),
        "hist_weight_input": weight_hist_ih,
        "hist_weight_recurrent": weight_hist_hh,
        "hist_effective_input": effective_hist_ih,
        "hist_effective_recurrent": effective_hist_hh,
    }
    metadata = {
        "distribution": base_summary,
        "context_means": {
            "input": [float(value) for value in context_mean_ih.mean(axis=(1, 2))],
            "recurrent": [float(value) for value in context_mean_hh.mean(axis=(1, 2))],
        },
        "task_relevance": relevance_summary,
        "sparsity": sparsity,
        "spectral_radius": spectral,
        "histogram_definition": {
            "gate_bins": hist_bins,
            "range": [0.0, 1.0],
            "mode": "center of the highest-count fixed-width bin",
            "modality": (
                "Gaussian-smoothed histogram, sigma=2 bins, internal and endpoint peaks "
                ">=2% prominence"
            ),
        },
    }
    return arrays, metadata


def main() -> None:
    """Run the trajectory, streaming gate audit, and save exact compact outputs."""

    args = parse_args()
    if args.batch_size <= 0 or args.gate_chunk_size <= 0:
        raise ValueError("batch_size and gate_chunk_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset_args = argparse.Namespace(
        data_dir=args.data_dir,
        data_suffix=args.data_suffix,
        use_mmap=args.use_mmap,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
    )
    test_ds, num_pos = build_test_dataset(dataset_args)
    model = build_model_from_ckpt(args.ckpt, num_pos, device, chan_num=args.chan_num)
    if not getattr(model, "is_gawf_model", False) or getattr(model, "is_gawf_multi_model", False):
        raise RuntimeError("This analysis currently requires a single-layer GaWF checkpoint")

    weight_ih = model.rnn.weight_ih_l0.detach().cpu().numpy().astype(np.float32)
    weight_hh = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
    u = model.U.detach().cpu().numpy().astype(np.float32)
    v = model.V.detach().cpu().numpy().astype(np.float32)
    trajectory_path = os.path.join(args.save_dir, "gawf_gate_trajectory.npz")
    intervention_path = os.path.join(args.save_dir, "gawf_gate_interventions.json")
    can_reuse = args.reuse_trajectory and os.path.isfile(trajectory_path) and os.path.isfile(
        intervention_path
    )
    if can_reuse:
        with np.load(trajectory_path) as loaded:
            trajectory = {key: loaded[key] for key in loaded.files}
        with open(intervention_path, encoding="utf-8") as file_obj:
            performance = json.load(file_obj)
        print(f"Reused trajectory: {trajectory_path}")
    else:
        trajectory, performance = collect_trajectory(test_ds, model, device, args.batch_size)
        trajectory.update({"U": u, "V": v, "weight_ih": weight_ih, "weight_hh": weight_hh})
        np.savez_compressed(trajectory_path, **trajectory)
        with open(intervention_path, "w", encoding="utf-8") as file_obj:
            json.dump(performance, file_obj, indent=2)

    arrays, metadata = compute_distribution_statistics(
        trajectory["feedback"],
        trajectory["labels"],
        u,
        v,
        weight_ih,
        weight_hh,
        float(model.gate_tau),
        args.gate_chunk_size,
        args.hist_bins,
        args.effective_hist_bins,
    )
    stats_path = os.path.join(args.save_dir, "gawf_gate_distribution_stats.npz")
    np.savez_compressed(stats_path, **arrays)

    n_sequences, frame_num, feedback_dim = trajectory["feedback"].shape
    full_gate_bytes = n_sequences * frame_num * (
        weight_ih.size + weight_hh.size
    ) * np.dtype(np.float32).itemsize
    metadata.update(
        {
            "checkpoint": os.path.abspath(args.ckpt),
            "data_dir": os.path.abspath(args.data_dir or os.path.join(PROJECT_ROOT, "stimuli")),
            "data_suffix": args.data_suffix,
            "n_sequences": n_sequences,
            "frames_per_sequence": frame_num,
            "n_frames": n_sequences * frame_num,
            "feedback_dim": feedback_dim,
            "input_gate_shape_per_frame": list(weight_ih.shape),
            "recurrent_gate_shape_per_frame": list(weight_hh.shape),
            "gate_tau": float(model.gate_tau),
            "layernorm_intervention": performance,
            "storage": {
                "representation": "exact feedback + U/V/W; gates streamed without sampling",
                "uncompressed_full_float32_gate_bytes": full_gate_bytes,
                "trajectory_file": os.path.abspath(trajectory_path),
                "statistics_file": os.path.abspath(stats_path),
            },
        }
    )
    metadata_path = os.path.join(args.save_dir, "gawf_gate_distribution_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)
    print(f"Saved trajectory: {trajectory_path}")
    print(f"Saved statistics: {stats_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
