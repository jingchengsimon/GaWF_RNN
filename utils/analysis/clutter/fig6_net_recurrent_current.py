"""Compute reset-excluded recurrent-current contributions by condition.

For each Digit or Sector context, this collector streams the exact GaWF recurrence and aggregates
``g_ij * W_ij * h_j(t-1)`` within context-specific TT/TR/RT/RR masks.  It retains only
condition means split by positive/negative ``W``; no dense gate, hidden, or current tensor is
saved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.clutter.multiseed_plotting import (
    add_seed_points,
)
from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors
from utils.analysis.clutter.fig6_encoder_sector_patterns import (
    CONDITIONS,
    ConditionConfig,
    _equal_n_condition_mask,
)
from utils.analysis.clutter.fig7_recurrent_gate_cache import group_masks


RESULT_NAME = "net_recurrent_current.npz"
GROUPS = ("TT", "TR", "RT", "RR")
SIGNS = ("excitatory", "inhibitory")
FULL_SIGNS = (*SIGNS, "total")
GROUP_LABELS = {"TT": "T→T", "TR": "T→R", "RT": "R→T", "RR": "R→R"}
GROUP_MARKERS = {"TT": "o", "TR": "s", "RT": "^", "RR": "D"}
GROUP_COLORS = {"TT": "#4477AA", "TR": "#EE6677", "RT": "#228833", "RR": "#CCBB44"}
CURRENT_COLORS = {"excitatory": "#c53030", "inhibitory": "#2b6cb0", "total": "#1a202c"}


def parse_args() -> argparse.Namespace:
    """Parse one-seed collection or ten-seed summary arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--ckpt", required=True, type=Path)
    collect_parser.add_argument("--data_dir", required=True, type=Path)
    collect_parser.add_argument("--compact", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--batch_size", type=int, default=16)
    collect_parser.add_argument("--num_workers", type=int, default=2)
    collect_parser.add_argument("--data_suffix", default="40h-uint8")
    collect_parser.add_argument("--chan_num", type=int, default=2)
    collect_parser.add_argument("--selection_seed", type=int, default=260718)
    collect_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="digit")

    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--data_root", required=True, type=Path)
    summary_parser.add_argument("--output_dir", required=True, type=Path)
    summary_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="digit")
    connection_parser = commands.add_parser("connection")
    connection_parser.add_argument("--data_root", required=True, type=Path)
    connection_parser.add_argument("--compact_root", required=True, type=Path)
    connection_parser.add_argument("--output_dir", required=True, type=Path)
    connection_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="digit")
    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--summary", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="digit")
    fig8_parser = commands.add_parser("fig8")
    fig8_parser.add_argument("--digit_long", required=True, type=Path)
    fig8_parser.add_argument("--sector_long", required=True, type=Path)
    fig8_parser.add_argument("--figure_dir", required=True, type=Path)
    fig8_parser.add_argument("--output_dir", required=True, type=Path)
    fig8_parser.add_argument(
        "--normalization", choices=("destination", "connection"), default="destination"
    )
    planes_parser = commands.add_parser("supple4_planes")
    planes_parser.add_argument("--unit_digit_long", required=True, type=Path)
    planes_parser.add_argument("--unit_sector_long", required=True, type=Path)
    planes_parser.add_argument("--connection_digit_long", required=True, type=Path)
    planes_parser.add_argument("--connection_sector_long", required=True, type=Path)
    planes_parser.add_argument("--figure_dir", required=True, type=Path)
    planes_parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def _condition_config(condition: str) -> ConditionConfig:
    """Return the selected label layout shared with Figure 6 condition analyses."""

    return CONDITIONS[condition]


def _load_compact(path: Path, config: ConditionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load condition masks, their gate baseline, and recurrent weights from one compact cache."""

    with np.load(path, allow_pickle=False) as arrays:
        tuned = np.asarray(arrays[f"{config.name}_tuned"], dtype=bool)
        gate_mean = np.asarray(arrays[f"{config.name}_gate_mean"], dtype=np.float32)
        weight = np.asarray(arrays["weight"], dtype=np.float32)
    if tuned.shape != (config.count, weight.shape[0]) or gate_mean.shape != (
        config.count,
        *weight.shape,
    ):
        raise ValueError(f"Unexpected compact shapes in {path}.")
    if weight.shape[0] != weight.shape[1]:
        raise ValueError("This analysis requires a square recurrent weight matrix.")
    return tuned, gate_mean, weight


def _context_masks(
    tuned: np.ndarray, weight: np.ndarray, device: torch.device
) -> tuple[list[list[list[torch.Tensor]]], np.ndarray]:
    """Build context-specific source/destination sign masks and destination denominators."""

    masks: list[list[list[torch.Tensor]]] = []
    destination_counts = np.zeros((tuned.shape[0], len(GROUPS)), dtype=np.int64)
    for context in range(tuned.shape[0]):
        groups = group_masks(tuned[context], ~tuned[context])
        context_masks: list[list[torch.Tensor]] = []
        for group_idx, group in enumerate(GROUPS):
            source, destination = groups[group]
            base = destination[:, None] & source[None, :] & (weight != 0.0)
            destination_counts[context, group_idx] = int(destination.sum())
            if destination_counts[context, group_idx] == 0:
                raise RuntimeError(f"{group} has no destination units for condition {context}.")
            context_masks.append(
                [
                    torch.as_tensor(base & (weight > 0.0), device=device),
                    torch.as_tensor(base & (weight < 0.0), device=device),
                ]
            )
        masks.append(context_masks)
    return masks, destination_counts


def collect(args: argparse.Namespace) -> Path:
    """Stream one seed's condition-specific recurrent-current sufficient statistics."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be nonnegative.")
    device = resolve_device(args.device, require_cuda_if_requested=True)
    config = _condition_config(args.condition)
    tuned, condition_gate_mean, compact_weight = _load_compact(args.compact, config)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir),
        data_suffix=args.data_suffix,
        use_mmap=True,
        use_sector_mode=True,
        predict_all_chars=False,
        chan_num=args.chan_num,
    )
    dataset, num_pos = build_test_dataset(dataset_args)
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    model.eval()
    weight = model.rnn.weight_hh_l0.detach().to(device=device, dtype=torch.float32)
    if not np.array_equal(weight.detach().cpu().numpy(), compact_weight):
        raise RuntimeError("Checkpoint recurrent weight differs from the registered compact cache.")
    total_frames = len(dataset) * int(dataset.frame_num)
    labels = np.asarray(
        dataset.labels_sector[args.chan_num : args.chan_num + total_frames], dtype=np.int64
    ).reshape(total_frames, 2)
    reset_mask = np.arange(total_frames) % int(dataset.frame_num) == 0
    valid_indices = np.flatnonzero(~reset_mask)
    selected_valid, target, original_counts = _equal_n_condition_mask(
        labels[valid_indices, config.label_column], config.count, args.selection_seed
    )
    selected = np.zeros(total_frames, dtype=bool)
    selected[valid_indices[selected_valid]] = True
    masks, destination_counts = _context_masks(tuned, compact_weight, device)
    gate_baseline = torch.from_numpy(condition_gate_mean.mean(axis=0)).to(device=device)
    sums_current = np.zeros((config.count, len(GROUPS), len(SIGNS)), dtype=np.float64)
    sums_gate = np.zeros_like(sums_current)
    counts = np.zeros(config.count, dtype=np.int64)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    frame_offset = 0
    with torch.no_grad():
        for batch in loader:
            frames, batch_labels = batch[0], batch[1]
            label_array = np.asarray(batch_labels, dtype=np.int64)
            batch_size, frame_num = label_array.shape[:2]
            frame_count = batch_size * frame_num
            expected = labels[frame_offset : frame_offset + frame_count]
            if not np.array_equal(label_array.reshape(-1, 2), expected):
                raise RuntimeError("DataLoader label order differs from current indexing.")
            batch_selected = torch.from_numpy(
                selected[frame_offset : frame_offset + frame_count].reshape(batch_size, frame_num)
            ).to(device=device)
            contexts = batch_labels[..., config.label_column].to(device=device, dtype=torch.int64)
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            encoded = model.encode_frames(frames)
            hidden = torch.zeros(
                batch_size, model.rnn.hidden_size, device=device, dtype=encoded.dtype
            )
            feedback = torch.zeros(
                batch_size, model.feedback_dim, device=device, dtype=torch.float32
            )
            for time_idx in range(frame_num):
                gate_ih, gate_hh = _gate_tensors(
                    feedback,
                    model.U,
                    model.V,
                    encoded.shape[-1],
                    model.gate_tau,
                )
                weighted_hidden = weight.unsqueeze(0) * hidden[:, None, :]
                for context in range(config.count):
                    use = batch_selected[:, time_idx] & (contexts[:, time_idx] == context)
                    if not bool(use.any()):
                        continue
                    current = gate_hh[use] * weighted_hidden[use]
                    gate_component = (gate_hh[use] - gate_baseline) * weighted_hidden[use]
                    for group_idx in range(len(GROUPS)):
                        for sign_idx in range(len(SIGNS)):
                            mask = masks[context][group_idx][sign_idx]
                            sums_current[context, group_idx, sign_idx] += float(
                                current[:, mask].sum().item()
                            )
                            sums_gate[context, group_idx, sign_idx] += float(
                                gate_component[:, mask].sum().item()
                            )
                    counts[context] += int(use.sum().item())
                input_term = torch.einsum(
                    "bi,bhi,hi->bh", encoded[:, time_idx], gate_ih, model.rnn.weight_ih_l0
                )
                hidden_term = torch.einsum("bi,bhi,hi->bh", hidden, gate_hh, weight)
                preactivation = input_term + hidden_term
                if model.rnn.bias_ih_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
                if model.rnn.bias_hh_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
                hidden = torch.relu(model.LNormRNN(torch.tanh(preactivation)))
                digit_logits, sector_logits = model.classifier(hidden)
                feedback = torch.cat([digit_logits, sector_logits], dim=-1).to(torch.float32)
            frame_offset += frame_count
    if frame_offset != total_frames or not np.all(counts == target):
        raise RuntimeError(f"Equal-n count mismatch: expected {target}, got {counts.tolist()}.")
    denominator = counts[:, None, None] * destination_counts[:, :, None]
    current = sums_current / denominator
    gate_component = sums_gate / denominator
    delta_current = current - current.mean(axis=0, keepdims=True)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / RESULT_NAME,
        seed=np.asarray(args.seed, dtype=np.int64),
        current=current.astype(np.float32),
        delta_current=delta_current.astype(np.float32),
        gate_component=gate_component.astype(np.float32),
        counts=counts,
        destination_counts=destination_counts,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt),
                "compact_source": str(args.compact),
                "condition": config.name,
                "reset_frames_excluded": int(reset_mask.sum()),
                "analysis_frames": int(valid_indices.size),
                "selection": "equal-n condition selection after reset exclusion",
                "selection_seed": args.selection_seed,
                "selected_frames_per_condition": int(target),
                "original_frames_by_condition": original_counts.astype(int).tolist(),
                "groups": list(GROUPS),
                "signs": list(SIGNS),
                "normalization": "divide every group/sign sum by its destination-unit count",
                "current_formula": "mean_t sum(g_ij(t) * W_ij * h_j(t-1))",
                "delta_formula": "current(c) - mean_cprime current(cprime)",
                "gate_component_formula": "mean_t sum((g_ij(t) - gbar_ij) * W_ij * h_j(t-1))",
                "gate_baseline": (
                    f"equal-weight mean of the {config.count} saved {config.name} gate means"
                ),
                "interpretation": (
                    "instantaneous decomposition; not a full frozen-gate counterfactual"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return args.output_dir / RESULT_NAME


def _with_total(values: np.ndarray) -> np.ndarray:
    """Append the signed total to the excitatory and inhibitory contributions."""

    return np.concatenate((values, values.sum(axis=-1, keepdims=True)), axis=-1)


def _connection_counts(tuned: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count destination units and nonzero signed recurrent connections by context/group."""

    destination_counts = np.zeros((tuned.shape[0], len(GROUPS)), dtype=np.int64)
    connection_counts = np.zeros((tuned.shape[0], len(GROUPS), len(SIGNS)), dtype=np.int64)
    for context in range(tuned.shape[0]):
        groups = group_masks(tuned[context], ~tuned[context])
        for group_idx, group in enumerate(GROUPS):
            source, destination = groups[group]
            base = destination[:, None] & source[None, :] & (weight != 0.0)
            destination_counts[context, group_idx] = int(destination.sum())
            connection_counts[context, group_idx, 0] = int(np.count_nonzero(base & (weight > 0.0)))
            connection_counts[context, group_idx, 1] = int(np.count_nonzero(base & (weight < 0.0)))
    if np.any(destination_counts == 0) or np.any(connection_counts == 0):
        raise RuntimeError(
            "Every context/group/sign requires at least one nonzero recurrent connection."
        )
    return destination_counts, connection_counts


def _per_connection_values(
    values: np.ndarray, destination_counts: np.ndarray, connection_counts: np.ndarray
) -> np.ndarray:
    """Normalize signed terms within sign and total terms over all nonzero connections."""

    signed = values * destination_counts[:, :, None] / connection_counts
    total = (
        (values * destination_counts[:, :, None]).sum(axis=-1, keepdims=True)
        / connection_counts.sum(axis=-1, keepdims=True)
    )
    return np.concatenate((signed, total), axis=-1)


def connection_normalize(args: argparse.Namespace) -> Path:
    """Convert retained per-destination currents to per-nonzero-connection currents."""

    config = _condition_config(args.condition)
    paths = sorted(args.data_root.glob(f"seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed outputs, found {len(paths)} in {args.data_root}.")
    rows: list[tuple[int, int, str, str, str, float]] = []
    metadata: dict[str, object] = {
        "condition": config.name,
        "source_data_root": str(args.data_root),
        "compact_root": str(args.compact_root),
        "normalization": (
            "divide E/I by their sign-specific nonzero connection counts and total by all counts"
        ),
        "units": "recurrent current per nonzero recurrent connection",
        "connection_counts": {},
    }
    seeds: list[int] = []
    compact_filename = "recurrent_gate_condition_means.npz"
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            seed = int(np.asarray(arrays["seed"]))
            saved_destinations = np.asarray(arrays["destination_counts"], dtype=np.int64)
            metrics = {
                metric: np.asarray(arrays[metric], dtype=np.float64)
                for metric in ("current", "gate_component")
            }
        compact = (
            args.compact_root
            / f"seed{seed:02d}"
            / "compact"
            / compact_filename
        )
        tuned, _, weight = _load_compact(compact, config)
        destination_counts, connection_counts = _connection_counts(tuned, weight)
        if not np.array_equal(saved_destinations, destination_counts):
            raise RuntimeError(f"Destination counts disagree between {path} and {compact}.")
        expected_shape = (config.count, len(GROUPS), len(SIGNS))
        if any(values.shape != expected_shape for values in metrics.values()):
            raise ValueError(f"Unexpected metric shape in {path}; expected {expected_shape}.")
        converted = {
            metric: _per_connection_values(values, destination_counts, connection_counts)
            for metric, values in metrics.items()
        }
        converted["delta_current"] = converted["current"] - converted["current"].mean(
            axis=0, keepdims=True
        )
        metadata["connection_counts"][f"seed{seed:02d}"] = connection_counts.tolist()
        seeds.append(seed)
        for metric, values in converted.items():
            for condition_idx in range(config.count):
                for group_idx, group in enumerate(GROUPS):
                    for sign_idx, sign in enumerate(FULL_SIGNS):
                        rows.append(
                            (
                                seed,
                                condition_idx,
                                group,
                                sign,
                                metric,
                                float(values[condition_idx, group_idx, sign_idx]),
                            )
                        )
    if sorted(seeds) != list(range(1, 11)):
        raise RuntimeError(f"Expected seeds 1–10, got {sorted(seeds)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "net_recurrent_current_connection_10seed_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(("seed", "condition", "group", "sign", "metric", "value"))
        writer.writerows(rows)
    (args.output_dir / "net_recurrent_current_connection_counts.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return long_path


def summarize(args: argparse.Namespace) -> Path:
    """Aggregate ten seed outputs to machine-readable mean ± SEM current summaries."""

    config = _condition_config(args.condition)
    paths = sorted(args.data_root.glob(f"seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed outputs, found {len(paths)} in {args.data_root}.")
    metrics = {"current": [], "delta_current": [], "gate_component": []}
    seeds: list[int] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            seeds.append(int(np.asarray(arrays["seed"])))
            for metric in metrics:
                metrics[metric].append(_with_total(np.asarray(arrays[metric], dtype=np.float64)))
    if sorted(seeds) != list(range(1, 11)):
        raise RuntimeError(f"Expected seeds 1–10, got {sorted(seeds)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"seeds": np.asarray(seeds, dtype=np.int64)}
    report: dict[str, object] = {
        "units": "recurrent current per destination unit",
        "condition": config.name,
        "conditions": list(range(config.count)),
        "groups": list(GROUPS),
        "signs": list(FULL_SIGNS),
        "metrics": {},
    }
    for metric, values in metrics.items():
        stacked = np.stack(values)
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        payload[f"{metric}_mean"] = mean.astype(np.float32)
        payload[f"{metric}_sem"] = sem.astype(np.float32)
        report["metrics"][metric] = {"mean": mean.tolist(), "sem": sem.tolist()}
    np.savez_compressed(args.output_dir / "net_recurrent_current_10seed_summary.npz", **payload)
    (args.output_dir / "net_recurrent_current_10seed_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "net_recurrent_current_10seed_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(("seed", "condition", "group", "sign", "metric", "value"))
        for seed_idx, seed in enumerate(seeds):
            for metric, values in metrics.items():
                for condition in range(config.count):
                    for group_idx, group in enumerate(GROUPS):
                        for sign_idx, sign in enumerate(FULL_SIGNS):
                            value = values[seed_idx][condition, group_idx, sign_idx]
                            writer.writerow((seed, condition, group, sign, metric, value))
    return args.output_dir / "net_recurrent_current_10seed_summary.json"


def plot(args: argparse.Namespace) -> Path:
    """Render the Supplementary 4 condition-wise current decomposition as a PNG."""

    config = _condition_config(args.condition)
    with np.load(args.summary, allow_pickle=False) as arrays:
        current_mean = np.asarray(arrays["current_mean"], dtype=np.float64)
        current_sem = np.asarray(arrays["current_sem"], dtype=np.float64)
        delta_mean = np.asarray(arrays["delta_current_mean"], dtype=np.float64)
        delta_sem = np.asarray(arrays["delta_current_sem"], dtype=np.float64)
        gate_mean = np.asarray(arrays["gate_component_mean"], dtype=np.float64)
        gate_sem = np.asarray(arrays["gate_component_sem"], dtype=np.float64)
    expected = (config.count, len(GROUPS), len(FULL_SIGNS))
    if any(
        array.shape != expected
        for array in (current_mean, current_sem, delta_mean, delta_sem, gate_mean, gate_sem)
    ):
        raise ValueError(f"Expected summary arrays with shape {expected}.")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    contexts = np.arange(config.count)
    rows = (
        (current_mean, current_sem, r"$I$"),
        (delta_mean, delta_sem, r"$\Delta I$"),
        (gate_mean, gate_sem, r"$\Delta I^{gate}$"),
    )
    row_limits = []
    for mean, sem, _ in rows:
        lower = min(float((mean - sem).min()), 0.0)
        upper = max(float((mean + sem).max()), 0.0)
        padding = 0.05 * (upper - lower)
        row_limits.append((lower - padding, upper + padding))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig, axes = plt.subplots(3, len(GROUPS), figsize=(14.2, 9.2), sharex=True)
        for row_idx, (mean, sem, label) in enumerate(rows):
            for group_idx, group in enumerate(GROUPS):
                axis = axes[row_idx, group_idx]
                for sign_idx, sign in enumerate(FULL_SIGNS):
                    values = mean[:, group_idx, sign_idx]
                    error = sem[:, group_idx, sign_idx]
                    axis.plot(
                        contexts, values, color=CURRENT_COLORS[sign], linewidth=2.0, label=sign
                    )
                    axis.fill_between(
                        contexts,
                        values - error,
                        values + error,
                        color=CURRENT_COLORS[sign],
                        alpha=0.18,
                    )
                axis.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
                axis.set_title(group)
                axis.set_ylim(*row_limits[row_idx])
                axis.set_xticks(contexts)
                axis.set_xlabel(config.label)
                if group_idx == 0:
                    axis.set_ylabel(f"{label} per destination unit")
                axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(
            f"Net recurrent-current decomposition ({config.label}, 10 seeds, reset-excluded)",
            y=0.995,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.973),
            ncol=3,
            frameon=False,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        stem = "Supple4_net_recurrent_current_3x4_10seed"
        if config.name != "digit":
            stem = f"{stem}_{config.name}"
        destination = args.figure_dir / f"{stem}.png"
        fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return destination


def _load_fig8_long(path: Path, condition_name: str) -> dict[str, np.ndarray]:
    """Load one condition family's current and gate term into dense seed-level arrays."""

    config = _condition_config(condition_name)
    required = {"seed", "condition", "group", "sign", "metric", "value"}
    with path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(f"Unexpected long CSV columns in {path}.")
        rows = list(reader)
    values = {
        metric: np.full((10, config.count, len(GROUPS), len(FULL_SIGNS)), np.nan)
        for metric in ("current", "gate_component")
    }
    for row in rows:
        metric = row["metric"]
        if metric not in values:
            continue
        seed_idx = int(row["seed"]) - 1
        condition_idx = int(row["condition"])
        group_idx = GROUPS.index(row["group"])
        sign_idx = FULL_SIGNS.index(row["sign"])
        if not 0 <= seed_idx < 10 or not 0 <= condition_idx < config.count:
            raise ValueError(f"Out-of-range seed or condition in {path}: {row}.")
        values[metric][seed_idx, condition_idx, group_idx, sign_idx] = float(row["value"])
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"Missing current or gate-component values in {path}.")
    values["frozen"] = values["current"] - values["gate_component"]
    return values


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the training-seed mean and SEM on the first array axis."""

    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def _one_sample_p(values: np.ndarray) -> np.ndarray:
    """Return uncorrected two-sided seed-level one-sample t-test p-values against zero."""

    from scipy.stats import ttest_1samp

    return np.asarray(ttest_1samp(values, 0.0, axis=0).pvalue, dtype=np.float64)


def _fig8_limits(
    reports: dict[str, dict[str, np.ndarray]]
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute shared Panel B x/y and Panel C y ranges across Digit and Sector."""

    gate_values: list[np.ndarray] = []
    for report in reports.values():
        gate_values.append(report["gate_component"][:, :, :, 2])
    gates = np.concatenate([value.ravel() for value in gate_values])
    gate_lower = min(float(gates.min()), 0.0)
    gate_upper = max(float(gates.max()), 0.0)
    gate_padding = 0.08 * (gate_upper - gate_lower)
    plane_values: list[np.ndarray] = []
    for report in reports.values():
        for metric in ("frozen", "current"):
            plane_values.extend((report[metric][..., 0], -report[metric][..., 1]))
    plane_upper = max(float(np.concatenate([value.ravel() for value in plane_values]).max()), 0.0)
    plane_limits = (0.0, 1.05 * plane_upper if plane_upper > 0.0 else 1.0)
    return plane_limits, plane_limits, (gate_lower - gate_padding, gate_upper + gate_padding)


def _plane_legend_handles() -> list[Line2D]:
    """Return the shared colored-group and state handles for E/I plane panels."""

    group_handles = [
        Line2D(
            (),
            (),
            marker="o",
            color=GROUP_COLORS[group],
            markerfacecolor=GROUP_COLORS[group],
            markersize=6.5,
            linestyle="None",
            label=GROUP_LABELS[group],
        )
        for group in GROUPS
    ]
    state_handles = [
        Line2D(
            (), (), marker="o", color="#1a202c", markerfacecolor="white", markersize=6.5,
            linestyle="None", label="frozen",
        ),
        Line2D(
            (), (), marker="o", color="#1a202c", markerfacecolor="#1a202c", markersize=6.5,
            linestyle="None", label="observed",
        ),
    ]
    return [*group_handles, *state_handles]


def _plot_fig8_plane(
    axis: plt.Axes,
    report: dict[str, np.ndarray],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    unit_label: str,
    group_colors: bool = False,
    ticks: tuple[float, ...] | None = None,
    show_legend: bool = True,
) -> list[dict[str, object]]:
    """Plot one condition-specific frozen-to-observed arrow for every training seed."""

    records: list[dict[str, object]] = []
    for group_idx, group in enumerate(GROUPS):
        for condition_idx in range(report["current"].shape[1]):
            for seed_idx in range(report["current"].shape[0]):
                frozen = report["frozen"][seed_idx, condition_idx, group_idx]
                observed = report["current"][seed_idx, condition_idx, group_idx]
                start = (-frozen[1], frozen[0])
                end = (-observed[1], observed[0])
                color = GROUP_COLORS[group] if group_colors else "#1a202c"
                marker = "o" if group_colors else GROUP_MARKERS[group]
                axis.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": 0.8,
                        "alpha": 0.25 if group_colors else 0.18,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                    zorder=2,
                )
                axis.plot(
                    *start,
                    marker=marker,
                    markersize=4.2,
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    alpha=0.32,
                    zorder=3,
                )
                axis.plot(
                    *end,
                    marker=marker,
                    markersize=4.2,
                    color=color,
                    markerfacecolor=color,
                    markeredgewidth=0.8,
                    alpha=0.75 if group_colors else 0.30,
                    zorder=3,
                )
                records.append(
                    {
                        "group": group,
                        "condition": condition_idx,
                        "seed": seed_idx + 1,
                        "frozen": list(start),
                        "observed": list(end),
                    }
                )
    diagonal_max = min(x_limits[1], y_limits[1])
    axis.plot((0.0, diagonal_max), (0.0, diagonal_max), color="0.55", linestyle="--", linewidth=0.9)
    axis.set(xlim=x_limits, ylim=y_limits)
    axis.set_aspect("equal", adjustable="box")
    if ticks is not None:
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel(f"Inhibitory-current magnitude per {unit_label}", fontsize=9)
    axis.set_ylabel(f"Excitatory current per {unit_label}", fontsize=11.5)
    axis.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles = _plane_legend_handles() if group_colors else []
        if not group_colors:
            handles = [
                Line2D(
                    (), (), marker=GROUP_MARKERS[group], color="#1a202c",
                    markerfacecolor="#1a202c", markersize=6.5, linestyle="None",
                    label=GROUP_LABELS[group],
                )
                for group in GROUPS
            ]
            handles.extend(_plane_legend_handles()[-2:])
        axis.legend(
            handles=handles,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=False,
            fontsize=8.5,
            columnspacing=0.8,
            handletextpad=0.35,
        )
    return records


def _plot_fig8_bars(
    axis: plt.Axes,
    report: dict[str, np.ndarray],
    condition_name: str,
    unit_label: str,
    shared_y_limits: tuple[float, float] | None = None,
    ticks: tuple[float, ...] | None = None,
    tick_format: str = "%.2f",
    show_legend: bool = True,
) -> list[dict[str, object]]:
    """Plot gate-change bars, SEM, and points from the same seed-level values."""

    values = report["gate_component"]
    seed_averages = values.mean(axis=1)
    mean, sem = _mean_sem(seed_averages)
    p_values = _one_sample_p(seed_averages)
    x = np.arange(len(GROUPS), dtype=np.float64)
    width = 0.78 / len(FULL_SIGNS)
    points: list[dict[str, object]] = []
    labels = {
        "excitatory": "W > 0",
        "inhibitory": "W < 0",
        "total": "All",
    }
    for sign_idx, sign in enumerate(FULL_SIGNS):
        centers = x + (sign_idx - 1) * width
        axis.bar(
            centers,
            mean[:, sign_idx],
            width=width,
            color=CURRENT_COLORS[sign],
            edgecolor="none",
            label=labels[sign],
            yerr=sem[:, sign_idx],
            capsize=3,
            error_kw={"linewidth": 1.0, "capthick": 1.0},
            zorder=2,
        )
        add_seed_points(
            axis,
            centers,
            seed_averages[:, :, sign_idx],
            bar_width=width,
        )
        y_span = (
            shared_y_limits[1] - shared_y_limits[0]
            if shared_y_limits is not None
            else axis.get_ylim()[1] - axis.get_ylim()[0]
        )
        for group_idx, group in enumerate(GROUPS):
            seed_values = seed_averages[:, group_idx, sign_idx]
            points.append(
                {
                    "group": group, "sign": sign, "mean": float(mean[group_idx, sign_idx]),
                    "sem": float(sem[group_idx, sign_idx]),
                    "p_value": float(p_values[group_idx, sign_idx]),
                    "seed_values": seed_values.tolist(),
                }
            )
            if p_values[group_idx, sign_idx] < 0.05:
                edge = (
                    mean[group_idx, sign_idx] + sem[group_idx, sign_idx]
                    if mean[group_idx, sign_idx] >= 0.0
                    else mean[group_idx, sign_idx] - sem[group_idx, sign_idx]
                )
                direction = 1.0 if edge >= 0.0 else -1.0
                axis.text(
                    centers[group_idx],
                    edge + direction * 0.025 * y_span,
                    "*",
                    ha="center",
                    va="bottom" if direction > 0.0 else "top",
                    fontsize=12,
                    fontweight="bold",
                    zorder=5,
                )
    axis.axhline(0.0, color="0.3", linewidth=0.9, zorder=1)
    if shared_y_limits is None:
        y_limits, y_ticks = {
            "digit": ((-0.5, 2.5), (-0.5, 0.0, 1.0, 2.0)),
            "sector": ((-0.5, 1.0), (-0.5, 0.0, 0.5, 1.0)),
        }[condition_name]
        axis.set(ylim=y_limits)
        axis.set_yticks(y_ticks)
    else:
        axis.set(ylim=shared_y_limits)
    if ticks is not None:
        axis.set_yticks(ticks)
        axis.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
    axis.set(xticks=x, xticklabels=[GROUP_LABELS[group] for group in GROUPS])
    axis.set_ylabel(rf"$\Delta I^{{gate}}$ per {unit_label}")
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    if show_legend:
        axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    return points


def _write_fig8_long(output: Path, reports: dict[str, dict[str, np.ndarray]]) -> Path:
    """Write the Figure 8 verification table with observed, frozen, and gate currents."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            (
                "condition_family",
                "seed",
                "condition",
                "group",
                "sign",
                "I",
                "I_frozen",
                "delta_I_gate",
            )
        )
        for family, report in reports.items():
            for seed_idx in range(10):
                for condition_idx in range(report["current"].shape[1]):
                    for group_idx, group in enumerate(GROUPS):
                        for sign_idx, sign in enumerate(FULL_SIGNS):
                            writer.writerow(
                                (
                                    family, seed_idx + 1, condition_idx, group, sign,
                                    report["current"][seed_idx, condition_idx, group_idx, sign_idx],
                                    report["frozen"][seed_idx, condition_idx, group_idx, sign_idx],
                                    report["gate_component"][
                                        seed_idx, condition_idx, group_idx, sign_idx
                                    ],
                                )
                            )
    return output


def plot_fig8(args: argparse.Namespace) -> tuple[Path, Path]:
    """Render one Figure 8 or Supplementary 4 two-panel current-bar PDF."""

    reports = {
        "digit": _load_fig8_long(args.digit_long, "digit"),
        "sector": _load_fig8_long(args.sector_long, "sector"),
    }
    connection_normalized = args.normalization == "connection"
    unit_description = (
        "nonzero recurrent connection" if connection_normalized else "destination unit"
    )
    unit_label = "connection" if connection_normalized else "destination unit"
    stem = (
        "Fig8_recurrent_current_connection"
        if connection_normalized
        else "Supple4_recurrent_current_unit"
    )
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    long_path = _write_fig8_long(args.output_dir / f"{stem}_long.csv", reports)
    caption_report: dict[str, object] = {
        "units": f"current per {unit_description}",
        "normalization": args.normalization,
        "reset_frames_excluded": True,
        "seed_count": 10,
        "point_unit": "training seed",
        "within_seed_aggregation": "equal mean across conditions",
        "layout": "Digit/Sector grouped bars",
        "families": {},
    }
    if connection_normalized:
        bar_limits = (-0.02, 0.08)
        bar_ticks: tuple[float, ...] | None = (-0.02, 0.0, 0.04, 0.08)
    else:
        reference_limits = (-0.5, 2.3)
        seed_bar_values = np.concatenate(
            [report["gate_component"].mean(axis=1).reshape(-1) for report in reports.values()]
        )
        data_limits = (float(seed_bar_values.min()), float(seed_bar_values.max()))
        within_reference = (
            data_limits[0] >= reference_limits[0]
            and data_limits[1] <= reference_limits[1]
        )
        if within_reference:
            bar_limits = reference_limits
            bar_ticks = (-0.5, 0.0, 1.0, 2.0)
        else:
            lower = min(reference_limits[0], data_limits[0])
            upper = max(reference_limits[1], data_limits[1])
            padding = 0.08 * (upper - lower)
            bar_limits = (lower - padding, upper + padding)
            bar_ticks = None
    caption_report["shared_bar_y_limits"] = bar_limits
    standalone_rc = {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    reference_rc = {
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(standalone_rc if connection_normalized else reference_rc):
        figure_size = (10.5, 5.4) if connection_normalized else (5.5, 2.35)
        fig, axes = plt.subplots(1, 2, figsize=figure_size, sharey=True)
        for axis, (family, report) in zip(axes, reports.items()):
            seed_gate = report["gate_component"].mean(axis=1)[:, :, 2]
            gate_mean, gate_sem = _mean_sem(seed_gate)
            seed_observed = report["current"].mean(axis=1)[:, :, 2]
            observed_mean, observed_sem = _mean_sem(seed_observed)
            bar_records = _plot_fig8_bars(
                axis,
                report,
                family,
                unit_label,
                bar_limits,
                bar_ticks,
                "%.2f" if connection_normalized else "%.1f",
                show_legend=False,
            )
            axis.set_title(family.capitalize(), pad=1 if not connection_normalized else None)
            axis.set_xlabel("Group")
            if not connection_normalized:
                for text in list(axis.texts):
                    if text.get_text() == "*":
                        text.remove()
                axis.tick_params(labelsize=6.7, length=2.5, width=0.7)
                axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            total_by_condition = report["gate_component"][:, :, :, 2].mean(axis=0)
            caption_report["families"][family] = {
                "gate_total_mean_sem": {
                    group: {"mean": float(gate_mean[idx]), "sem": float(gate_sem[idx])}
                    for idx, group in enumerate(GROUPS)
                },
                "observed_total_mean_sem": {
                    group: {"mean": float(observed_mean[idx]), "sem": float(observed_sem[idx])}
                    for idx, group in enumerate(GROUPS)
                },
                "gate_total_condition_ranges": {
                    group: {
                        "min": float(total_by_condition[:, idx].min()),
                        "max": float(total_by_condition[:, idx].max()),
                        "all_same_sign": bool(
                            np.all(total_by_condition[:, idx] > 0.0)
                            or np.all(total_by_condition[:, idx] < 0.0)
                        ),
                    }
                    for idx, group in enumerate(GROUPS)
                },
                "bar_summary": bar_records,
            }
        axes[1].set_ylabel("")
        if connection_normalized:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
            fig.subplots_adjust(
                left=0.08, right=0.99, bottom=0.14, top=0.84, wspace=0.08
            )
        else:
            fig.legend(
                handles=(
                    Patch(
                        facecolor=CURRENT_COLORS["excitatory"],
                        edgecolor="none",
                        label=r"$w^{\mathrm{rec}}_{+}$",
                    ),
                    Patch(
                        facecolor=CURRENT_COLORS["inhibitory"],
                        edgecolor="none",
                        label=r"$w^{\mathrm{rec}}_{-}$",
                    ),
                    Patch(
                        facecolor=CURRENT_COLORS["total"],
                        edgecolor="none",
                        label="All",
                    ),
                ),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=3,
                frameon=False,
                handlelength=1.0,
                columnspacing=1.2,
            )
            fig.subplots_adjust(
                left=0.105, right=0.995, bottom=0.20, top=0.82, wspace=0.12
            )
            for label, axis in zip("AB", axes):
                position = axis.get_position()
                fig.text(
                    position.x0 - 0.025,
                    position.y1 + 0.02,
                    label,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        destination = args.figure_dir / f"{stem}.pdf"
        if connection_normalized:
            fig.savefig(destination, bbox_inches="tight", pad_inches=0.04)
        else:
            fig.savefig(destination)
        plt.close(fig)
    (args.output_dir / f"{stem}_caption_stats.json").write_text(
        json.dumps(caption_report, indent=2) + "\n", encoding="utf-8"
    )
    return destination, long_path


def plot_supple4_planes(args: argparse.Namespace) -> Path:
    """Render a title-free 2-by-2 supplementary E/I-plane record for both normalizations."""

    reports = {
        "unit": {
            "digit": _load_fig8_long(args.unit_digit_long, "digit"),
            "sector": _load_fig8_long(args.unit_sector_long, "sector"),
        },
        "connection": {
            "digit": _load_fig8_long(args.connection_digit_long, "digit"),
            "sector": _load_fig8_long(args.connection_sector_long, "sector"),
        },
    }
    unit_limits, _, _ = _fig8_limits(reports["unit"])
    connection_limits = {"digit": (0.0, 0.16), "sector": (0.0, 0.12)}
    connection_ticks = {
        "digit": (0.0, 0.05, 0.10, 0.15),
        "sector": (0.0, 0.05, 0.10),
    }
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    arrow_records: dict[str, dict[str, list[dict[str, object]]]] = {"unit": {}, "connection": {}}
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.4))
        for row, normalization in enumerate(("unit", "connection")):
            for column, family in enumerate(("digit", "sector")):
                axis = axes[row, column]
                if normalization == "unit":
                    limits = unit_limits
                    ticks = None
                    label = "destination unit"
                else:
                    limits = connection_limits[family]
                    ticks = connection_ticks[family]
                    label = "connection"
                arrow_records[normalization][family] = _plot_fig8_plane(
                    axis,
                    reports[normalization][family],
                    limits,
                    limits,
                    label,
                    group_colors=True,
                    ticks=ticks,
                    show_legend=False,
                )
                if row == 0:
                    axis.set_title(family.capitalize())
        fig.legend(
            _plane_legend_handles(),
            [handle.get_label() for handle in _plane_legend_handles()],
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.91, hspace=0.28, wspace=0.22)
        destination = args.figure_dir / "Supple4_recurrent_current_ei_planes.pdf"
        fig.savefig(destination, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    (args.output_dir / "Supple4_recurrent_current_ei_planes_caption_stats.json").write_text(
        json.dumps({"ei_arrows": arrow_records, "reset_frames_excluded": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Dispatch one-seed collection or ten-seed aggregation."""

    args = parse_args()
    if args.command == "collect":
        result = collect(args)
    elif args.command == "summarize":
        result = summarize(args)
    elif args.command == "connection":
        result = connection_normalize(args)
    elif args.command == "plot":
        result = plot(args)
    elif args.command == "fig8":
        figure, long_csv = plot_fig8(args)
        result = f"{figure} (long CSV: {long_csv})"
    elif args.command == "supple4_planes":
        result = plot_supple4_planes(args)
    else:
        raise ValueError(f"Unhandled command: {args.command}")
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
