"""Summarize reset-excluded input-gate current by matching versus other encoder sources.

The collector streams the exact instantaneous input term ``g_ik W_ik x_k``.  It stores only
condition means split by weight sign and source group, normalized by all 256 destinations.
``digit`` uses reset-excluded validation FDR selectivity to choose the top 10% eligible encoder
features for each digit; ``sector`` uses the geometric 128-feature receptive-field block.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.clutter.fig3_gate_distribution import _gate_tensors, _spatial_sector_indices
from utils.analysis.clutter.fig6_encoder_sector_patterns import CONDITIONS, ConditionConfig
from utils.analysis.clutter.fig6_encoder_sector_patterns import _equal_n_condition_mask
from utils.analysis.clutter.fig7_relevance_stats import relevance_masks


RESULT_NAME = "net_input_current.npz"
GROUPS = ("matching", "other")
SIGNS = ("excitatory", "inhibitory")
FULL_SIGNS = (*SIGNS, "total")


def parse_args() -> argparse.Namespace:
    """Parse collection, ten-seed summary, or plotting options."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--ckpt", required=True, type=Path)
    collect_parser.add_argument("--data_dir", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--seed", required=True, type=int)
    collect_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    collect_parser.add_argument("--selectivity", type=Path)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--batch_size", type=int, default=16)
    collect_parser.add_argument("--num_workers", type=int, default=2)
    collect_parser.add_argument("--data_suffix", default="40h-uint8")
    collect_parser.add_argument("--chan_num", type=int, default=2)
    collect_parser.add_argument("--selection_seed", type=int, default=260718)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--data_root", required=True, type=Path)
    summary_parser.add_argument("--output_dir", required=True, type=Path)
    summary_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--summary", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    return parser.parse_args()


def _config(name: str) -> ConditionConfig:
    """Return the common condition layout."""

    return CONDITIONS[name]


def _source_masks(
    condition: str, input_size: int, selectivity: Path | None
) -> tuple[np.ndarray, dict[str, object]]:
    """Return condition-by-input masks and auditable top-10% selection metadata."""

    config = _config(condition)
    if condition == "sector":
        masks = np.zeros((config.count, input_size), dtype=bool)
        for sector, indices in enumerate(_spatial_sector_indices(input_size)):
            masks[sector, indices] = True
        if not np.all(masks.sum(axis=1) == 128):
            raise RuntimeError("Each sector must have exactly 128 matching encoder sources.")
        return masks, {"definition": "sector receptive-field block", "matching_per_context": 128}
    if selectivity is None:
        raise ValueError("--selectivity is required for digit input-current analysis.")
    with np.load(selectivity, allow_pickle=False) as arrays:
        tuning = np.asarray(arrays["primary_encoder_tuning_digit"], dtype=np.float64)
        passed = np.asarray(arrays["primary_encoder_passed_digit"], dtype=bool)
        dominant = np.asarray(arrays["primary_encoder_interaction_dominant"], dtype=bool)
    if tuning.shape != (config.count, input_size) or passed.shape != (input_size,):
        raise ValueError("Encoder selectivity dimensions do not match the checkpoint input size.")
    eligible = passed & ~dominant
    masks = relevance_masks(tuning, eligible, 0.10)
    return masks, {
        "definition": "top 10% of validation-FDR eligible encoder units per digit",
        "eligible_encoder_units": int(eligible.sum()),
        "interaction_dominant_encoder_units": int(dominant.sum()),
        "matching_per_context": masks.sum(axis=1).astype(int).tolist(),
    }


def _selected_frames(
    dataset: object, config: ConditionConfig, selection_seed: int
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Select equal condition counts after excluding all reset frames."""

    total = len(dataset) * int(dataset.frame_num)
    labels = np.asarray(dataset.labels_sector[2 : 2 + total], dtype=np.int64).reshape(total, 2)
    reset = np.arange(total) % int(dataset.frame_num) == 0
    valid = np.flatnonzero(~reset)
    chosen, target, original = _equal_n_condition_mask(
        labels[valid, config.label_column], config.count, selection_seed
    )
    selected = np.zeros(total, dtype=bool)
    selected[valid[chosen]] = True
    return labels, selected, target, original, int(reset.sum())


def collect(args: argparse.Namespace) -> Path:
    """Stream one seed's exact input-current sufficient statistics."""

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be nonnegative.")
    config = _config(args.condition)
    device = resolve_device(args.device, require_cuda_if_requested=True)
    dataset_args = argparse.Namespace(
        data_dir=str(args.data_dir), data_suffix=args.data_suffix, use_mmap=True,
        use_sector_mode=True, predict_all_chars=False, chan_num=args.chan_num,
    )
    dataset, num_pos = build_test_dataset(dataset_args)
    model = build_model_from_ckpt(str(args.ckpt), num_pos, device, chan_num=args.chan_num)
    model.eval()
    input_size = int(model.rnn.weight_ih_l0.shape[1])
    source_masks, source_metadata = _source_masks(args.condition, input_size, args.selectivity)
    labels, selected, target, original, reset_count = _selected_frames(
        dataset, config, args.selection_seed
    )
    weight = model.rnn.weight_ih_l0.detach().to(device=device, dtype=torch.float32)
    # The masks vary by context; materialize the small H-by-I boolean matrices once per context.
    weight_np = weight.detach().cpu().numpy()
    masks = []
    for source in source_masks:
        masks.append([
            [torch.as_tensor((source if group == "matching" else ~source)[None, :] &
                             (weight_np > 0.0 if sign == "excitatory" else weight_np < 0.0),
                             device=device)
             for sign in SIGNS]
            for group in GROUPS
        ])
    sums_current = np.zeros((config.count, len(GROUPS), len(SIGNS)), dtype=np.float64)
    sums_gate = np.zeros((config.count, *weight_np.shape), dtype=np.float64)
    sums_drive = np.zeros_like(sums_gate)
    counts = np.zeros(config.count, dtype=np.int64)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
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
                raise RuntimeError("DataLoader labels differ from selection indexing.")
            batch_selected = torch.from_numpy(selected[frame_offset:frame_offset + frame_count]
                                               .reshape(batch_size, frame_num)).to(device)
            contexts = batch_labels[..., config.label_column].to(device=device, dtype=torch.int64)
            encoded = model.encode_frames(
                frames.to(device=device, dtype=torch.float32, non_blocking=True)
            )
            hidden = torch.zeros(
                batch_size, model.rnn.hidden_size, device=device, dtype=encoded.dtype
            )
            feedback = torch.zeros(
                batch_size, model.feedback_dim, device=device, dtype=torch.float32
            )
            for time_idx in range(frame_num):
                gate_ih, gate_hh = _gate_tensors(
                    feedback, model.U, model.V, input_size, model.gate_tau
                )
                use_base = batch_selected[:, time_idx]
                for context in range(config.count):
                    use = use_base & (contexts[:, time_idx] == context)
                    if not bool(use.any()):
                        continue
                    drive = weight.unsqueeze(0) * encoded[use, time_idx, None, :]
                    current = gate_ih[use] * drive
                    sums_gate[context] += gate_ih[use].sum(dim=0).cpu().numpy()
                    sums_drive[context] += drive.sum(dim=0).cpu().numpy()
                    for group_idx in range(len(GROUPS)):
                        for sign_idx in range(len(SIGNS)):
                            sums_current[context, group_idx, sign_idx] += float(
                                current[:, masks[context][group_idx][sign_idx]].sum().item()
                            )
                    counts[context] += int(use.sum().item())
                input_term = torch.einsum("bi,bhi,hi->bh", encoded[:, time_idx], gate_ih, weight)
                recurrent_term = torch.einsum(
                    "bi,bhi,hi->bh", hidden, gate_hh, model.rnn.weight_hh_l0
                )
                preactivation = input_term + recurrent_term
                if model.rnn.bias_ih_l0 is not None:
                    preactivation += model.rnn.bias_ih_l0
                if model.rnn.bias_hh_l0 is not None:
                    preactivation += model.rnn.bias_hh_l0
                hidden = torch.relu(model.LNormRNN(preactivation))
                digit_logits, sector_logits = model.classifier(hidden)
                feedback = torch.cat([digit_logits, sector_logits], dim=-1).to(torch.float32)
            frame_offset += frame_count
    if frame_offset != labels.shape[0] or not np.all(counts == target):
        raise RuntimeError(f"Equal-n mismatch: expected {target}, got {counts.tolist()}.")
    gate_mean = sums_gate / counts[:, None, None]
    baseline = gate_mean.mean(axis=0)
    sums_baseline = np.zeros_like(sums_current)
    for context in range(config.count):
        for group_idx in range(len(GROUPS)):
            for sign_idx in range(len(SIGNS)):
                mask = masks[context][group_idx][sign_idx].cpu().numpy()
                sums_baseline[context, group_idx, sign_idx] = (
                    baseline * sums_drive[context]
                )[mask].sum()
    denominator = counts[:, None, None] * int(weight.shape[0])
    current = sums_current / denominator
    gate_component = (sums_current - sums_baseline) / denominator
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / RESULT_NAME,
        seed=np.asarray(args.seed), current=current.astype(np.float32),
        delta_current=(current - current.mean(axis=0, keepdims=True)).astype(np.float32),
        gate_component=gate_component.astype(np.float32), counts=counts,
        source_matching=source_masks.astype(np.uint8), gate_mean=gate_mean.astype(np.float32),
        weight=weight_np.astype(np.float32),
    )
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "condition": config.name, "reset_frames_excluded": reset_count,
        "selection": "equal-n condition selection after reset exclusion",
        "selected_frames_per_condition": int(target),
        "original_frames_by_condition": original.astype(int).tolist(),
        "source_group": source_metadata,
        "normalization": "divide by all 256 destination units",
        "current_formula": "mean_t sum(g_ik W_ik x_k)",
        "delta_formula": "I(c) - mean_cprime I(cprime)",
        "gate_component_formula": "mean_t sum((g_ik-gbar_ik) W_ik x_k)",
        "interpretation": "instantaneous decomposition, not a frozen-gate counterfactual",
    }, indent=2) + "\n", encoding="utf-8")
    return args.output_dir / RESULT_NAME


def _with_total(values: np.ndarray) -> np.ndarray:
    """Append the signed E plus I total."""

    return np.concatenate((values, values.sum(axis=-1, keepdims=True)), axis=-1)


def summarize(args: argparse.Namespace) -> Path:
    """Aggregate exactly ten seeds into machine-readable mean and SEM arrays."""

    config = _config(args.condition)
    paths = sorted(args.data_root.glob(f"seed*/{RESULT_NAME}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten seed files in {args.data_root}, found {len(paths)}.")
    raw = {metric: [] for metric in ("current", "delta_current", "gate_component")}
    seeds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            seeds.append(int(np.asarray(arrays["seed"])))
            for metric in raw:
                raw[metric].append(_with_total(np.asarray(arrays[metric], dtype=np.float64)))
    if sorted(seeds) != list(range(1, 11)):
        raise RuntimeError(f"Expected seeds 1-10, got {sorted(seeds)}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "condition": config.name, "groups": list(GROUPS), "signs": list(FULL_SIGNS), "metrics": {}
    }
    payload: dict[str, np.ndarray] = {"seeds": np.asarray(seeds, dtype=np.int64)}
    for metric, entries in raw.items():
        values = np.stack(entries)
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(10)
        payload[f"{metric}_mean"] = mean.astype(np.float32)
        payload[f"{metric}_sem"] = sem.astype(np.float32)
        report["metrics"][metric] = {"mean": mean.tolist(), "sem": sem.tolist()}
    np.savez_compressed(args.output_dir / "net_input_current_10seed_summary.npz", **payload)
    (args.output_dir / "net_input_current_10seed_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "net_input_current_10seed_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(("seed", "condition", "group", "sign", "metric", "value"))
        for seed_idx, seed in enumerate(seeds):
            for metric, entries in raw.items():
                for context in range(config.count):
                    for group_idx, group in enumerate(GROUPS):
                        for sign_idx, sign in enumerate(FULL_SIGNS):
                            value = entries[seed_idx][context, group_idx, sign_idx]
                            writer.writerow((seed, context, group, sign, metric, value))
    return args.output_dir / "net_input_current_10seed_summary.json"


def plot(args: argparse.Namespace) -> Path:
    """Render the requested three-row, matching-versus-other supplementary panel."""

    config = _config(args.condition)
    with np.load(args.summary, allow_pickle=False) as arrays:
        rows = [
            (np.asarray(arrays[f"{metric}_mean"]), np.asarray(arrays[f"{metric}_sem"]), label)
            for metric, label in (
                ("current", r"$I^{input}$"),
                ("delta_current", r"$\Delta I^{input}$"),
                ("gate_component", r"$\Delta I^{input,gate}$"),
            )
        ]
    expected = (config.count, len(GROUPS), len(FULL_SIGNS))
    if any(mean.shape != expected or sem.shape != expected for mean, sem, _ in rows):
        raise ValueError(f"Expected summary shape {expected}.")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#c53030", "#2b6cb0", "#1a202c")
    contexts = np.arange(config.count)
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 8.5), sharex="col")
    for row_idx, (mean, sem, label) in enumerate(rows):
        low, high = (mean-sem).min(), (mean+sem).max()
        margin = max((high-low)*0.12, 1e-8)
        for group_idx, group in enumerate(GROUPS):
            axis = axes[row_idx, group_idx]
            for sign_idx, sign in enumerate(FULL_SIGNS):
                axis.errorbar(
                    contexts, mean[:, group_idx, sign_idx], yerr=sem[:, group_idx, sign_idx],
                    color=colors[sign_idx], marker="o", markersize=3, linewidth=1.2, capsize=2,
                    label=sign if row_idx == 0 and group_idx == 0 else None,
                )
            axis.axhline(0, color="#718096", linewidth=.7)
            axis.set_ylim(low-margin, high+margin)
            axis.set_title(group.capitalize() if row_idx == 0 else "")
            if group_idx == 0:
                axis.set_ylabel(f"{label} per destination")
            if row_idx == 2:
                axis.set_xlabel(config.label)
            axis.set_xticks(contexts)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False
    )
    fig.suptitle(
        f"Net input-current decomposition ({config.label}, 10 seeds, reset-excluded)", y=.995
    )
    fig.tight_layout(rect=(0, 0, 1, .88))
    destination = args.figure_dir / f"Supple4_net_input_current_3x2_10seed_{config.name}.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    return destination


def main() -> None:
    """Dispatch the requested operation."""

    args = parse_args()
    if args.command == "collect":
        result = collect(args)
    elif args.command == "summarize":
        result = summarize(args)
    else:
        result = plot(args)
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
