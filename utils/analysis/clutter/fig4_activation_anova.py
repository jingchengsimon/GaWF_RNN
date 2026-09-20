"""Compute and plot six-model Clutter activation ANOVA summaries.

Each ``collect`` run streams one checkpoint over the held-out Clutter split and writes only
the 20 balanced-draw aggregate condition-mean fractions for encoder and hidden activations.
``plot`` combines six models and ten training seeds into the Figure 4 preliminary 1-by-2 panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
from utils.analysis.clutter.fg_switch_offset_acc import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from utils.analysis.clutter.multiseed_plotting import add_seed_points
from utils.analysis.model_train_single_result import parse_hparams_from_filename
from utils.analysis.variance_decomposition import (
    CM_FACTORS,
    StreamingMoments,
    balanced_subsample_indices,
)


OBJECTS = ("input_activation", "hidden_activation")
PLOT_COMPONENTS = (*CM_FACTORS, "residual")
RESULT_NAME = "activation_anova.npz"


def parse_args() -> argparse.Namespace:
    """Parse the ``collect`` and ``plot`` commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Stream one checkpoint into compact ANOVA data.")
    collect.add_argument("--ckpt", required=True, type=Path)
    collect.add_argument("--data_dir", required=True, type=Path)
    collect.add_argument("--output_dir", required=True, type=Path)
    collect.add_argument("--seed", required=True, type=int)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--batch_size", type=int, default=16)
    collect.add_argument("--num_workers", type=int, default=2)
    collect.add_argument("--data_suffix", default="40h-uint8")
    collect.add_argument("--chan_num", type=int, default=2)
    collect.add_argument("--repeats", type=int, default=20)

    plot = commands.add_parser("plot", help="Render the 6-model, 10-seed Figure 4 panel.")
    plot.add_argument("--data_root", required=True, type=Path)
    plot.add_argument("--figure_dir", required=True, type=Path)
    plot.add_argument("--stem", default="activation_anova_1x2_6model_10seed")
    plot.add_argument(
        "--show-seed-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay one neutral-gray point per training seed on each bar.",
    )
    plot.add_argument(
        "--with-residual",
        action="store_true",
        help="Add the trial-level residual bar to an extra Figure 4 rendering.",
    )
    plot.add_argument(
        "--iclr-width",
        action="store_true",
        help="Render at the 5.5-inch ICLR text width with publication-scale typography.",
    )
    return parser.parse_args()


def _model_key(checkpoint: Path) -> str:
    model_type = parse_hparams_from_filename(checkpoint.name).get("model_type", "").lower()
    keys = {key: key for key in MODEL_ORDER}
    if model_type not in keys:
        raise ValueError(f"Unsupported checkpoint model type {model_type!r}: {checkpoint}")
    return keys[model_type]


def _hidden_activations(model: torch.nn.Module, encoded: torch.Tensor) -> torch.Tensor:
    """Return the readout-facing hidden activation for every encoded frame."""

    if not getattr(model, "is_gawf_model", False):
        hidden, _state = model.core(encoded)
        return hidden
    if getattr(model, "is_gawf_multi_model", False):
        raise ValueError("The Figure 4 activation ANOVA supports single-layer GaWF only.")
    batch_size, frame_num = encoded.shape[:2]
    hidden = model.core.initial_state(batch_size, encoded.device, encoded.dtype)
    feedback = torch.zeros(
        batch_size, model.feedback_dim, device=encoded.device, dtype=torch.float32
    )
    outputs = []
    for time_idx in range(frame_num):
        hidden = model.core.step(encoded[:, time_idx], hidden, feedback)
        char_logits, sector_logits = model.classifier(hidden)
        feedback = model._compute_feedback(char_logits, sector_logits).to(dtype=torch.float32)
        outputs.append(hidden)
    return torch.stack(outputs, dim=1)


def _aggregate_draws(
    accumulators: list[StreamingMoments],
) -> dict[str, np.ndarray]:
    """Convert per-draw streaming moments to aggregate condition-mean fractions."""

    values = {factor: np.empty(len(accumulators), dtype=np.float32) for factor in PLOT_COMPONENTS}
    for index, accumulator in enumerate(accumulators):
        result = accumulator.finalize()
        total = float(result.sum_squares["total_cm"].sum())
        if total <= 0:
            raise RuntimeError("Activation condition-mean sum of squares must be positive.")
        for factor in CM_FACTORS:
            values[factor][index] = float(result.sum_squares[factor].sum() / total)
        values["residual"][index] = float(result.aggregate_trial["residual"])
    return values


def collect(args: argparse.Namespace) -> Path:
    """Write compact balanced ANOVA summaries for one trained model seed."""

    if args.batch_size <= 0 or args.num_workers < 0 or args.repeats <= 0:
        raise ValueError(
            "batch_size and repeats must be positive; num_workers must be nonnegative."
        )
    destination = args.output_dir / RESULT_NAME
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing activation ANOVA data: {destination}"
        )
    device = resolve_device(args.device, require_cuda_if_requested=True)
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
    model_key = _model_key(args.ckpt)
    total_frames = len(dataset) * int(dataset.frame_num)
    labels = np.asarray(
        dataset.labels_sector[args.chan_num : args.chan_num + total_frames], dtype=np.int64
    )
    reset_mask = np.arange(total_frames) % int(dataset.frame_num) == 0
    valid_indices = np.flatnonzero(~reset_mask)
    draws, balance = balanced_subsample_indices(
        labels[valid_indices], repeats=args.repeats, seed=args.seed
    )
    membership = np.zeros((args.repeats, total_frames), dtype=bool)
    for repeat, indices in enumerate(draws):
        membership[repeat, valid_indices[indices]] = True
    input_accumulators = [StreamingMoments(model.encoder_flatten_size) for _ in draws]
    hidden_accumulators: list[StreamingMoments] | None = None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    frame_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames, batch_labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            encoded = model.encode_frames(frames)
            hidden = _hidden_activations(model, encoded)
            flat_labels = np.asarray(batch_labels, dtype=np.int64).reshape(-1, 2)
            frame_count = flat_labels.shape[0]
            expected = labels[frame_offset : frame_offset + frame_count]
            if not np.array_equal(flat_labels, expected):
                raise RuntimeError(
                    "DataLoader label order differs from the balanced-draw index order."
                )
            if hidden_accumulators is None:
                hidden_accumulators = [StreamingMoments(hidden.shape[-1]) for _ in draws]
            input_values = encoded.detach().cpu().numpy().reshape(frame_count, -1)
            hidden_values = hidden.detach().cpu().numpy().reshape(frame_count, -1)
            active = membership[:, frame_offset : frame_offset + frame_count]
            for repeat, selected in enumerate(active):
                if np.any(selected):
                    input_accumulators[repeat].update(input_values[selected], flat_labels[selected])
                    hidden_accumulators[repeat].update(
                        hidden_values[selected], flat_labels[selected]
                    )
            frame_offset += frame_count
    if frame_offset != total_frames or hidden_accumulators is None:
        raise RuntimeError(
            f"Activation collection stopped at {frame_offset}/{total_frames} frames."
        )
    collected = {
        "input_activation": _aggregate_draws(input_accumulators),
        "hidden_activation": _aggregate_draws(hidden_accumulators),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        destination,
        **{
            f"{object_name}_{factor}": values
            for object_name, by_factor in collected.items()
            for factor, values in by_factor.items()
        },
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt),
                "model": model_key,
                "seed": args.seed,
                "objects": list(OBJECTS),
                "factors": list(CM_FACTORS),
                "repeats": args.repeats,
                "balance": balance.__dict__,
                "n_frames": total_frames,
                "reset_frames_excluded": int(reset_mask.sum()),
                "analysis_n_frames": int((~reset_mask).sum()),
                "input_units": int(model.encoder_flatten_size),
                "hidden_units": int(hidden_accumulators[0].num_units),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_multiseed(
    data_root: Path, *, include_residual: bool
) -> dict[str, dict[str, np.ndarray]]:
    """Load one repeat-mean estimate per model and training seed."""

    result: dict[str, dict[str, np.ndarray]] = {}
    components = PLOT_COMPONENTS if include_residual else CM_FACTORS
    for model in MODEL_ORDER:
        files = sorted(data_root.glob(f"{model}-seed*/{RESULT_NAME}"))
        if len(files) != 10:
            raise RuntimeError(
                f"Expected 10 {model} seed files under {data_root}, found {len(files)}."
            )
        for object_name in OBJECTS:
            for factor in components:
                key = f"{object_name}_{factor}"
                values = []
                for path in files:
                    with np.load(path, allow_pickle=False) as arrays:
                        values.append(float(np.asarray(arrays[key], dtype=np.float64).mean()))
                result.setdefault(object_name, {}).setdefault(factor, []).append(values)
    return {
        object_name: {
            factor: np.asarray(by_model[factor], dtype=np.float64)
            for factor in components
        }
        for object_name, by_model in result.items()
    }


def plot(args: argparse.Namespace) -> tuple[Path, Path]:
    """Render cross-seed mean ± SEM and seed points for the requested Figure 4 panel."""

    data = _load_multiseed(args.data_root, include_residual=args.with_residual)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    # Match Fig5's within-panel category geometry: each three-factor group has a total bar
    # width of 0.78 and adjacent group centers are one unit apart, leaving 0.22 clear units.
    components = PLOT_COMPONENTS if args.with_residual else CM_FACTORS
    category_centers = np.arange(len(components), dtype=np.float64)
    bar_width = 0.13
    rng = np.random.default_rng(0)
    style = (
        {
            "font.size": 7.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        if args.iclr_width
        else {
            "font.size": 13,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )
    with plt.rc_context(style):
        figsize = (5.5, 2.65) if args.iclr_width else (10.4, 5.2)
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        for axis, object_name, title in zip(
            axes, OBJECTS, ("Input activation", "Hidden activation")
        ):
            for model_index, model in enumerate(MODEL_ORDER):
                offset = (model_index - (len(MODEL_ORDER) - 1) / 2.0) * bar_width
                if args.with_residual:
                    residual = data[object_name]["residual"][model_index]
                    component_values = {
                        factor: (
                            residual
                            if factor == "residual"
                            else data[object_name][factor][model_index] * (1.0 - residual)
                        )
                        for factor in components
                    }
                else:
                    component_values = {
                        factor: data[object_name][factor][model_index] for factor in components
                    }
                means = np.asarray(
                    [component_values[factor].mean() for factor in components]
                )
                errors = np.asarray(
                    [
                        component_values[factor].std(ddof=1)
                        / np.sqrt(component_values[factor].size)
                        for factor in components
                    ]
                )
                positions = category_centers + offset
                axis.bar(
                    positions,
                    100.0 * means,
                    width=bar_width,
                    color=MODEL_COLORS[model],
                    edgecolor="none",
                    yerr=100.0 * errors,
                    capsize=2.3,
                    error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#333333"},
                    zorder=2,
                )
                seed_values = np.column_stack(
                    [100.0 * component_values[factor] for factor in components]
                )
                add_seed_points(
                    axis,
                    positions,
                    seed_values,
                    bar_width=bar_width,
                    show=args.show_seed_points,
                    rng=rng,
                )
            axis.set_title(
                title,
                fontsize=8.2 if args.iclr_width else 15,
                pad=24 if args.iclr_width else 43,
            )
            axis.set_xticks(
                category_centers,
                ("Sector", "Digit", "Interaction", "Residual\n(trial-level)")
                if args.with_residual
                else ("Sector", "Digit", "Interaction"),
            )
            axis.set_xlim(category_centers[0] - 0.55, category_centers[-1] + 0.55)
            axis.set_ylim(0.0, 100.0)
            axis.set_yticks(np.arange(0.0, 101.0, 20.0))
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        axes[0].set_ylabel(
            "Variance component (%)" if args.with_residual else "Explained variance (%)"
        )
        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model], linewidth=5) for model in MODEL_ORDER
        ]
        fig.legend(
            handles,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            loc="upper center",
            ncol=len(MODEL_ORDER),
            frameon=False,
            bbox_to_anchor=(0.5, 0.975),
            handlelength=1.1,
            columnspacing=1.0,
        )
        fig.subplots_adjust(
            left=0.085,
            right=0.995,
            bottom=0.16 if args.iclr_width else 0.14,
            top=0.72,
            wspace=0.20,
        )
        if args.iclr_width:
            for label, axis in zip("AB", axes):
                position = axis.get_position()
                fig.text(
                    position.x0 - 0.012,
                    0.835,
                    label,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        stem = f"{args.stem}_with_residual" if args.with_residual else args.stem
        png = args.figure_dir / f"{stem}.png"
        pdf = args.figure_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return png, pdf


def main() -> None:
    """Dispatch collection or plotting and report exact output paths."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
        return
    png, pdf = plot(args)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
