"""Plot encoder activation patterns for the Clutter Sector or Digit conditions.

``collect`` removes the first (zero-feedback reset) frame of every held-out rollout, then streams
one GaWF checkpoint and saves only equal-n condition means with shape ``(classes, 32, 6, 6)``.
``plot`` averages ten seed files and writes the
condition-specific spatial and feature-channel pattern grids plus a similarity matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from utils.analysis.anal_helpers import build_model_from_ckpt, build_test_dataset, resolve_device
ENCODER_SHAPE = (32, 6, 6)


@dataclass(frozen=True)
class ConditionConfig:
    """Describe label selection and requested Figure 6 panel layouts for one condition."""

    name: str
    label: str
    count: int
    label_column: int
    spatial_grid: tuple[int, int]
    channel_grid: tuple[int, int]


CONDITIONS = {
    "sector": ConditionConfig("sector", "Sector", 9, 1, (3, 3), (2, 5)),
    "digit": ConditionConfig("digit", "Digit", 10, 0, (2, 5), (3, 4)),
}


def parse_args() -> argparse.Namespace:
    """Parse the per-seed collector and ten-seed plotter commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--ckpt", required=True, type=Path)
    collect_parser.add_argument("--data_dir", required=True, type=Path)
    collect_parser.add_argument("--output_dir", required=True, type=Path)
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--batch_size", type=int, default=16)
    collect_parser.add_argument("--num_workers", type=int, default=2)
    collect_parser.add_argument("--data_suffix", default="40h-uint8")
    collect_parser.add_argument("--chan_num", type=int, default=2)
    collect_parser.add_argument("--selection_seed", type=int, default=260718)
    collect_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="sector")

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--data_root", required=True, type=Path)
    plot_parser.add_argument("--figure_dir", required=True, type=Path)
    plot_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="sector")
    spatial_parser = commands.add_parser("plot-spatial")
    spatial_parser.add_argument("--summary", required=True, type=Path)
    spatial_parser.add_argument("--figure_dir", required=True, type=Path)
    spatial_parser.add_argument("--condition", choices=tuple(CONDITIONS), default="sector")
    spatial_parser.add_argument("--stem", default="encoder_sector_spatial_patterns_3x3_10seed")
    return parser.parse_args()


def _condition_config(condition: str) -> ConditionConfig:
    """Return the validated condition-specific Figure 6 configuration."""

    return CONDITIONS[condition]


def _result_name(config: ConditionConfig) -> str:
    """Return the compact per-seed result filename for one condition."""

    return f"encoder_{config.name}_patterns.npz"


def _equal_n_condition_mask(
    labels: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """Select the same minimum observed frame count independently from each condition value."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((labels < 0) | (labels >= count)):
        raise ValueError(f"condition labels must lie in [0, {count - 1}]")
    original_counts = np.bincount(labels, minlength=count).astype(np.int64)
    if np.any(original_counts == 0):
        missing = np.flatnonzero(original_counts == 0).tolist()
        raise RuntimeError(f"No frames found for condition values: {missing}")
    target = int(original_counts.min())
    selected = np.zeros(labels.size, dtype=bool)
    rng = np.random.default_rng(seed)
    for value in range(count):
        indices = np.flatnonzero(labels == value)
        selected[rng.choice(indices, size=target, replace=False)] = True
    return selected, target, original_counts


def collect(args: argparse.Namespace) -> Path:
    """Stream equal-n condition means for one checkpoint without retaining frame activations."""

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be nonnegative.")
    config = _condition_config(args.condition)
    destination = args.output_dir / _result_name(config)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
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
    sums = np.zeros((config.count, int(np.prod(ENCODER_SHAPE))), dtype=np.float64)
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
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames, batch_labels = batch[0], batch[1]
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
            encoded = model.encode_frames(frames)
            values = encoded.detach().cpu().numpy().reshape(-1, int(np.prod(ENCODER_SHAPE)))
            batch_labels = np.asarray(batch_labels, dtype=np.int64).reshape(-1, 2)
            frame_count = batch_labels.shape[0]
            expected = labels[frame_offset : frame_offset + frame_count]
            if not np.array_equal(batch_labels, expected):
                raise RuntimeError(
                    "DataLoader label order differs from condition-pattern indexing."
                )
            batch_selected = selected[frame_offset : frame_offset + frame_count]
            for value in range(config.count):
                use = batch_selected & (batch_labels[:, config.label_column] == value)
                if np.any(use):
                    sums[value] += values[use].sum(axis=0, dtype=np.float64)
                    counts[value] += int(use.sum())
            frame_offset += frame_count
    if frame_offset != total_frames or not np.all(counts == target):
        raise RuntimeError(
            f"Equal-n accumulation mismatch: expected {target}, got {counts.tolist()}."
        )
    patterns = (sums / counts[:, None]).reshape(config.count, *ENCODER_SHAPE).astype(np.float32)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(destination, patterns=patterns, counts=counts)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.ckpt),
                "split": "test",
                "condition": config.name,
                "activation": "CNN encoder output before recurrent input gates",
                "selection": "equal-n random subsample independently within each condition",
                "selection_seed": args.selection_seed,
                "reset_frames_excluded": int(reset_mask.sum()),
                "analysis_frames": int(valid_indices.size),
                "original_frames_by_condition": original_counts.astype(int).tolist(),
                "selected_frames_per_condition": target,
                "pattern_shape": list(patterns.shape),
                "spatial_view": "mean over 32 feature channels",
                "channel_view": "mean over 6-by-6 spatial positions",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _load_patterns(data_root: Path, config: ConditionConfig) -> np.ndarray:
    """Load the exactly ten compact seed pattern files."""

    result_name = _result_name(config)
    paths = sorted(data_root.glob(f"gawf-seed*/{result_name}"))
    if not paths:
        paths = sorted(data_root.glob(f"seed*/{result_name}"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected ten GaWF seed outputs in {data_root}, found {len(paths)}.")
    values = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            pattern = np.asarray(arrays["patterns"], dtype=np.float64)
        if pattern.shape != (config.count, *ENCODER_SHAPE):
            raise RuntimeError(f"Unexpected pattern shape in {path}: {pattern.shape}.")
        values.append(pattern)
    return np.stack(values, axis=0)


def _draw_combined(
    patterns: np.ndarray, figure_dir: Path, config: ConditionConfig
) -> tuple[Path, Path, np.ndarray]:
    """Write one condition figure with spatial and channel grids sharing an activation scale."""

    spatial_maps = patterns.mean(axis=(0, 2))
    channels = patterns.mean(axis=(0, 3, 4))
    channel_maps = channels.reshape(config.count, 4, 8)
    similarity = np.corrcoef(channels)
    if not np.isfinite(similarity).all():
        raise RuntimeError("At least one condition channel pattern has zero variance.")
    activation_vmin = float(min(spatial_maps.min(), channel_maps.min()))
    activation_vmax = float(max(spatial_maps.max(), channel_maps.max()))
    spatial_rows, spatial_columns = config.spatial_grid
    channel_rows, channel_columns = config.channel_grid
    figure_width = 2.5 * (spatial_columns + channel_columns) + 0.8
    figure_height = 2.1 * max(spatial_rows, channel_rows) + 0.4
    fig = plt.figure(figsize=(figure_width, figure_height))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=(spatial_columns, channel_columns, 0.12),
        wspace=0.16,
    )
    spatial_grid = outer[0, 0].subgridspec(spatial_rows, spatial_columns, wspace=0.10, hspace=0.20)
    channel_grid = outer[0, 1].subgridspec(channel_rows, channel_columns, wspace=0.12, hspace=0.28)
    spatial_axes = [
        fig.add_subplot(spatial_grid[row, column])
        for row in range(spatial_rows)
        for column in range(spatial_columns)
    ]
    channel_axes = [
        fig.add_subplot(channel_grid[row, column])
        for row in range(channel_rows)
        for column in range(channel_columns)
    ]
    activation_colorbar_axis = fig.add_subplot(outer[0, 2])
    image = None
    for value, axis in enumerate(spatial_axes[: config.count]):
        image = axis.pcolormesh(
            spatial_maps[value],
            cmap="viridis",
            vmin=activation_vmin,
            vmax=activation_vmax,
            shading="flat",
        )
        axis.set_aspect("equal")
        axis.set_title(f"{config.label} {value}", fontsize=15)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in spatial_axes[config.count :]:
        axis.set_visible(False)

    for value, axis in enumerate(channel_axes[: config.count]):
        image = axis.pcolormesh(
            channel_maps[value],
            cmap="viridis",
            vmin=activation_vmin,
            vmax=activation_vmax,
            shading="flat",
        )
        axis.set_aspect("equal")
        axis.set_title(f"{config.label} {value}", fontsize=14)
        axis.set_xticks([])
        axis.set_yticks([])
    assert image is not None
    fig.colorbar(image, cax=activation_colorbar_axis, label="Mean encoder activation")
    matrix_axis = channel_axes[config.count]
    matrix = matrix_axis.imshow(
        similarity,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        interpolation="nearest",
    )
    matrix_axis.set_title("Similarity", fontsize=12, pad=2)
    matrix_axis.set_xticks(range(config.count), range(config.count), fontsize=9)
    matrix_axis.set_yticks(range(config.count), range(config.count), fontsize=9)
    matrix_axis.set_xlabel(config.label)
    matrix_axis.set_ylabel(config.label)
    fig.colorbar(matrix, ax=matrix_axis, shrink=0.80, pad=0.04, label="Pearson $r$")
    for axis in channel_axes[config.count + 1 :]:
        axis.set_visible(False)
    stem = figure_dir / (
        f"encoder_{config.name}_spatial_channel_patterns_"
        f"{spatial_rows}x{spatial_columns}_{channel_rows}x{channel_columns}_10seed"
    )
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png, pdf, similarity


def _spatial_activation_limits(
    spatial_maps: np.ndarray, config: ConditionConfig
) -> tuple[float, float]:
    """Validate condition spatial maps and return their own sequential colour limits."""

    values = np.asarray(spatial_maps, dtype=np.float64)
    expected = (config.count, *ENCODER_SHAPE[1:])
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Expected finite spatial maps with shape {expected}, got {values.shape}.")
    vmin, vmax = float(values.min()), float(values.max())
    if vmin >= vmax:
        raise ValueError("Spatial activation maps must have a nonzero value range.")
    return vmin, vmax


def plot_spatial(args: argparse.Namespace) -> tuple[Path, Path]:
    """Render a standalone condition spatial grid from the saved ten-seed summary."""

    config = _condition_config(args.condition)
    with np.load(args.summary, allow_pickle=False) as arrays:
        spatial_maps = np.asarray(arrays["spatial_maps"], dtype=np.float32)
    vmin, vmax = _spatial_activation_limits(spatial_maps, config)
    rows, columns = config.spatial_grid
    if rows * columns < config.count:
        raise RuntimeError(
            f"Spatial grid {rows}x{columns} cannot contain {config.count} conditions."
        )
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 13, "axes.titlesize": 16}):
        fig, axes = plt.subplots(rows, columns, figsize=(7.2, 6.8), constrained_layout=True)
        image = None
        for value, axis in enumerate(np.ravel(axes)):
            if value >= config.count:
                axis.set_visible(False)
                continue
            image = axis.pcolormesh(
                spatial_maps[value],
                cmap="Reds",
                vmin=vmin,
                vmax=vmax,
                shading="flat",
                edgecolors="face",
                linewidth=0.01,
                antialiased=False,
                rasterized=False,
                snap=True,
            )
            axis.set_xlim(0, ENCODER_SHAPE[2])
            axis.set_ylim(ENCODER_SHAPE[1], 0)
            axis.set_aspect("equal")
            axis.set_title(f"{config.label} {value}")
            axis.set_xticks([])
            axis.set_yticks([])
        assert image is not None
        fig.suptitle(f"Mean encoder activation (equal-n {config.name}s)")
        fig.colorbar(
            image,
            ax=np.ravel(axes).tolist(),
            shrink=0.82,
            label="Mean encoder activation",
        )
        png, pdf = (args.figure_dir / args.stem).with_suffix(".png"), (
            args.figure_dir / args.stem
        ).with_suffix(".pdf")
        fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return png, pdf


def plot(args: argparse.Namespace) -> tuple[Path, Path]:
    """Aggregate ten seeds and render one shared-scale condition-pattern visualisation."""

    config = _condition_config(args.condition)
    patterns = _load_patterns(args.data_root, config)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    png, pdf, similarity = _draw_combined(patterns, args.figure_dir, config)
    np.savez_compressed(
        args.figure_dir / f"encoder_{config.name}_patterns_10seed_summary.npz",
        spatial_maps=patterns.mean(axis=(0, 2)).astype(np.float32),
        channel_maps=patterns.mean(axis=(0, 3, 4)).astype(np.float32),
        channel_similarity=similarity.astype(np.float32),
    )
    return png, pdf


def main() -> None:
    """Dispatch per-seed collection or ten-seed plotting."""

    args = parse_args()
    if args.command == "collect":
        print(f"Saved {collect(args)}")
    elif args.command == "plot":
        for path in plot(args):
            print(f"Saved {path}")
    else:
        for path in plot_spatial(args):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
