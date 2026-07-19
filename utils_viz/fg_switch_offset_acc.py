"""Plot multi-model fg/bg-switch recovery curves from offset-accuracy exports.

Reads all matching ``fg_switch_offset_acc_*.npz`` or ``bg_switch_offset_acc_*.npz``
files in ``--data_dir``. Each switch kind is rendered as one two-panel figure with every
model overlaid: foreground digit accuracy on the left and sector accuracy on the right.

Outputs (in ``--save_dir/fg`` or ``--save_dir/bg``):
- ``[<condition_tag>_]fg_switch_offset_acc_models.png`` — models aligned to fg switches.
- ``[<condition_tag>_]bg_switch_offset_acc_models.png`` — models aligned to bg switches.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np  # noqa: E402


MODEL_ORDER = ("gawf", "rnn", "lstm", "gru", "mamba", "s5")
MODEL_LABELS = {
    "gawf": "GaWF",
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "mamba": "Mamba",
    "s5": "S5",
}
MODEL_COLORS = {
    "gawf": "#4C78A8",
    "rnn": "#F58518",
    "lstm": "#54A24B",
    "gru": "#E45756",
    "mamba": "#B279A2",
    "s5": "#72B7B2",
}
MODEL_MARKERS = {
    "gawf": "o",
    "rnn": "s",
    "lstm": "^",
    "gru": "D",
    "mamba": "P",
    "s5": "X",
}


def parse_args() -> argparse.Namespace:
    """Parse input, output, title, kind, and legacy-cleanup options."""

    parser = argparse.ArgumentParser(
        description="Plot all switch-offset model curves together in a two-panel figure."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/fg_switch_offset_acc",
        help="Directory containing fg/bg_switch_offset_acc_*.npz files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/fg_switch_offset_acc",
        help="Output directory for combined model figures.",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "fg", "bg"),
        default="auto",
        help="Plot both kinds found in the directory, or restrict to fg/bg.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Default identifies the switch kind.",
    )
    parser.add_argument(
        "--condition_tag",
        type=str,
        default="",
        help="Optional filename prefix used to distinguish conditions in one figure folder.",
    )
    parser.add_argument(
        "--clean_legacy_individuals",
        action="store_true",
        help="After saving, delete legacy per-model bar-chart PNGs for the plotted kind.",
    )
    return parser.parse_args()


def _kind_and_tag(npz_path: str) -> Tuple[str, str]:
    """Return switch kind and checkpoint tag from an exported NPZ path."""

    base = os.path.basename(npz_path)
    for kind in ("fg", "bg"):
        prefix = f"{kind}_switch_offset_acc_"
        if base.startswith(prefix) and base.endswith(".npz"):
            return kind, base[len(prefix) : -len(".npz")]
    raise ValueError(f"Unrecognized switch-offset filename: {base}")


def _model_key(ckpt_tag: str) -> str:
    """Extract the stable model key encoded at the start of a checkpoint tag."""

    lowered = ckpt_tag.lower()
    for key in MODEL_ORDER:
        if lowered.startswith(f"{key}_"):
            return key
    return lowered.split("_", 1)[0]


def _model_sort_key(npz_path: str) -> Tuple[int, str]:
    """Sort known models in the project-standard order, then unknown models by name."""

    _, tag = _kind_and_tag(npz_path)
    key = _model_key(tag)
    try:
        return MODEL_ORDER.index(key), key
    except ValueError:
        return len(MODEL_ORDER), key


def _collect_by_kind(data_dir: str, requested_kind: str) -> Dict[str, List[str]]:
    """Collect and model-sort exported NPZ paths for each requested switch kind."""

    kinds = ("fg", "bg") if requested_kind == "auto" else (requested_kind,)
    collected: Dict[str, List[str]] = {}
    for kind in kinds:
        paths = glob.glob(os.path.join(data_dir, f"{kind}_switch_offset_acc_*.npz"))
        if paths:
            collected[kind] = sorted(paths, key=_model_sort_key)
    if not collected:
        raise RuntimeError(f"No requested switch-offset NPZ files found in {data_dir}")
    return collected


def _style_axis(ax: Axes) -> None:
    """Apply the shared compact, borderless recovery-curve style."""

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _load_curves(
    npz_paths: List[str],
) -> Tuple[np.ndarray, List[str], List[Tuple[str, np.ndarray, np.ndarray]]]:
    """Load aligned offsets and char/sector curves for every checkpoint."""

    reference_offsets: np.ndarray | None = None
    reference_labels: List[str] | None = None
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    seen_models: set[str] = set()

    for npz_path in npz_paths:
        _, ckpt_tag = _kind_and_tag(npz_path)
        model_key = _model_key(ckpt_tag)
        if model_key in seen_models:
            raise RuntimeError(
                f"Multiple NPZ files map to model {model_key!r} in one data directory"
            )
        seen_models.add(model_key)
        with np.load(npz_path) as data:
            if "sector_acc" not in data.files:
                raise RuntimeError(f"Missing sector_acc required for two-panel plot: {npz_path}")
            offsets = data["offset_order"].astype(np.int64)
            labels = [str(value) for value in data["offset_labels"].tolist()]
            char_acc = data["char_acc"].astype(np.float32)
            sector_acc = data["sector_acc"].astype(np.float32)
        if reference_offsets is None:
            reference_offsets = offsets
            reference_labels = labels
        elif not np.array_equal(offsets, reference_offsets) or labels != reference_labels:
            raise RuntimeError(f"Offset layout differs across model files: {npz_path}")
        curves.append((model_key, char_acc, sector_acc))

    if reference_offsets is None or reference_labels is None:
        raise RuntimeError("No model curves were loaded")
    return reference_offsets, reference_labels, curves


def _plot_kind(
    kind: str,
    npz_paths: List[str],
    save_dir: str,
    title: str,
    condition_tag: str,
    clean_legacy_individuals: bool,
) -> str:
    """Render full recovery curves with markers only at key switch offsets."""

    offsets, labels, curves = _load_curves(npz_paths)
    wanted = {-10: "pre10", -5: "pre5", 1: "switch", 5: "post5", 10: "post10"}
    selected = [(index, wanted[int(offset)]) for index, offset in enumerate(offsets) if int(offset) in wanted]
    if not selected:
        raise RuntimeError(f"No key offsets found in {npz_paths[0]}")
    selected_indices = np.asarray([index for index, _ in selected], dtype=np.int64)
    selected_labels = [label for _, label in selected]
    x = np.arange(offsets.size, dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True)

    for ax, value_idx, panel_title, chance_level, chance_label in (
        (axes[0], 1, "Character readout (FG digit)", 10.0, "chance = 10%"),
        (axes[1], 2, "Sector readout", 100.0 / 9.0, "chance = 11.1%"),
    ):
        for model_key, char_acc, sector_acc in curves:
            values = char_acc if value_idx == 1 else sector_acc
            ax.plot(
                x,
                values,
                marker=MODEL_MARKERS.get(model_key, "o"),
                markevery=selected_indices.tolist(),
                linewidth=1.9,
                markersize=4.5,
                label=MODEL_LABELS.get(model_key, model_key.upper()),
                color=MODEL_COLORS.get(model_key),
            )
        ax.set_title(panel_title)
        ax.set_xlabel("Frame relative to switch")
        ax.set_ylim(0.0, 100.0)
        ax.set_xticks(selected_indices, selected_labels)
        ax.axhline(chance_level, color="0.3", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.text(0.99, chance_level + 1.2, chance_label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", color="0.3", fontsize=8)
        if "switch" in selected_labels:
            switch_index = selected_indices[selected_labels.index("switch")]
            ax.axvline(switch_index, color="0.35", linewidth=1.0, linestyle="--")
        _style_axis(ax)

    axes[0].set_ylabel("Accuracy (%)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=min(len(legend_labels), 6), fontsize=9)
    figure_title = title or f"{kind.upper()}-switch recovery across models"
    fig.suptitle(figure_title, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86])
    kind_dir = os.path.join(save_dir, kind)
    os.makedirs(kind_dir, exist_ok=True)
    filename_prefix = f"{condition_tag}_" if condition_tag else ""
    out_path = os.path.join(kind_dir, f"{filename_prefix}{kind}_switch_offset_acc_models.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    if os.path.isfile(pdf_path):
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {out_path}")

    if clean_legacy_individuals:
        legacy_pattern = os.path.join(kind_dir, f"{kind}_*_switch_offset_acc.png")
        for legacy_path in glob.glob(legacy_pattern):
            os.remove(legacy_path)
            print(f"Removed legacy per-model figure: {legacy_path}")
    return out_path


def main() -> None:
    """Create one combined recovery figure for every requested switch kind."""

    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    save_dir = os.path.abspath(args.save_dir)
    if args.condition_tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.condition_tag):
        raise ValueError(
            "--condition_tag must start with an alphanumeric character and contain only "
            "letters, digits, underscores, or hyphens"
        )
    grouped_paths = _collect_by_kind(data_dir, args.kind)
    for kind, npz_paths in grouped_paths.items():
        _plot_kind(
            kind,
            npz_paths,
            save_dir,
            args.title,
            args.condition_tag,
            args.clean_legacy_individuals,
        )


if __name__ == "__main__":
    main()
