"""Plot digit-conditioned GaWF distributions and sparsity from compact statistics."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse digit-conditioned plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="./results/anal_data/gawf_gate_audit_digit",
    )
    parser.add_argument(
        "--save_dir",
        default="./results/anal_figs/gawf_gate_audit_digit",
    )
    parser.add_argument(
        "--centered_save_dir",
        default="./results/anal_figs/gawf_gate_audit",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    return parser.parse_args()


def _density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    return counts.astype(np.float64) / (float(counts.sum()) * widths)


def _finish(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure: {path}")


def main() -> None:
    """Save the digit versions of the original sector Figures 4 and 6."""

    args = parse_args()
    stats_path = os.path.join(args.data_dir, "gawf_gate_digit_stats.npz")
    metadata_path = os.path.join(args.data_dir, "gawf_gate_digit_meta.json")
    with np.load(stats_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with open(metadata_path, encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.centered_save_dir, exist_ok=True)

    delta_edges = arrays["delta_edges"]
    delta_centers = (delta_edges[:-1] + delta_edges[1:]) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    kinds = ("input", "recurrent")
    titles = ("Input gate", "Recurrent gate")
    for axis, kind, title in zip(axes, kinds, titles):
        counts = arrays[f"hist_{kind}_delta"]
        axis.plot(delta_centers, _density(counts, delta_edges), color="#805ad5")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set(
            title=title,
            xlabel=r"Group-mean gate $\Delta g$",
            ylabel="Density",
            xlim=(-0.75, 0.75),
        )
    fig.suptitle("Digit-centered gate distributions")
    centered_name = f"03_digit_centered_gate_histogram.{args.format}"
    _finish(fig, os.path.join(args.centered_save_dir, centered_name))

    edges = arrays["gate_edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    colors = plt.get_cmap("tab10")(np.arange(10))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, kind, title in zip(axes, kinds, titles):
        digit_histograms = arrays[f"hist_{kind}_digit"]
        for digit in range(10):
            axis.plot(
                centers,
                _density(digit_histograms[digit], edges),
                color=colors[digit],
                linewidth=1.0,
                label=str(digit),
            )
        axis.set(title=title, xlabel="Gate value", ylabel="Density", xlim=(0.0, 1.0))
    axes[1].legend(title="Digit", ncol=2, fontsize=8)
    fig.suptitle("Gate distributions by foreground digit identity")
    figure4_path = os.path.join(args.save_dir, f"04_per_digit_histogram.{args.format}")
    _finish(fig, figure4_path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    digits = np.arange(10)
    for column, kind in enumerate(kinds):
        records = metadata["sparsity"][kind]
        axes[0, column].plot(
            digits,
            [row["top_5pct_mass_fraction"] for row in records],
            marker="o",
            label="top 5%",
        )
        axes[0, column].plot(
            digits,
            [row["top_10pct_mass_fraction"] for row in records],
            marker="s",
            label="top 10%",
        )
        axes[0, column].set(title=f"{kind.capitalize()} gate mass", ylabel="Mass fraction")
        axes[0, column].legend()
        axes[1, column].plot(
            digits,
            [row["gini"] for row in records],
            marker="o",
            label="Gini",
        )
        axes[1, column].plot(
            digits,
            [row["normalized_participation_ratio"] for row in records],
            marker="s",
            label="Normalized PR",
        )
        axes[1, column].set(xlabel="Foreground digit", ylabel="Index", xticks=digits)
        axes[1, column].legend()
    fig.suptitle("Gate sparsity and concentration by foreground digit identity")
    figure6_path = os.path.join(args.save_dir, f"06_sparsity_by_digit.{args.format}")
    _finish(fig, figure6_path)


if __name__ == "__main__":
    main()
