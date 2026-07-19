"""Plot condition-level RDMs and cross-model representation similarities.

Reads analysis files produced by ``utils_anal/pop_act_representation_similarity.py``.
Each model receives a digit-blocked 90x90 RDM heatmap below this script's
``D_variance_decomposition`` figure directory. A paired RSA/Linear CKA
heatmap is saved at the figure-root level. No arrays or metadata are written here.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import json
import os
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot per-model condition RDMs and cross-model RSA/Linear CKA."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(
            output_dir(
                "D_variance_decomposition",
                "pop_act_representation_similarity",
                "data",
            )
        ),
        help="Directory containing representation similarity analysis outputs.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default=str(
            output_dir(
                "D_variance_decomposition",
                "pop_act_representation_similarity",
                "figs",
            )
        ),
        help="Figure root; only PNG output is written.",
    )
    return parser.parse_args()


def _load_metadata(data_dir: str) -> dict[str, Any]:
    path = os.path.join(data_dir, "representation_similarity_meta.json")
    with open(path) as file:
        return json.load(file)


def plot_condition_rdm(
    rdm: np.ndarray,
    model_label: str,
    normalization: str,
    output_path: str,
) -> None:
    """Save one digit-blocked 90-condition Euclidean RDM heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(rdm, dtype=np.float64)
    if values.shape != (90, 90):
        raise ValueError(f"RDM must have shape (90, 90), got {values.shape}")

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    image = ax.imshow(values, cmap="viridis", vmin=0.0, interpolation="nearest")
    digit_centers = np.arange(10) * 9 + 4
    ax.set_xticks(digit_centers)
    ax.set_yticks(digit_centers)
    ax.set_xticklabels([str(digit) for digit in range(10)])
    ax.set_yticklabels([str(digit) for digit in range(10)])
    for boundary in np.arange(9, 90, 9) - 0.5:
        ax.axhline(boundary, color="white", linewidth=0.45, alpha=0.75)
        ax.axvline(boundary, color="white", linewidth=0.45, alpha=0.75)
    ax.set_xlabel("digit block (sector 0–8 within each block)")
    ax.set_ylabel("digit block (sector 0–8 within each block)")
    ax.set_title(f"{model_label}: 90-condition Euclidean RDM\n{normalization} activation")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Euclidean distance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _annotate_similarity(
    ax: Any,
    matrix: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Annotate cells with text contrast derived from the rendered colormap luminance."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    colormap = plt.get_cmap(cmap)
    normalizer = Normalize(vmin=vmin, vmax=vmax)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            red, green, blue, _ = colormap(normalizer(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "black" if luminance > 0.55 else "white"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)


def plot_cross_model_similarity(
    rsa: np.ndarray,
    linear_cka: np.ndarray,
    labels: list[str],
    output_path: str,
) -> None:
    """Save aligned RSA and Linear CKA model-by-model heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rsa_values = np.asarray(rsa, dtype=np.float64)
    cka_values = np.asarray(linear_cka, dtype=np.float64)
    expected = (len(labels), len(labels))
    if rsa_values.shape != expected or cka_values.shape != expected:
        raise ValueError(
            f"Similarity matrices must have shape {expected}, got "
            f"{rsa_values.shape}, {cka_values.shape}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    panels = [
        (axes[0], rsa_values, "RDM RSA (Spearman)", "coolwarm", -1.0, 1.0),
        (axes[1], cka_values, "Linear CKA", "viridis", 0.0, 1.0),
    ]
    for ax, matrix, title, cmap, vmin, vmax in panels:
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title)
        _annotate_similarity(ax, matrix, cmap, vmin, vmax)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("similarity")
    fig.suptitle("Cross-model similarity of the original 90-condition representations")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def generate_figures(data_dir: str, fig_dir: str) -> list[str]:
    """Read saved analysis outputs and generate all requested heatmaps."""
    metadata = _load_metadata(data_dir)
    normalization = str(metadata["normalization"])
    records = metadata["models"]
    labels = [str(record["model_label"]) for record in records]
    output_paths = []

    for record in records:
        model = str(record["model"])
        rdm_path = os.path.join(data_dir, model, "condition_rdm.npy")
        output_path = os.path.join(fig_dir, model, "condition_rdm.png")
        plot_condition_rdm(
            np.load(rdm_path),
            str(record["model_label"]),
            normalization,
            output_path,
        )
        output_paths.append(output_path)
        print(f"Saved {output_path}")

    rsa_path = os.path.join(data_dir, "representation_similarity_rsa_spearman.npy")
    cka_path = os.path.join(data_dir, "representation_similarity_linear_cka.npy")
    comparison_path = os.path.join(fig_dir, "representation_similarity_rsa_cka.png")
    plot_cross_model_similarity(
        np.load(rsa_path),
        np.load(cka_path),
        labels,
        comparison_path,
    )
    output_paths.append(comparison_path)
    print(f"Saved {comparison_path}")
    return output_paths


def main() -> None:
    """Run the command-line plot generator."""
    args = parse_args()
    generate_figures(args.data_dir, args.fig_dir)


if __name__ == "__main__":
    main()
