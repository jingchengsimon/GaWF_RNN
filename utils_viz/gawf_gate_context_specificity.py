"""Plot compact GaWF gate context-specificity results from Parts 1--3.

Inputs are ``parts123_compact.npz`` and ``parts123_results.json``.  Outputs are separate PNG
figures for delta histograms, input spatial maps, variance decomposition, and proxy-confound
checks.  The script never loads a model or reconstructs gates.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse visualization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="./results/anal_data/gawf_gate_context_specificity",
    )
    parser.add_argument(
        "--fig_dir",
        default="./results/anal_figs/gawf_gate_context_specificity",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {path}")


def plot_delta_histograms(
    arrays: dict[str, np.ndarray], point_key: str, fig_dir: str, dpi: int
) -> None:
    """Plot the requested 2x2 shared-limit delta histograms."""

    edges = arrays["delta_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True, sharey=False)
    for row, gate in enumerate(("input", "recurrent")):
        for col, factor in enumerate(("sector", "digit")):
            ax = axes[row, col]
            for balance, label, style, width in (
                ("equal_n", "Equal-n", "-", 1.8),
                ("full", "Full data", "--", 1.1),
            ):
                counts = arrays[f"hist_delta_{gate}_{factor}_{balance}_{point_key}"]
                density = counts / (counts.sum() * np.diff(edges))
                ax.plot(centers, density, style, linewidth=width, label=label)
            ax.axvline(0.0, color="0.25", linewidth=0.8)
            ax.set_xlim(-0.75, 0.75)
            ax.set_yscale("log")
            ax.set_title(f"{gate.title()} gate — {factor}")
            ax.set_xlabel(r"$\Delta g$")
            ax.set_ylabel("Density (log scale)")
            ax.grid(alpha=0.2, linewidth=0.5)
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    suffix = "included" if point_key == "point_included" else "excluded"
    fig.suptitle(f"Context-mean gate deviations — 0.5 point mass {suffix}")
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, f"01_delta_histograms_point_{suffix}.png"), dpi)


def plot_spatial_maps(
    arrays: dict[str, np.ndarray], point_key: str, fig_dir: str, dpi: int
) -> None:
    """Plot 9 sector and 10 digit input-gate maps with one symmetric color scale."""

    sector = arrays[f"spatial_sector_{point_key}"]
    digit = arrays[f"spatial_digit_{point_key}"]
    limit = float(max(np.max(np.abs(sector)), np.max(np.abs(digit))))
    fig, axes = plt.subplots(4, 5, figsize=(11.2, 8.5))
    for index, ax in enumerate(axes.flat):
        if index < 9:
            values = sector[index]
            title = f"Sector {index}"
        elif index == 9:
            ax.axis("off")
            continue
        else:
            digit_index = index - 10
            values = digit[digit_index]
            title = f"Digit {digit_index}"
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.tick_params(labelsize=6, length=2)
    suffix = "included" if point_key == "point_included" else "excluded"
    fig.suptitle(
        r"Input-gate spatial $\Delta g$ (mean over 256 hidden units and 32 channels)" "\n"
        f"0.5 point mass {suffix}"
    )
    colorbar_axis = fig.add_axes([0.915, 0.18, 0.018, 0.62])
    fig.colorbar(image, cax=colorbar_axis, label=r"Mean $\Delta g$")
    fig.subplots_adjust(top=0.88, wspace=0.32, hspace=0.38, right=0.88)
    _save(fig, os.path.join(fig_dir, f"02_input_spatial_maps_point_{suffix}.png"), dpi)


def plot_variance(report: dict, fig_dir: str, dpi: int) -> None:
    """Plot condition-mean fractions and trial-total percentages side by side."""

    factors = ("sector", "digit", "interaction")
    gates = ("input", "recurrent")
    x = np.arange(len(factors))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for gate_idx, gate in enumerate(gates):
        condition = report["part2"][gate]["point_included"][
            "equal_cell_condition_mean"
        ]["fractions"]
        trial = report["part2"][gate]["point_included"]["equal_cell_trial_total"][
            "percent"
        ]
        axes[0].bar(
            x + (gate_idx - 0.5) * width,
            [100 * condition[factor] for factor in factors],
            width,
            label=gate.title(),
        )
        trial_factors = factors + ("residual",)
        axes[1].bar(
            np.arange(4) + (gate_idx - 0.5) * width,
            [trial[factor] for factor in trial_factors],
            width,
            label=gate.title(),
        )
    axes[0].set_xticks(x, [value.title() for value in factors])
    axes[0].set_ylabel("Condition-mean variance (%)")
    axes[0].set_title("Hidden-state-compatible marginalization")
    axes[1].set_xticks(np.arange(4), ["Sector", "Digit", "Interaction", "Residual"])
    axes[1].set_ylabel("Trial-total variance (%)")
    axes[1].set_title("Balanced trial-level ANOVA")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "03_variance_decomposition.png"), dpi)


def plot_encoder_control(report: dict, fig_dir: str, dpi: int) -> None:
    """Plot the raw CNN encoder condition-mean decomposition."""

    fractions = report["part3"]["encoder_control"]["equal_cell_condition_mean"][
        "fractions"
    ]
    factors = ("sector", "digit", "interaction")
    values = [100 * fractions[factor] for factor in factors]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar([factor.title() for factor in factors], values)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Condition-mean variance (%)")
    ax.set_title("Raw CNN encoder control")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "04_encoder_control_decomposition.png"), dpi)


def _scatter_grid(
    arrays: dict[str, np.ndarray], factor: str, response_prefix: str, output: str, dpi: int
) -> None:
    metrics = (
        "ink_area_proxy",
        "ink_intensity_proxy",
        "encoder_l1",
        "encoder_l2",
        "encoder_active_gt_0",
    )
    labels = np.arange(10 if factor == "digit" else 9)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2))
    for row, gate in enumerate(("input", "recurrent")):
        y = arrays[f"{factor}_{response_prefix}_{gate}"]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            x = arrays[f"{factor}_{metric}"]
            ax.scatter(x, y, s=28)
            for label, xv, yv in zip(labels, x, y):
                ax.annotate(
                    str(label),
                    (xv, yv),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.set_xlabel(metric.replace("_", " "), fontsize=8)
            ax.set_ylabel(f"{gate.title()} {response_prefix}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, linewidth=0.5)
    fig.suptitle(f"{factor.title()}-level low-level metrics versus {response_prefix}")
    fig.tight_layout()
    _save(fig, output, dpi)


def plot_digit_regression(arrays: dict[str, np.ndarray], fig_dir: str, dpi: int) -> None:
    """Plot the decisive digit-level Gini-on-ink proxy regression."""

    ink = arrays["digit_ink_area_proxy"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for ax, gate in zip(axes, ("input", "recurrent")):
        gini = arrays[f"digit_gini_{gate}"]
        slope, intercept = np.polyfit(ink, gini, 1)
        order = np.argsort(ink)
        ax.scatter(ink, gini, s=34)
        ax.plot(ink[order], intercept + slope * ink[order], linewidth=1.3)
        for digit, x, y in zip(range(10), ink, gini):
            ax.annotate(str(digit), (x, y), xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("Composite-crop ink area proxy")
        ax.set_ylabel("Per-digit Gini")
        ax.set_title(f"{gate.title()} gate")
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "06_digit_gini_ink_regression.png"), dpi)


def main() -> None:
    """Load compact outputs and write every requested figure."""

    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    with np.load(os.path.join(args.data_dir, "parts123_compact.npz")) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with open(
        os.path.join(args.data_dir, "parts123_results.json"), "r", encoding="utf-8"
    ) as file_obj:
        report = json.load(file_obj)
    for point_key in ("point_included", "point_excluded"):
        plot_delta_histograms(arrays, point_key, args.fig_dir, args.dpi)
        plot_spatial_maps(arrays, point_key, args.fig_dir, args.dpi)
    plot_variance(report, args.fig_dir, args.dpi)
    plot_encoder_control(report, args.fig_dir, args.dpi)
    _scatter_grid(
        arrays,
        "digit",
        "gini",
        os.path.join(args.fig_dir, "05_digit_metric_gini_scatter.png"),
        args.dpi,
    )
    plot_digit_regression(arrays, args.fig_dir, args.dpi)
    _scatter_grid(
        arrays,
        "sector",
        "gini",
        os.path.join(args.fig_dir, "07_sector_symmetric_control.png"),
        args.dpi,
    )
    _scatter_grid(
        arrays,
        "digit",
        "variance_contribution",
        os.path.join(args.fig_dir, "08_digit_variance_contribution_scatter.png"),
        args.dpi,
    )


if __name__ == "__main__":
    main()
