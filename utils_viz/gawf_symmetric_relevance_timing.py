"""Plot saved symmetric GaWF relevance and switch-timing outputs.

Inputs are the compact NPZ/JSON artifacts produced by
``utils_anal/gawf_symmetric_relevance_timing.py``. Outputs are seven independent PNG figures plus
a PDF copy of the continuous-alignment figure; this module never loads a model or raw stimulus.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"sector": "#3b82f6", "digit": "#f97316"}
ALIGNMENT_COLOR_LIMIT = 0.6


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    for factor, category in (
        ("decomposition", "D_variance_decomposition"),
        ("relevance", "E_relevance_alignment"),
        ("timing", "F_timing"),
    ):
        parser.add_argument(
            f"--{factor}_data_dir",
            default=str(output_dir(category, "gawf_symmetric_relevance_timing", "data")),
        )
        parser.add_argument(
            f"--{factor}_fig_dir",
            default=str(output_dir(category, "gawf_symmetric_relevance_timing", "figs")),
        )
    parser.add_argument(
        "--data_dir",
        default="",
        help="Deprecated compatibility override; reads every part from one directory.",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="Deprecated compatibility override; writes every figure to one directory.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_part1_selectivity(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot per-unit eta-squared distributions and interaction-dominant fractions."""

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    for axis, population in zip(axes, ("encoder", "hidden")):
        values = [
            data[f"primary_{population}_eta_sector"],
            data[f"primary_{population}_eta_digit"],
            data[f"primary_{population}_eta_interaction"],
            data[f"primary_{population}_eta_residual"],
        ]
        parts = axis.violinplot(values, showmeans=True, showextrema=False)
        for body, color in zip(
            parts["bodies"], (COLORS["sector"], COLORS["digit"], "#8b5cf6", "#64748b")
        ):
            body.set_facecolor(color)
            body.set_alpha(0.72)
        parts["cmeans"].set_color("black")
        dominant = data[f"primary_{population}_interaction_dominant"].astype(bool)
        axis.set_title(
            f"{population.capitalize()} units\ninteraction-dominant: {dominant.mean():.1%}"
        )
        axis.set_xticks(range(1, 5), ["sector", "digit", "interaction", "residual"], rotation=20)
        axis.set_ylabel(r"$\eta^2$")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Part 1 — activation-defined unit selectivity")
    fig.subplots_adjust(top=0.76, wspace=0.20)
    _save(fig, os.path.join(save_dir, "part1_selectivity.png"), dpi)


def plot_architecture_axis(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot encoder selectivity over 6x6 space and 32 channels."""

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for row, factor in enumerate(("sector", "digit")):
        eta = data[f"primary_encoder_eta_{factor}"].reshape(32, 6, 6)
        spatial = eta.mean(axis=0)
        channel = eta.mean(axis=(1, 2))
        image = axes[row, 0].imshow(spatial, cmap="viridis", aspect="equal")
        fig.colorbar(image, ax=axes[row, 0], fraction=0.046, pad=0.04)
        axes[row, 0].set_title(f"{factor}: mean across channels")
        axes[row, 0].set_xlabel("6x6 x-position")
        axes[row, 0].set_ylabel("6x6 y-position")
        axes[row, 1].bar(np.arange(32), channel, color=COLORS[factor])
        axes[row, 1].set_title(f"{factor}: mean across space")
        axes[row, 1].set_xlabel("encoder channel")
        axes[row, 1].set_ylabel(r"mean $\eta^2$")
        axes[row, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("Part 1 — tested spatial/channel architectural assumption")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "part1_architecture_axis.png"), dpi)


def plot_part2_effects(report: dict, save_dir: str, dpi: int) -> None:
    """Plot Cohen's d for all cells, thresholds, and interaction policies."""

    primary = report["primary_validation_estimate_test_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for axis, policy in zip(axes, ("interaction_excluded", "interaction_included")):
        positions = np.arange(3)
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for offset, gate, factor in zip(
            offsets,
            ("input", "input", "recurrent", "recurrent"),
            ("sector", "digit", "sector", "digit"),
        ):
            points, lower, upper = [], [], []
            for percent in (10, 20, 30):
                cell = primary[policy]["cells"][f"{gate}_{factor}"]["top_percent"][str(percent)]
                points.append(cell["cohens_d"])
                lower.append(cell["cohens_d"] - cell["bootstrap_ci95"][0])
                upper.append(cell["bootstrap_ci95"][1] - cell["cohens_d"])
            label = f"{gate} / {factor}"
            hatch = "//" if gate == "recurrent" else None
            axis.bar(
                positions + offset * width,
                points,
                width,
                color=COLORS[factor],
                alpha=0.78,
                hatch=hatch,
                label=label,
            )
            axis.errorbar(
                positions + offset * width,
                points,
                yerr=np.asarray([lower, upper]),
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, ["top 10%", "top 20%", "top 30%"])
        axis.set_title(policy.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Cohen's d: relevant minus other selective units")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Part 2 — symmetric 2x2 relevance effects")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "part2_relevance_effects.png"), dpi)


def plot_part2_top10_excluded_effects(report: dict, save_dir: str, dpi: int) -> None:
    """Plot top-10% interaction-excluded effects with gate type on the category axis."""

    cells = report["primary_validation_estimate_test_effect"]["interaction_excluded"]["cells"]
    positions = np.arange(2)
    width = 0.34
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    for factor_index, factor in enumerate(("sector", "digit")):
        points, lower, upper = [], [], []
        for gate in ("input", "recurrent"):
            cell = cells[f"{gate}_{factor}"]["top_percent"]["10"]
            points.append(cell["cohens_d"])
            lower.append(cell["cohens_d"] - cell["bootstrap_ci95"][0])
            upper.append(cell["bootstrap_ci95"][1] - cell["cohens_d"])
        offset = (factor_index - 0.5) * width
        bars = axis.bar(
            positions + offset,
            points,
            width,
            color=COLORS[factor],
            alpha=0.82,
            label=factor.capitalize(),
        )
        axis.errorbar(
            positions + offset,
            points,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, ["Input gate", "Recurrent gate"])
    axis.set_ylabel("Cohen's d: relevant minus other selective units")
    axis.set_title("Top 10% relevance effects (interaction excluded)")
    axis.grid(axis="y", alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    axis.margins(y=0.16)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "part2_relevance_effects_top10_excluded.png"), dpi)


def plot_part2_alignment(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot the four primary continuous alignment matrices and diagonal contrasts."""

    cells = tuple(
        zip(
            ("input", "input", "recurrent", "recurrent"),
            ("sector", "digit", "sector", "digit"),
        )
    )
    matrices = [
        data[f"primary_interaction_excluded_{gate}_{factor}_alignment_matrix"]
        for gate, factor in cells
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.7))
    for axis, (gate, factor), matrix in zip(axes.flat, cells, matrices):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Alignment matrix must be square, got {matrix.shape}")
        diagonal_mask = np.eye(matrix.shape[0], dtype=bool)
        diagonal_minus_off_diagonal = float(
            matrix[diagonal_mask].mean() - matrix[~diagonal_mask].mean()
        )
        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-ALIGNMENT_COLOR_LIMIT,
            vmax=ALIGNMENT_COLOR_LIMIT,
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(
            f"{gate} gate / {factor}\n" f"diag-offdiag = {diagonal_minus_off_diagonal:.3f}"
        )
        axis.set_xlabel("gate context")
        axis.set_ylabel("activation context")
    fig.suptitle("Part 2 — activation/gate tuning cosine alignment")
    fig.tight_layout()
    pdf_path = os.path.join(save_dir, "part2_continuous_alignment.pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    print(f"Saved: {pdf_path}")
    _save(fig, os.path.join(save_dir, "part2_continuous_alignment.png"), dpi)


def plot_part3_timing(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot switch-aligned gate differences, argmax accuracy, and graded evidence."""

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    post = np.arange(1, 11)
    for axis, gate, factor in zip(
        axes.flat,
        ("input", "input", "recurrent", "recurrent"),
        ("sector", "digit", "sector", "digit"),
    ):
        alignment = data[f"primary_{gate}_{factor}_alignment_difference"].mean(axis=0)
        correct = data[f"primary_{factor}_readout_correct"].mean(axis=0)
        graded = data[f"primary_{factor}_readout_graded"].mean(axis=0)
        axis.plot(post, alignment, color="#7c3aed", marker="o", label="gate new-old align")
        axis.plot(post, correct, color="#111827", marker="s", label="readout argmax")
        axis.plot(post, graded, color="#10b981", marker="^", label="graded probability")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(3, color="red", linewidth=0.7, linestyle="--", label="post3 floor")
        axis.set_title(f"{gate} gate / {factor}")
        axis.set_xlabel("post-switch frame")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("mean value")
    axes[1, 0].set_ylabel("mean value")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Part 3 — gate reconfiguration versus readout recovery")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "part3_switch_timing.png"), dpi)


def plot_part3_event_leads(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot paired per-event readout-minus-gate timing differences."""

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True, sharey=False)
    bins = np.arange(-9.5, 10.5, 1)
    for axis, gate, factor in zip(
        axes.flat,
        ("input", "input", "recurrent", "recurrent"),
        ("sector", "digit", "sector", "digit"),
    ):
        argmax = data[f"primary_{gate}_{factor}_argmax_minus_gate"]
        graded = data[f"primary_{gate}_{factor}_graded_rise_minus_gate"]
        axis.hist(argmax, bins=bins, alpha=0.65, color="#111827", label="first correct")
        axis.hist(graded, bins=bins, alpha=0.55, color="#10b981", label="first prob. rise")
        axis.axvline(0, color="red", linewidth=0.7)
        axis.set_title(f"{gate} gate / {factor}")
        axis.set_xlabel("readout frame - gate zero-crossing")
        axis.set_ylabel("events")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Part 3 — paired event timing (positive means gate leads)")
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, "part3_per_event_lead.png"), dpi)


def main() -> None:
    """Render every saved analysis figure."""

    args = parse_args()
    if args.data_dir:
        args.decomposition_data_dir = args.relevance_data_dir = args.timing_data_dir = args.data_dir
    if args.save_dir:
        args.decomposition_fig_dir = args.relevance_fig_dir = args.timing_fig_dir = args.save_dir
    for directory in (
        args.decomposition_fig_dir,
        args.relevance_fig_dir,
        args.timing_fig_dir,
    ):
        os.makedirs(directory, exist_ok=True)
    part1_path = os.path.join(args.decomposition_data_dir, "part1_selectivity.npz")
    part2_npz_path = os.path.join(args.relevance_data_dir, "part2_inference.npz")
    part2_json_path = os.path.join(args.relevance_data_dir, "part2_results.json")
    part3_path = os.path.join(args.timing_data_dir, "part3_events.npz")
    with np.load(part1_path) as part1:
        plot_part1_selectivity(part1, args.decomposition_fig_dir, args.dpi)
        plot_architecture_axis(part1, args.decomposition_fig_dir, args.dpi)
    part2_report = _load_json(part2_json_path)
    plot_part2_effects(part2_report, args.relevance_fig_dir, args.dpi)
    plot_part2_top10_excluded_effects(part2_report, args.relevance_fig_dir, args.dpi)
    with np.load(part2_npz_path) as part2:
        plot_part2_alignment(part2, args.relevance_fig_dir, args.dpi)
    with np.load(part3_path) as part3:
        plot_part3_timing(part3, args.timing_fig_dir, args.dpi)
        plot_part3_event_leads(part3, args.timing_fig_dir, args.dpi)


if __name__ == "__main__":
    main()
