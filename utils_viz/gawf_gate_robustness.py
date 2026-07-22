"""Plot GaWF gate robustness survival curves and CI convergence.

Input is ``robustness_compact.npz`` from ``utils_anal/gawf_gate_robustness.py``.
Outputs are two PNG figures in ``--save_dir``; no analysis arrays are modified.
"""

from __future__ import annotations

import os as _anal_os
import sys as _anal_sys

_ANAL_PROJECT_ROOT = _anal_os.path.dirname(_anal_os.path.dirname(_anal_os.path.abspath(__file__)))
if _ANAL_PROJECT_ROOT not in _anal_sys.path:
    _anal_sys.path.insert(0, _ANAL_PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse plotting arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(
            output_dir(
                "H_controls",
                "gawf_gate_robustness",
                "data",
            )
            / "robustness_compact.npz"
        ),
    )
    parser.add_argument(
        "--save_dir", default=str(output_dir("H_controls", "gawf_gate_robustness", "figs"))
    )
    parser.add_argument(
        "--delta_dir", default=str(output_dir("C_delta_gate", "gawf_gate_robustness", "figs"))
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {path}")


def plot_survival(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot all four absolute group-mean deviation survival functions."""

    thresholds = data["survival_thresholds"]
    styles = {
        "input_sector": ("Input × sector", "tab:red", "-"),
        "input_digit": ("Input × digit", "tab:red", "--"),
        "recurrent_sector": ("Recurrent × sector", "tab:blue", "-"),
        "recurrent_digit": ("Recurrent × digit", "tab:blue", "--"),
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for key, (label, color, linestyle) in styles.items():
        survival = np.maximum(data[f"survival_{key}"], 1e-7)
        ax.plot(
            thresholds,
            survival,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel(r"Threshold $t$")
    ax.set_ylabel(r"Fraction with $|\Delta g| > t$")
    ax.set_title("Group-mean gate-deviation survival functions")
    ax.grid(True, which="both", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    _save(fig, os.path.join(save_dir, "01_delta_survival.png"), dpi)


def plot_ci_width(data: np.lib.npyio.NpzFile, save_dir: str, dpi: int) -> None:
    """Plot 95% bootstrap-CI width against sampled-synapse count."""

    sizes = np.asarray([128, 512, 2048, 8192], dtype=np.int64)
    styles = {
        ("input", 0): ("Input sector", "tab:red", "o", "-"),
        ("input", 1): ("Input digit", "tab:red", "s", "--"),
        ("recurrent", 0): ("Recurrent sector", "tab:blue", "o", "-"),
        ("recurrent", 1): ("Recurrent digit", "tab:blue", "s", "--"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (gate, factor_index), (label, color, marker, linestyle) in styles.items():
        widths = []
        for size in sizes:
            draws = data[f"ci_draws_{gate}_{size}"][:, factor_index]
            low, high = np.quantile(draws, [0.025, 0.975])
            widths.append(100.0 * (high - low))
        ax.plot(
            sizes,
            widths,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.set_xlabel("Sampled synapses")
    ax.set_ylabel("95% CI width (percentage points)")
    ax.set_title("Variance-fraction CI convergence")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    _save(fig, os.path.join(save_dir, "02_ci_width_convergence.png"), dpi)


def main() -> None:
    """Load compact arrays and save both requested figures."""

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.delta_dir, exist_ok=True)
    with np.load(args.data) as data:
        plot_survival(data, args.delta_dir, args.dpi)
        plot_ci_width(data, args.save_dir, args.dpi)


if __name__ == "__main__":
    main()
