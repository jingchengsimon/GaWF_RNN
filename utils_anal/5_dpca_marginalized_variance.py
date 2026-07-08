"""Summarize dPCA marginalized variance across multiple population-activity segments.

Reads per-segment ``pop_act_dpca.npy`` arrays with shape ``(hidden, 10, 9)`` from
``--data_root/<model>/seed<seed>/``. For each segment it computes raw digit, sector,
and digit-sector interaction variance fractions from condition means while excluding
empty digit-sector cells. It also fits official dPCA RRR as a side check and stores
summed explained-variance ratios for d/s/ds marginalizations.

Outputs (in --output_dir):
- ``<model>/dpca_variance.json``  per-model per-seed records and summary statistics
- ``dpca_marginalized_variance_table.csv``  model × factor summary table
- ``dpca_marginalized_variance_tests.csv``  paired t-test table
- ``dpca_marginalized_variance_tests.json``  paired t-test records
- ``dpca_marginalized_variance_compare.png``  grouped bar chart with CI error bars
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_MODELS = [
    "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model",
    "rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5_model",
    "lstm_sector_acc_h80_lr0.001_wd0.001_cdo0.0_rdo0.5_model",
    "gru_sector_acc_h105_lr0.005_wd0.001_cdo0.0_rdo0.5_model",
    "mamba_sector_acc_dmodel170_lr0.001_wd0.001_cdo0.0_rdo0.5_model",
    "s5_sector_acc_dmodel256_state128_lr0.001_wd0.0_cdo0.0_rdo0.5_model",
]

MODEL_LABELS = {
    DEFAULT_MODELS[0]: "GaWF",
    DEFAULT_MODELS[1]: "RNN",
    DEFAULT_MODELS[2]: "LSTM",
    DEFAULT_MODELS[3]: "GRU",
    DEFAULT_MODELS[4]: "Mamba",
    DEFAULT_MODELS[5]: "S5",
}

FACTORS = ("digit", "sector", "interaction")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-segment raw marginalized variance summary for pop_act_dpca outputs."
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="results/anal_data/pop_act_multiseg",
        help="Root containing <model>/seed<seed>/pop_act_dpca.npy.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="results/anal_figs/5_pop_act_umap_multiseg",
        help="Output directory for JSON, CSV, and figure artifacts.",
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--T", type=int, default=57576, help="Fixed segment length for metadata.")
    p.add_argument(
        "--dpca_regularizer",
        type=float,
        default=1e-4,
        help="Fixed official dPCA RRR regularizer used for side-check EVR.",
    )
    p.add_argument(
        "--errorbar",
        choices=["ci95", "std"],
        default="ci95",
        help="Error bars to draw on the grouped bar chart.",
    )
    return p.parse_args()


def load_counts(path: str, shape: tuple[int, int]) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    counts = np.load(path)
    if counts.shape != shape:
        raise ValueError(f"{path} must have shape {shape}, got {counts.shape}")
    return counts.astype(np.int64, copy=False)


def condition_mask(X: np.ndarray, counts: np.ndarray | None) -> np.ndarray:
    if counts is not None:
        return counts > 0
    return ~np.all(np.isnan(X), axis=0)


def raw_marginalized_variance(X_dpca: np.ndarray, counts: np.ndarray | None) -> dict[str, Any]:
    """Compute raw digit/sector/interaction variance fractions over nonempty cells only."""
    X = np.asarray(X_dpca, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (10, 9):
        raise ValueError(f"Expected X_dpca (N, 10, 9), got {X.shape}")
    mask = condition_mask(X, counts)
    if not np.any(mask):
        raise ValueError("No nonempty digit-sector cells")

    X_masked = np.where(mask[np.newaxis, :, :], X, np.nan)
    grand = np.nanmean(X_masked, axis=(1, 2))
    digit_mean = np.nanmean(X_masked, axis=2) - grand[:, np.newaxis]
    sector_mean = np.nanmean(X_masked, axis=1) - grand[:, np.newaxis]
    digit_bc = np.broadcast_to(digit_mean[:, :, np.newaxis], X.shape)
    sector_bc = np.broadcast_to(sector_mean[:, np.newaxis, :], X.shape)
    interaction = X_masked - grand[:, np.newaxis, np.newaxis] - digit_bc - sector_bc

    valid = mask[np.newaxis, :, :]
    centered = X_masked - grand[:, np.newaxis, np.newaxis]
    var_digit = float(np.nansum(np.where(valid, digit_bc, np.nan) ** 2))
    var_sector = float(np.nansum(np.where(valid, sector_bc, np.nan) ** 2))
    var_interaction = float(np.nansum(np.where(valid, interaction, np.nan) ** 2))
    component_sum = var_digit + var_sector + var_interaction
    var_total = float(np.nansum(np.where(valid, centered, np.nan) ** 2))
    if var_total <= 0:
        fractions = {factor: float("nan") for factor in FACTORS}
    else:
        fractions = {
            "digit": var_digit / var_total,
            "sector": var_sector / var_total,
            "interaction": var_interaction / var_total,
        }
    return {
        "counts_nonempty": int(np.sum(mask)),
        "counts_total_cells": int(mask.size),
        "marg_frac": fractions,
        "var_digit": var_digit,
        "var_sector": var_sector,
        "var_interaction": var_interaction,
        "var_total": var_total,
        "sum_check": component_sum,
        "sum_check_ratio": component_sum / var_total if var_total > 0 else float("nan"),
        "sum_check_residual": component_sum - var_total,
    }


def fill_nan_digit_sector_cells(
    X_dpca: np.ndarray,
    counts: np.ndarray | None,
) -> np.ndarray:
    """Impute empty cells for official dPCA using digit + sector - grand per feature."""
    X = np.asarray(X_dpca, dtype=np.float64).copy()
    mask = condition_mask(X, counts)
    nan_mask = np.isnan(X)
    empty_cells = ~mask
    with np.errstate(invalid="ignore", divide="ignore"):
        X_masked = np.where(mask[np.newaxis, :, :], X, np.nan)
        digit_mean = np.nanmean(X_masked, axis=2)
        sector_mean = np.nanmean(X_masked, axis=1)
        grand = np.nanmean(X_masked, axis=(1, 2))
    grand = np.where(np.isnan(grand), 0.0, grand)
    for d in range(10):
        for s in range(9):
            needs_fill = bool(empty_cells[d, s]) or bool(np.any(np.isnan(X[:, d, s])))
            if not needs_fill:
                continue
            estimate = digit_mean[:, d] + sector_mean[:, s] - grand
            estimate = np.where(np.isnan(estimate), grand, estimate)
            elem_mask = nan_mask[:, d, s]
            if bool(empty_cells[d, s]):
                X[:, d, s] = estimate
            elif np.any(elem_mask):
                X[elem_mask, d, s] = estimate[elem_mask]
    if np.isnan(X).any():
        raise ValueError("NaN imputation for official dPCA failed")
    return X


def official_dpca_evr(
    X_dpca: np.ndarray,
    counts: np.ndarray | None,
    regularizer: float,
) -> dict[str, float]:
    try:
        from dPCA.dPCA import dPCA
    except ImportError as e:
        raise ImportError("Official dPCA side check requires: pip install dpca") from e

    X = fill_nan_digit_sector_cells(X_dpca, counts)
    dpca = dPCA(labels="ds", n_components=2, regularizer=float(regularizer))
    dpca.fit_transform(X)
    evr = {}
    for key, out_key in (("d", "digit"), ("s", "sector"), ("ds", "interaction")):
        vals = np.asarray(dpca.explained_variance_ratio_.get(key, []), dtype=np.float64)
        evr[out_key] = float(np.sum(vals))
    return evr


def t_critical_975(df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except Exception:
        lookup = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }
        return lookup.get(df, 1.96)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    k = int(vals.size)
    if k == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if k > 1 else 0.0
    ci = t_critical_975(k - 1) * std / math.sqrt(k) if k > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci)}


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    vals_a = np.asarray(a, dtype=np.float64)
    vals_b = np.asarray(b, dtype=np.float64)
    valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
    vals_a = vals_a[valid]
    vals_b = vals_b[valid]
    if vals_a.size < 2:
        return {"t": float("nan"), "p": float("nan"), "n": int(vals_a.size)}
    try:
        from scipy import stats

        res = stats.ttest_rel(vals_a, vals_b)
        return {"t": float(res.statistic), "p": float(res.pvalue), "n": int(vals_a.size)}
    except Exception:
        diff = vals_a - vals_b
        std = float(np.std(diff, ddof=1))
        if std <= 0:
            return {"t": float("inf"), "p": 0.0, "n": int(vals_a.size)}
        tval = float(np.mean(diff) / (std / math.sqrt(vals_a.size)))
        return {"t": tval, "p": float("nan"), "n": int(vals_a.size)}


def analyze_model(
    data_root: str,
    output_dir: str,
    model: str,
    seeds: list[int],
    segment_t: int,
    regularizer: float,
) -> dict[str, Any]:
    model_dir = os.path.join(data_root, model)
    per_seed = []
    n_features: int | None = None
    for seed in seeds:
        seg_dir = os.path.join(model_dir, f"seed{seed}")
        x_path = os.path.join(seg_dir, "pop_act_dpca.npy")
        c_path = os.path.join(seg_dir, "pop_act_digitxsector_counts.npy")
        if not os.path.isfile(x_path):
            raise FileNotFoundError(x_path)
        X = np.load(x_path)
        n_features = int(X.shape[0])
        counts = load_counts(c_path, (10, 9))
        raw = raw_marginalized_variance(X, counts)
        raw["dpca_evr"] = official_dpca_evr(X, counts, regularizer)
        raw["seed"] = int(seed)
        per_seed.append(raw)

    summary = {}
    for factor in FACTORS:
        vals = np.asarray([row["marg_frac"][factor] for row in per_seed], dtype=np.float64)
        summary[factor] = summarize_values(vals)

    payload = {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "N": n_features,
        "T": int(segment_t),
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "summary": summary,
    }
    out_model_dir = os.path.join(output_dir, model)
    os.makedirs(out_model_dir, exist_ok=True)
    with open(os.path.join(out_model_dir, "dpca_variance.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_summary_table(payloads: list[dict[str, Any]], output_dir: str) -> str:
    path = os.path.join(output_dir, "dpca_marginalized_variance_table.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "model_label", "factor", "mean", "std", "ci95_lo", "ci95_hi", "K"],
        )
        writer.writeheader()
        for payload in payloads:
            k = len(payload["per_seed"])
            for factor in FACTORS:
                stats = payload["summary"][factor]
                writer.writerow(
                    {
                        "model": payload["model"],
                        "model_label": payload["model_label"],
                        "factor": factor,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci95_lo": stats["mean"] - stats["ci95"],
                        "ci95_hi": stats["mean"] + stats["ci95"],
                        "K": k,
                    }
                )
    return path


def write_tests(payloads: list[dict[str, Any]], output_dir: str) -> tuple[str, str]:
    tests = []
    by_model = {payload["model"]: payload for payload in payloads}
    for payload in payloads:
        digit = np.asarray([r["marg_frac"]["digit"] for r in payload["per_seed"]])
        sector = np.asarray([r["marg_frac"]["sector"] for r in payload["per_seed"]])
        res = paired_ttest(sector, digit)
        tests.append(
            {
                "test": "within_model_sector_gt_digit",
                "model": payload["model"],
                "comparison": "sector - digit",
                **res,
            }
        )

    gawf = next((p for p in payloads if p["model"].startswith("gawf_")), None)
    if gawf is not None:
        gawf_sector = np.asarray([r["marg_frac"]["sector"] for r in gawf["per_seed"]])
        for payload in payloads:
            if payload is gawf:
                continue
            sector = np.asarray([r["marg_frac"]["sector"] for r in payload["per_seed"]])
            res = paired_ttest(gawf_sector, sector)
            tests.append(
                {
                    "test": "gawf_sector_vs_baseline_sector",
                    "model": f"{gawf['model']} vs {payload['model']}",
                    "comparison": "gawf sector - baseline sector",
                    **res,
                }
            )

    csv_path = os.path.join(output_dir, "dpca_marginalized_variance_tests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test", "model", "comparison", "t", "p", "n"],
        )
        writer.writeheader()
        for row in tests:
            writer.writerow(row)
    json_path = os.path.join(output_dir, "dpca_marginalized_variance_tests.json")
    with open(json_path, "w") as f:
        json.dump(tests, f, indent=2)
    return csv_path, json_path


def plot_grouped_bars(payloads: list[dict[str, Any]], output_dir: str, errorbar: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [payload["model_label"] for payload in payloads]
    x = np.arange(len(payloads), dtype=np.float64)
    width = 0.24
    colors = {"digit": "#4C78A8", "sector": "#F58518", "interaction": "#54A24B"}
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for idx, factor in enumerate(FACTORS):
        means = np.asarray([p["summary"][factor]["mean"] for p in payloads]) * 100.0
        if errorbar == "std":
            err = np.asarray([p["summary"][factor]["std"] for p in payloads]) * 100.0
        else:
            err = np.asarray([p["summary"][factor]["ci95"] for p in payloads]) * 100.0
        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=err,
            label=factor,
            color=colors[factor],
            edgecolor="white",
            linewidth=0.6,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of condition-mean variance")
    ax.set_title("Raw marginalized variance fractions across fixed-length segments")
    ax.legend(title="factor")
    ax.grid(axis="y", alpha=0.25)
    note = (
        f"Error bars = cross-segment {errorbar}; T fixed, only segment start changes. "
        "Raw marginalized fractions, not whitened dPCA EVR."
    )
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, "dpca_marginalized_variance_compare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    payloads = [
        analyze_model(
            args.data_root,
            args.output_dir,
            model,
            args.seeds,
            args.T,
            args.dpca_regularizer,
        )
        for model in args.models
    ]
    table_path = write_summary_table(payloads, args.output_dir)
    tests_csv, tests_json = write_tests(payloads, args.output_dir)
    fig_path = plot_grouped_bars(payloads, args.output_dir, args.errorbar)
    print(f"Saved {table_path}")
    print(f"Saved {tests_csv}")
    print(f"Saved {tests_json}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
