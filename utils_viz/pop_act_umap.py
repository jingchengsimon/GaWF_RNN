"""
Load export_pop_act output: pop_act.npy + labels.tsv (UMAP/PCA), or pop_act_dpca.npy (dPCA).

Reduce (T, D) -> (T, 3) via UMAP or PCA; save figures directly under ``<save_dir>``.

PCA mode also saves explained-variance bar chart. ``--reducer dpca`` keeps PNG/HTML under the
figure tree and saves JSON/NPZ under the matching analysis-data tree. Interactive dPCA views use
90 condition points with small visual-only offsets while preserving raw coordinates.

Reducer implementations: utils_viz.dimred_reducer (UMAPReducer, PCAReducer); dPCA RRR uses
the optional ``dpca`` PyPI package (machenslab/dPCA).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils_anal.anal_paths import output_dir

from utils_viz.dimred_reducer import PCAReducer, UMAPReducer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D trajectory for pop_act export (UMAP or PCA + optional variance bar chart). "
    )
    p.add_argument(
        "--reducer",
        type=str,
        default="umap",
        choices=["umap", "pca", "dpca"],
        help=(
            "UMAP / PCA (3D Plotly) or dPCA (2D PNG + interactive 3D HTML from "
            "pop_act_dpca.npy in --pop_act_dir)."
        ),
    )
    p.add_argument(
        "--dpca_method",
        type=str,
        default="rrr",
        choices=["rrr"],
        help=(
            "dPCA method when --reducer dpca. Production figures use the official imported "
            "reduced-rank-regression implementation (rrr); condensed SVD remains internal "
            "as a diagnostic reference only."
        ),
    )
    p.add_argument(
        "--dpca_regularizer",
        type=float,
        default=1e-4,
        help=(
            "Fixed official dPCA regularizer for --dpca_method rrr. Do not use 'auto' here "
            "because pop_act_dpca contains condition means, not trial-by-trial data."
        ),
    )
    p.add_argument(
        "--dpca_components",
        type=int,
        default=10,
        help=(
            "When --reducer dpca: number of dPCs to fit/save for explained variance. "
            "At least three are always fit so the interactive 3D output has dPC1-dPC3."
        ),
    )
    p.add_argument(
        "--dpca_dodge_fraction",
        type=float,
        default=0.018,
        help=(
            "Visual-only spacing for overlapping dPCA condition points, as a fraction of the "
            "symmetric coordinate span (default 0.018; raw coordinates remain unchanged)."
        ),
    )
    p.add_argument(
        "--pca_variance_bars",
        type=int,
        default=20,
        help="When --reducer pca: number of PCs in the explained-variance bar chart (default 20).",
    )
    p.add_argument(
        "--color_by",
        type=str,
        default="digit",
        choices=["digit", "sector"],
        help="Marker color: fg digit (0-9) or 3x3 sector (0-8) from fg_char_x / fg_char_y.",
    )
    p.add_argument(
        "--frame_height",
        type=int,
        default=96,
        help="Stimulus height (pixels); must match data used for pop_act (sector mapping).",
    )
    p.add_argument(
        "--frame_width",
        type=int,
        default=96,
        help="Stimulus width (pixels); must match data used for pop_act.",
    )
    p.add_argument(
        "--num_sectors",
        type=int,
        default=9,
        help="Sector count (default 9 = 3x3); must be a perfect square.",
    )
    p.add_argument(
        "--pop_act_dir",
        type=str,
        default=str(
            output_dir(
                "D_variance_decomposition",
                "export_pop_act",
                "data",
            ) / "gru_sector_acc_h105_lr0.0005_wd0.0001_do0_model"
        ),
        help=(
            "Directory with pop_act.npy and labels.tsv "
            "(typically export_pop_act: <save_dir>/<run_tag>/)."
        ),
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("D_variance_decomposition", "pop_act_umap", "figs")),
        help=(
            "Category-level directory for PNG/PDF/HTML figures; writes directly under "
            "<save_dir>. "
            "dPCA arrays/JSON are written to the matching anal_data tree."
        ),
    )
    p.add_argument(
        "--anal_data_dir",
        type=str,
        default="",
        help=(
            "Parent directory for dPCA arrays/JSON. Default: replace the canonical "
            "figs component in --save_dir with its sibling data component."
        ),
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Run tag used to identify the matching data directory and output names.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (UMAP random_state; passed to PCA for reproducibility where used).",
    )
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument(
        "--out_html",
        type=str,
        default="",
        help=(
            "Output HTML filename inside <save_dir>. "
            "Default: trajectory_<color_by>.html (umap) or trajectory_pca_<color_by>.html (pca)."
        ),
    )
    return p.parse_args()


def resolve_dpca_output_dirs(
    save_dir: str,
    anal_data_dir: str,
    run_tag: str,
) -> tuple[str, str]:
    """Resolve matching figure/data run directories for one dPCA invocation."""
    fig_parent = os.path.normpath(save_dir)
    if anal_data_dir.strip():
        data_parent = os.path.normpath(anal_data_dir.strip())
    else:
        parts = fig_parent.split(os.sep)
        matching = [idx for idx, part in enumerate(parts) if part == "anal_figs"]
        if matching:
            index = matching[-1]
            if len(parts) <= index + 1:
                raise ValueError("Canonical anal_figs path is missing its category component")
            category = parts[index + 1]
            script_name = parts[index + 2] if len(parts) > index + 2 else "pop_act_umap"
            fig_parent = os.sep.join(parts[: index + 2]) or os.sep
            data_parent = os.sep.join(
                [*parts[:index], "anal_data", category, script_name]
            ) or os.sep
            data_path = os.path.join(data_parent, run_tag) if run_tag else data_parent
            return fig_parent, data_path
        matching = [idx for idx, part in enumerate(parts) if part == "figs"]
        if not matching:
            raise ValueError(
                "Cannot derive --anal_data_dir because --save_dir has neither an "
                "'anal_figs' nor a legacy 'figs' path component; pass --anal_data_dir explicitly."
            )
        parts[matching[-1]] = "data"
        data_parent = os.sep.join(parts) or os.sep
    return os.path.join(fig_parent, run_tag), os.path.join(data_parent, run_tag)


def save_pca_explained_variance_bar_chart(
    out_path: str,
    explained_var_ratio: np.ndarray,
    max_components: int = 20,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(int(max_components), int(len(explained_var_ratio)))
    if k < 1:
        raise ValueError("No PCA components to plot")
    ratios = explained_var_ratio[:k]
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x, ratios * 100.0, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"PCA — explained variance (first {k} components)")
    ax.set_xticks(x)
    ax.set_xlim(0.5, k + 0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dpca_explained_variance_bars(
    out_dir: str,
    summary: dict,
    max_components: int = 10,
) -> None:
    """Save dPCA explained-variance bars for digit and sector components."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = summary.get("explained_variance_ratio", {})
    panels = [
        ("d", "Digit dPC explained variance", "#4472C4"),
        ("s", "Sector dPC explained variance", "#ED7D31"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)

    max_percent = 0.0
    values_by_key: dict[str, np.ndarray] = {}
    for key, _, _ in panels:
        vals = np.asarray(ev.get(key, []), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        vals = vals[: max(1, int(max_components))]
        values_by_key[key] = vals
        if vals.size:
            max_percent = max(max_percent, float(np.max(vals) * 100.0))
    ylim_top = max(5.0, max_percent * 1.25)

    for ax, (key, title, color) in zip(axes, panels):
        vals = values_by_key[key]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No components", ha="center", va="center")
            ax.set_xticks([])
            ax.set_ylim(0.0, ylim_top)
            ax.set_title(title, fontsize=11)
            continue

        x = np.arange(vals.size)
        perc = vals * 100.0
        bars = ax.bar(x, perc, color=color, alpha=0.9, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels([f"dPC{i + 1}" for i in x])
        ax.set_ylim(0.0, ylim_top)
        ax.set_title(f"{title}\ntotal={float(np.sum(perc)):.2f}%", fontsize=11)
        ax.grid(axis="y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, perc):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ylim_top * 0.025,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Explained variance (%)")
    method = summary.get("method", "dpca")
    fig.suptitle(f"dPCA explained variance ({method}; first {max_components} dPCs)", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpca_explained_variance_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_path}")


def sector_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    num_sectors: int = 9,
) -> np.ndarray:
    """
    Same 3x3 mapping as MC_RNN_Dataset (train_model.py): sector in 0..num_sectors-1.
    """
    grid_size = int(np.sqrt(num_sectors))
    if grid_size * grid_size != num_sectors:
        raise ValueError(f"num_sectors={num_sectors} must be a perfect square")
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    col = np.clip((xf / max(width - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
    row = np.clip((yf / max(height - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
    return (row * grid_size + col).astype(np.int64)


def load_fg_char_ids(labels_tsv: str) -> np.ndarray:
    """Read fg_char_id column (int per frame, same length as T)."""
    fg: list[int] = []
    with open(labels_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames is None or "fg_char_id" not in r.fieldnames:
            raise ValueError(f"labels.tsv must have fg_char_id column, got {r.fieldnames}")
        for row in r:
            fg.append(int(float(row["fg_char_id"])))
    return np.asarray(fg, dtype=np.int64)


def load_xy_for_sector(labels_tsv: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    with open(labels_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        has_xy = (
            r.fieldnames is not None and "fg_char_x" in r.fieldnames and "fg_char_y" in r.fieldnames
        )
        if not has_xy:
            raise ValueError(f"labels.tsv must have fg_char_x, fg_char_y; got {r.fieldnames}")
        for row in r:
            xs.append(float(row["fg_char_x"]))
            ys.append(float(row["fg_char_y"]))
    return np.asarray(xs), np.asarray(ys)


def load_color_array(
    labels_tsv: str,
    color_by: str,
    frame_h: int,
    frame_w: int,
    num_sectors: int,
) -> tuple[np.ndarray, str, list[str], float, float]:
    """
    Returns:
        cat_float: shape (T,), values in [0..n_cat-1] for Plotly color
        cbar_title: colorbar title string
        colors: list of hex length n_cat
        cmin, cmax: Plotly color limits (half-open band centers: -0.5 .. n_cat-0.5 so each
            integer class maps to one equal color strip on the bar; avoids N ticks -> N-1 bands).
    """
    if color_by == "digit":
        v = load_fg_char_ids(labels_tsv)
        v = np.clip(v.astype(np.float64), 0.0, 9.0)
        colors = list(DIGIT_COLORS[:10])
        n = len(colors)
        return v, "fg digit", colors, -0.5, float(n) - 0.5
    if color_by == "sector":
        x, y = load_xy_for_sector(labels_tsv)
        v = sector_from_xy(x, y, frame_h, frame_w, num_sectors).astype(np.float64)
        n = num_sectors
        return (
            v,
            "fg sector (3x3)",
            list(SECTOR_COLORS[:n]),
            -0.5,
            float(n) - 0.5,
        )
    raise ValueError(f"Unknown color_by={color_by!r}")


DIGIT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabed4",
]

# 9-class palette (distinct from digit list)
SECTOR_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


def _load_official_dpca_class():
    try:
        from dPCA.dPCA import dPCA
    except ImportError as e:
        raise ImportError(
            "Official dPCA RRR mode requires the PyPI package 'dpca'. "
            "Install it with: pip install dpca"
        ) from e
    return dPCA


def _counts_from_dir(pop_act_dir: str) -> np.ndarray | None:
    candidates = [
        os.path.join(pop_act_dir, "pop_act_digitxsector_counts.npy"),
        os.path.join(pop_act_dir, "pop_act_counts.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            counts = np.load(path)
            if counts.shape != (10, 9):
                raise ValueError(f"{path} must have shape (10, 9), got {counts.shape}")
            return counts.astype(np.int64, copy=False)
    return None


def fill_nan_digit_sector_cells(
    X_dpca: np.ndarray,
    counts: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Fill empty or NaN digit×sector cells before official dPCA.

    ``X_dpca`` is (D, 10, 9), where D is hidden units. Official dPCA rejects NaNs. Empty
    cells are identified from counts when available, otherwise from any NaN in a cell. Each
    empty cell is imputed per feature with an additive estimate:
    ``digit_marginal[d] + sector_marginal[s] - grand_mean``. If either marginal is undefined
    for a feature, that feature falls back to its grand mean. The function prints how many
    condition cells/elements were imputed so missing data never silently becomes zero.
    """
    X = np.asarray(X_dpca, dtype=np.float64).copy()
    if X.ndim != 3 or X.shape[1:] != (10, 9):
        raise ValueError(f"X_dpca must be (D, 10, 9), got {X.shape}")

    nan_mask = np.isnan(X)
    if counts is not None:
        empty_cells = counts <= 0
        if empty_cells.shape != (10, 9):
            raise ValueError(f"counts must have shape (10, 9), got {empty_cells.shape}")
    else:
        empty_cells = np.any(nan_mask, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        digit_mean = np.nanmean(X, axis=2)
        sector_mean = np.nanmean(X, axis=1)
        grand = np.nanmean(X, axis=(1, 2))
    grand = np.where(np.isnan(grand), 0.0, grand)

    element_fill_count = int(np.sum(nan_mask))
    cell_fill_count = int(np.sum(empty_cells))
    for d in range(10):
        for s in range(9):
            needs_fill = bool(empty_cells[d, s]) or bool(np.any(np.isnan(X[:, d, s])))
            if not needs_fill:
                continue
            estimate = digit_mean[:, d] + sector_mean[:, s] - grand
            estimate = np.where(np.isnan(estimate), grand, estimate)
            elem_mask = np.isnan(X[:, d, s])
            if bool(empty_cells[d, s]):
                X[:, d, s] = estimate
            elif np.any(elem_mask):
                X[elem_mask, d, s] = estimate[elem_mask]

    remaining_nan = int(np.isnan(X).sum())
    if remaining_nan:
        raise ValueError(f"NaN imputation failed; remaining NaN elements={remaining_nan}")

    print(
        "[dPCA] NaN/empty-cell imputation: "
        f"cells={cell_fill_count}, nan_elements={element_fill_count}, "
        f"counts_available={counts is not None}"
    )
    return X.astype(np.float64, copy=False), {
        "empty_cell_count": cell_fill_count,
        "nan_element_count": element_fill_count,
        "counts_available": counts is not None,
    }


def _flatten_component_scores(scores: np.ndarray, n_components: int = 3) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (10, 9):
        raise ValueError(f"dPCA scores must be (components, 10, 9), got {arr.shape}")
    n_components = max(1, int(n_components))
    out = np.zeros((90, n_components), dtype=np.float64)
    n = min(n_components, arr.shape[0])
    for comp in range(n):
        out[:, comp] = arr[comp].reshape(90)
    return out.astype(np.float32)


def _ensure_component_vectors(
    vectors: np.ndarray,
    n_features: int,
    n_components: int = 3,
) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n_features:
        raise ValueError(
            f"Axis vectors must have shape ({n_features}, n_components), got {arr.shape}"
        )
    n_components = max(1, int(n_components))
    out = np.zeros((n_features, n_components), dtype=np.float64)
    n = min(n_components, arr.shape[1])
    out[:, :n] = arr[:, :n]
    return out


def _variance_ratio_for_json(explained: dict, max_components: int = 10) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key in ("d", "s", "ds"):
        vals = np.asarray(explained.get(key, []), dtype=np.float64)
        out[key] = [float(v) for v in vals[: max(1, int(max_components))]]
    return out


def run_dpca_condensed(X_dpca: np.ndarray, *, n_components: int = 10) -> dict:
    """
    X_dpca: (D, 10, 9) condition-averaged pop act (may contain NaN for empty cells).

    Legacy condensed baseline: first marginalize over sector/digit, then take left singular
    vectors for each marginalization. This does not include the official dPCA
    reduced-rank-regression whitening/regularization step. Static scatter diagnostics still use
    the first two axes, while interactive output uses the first three and explained-variance
    output keeps up to ``n_components`` entries.
    """
    X = np.asarray(X_dpca, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (10, 9):
        raise ValueError(f"run_dpca_condensed expects (D, 10, 9), got {X.shape}")

    X_digit = np.nanmean(X, axis=2)
    X_digit -= X_digit.mean(axis=1, keepdims=True)
    U_d, S_d, _ = np.linalg.svd(X_digit, full_matrices=False)
    n_plot_components = 3
    n_d = min(n_plot_components, U_d.shape[1])
    W_digit = np.zeros((U_d.shape[0], n_plot_components), dtype=np.float64)
    W_digit[:, :n_d] = U_d[:, :n_d]

    X_sector = np.nanmean(X, axis=1)
    X_sector -= X_sector.mean(axis=1, keepdims=True)
    U_s, S_s, _ = np.linalg.svd(X_sector, full_matrices=False)
    n_s = min(n_plot_components, U_s.shape[1])
    W_sector = np.zeros((U_s.shape[0], n_plot_components), dtype=np.float64)
    W_sector[:, :n_s] = U_s[:, :n_s]

    D = X.shape[0]
    X_flat = np.nan_to_num(X.reshape(D, 90).T.astype(np.float64), nan=0.0)
    coords_digit = (X_flat @ W_digit).astype(np.float32)
    coords_sector = (X_flat @ W_sector).astype(np.float32)
    denom_d = float(np.sum(S_d**2)) or 1.0
    denom_s = float(np.sum(S_s**2)) or 1.0
    k = max(1, int(n_components))
    explained = {
        "d": [float((s**2) / denom_d) for s in S_d[:k]],
        "s": [float((s**2) / denom_s) for s in S_s[:k]],
        "ds": [],
    }
    return {
        "method": "condensed",
        "coords_digit": coords_digit,
        "coords_sector": coords_sector,
        "explained_variance_ratio": explained,
        "digit_axes": W_digit,
        "sector_axes": W_sector,
        "imputation": {
            "empty_cell_count": int(np.any(np.isnan(X), axis=0).sum()),
            "nan_element_count": int(np.isnan(X).sum()),
            "counts_available": False,
        },
    }


def run_dpca_rrr(
    X_dpca: np.ndarray,
    *,
    regularizer: float,
    n_components: int = 10,
    counts: np.ndarray | None = None,
) -> dict:
    """
    Official machenslab/dPCA reduced-rank-regression dPCA for (D, digit, sector) means.

    Empty or NaN cells are explicitly imputed with
    ``digit_marginal[d] + sector_marginal[s] - grand_mean`` (falling back to grand mean when
    a marginal is unavailable) before fitting, because official dPCA does not accept NaNs.
    The fixed ``regularizer`` is used directly; ``regularizer='auto'`` is deliberately not
    used because these inputs are condition means rather than trial-by-trial data.
    """
    dPCA = _load_official_dpca_class()
    X, imputation = fill_nan_digit_sector_cells(X_dpca, counts=counts)
    n_components = max(3, int(n_components))
    print(f"[dPCA] method=rrr regularizer={regularizer:g} n_components={n_components}")
    dpca = dPCA(labels="ds", n_components=n_components, regularizer=float(regularizer))
    Z = dpca.fit_transform(X)
    for key in ("d", "s", "ds"):
        if key not in Z:
            raise KeyError(f"Official dPCA result missing marginalization {key!r}; got {list(Z)}")
    return {
        "method": "rrr",
        "coords_digit": _flatten_component_scores(Z["d"], n_components=3),
        "coords_sector": _flatten_component_scores(Z["s"], n_components=3),
        "explained_variance_ratio": _variance_ratio_for_json(
            dpca.explained_variance_ratio_,
            max_components=n_components,
        ),
        "digit_axes": _ensure_component_vectors(dpca.D["d"], X.shape[0], n_components=3),
        "sector_axes": _ensure_component_vectors(dpca.D["s"], X.shape[0], n_components=3),
        "imputation": imputation,
        "regularizer": float(regularizer),
    }


def run_dpca(
    X_dpca: np.ndarray,
    *,
    method: str,
    regularizer: float,
    n_components: int = 10,
    counts: np.ndarray | None = None,
) -> dict:
    if method == "rrr":
        return run_dpca_rrr(
            X_dpca,
            regularizer=regularizer,
            n_components=n_components,
            counts=counts,
        )
    if method == "condensed":
        return run_dpca_condensed(X_dpca, n_components=n_components)
    raise ValueError(f"Unknown dPCA method {method!r}")


def _principal_angles_deg(A: np.ndarray, B: np.ndarray) -> list[float]:
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(v, -1.0, 1.0)))) for v in sv]


def _first_axis_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A, dtype=np.float64)[:, 0]
    b = np.asarray(B, dtype=np.float64)[:, 0]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-30:
        return 90.0
    cos = abs(float(np.dot(a, b) / denom))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _group_between_variance_ratio(coords: np.ndarray, labels: np.ndarray) -> float:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy[:, np.newaxis]
    labels = np.asarray(labels, dtype=np.int64)
    grand = xy.mean(axis=0)
    total = float(np.sum((xy - grand) ** 2))
    if total <= 1e-30:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = xy[labels == label]
        diff = group.mean(axis=0) - grand
        between += float(group.shape[0] * np.sum(diff**2))
    return between / total


def summarize_dpca_result(
    result: dict,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
) -> dict:
    digit_axes = np.asarray(result["digit_axes"], dtype=np.float64)
    sector_axes = np.asarray(result["sector_axes"], dtype=np.float64)
    coords_digit_2d = np.asarray(result["coords_digit"], dtype=np.float64)[:, :2]
    coords_sector_2d = np.asarray(result["coords_sector"], dtype=np.float64)[:, :2]
    summary = {
        "method": result["method"],
        "regularizer": result.get("regularizer"),
        "first_axis_angle_deg": _first_axis_angle_deg(digit_axes[:, :2], sector_axes[:, :2]),
        "subspace_angles_deg": _principal_angles_deg(digit_axes[:, :2], sector_axes[:, :2]),
        "sector_leakage_on_digit_plane": _group_between_variance_ratio(
            coords_digit_2d, sector_labels
        ),
        "digit_leakage_on_sector_plane": _group_between_variance_ratio(
            coords_sector_2d, digit_labels
        ),
        "explained_variance_ratio": result["explained_variance_ratio"],
        "imputation": result.get("imputation", {}),
    }
    return summary


def print_dpca_summary(summary: dict) -> None:
    ev = summary["explained_variance_ratio"]
    ev_text = ", ".join(
        f"{key}={sum(ev.get(key, [])) * 100.0:.2f}% "
        f"({', '.join(f'{v * 100.0:.2f}' for v in ev.get(key, []))})"
        for key in ("d", "s", "ds")
    )
    print(
        f"[dPCA:{summary['method']}] first-axis angle="
        f"{summary['first_axis_angle_deg']:.2f} deg; subspace angles="
        f"{[round(v, 2) for v in summary['subspace_angles_deg']]}"
    )
    print(
        f"[dPCA:{summary['method']}] leakage sector-on-digit="
        f"{summary['sector_leakage_on_digit_plane']:.4f}, digit-on-sector="
        f"{summary['digit_leakage_on_sector_plane']:.4f}"
    )
    print(f"[dPCA:{summary['method']}] explained variance ratio: {ev_text}")


def save_dpca_variance_json(data_dir: str, payload: dict) -> str:
    """Save method metadata under anal_data without a method suffix."""
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "dpca_variance.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")
    return out_path


def save_dpca_scatter(
    coords_digit: np.ndarray,
    coords_sector: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    out_dir: str,
) -> None:
    """
    Single 1×2 figure: left = color by digit (0–9), right = by sector (0–8);
    shared axis limits and zero lines; saved as dpca_scatter.png.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c_digit = np.asarray(coords_digit, dtype=np.float64)
    c_sector = np.asarray(coords_sector, dtype=np.float64)
    if c_digit.ndim != 2 or c_digit.shape[0] != 90 or c_digit.shape[1] < 1:
        raise ValueError(f"digit coords must be (90, >=1), got {c_digit.shape}")
    if c_sector.ndim != 2 or c_sector.shape[0] != 90 or c_sector.shape[1] < 1:
        raise ValueError(f"sector coords must be (90, >=1), got {c_sector.shape}")
    c = np.stack([c_digit[:, 0], c_sector[:, 0]], axis=1)
    digit_labels = np.asarray(digit_labels, dtype=np.int64).reshape(90)
    sector_labels = np.asarray(sector_labels, dtype=np.int64).reshape(90)

    vx = np.max(np.abs(c[:, 0]))
    vy = np.max(np.abs(c[:, 1]))
    m = float(max(vx, vy, 1e-12))
    pad = m * 0.12
    lo, hi = -m - pad, m + pad

    def _decorate(ax) -> None:
        ax.axhline(0.0, color="0.55", linewidth=0.9, zorder=1)
        ax.axvline(0.0, color="0.55", linewidth=0.9, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("dPC digit-1")
        ax.set_ylabel("dPC sector-1")
        ax.grid(True, alpha=0.22)

    os.makedirs(out_dir, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.8, 6.2))

    for d in range(10):
        mask = digit_labels == d
        ax0.scatter(
            c[mask, 0],
            c[mask, 1],
            c=DIGIT_COLORS[d],
            s=48,
            edgecolors="0.35",
            linewidths=0.35,
            zorder=2,
        )
    digit_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(d),
            markerfacecolor=DIGIT_COLORS[d],
            markeredgecolor="0.35",
            markersize=9,
        )
        for d in range(10)
    ]
    ax0.legend(handles=digit_legend, title="digit", fontsize=8, title_fontsize=9)
    ax0.set_title("By digit", fontsize=11)
    _decorate(ax0)

    for s in range(9):
        mask = sector_labels == s
        ax1.scatter(
            c[mask, 0],
            c[mask, 1],
            c=SECTOR_COLORS[s],
            s=48,
            edgecolors="0.35",
            linewidths=0.35,
            zorder=2,
        )
    sector_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(s),
            markerfacecolor=SECTOR_COLORS[s],
            markeredgecolor="0.35",
            markersize=9,
        )
        for s in range(9)
    ]
    ax1.legend(handles=sector_legend, title="sector", fontsize=8, title_fontsize=9)
    ax1.set_title("By sector", fontsize=11)
    _decorate(ax1)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "dpca_scatter.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dpca_2x2_orthogonality(
    coords_digit_90xn: np.ndarray,
    coords_sector_90xn: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    out_dir: str,
    filename: str = "dpca_2x2_orthogonality.png",
    dodge_conditions: bool = False,
) -> None:
    """
    2×2 panels: digit-PC vs sector-PC plane, each colored by digit or sector.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    coords_digit = np.asarray(coords_digit_90xn, dtype=np.float64)
    coords_sector = np.asarray(coords_sector_90xn, dtype=np.float64)
    if (
        coords_digit.ndim != 2
        or coords_sector.ndim != 2
        or coords_digit.shape[0] != 90
        or coords_sector.shape[0] != 90
        or coords_digit.shape[1] < 2
        or coords_sector.shape[1] < 2
    ):
        raise ValueError(
            f"coords_digit / coords_sector must be (90, >=2), got "
            f"{coords_digit.shape} and {coords_sector.shape}"
        )
    coords_digit = coords_digit[:, :2]
    coords_sector = coords_sector[:, :2]
    digit_labels = np.asarray(digit_labels, dtype=np.int64).reshape(90)
    sector_labels = np.asarray(sector_labels, dtype=np.int64).reshape(90)

    def _sym_lim(c: np.ndarray) -> tuple[float, float]:
        m = float(np.max(np.abs(c))) if c.size else 0.0
        if m < 1e-30:
            m = 1e-12
        pad = m * 0.12
        return -m - pad, m + pad

    lo_d, hi_d = _sym_lim(coords_digit)
    lo_s, hi_s = _sym_lim(coords_sector)

    def _dodged(xy: np.ndarray, labels: np.ndarray, lim: tuple[float, float]) -> np.ndarray:
        if not dodge_conditions:
            return xy
        vals = sorted(int(v) for v in np.unique(labels))
        n_cols = int(np.ceil(np.sqrt(len(vals))))
        n_rows = int(np.ceil(len(vals) / n_cols))
        span = max(float(lim[1] - lim[0]), 1e-12)
        step = span * 0.018
        offsets = {}
        for idx, val in enumerate(vals):
            col = idx % n_cols
            row = idx // n_cols
            offsets[val] = np.array(
                [
                    (col - (n_cols - 1) / 2.0) * step,
                    (row - (n_rows - 1) / 2.0) * step,
                ],
                dtype=np.float64,
            )
        out = xy.copy()
        for idx, val in enumerate(labels.astype(np.int64)):
            out[idx] += offsets[int(val)]
        return out

    coords_digit_plot = _dodged(coords_digit, sector_labels, (lo_d, hi_d))
    coords_sector_plot = _dodged(coords_sector, digit_labels, (lo_s, hi_s))

    digit_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(d),
            markerfacecolor=DIGIT_COLORS[d],
            markeredgecolor="0.35",
            markersize=7,
        )
        for d in range(10)
    ]
    sector_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=str(s),
            markerfacecolor=SECTOR_COLORS[s],
            markeredgecolor="0.35",
            markersize=7,
        )
        for s in range(9)
    ]

    legend_kw = dict(
        fontsize=6,
        title_fontsize=7,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    def _draw_points(ax, xy: np.ndarray, by_digit: bool) -> None:
        if by_digit:
            for d in range(10):
                m = digit_labels == d
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    c=DIGIT_COLORS[d],
                    s=36,
                    edgecolors="0.35",
                    linewidths=0.3,
                    zorder=2,
                )
            ax.legend(handles=digit_handles, title="digit", **legend_kw)
        else:
            for s in range(9):
                m = sector_labels == s
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    c=SECTOR_COLORS[s],
                    s=36,
                    edgecolors="0.35",
                    linewidths=0.3,
                    zorder=2,
                )
            ax.legend(handles=sector_handles, title="sector", **legend_kw)

    def _decorate(
        ax,
        xlabel: str,
        ylabel: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        ax.axhline(0.0, color="0.5", linewidth=0.85, linestyle="--", zorder=1)
        ax.axvline(0.0, color="0.5", linewidth=0.85, linestyle="--", zorder=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex="row", sharey="row")

    axes[0, 0].set_title("[expected: clustered]", fontsize=9)
    _draw_points(axes[0, 0], coords_digit_plot, by_digit=True)
    _decorate(axes[0, 0], "dPC digit-1", "dPC digit-2", (lo_d, hi_d), (lo_d, hi_d))

    axes[0, 1].set_title("[expected: mixed if orthogonal]", fontsize=9)
    _draw_points(axes[0, 1], coords_digit_plot, by_digit=False)
    _decorate(axes[0, 1], "dPC digit-1", "dPC digit-2", (lo_d, hi_d), (lo_d, hi_d))

    axes[1, 0].set_title("[expected: mixed if orthogonal]", fontsize=9)
    _draw_points(axes[1, 0], coords_sector_plot, by_digit=True)
    _decorate(axes[1, 0], "dPC sector-1", "dPC sector-2", (lo_s, hi_s), (lo_s, hi_s))

    axes[1, 1].set_title("[expected: clustered]", fontsize=9)
    _draw_points(axes[1, 1], coords_sector_plot, by_digit=False)
    _decorate(axes[1, 1], "dPC sector-1", "dPC sector-2", (lo_s, hi_s), (lo_s, hi_s))

    title = "dPCA orthogonality check: digit vs sector axes"
    if dodge_conditions:
        title += " (condition-dodged)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prepare_dpca_3d_inputs(
    coords_digit: np.ndarray,
    coords_sector: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digit_xyz = np.asarray(coords_digit, dtype=np.float32)
    sector_xyz = np.asarray(coords_sector, dtype=np.float32)
    if digit_xyz.ndim != 2 or digit_xyz.shape[0] != 90 or digit_xyz.shape[1] < 3:
        raise ValueError(f"digit coords must be (90, >=3), got {digit_xyz.shape}")
    if sector_xyz.ndim != 2 or sector_xyz.shape[0] != 90 or sector_xyz.shape[1] < 3:
        raise ValueError(f"sector coords must be (90, >=3), got {sector_xyz.shape}")
    digit_ids = np.asarray(digit_labels, dtype=np.int64).reshape(90)
    sector_ids = np.asarray(sector_labels, dtype=np.int64).reshape(90)
    return digit_xyz[:, :3], sector_xyz[:, :3], digit_ids, sector_ids


def _dodge_condition_points(
    coordinates: np.ndarray,
    offset_labels: np.ndarray,
    dodge_fraction: float,
) -> np.ndarray:
    """Apply the existing 90-point x/y grid dodge without altering raw dPC coordinates."""
    xyz = np.asarray(coordinates, dtype=np.float32)
    labels = np.asarray(offset_labels, dtype=np.int64).reshape(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"coordinates must be (N, 3), got {xyz.shape}")
    if not np.isfinite(dodge_fraction) or dodge_fraction < 0.0:
        raise ValueError(f"dodge_fraction must be finite and non-negative, got {dodge_fraction}")
    if dodge_fraction == 0.0:
        return xyz.copy()

    values = sorted(int(value) for value in np.unique(labels))
    n_cols = int(np.ceil(np.sqrt(len(values))))
    n_rows = int(np.ceil(len(values) / n_cols))
    span = max(float(np.max(np.abs(xyz))) * 2.0, 1e-12)
    step = span * float(dodge_fraction)
    offsets: dict[int, np.ndarray] = {}
    for idx, value in enumerate(values):
        col = idx % n_cols
        row = idx // n_cols
        offsets[value] = np.asarray(
            [
                (col - (n_cols - 1) / 2.0) * step,
                (row - (n_rows - 1) / 2.0) * step,
                0.0,
            ],
            dtype=np.float32,
        )

    plotted = xyz.copy()
    for idx, value in enumerate(labels):
        plotted[idx] += offsets[int(value)]
    return plotted.astype(np.float32, copy=False)


def build_dpca_3d_plot_coordinates(
    coords_digit: np.ndarray,
    coords_sector: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    dodge_fraction: float = 0.018,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw and visually dodged dPCA coordinates plus stable condition labels."""
    digit_xyz, sector_xyz, digit_ids, sector_ids = _prepare_dpca_3d_inputs(
        coords_digit,
        coords_sector,
        digit_labels,
        sector_labels,
    )
    digit_plot = _dodge_condition_points(digit_xyz, sector_ids, dodge_fraction)
    sector_plot = _dodge_condition_points(sector_xyz, digit_ids, dodge_fraction)
    return digit_xyz, sector_xyz, digit_plot, sector_plot, digit_ids, sector_ids


def save_dpca_3d_coordinates(
    coords_digit: np.ndarray,
    coords_sector: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    data_dir: str,
    fig_dir: str,
    method: str,
    source_path: str,
    dodge_fraction: float = 0.018,
) -> tuple[str, str]:
    """Save reusable dPC1-dPC3 coordinates and JSON provenance metadata."""
    digit_xyz, sector_xyz, digit_plot, sector_plot, digit_ids, sector_ids = (
        build_dpca_3d_plot_coordinates(
            coords_digit,
            coords_sector,
            digit_labels,
            sector_labels,
            dodge_fraction=dodge_fraction,
        )
    )
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "dpca_3d_coordinates.npz")
    meta_path = os.path.join(data_dir, "dpca_3d_coordinates_meta.json")
    html_name = "dpca_3d_interactive.html"
    np.savez_compressed(
        data_path,
        coords_digit=digit_xyz.astype(np.float32, copy=False),
        coords_sector=sector_xyz.astype(np.float32, copy=False),
        coords_digit_plot=digit_plot.astype(np.float32, copy=False),
        coords_sector_plot=sector_plot.astype(np.float32, copy=False),
        digit_labels=digit_ids.astype(np.int64, copy=False),
        sector_labels=sector_ids.astype(np.int64, copy=False),
    )
    metadata = {
        "method": method,
        "source": os.path.abspath(source_path),
        "coordinate_file": os.path.basename(data_path),
        "interactive_html": os.path.abspath(os.path.join(fig_dir, html_name)),
        "coords_digit": {
            "shape": list(digit_xyz.shape),
            "dtype": "float32",
            "axes": ["digit dPC1", "digit dPC2", "digit dPC3"],
        },
        "coords_sector": {
            "shape": list(sector_xyz.shape),
            "dtype": "float32",
            "axes": ["sector dPC1", "sector dPC2", "sector dPC3"],
        },
        "plot_coordinates": {
            "digit_key": "coords_digit_plot",
            "sector_key": "coords_sector_plot",
            "dtype": "float32",
            "visual_dodge_only": True,
            "dodge_fraction": float(dodge_fraction),
            "digit_space_offset_by": "sector",
            "sector_space_offset_by": "digit",
            "offset_axes": ["dPC1", "dPC2"],
        },
        "digit_labels": {"shape": list(digit_ids.shape), "dtype": "int64"},
        "sector_labels": {"shape": list(sector_ids.shape), "dtype": "int64"},
        "condition_order": "digit-major: flat_index = digit * 9 + sector",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {data_path}")
    print(f"Saved {meta_path}")
    return data_path, meta_path


def save_dpca_3d_interactive_html(
    coords_digit: np.ndarray,
    coords_sector: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    fig_dir: str,
    method: str,
    dodge_fraction: float = 0.018,
) -> str:
    """Save an offline Plotly HTML with four interactive dPC1-dPC3 condition views."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "Interactive dPCA output requires Plotly; install the viz extra: "
            "pip install -e '.[viz]'"
        ) from e

    digit_xyz, sector_xyz, digit_plot, sector_plot, digit_ids, sector_ids = (
        build_dpca_3d_plot_coordinates(
            coords_digit,
            coords_sector,
            digit_labels,
            sector_labels,
            dodge_fraction=dodge_fraction,
        )
    )
    subplot_titles = (
        "Digit dPC space — colored by digit",
        "Digit dPC space — colored by sector",
        "Sector dPC space — colored by digit",
        "Sector dPC space — colored by sector",
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.19,
        vertical_spacing=0.08,
    )
    panels = [
        (digit_plot, digit_xyz, "digit", digit_ids, DIGIT_COLORS, "digit", 1, 1, "legend"),
        (
            digit_plot,
            digit_xyz,
            "digit",
            sector_ids,
            SECTOR_COLORS,
            "sector",
            1,
            2,
            "legend2",
        ),
        (
            sector_plot,
            sector_xyz,
            "sector",
            digit_ids,
            DIGIT_COLORS,
            "digit",
            2,
            1,
            "legend3",
        ),
        (
            sector_plot,
            sector_xyz,
            "sector",
            sector_ids,
            SECTOR_COLORS,
            "sector",
            2,
            2,
            "legend4",
        ),
    ]
    for xyz, raw_xyz, space, color_ids, palette, color_name, row, col, legend_id in panels:
        customdata = np.column_stack([digit_ids, sector_ids, raw_xyz]).astype(
            np.float64,
            copy=False,
        )
        for class_id, color in enumerate(palette):
            mask = color_ids == class_id
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[mask, 0],
                    y=xyz[mask, 1],
                    z=xyz[mask, 2],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": color,
                        "opacity": 0.88,
                        "line": {"color": "rgba(50,50,50,0.65)", "width": 0.7},
                    },
                    customdata=customdata[mask],
                    name=str(class_id),
                    legend=legend_id,
                    legendgroup=f"{row}-{col}-{color_name}-{class_id}",
                    showlegend=True,
                    hovertemplate=(
                        "digit=%{customdata[0]:.0f}<br>sector=%{customdata[1]:.0f}<br>"
                        f"raw {space} dPC1=%{{customdata[2]:.4f}}<br>"
                        f"raw {space} dPC2=%{{customdata[3]:.4f}}<br>"
                        f"raw {space} dPC3=%{{customdata[4]:.4f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    for row, space, xyz in ((1, "digit", digit_plot), (2, "sector", sector_plot)):
        limit = max(float(np.max(np.abs(xyz))) * 1.08, 1e-12)
        axis_common = {
            "range": [-limit, limit],
            "zeroline": True,
            "zerolinecolor": "rgba(80,80,80,0.55)",
            "gridcolor": "rgba(160,160,160,0.28)",
        }
        for col in (1, 2):
            fig.update_scenes(
                xaxis={**axis_common, "title": f"{space} dPC1"},
                yaxis={**axis_common, "title": f"{space} dPC2"},
                zaxis={**axis_common, "title": f"{space} dPC3"},
                aspectmode="cube",
                dragmode="orbit",
                row=row,
                col=col,
            )

    method_label = "official dPCA" if method == "rrr" else "condensed SVD baseline"
    legend_common = {
        "orientation": "v",
        "itemsizing": "constant",
        "font": {"size": 9},
        "bgcolor": "rgba(255,255,255,0.82)",
        "bordercolor": "rgba(120,120,120,0.45)",
        "borderwidth": 1,
        "xanchor": "left",
        "yanchor": "top",
    }
    fig.update_layout(
        title=(
            f"Interactive dPCA condition geometry ({method_label}; 90 points)"
            "<br><sup>Small x/y offsets separate overlaps; hover reports raw dPC values. "
            "Drag to rotate, shift-drag to pan, and scroll to zoom.</sup>"
        ),
        width=1780,
        height=1080,
        margin={"l": 10, "r": 150, "t": 105, "b": 10},
        legend={**legend_common, "title": {"text": "digit"}, "x": 0.41, "y": 0.98},
        legend2={**legend_common, "title": {"text": "sector"}, "x": 1.005, "y": 0.98},
        legend3={**legend_common, "title": {"text": "digit"}, "x": 0.41, "y": 0.44},
        legend4={**legend_common, "title": {"text": "sector"}, "x": 1.005, "y": 0.44},
    )
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "dpca_3d_interactive.html")
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "responsive": True, "displaylogo": False},
    )
    print(f"Saved {out_path}")
    return out_path


def discrete_equal_bins_colorscale(colors: list[str]) -> list[list]:
    """
    n equal-height strips on [0, 1]. Use with cmin=-0.5, cmax=n-0.5 so class v maps to strip v.
    """
    n = len(colors)
    if n == 0:
        raise ValueError("colors must be non-empty")
    if n == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    scale: list[list] = []
    for k in range(n):
        lo = k / n
        hi = (k + 1) / n
        c = colors[k]
        scale.append([lo, c])
        scale.append([hi, c])
    return scale


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag.strip() or os.path.basename(os.path.normpath(args.pop_act_dir))
    fig_dir = args.save_dir
    os.makedirs(fig_dir, exist_ok=True)

    if args.reducer == "dpca":
        fig_dir, data_dir = resolve_dpca_output_dirs(
            args.save_dir,
            args.anal_data_dir,
            run_tag,
        )
        os.makedirs(data_dir, exist_ok=True)
        candidates = [
            os.path.join(args.pop_act_dir, "pop_act_dpca.npy"),
            os.path.join(args.pop_act_dir, "pop_act_digitxsector_mean.npy"),
            os.path.join(args.pop_act_dir, "pop_act_digit_sector_mean.npy"),
        ]
        primary = next((p for p in candidates if os.path.isfile(p)), "")
        if not primary:
            raise FileNotFoundError(
                f"pop_act_dpca.npy not found under {args.pop_act_dir} "
                f"(tried alternate aggregation filenames too)."
            )
        X_dpca = np.load(primary)
        if X_dpca.ndim != 3 or X_dpca.shape[1:] != (10, 9):
            raise ValueError(f"pop_act_dpca must be (D, 10, 9), got {X_dpca.shape}")

        counts = _counts_from_dir(args.pop_act_dir)
        digit_labels = np.arange(90, dtype=np.int64) // 9
        sector_labels = np.arange(90, dtype=np.int64) % 9

        condensed = run_dpca_condensed(X_dpca, n_components=args.dpca_components)
        condensed_summary = summarize_dpca_result(condensed, digit_labels, sector_labels)
        print_dpca_summary(condensed_summary)

        stability_summary = None
        if args.dpca_method == "rrr":
            if float(args.dpca_regularizer) != 0.0:
                rrr_zero = run_dpca_rrr(
                    X_dpca,
                    regularizer=0.0,
                    n_components=args.dpca_components,
                    counts=counts,
                )
                stability_summary = summarize_dpca_result(rrr_zero, digit_labels, sector_labels)
                print_dpca_summary(stability_summary)
            result = run_dpca(
                X_dpca,
                method="rrr",
                regularizer=float(args.dpca_regularizer),
                n_components=args.dpca_components,
                counts=counts,
            )
        else:
            result = condensed

        summary = summarize_dpca_result(result, digit_labels, sector_labels)
        print_dpca_summary(summary)
        if args.dpca_method == "rrr":
            delta_angle = (
                summary["first_axis_angle_deg"] - condensed_summary["first_axis_angle_deg"]
            )
            delta_leak = (
                summary["sector_leakage_on_digit_plane"]
                - condensed_summary["sector_leakage_on_digit_plane"]
            )
            print(
                "[dPCA compare] rrr - condensed: "
                f"first-axis angle delta={delta_angle:.2f} deg, "
                f"sector-on-digit leakage delta={delta_leak:.4f}"
            )

        save_dpca_scatter(
            result["coords_digit"],
            result["coords_sector"],
            digit_labels,
            sector_labels,
            fig_dir,
        )
        save_dpca_2x2_orthogonality(
            result["coords_digit"],
            result["coords_sector"],
            digit_labels,
            sector_labels,
            fig_dir,
        )
        save_dpca_2x2_orthogonality(
            result["coords_digit"],
            result["coords_sector"],
            digit_labels,
            sector_labels,
            fig_dir,
            filename="dpca_2x2_orthogonality_90points.png",
            dodge_conditions=True,
        )
        coordinate_path, coordinate_meta_path = save_dpca_3d_coordinates(
            result["coords_digit"],
            result["coords_sector"],
            digit_labels,
            sector_labels,
            data_dir,
            fig_dir,
            method=args.dpca_method,
            source_path=primary,
            dodge_fraction=float(args.dpca_dodge_fraction),
        )
        interactive_html_path = save_dpca_3d_interactive_html(
            result["coords_digit"],
            result["coords_sector"],
            digit_labels,
            sector_labels,
            fig_dir,
            method=args.dpca_method,
            dodge_fraction=float(args.dpca_dodge_fraction),
        )
        payload = {
            "method": args.dpca_method,
            "pop_act_dpca": os.path.abspath(primary),
            "regularizer": float(args.dpca_regularizer) if args.dpca_method == "rrr" else None,
            "selected": summary,
            "condensed_reference": condensed_summary,
            "rrr_zero_regularizer_reference": stability_summary,
            "interactive_3d": {
                "html": os.path.abspath(interactive_html_path),
                "coordinates": os.path.abspath(coordinate_path),
                "metadata": os.path.abspath(coordinate_meta_path),
            },
        }
        save_dpca_variance_json(data_dir, payload)
        save_dpca_explained_variance_bars(
            fig_dir,
            summary,
            max_components=args.dpca_components,
        )
        print(f"Saved {os.path.join(fig_dir, 'dpca_scatter.png')}")
        print(f"Saved {os.path.join(fig_dir, 'dpca_2x2_orthogonality.png')}")
        print(f"Saved {os.path.join(fig_dir, 'dpca_2x2_orthogonality_90points.png')}")
        return

    pop_path = os.path.join(args.pop_act_dir, "pop_act.npy")
    lbl_path = os.path.join(args.pop_act_dir, "labels.tsv")
    if not os.path.isfile(pop_path):
        raise FileNotFoundError(pop_path)
    if not os.path.isfile(lbl_path):
        raise FileNotFoundError(lbl_path)

    X = np.load(pop_path)
    if X.ndim != 2:
        raise ValueError(f"Expected pop_act (T, D), got {X.shape}")
    T, _ = X.shape

    cat_float, cbar_title, cat_colors, cmin, cmax = load_color_array(
        lbl_path,
        args.color_by,
        args.frame_height,
        args.frame_width,
        args.num_sectors,
    )
    if cat_float.shape[0] != T:
        raise ValueError(f"labels rows {cat_float.shape[0]} != pop_act T={T}")

    Xf = X.astype(np.float32, copy=False)
    pca_ratios: np.ndarray | None = None
    if args.reducer == "umap":
        n_neighbors = min(args.n_neighbors, max(2, T - 1))
        reducer = UMAPReducer(
            n_components=3,
            random_state=args.seed,
            n_neighbors=n_neighbors,
            min_dist=args.min_dist,
        )
        xyz = reducer.fit_transform(Xf)
        method_label = "UMAP"
        axis_labels = ("dim1", "dim2", "dim3")
    elif args.reducer == "pca":
        reducer = PCAReducer(
            n_components=3,
            fit_max_components=max(3, args.pca_variance_bars),
            random_state=args.seed,
        )
        xyz = reducer.fit_transform(Xf)
        pca_ratios = reducer.explained_variance_ratio_
        method_label = "PCA"
        axis_labels = ("PC1", "PC2", "PC3")
    else:
        raise ValueError(f"Unexpected reducer {args.reducer!r} (handled above)")

    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly is required: pip install plotly") from e

    line_x, line_y, line_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    tickvals = list(range(len(cat_colors)))
    ticktext = [str(i) for i in tickvals]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(color="rgba(80,80,80,0.25)", width=2),
            name="trajectory",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="markers",
            marker=dict(
                size=3,
                color=cat_float,
                colorscale=discrete_equal_bins_colorscale(cat_colors),
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=cbar_title),
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    len=0.75,
                    thickness=18,
                ),
                showscale=True,
            ),
            name="frames",
            showlegend=False,
        )
    )
    subt = f"marker color = {cbar_title}"
    fig.update_layout(
        title=f"Population activity 3D embedding ({method_label})<br><sub>{subt}</sub>",
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if args.out_html.strip():
        html_name = args.out_html.strip()
    elif args.reducer == "pca":
        html_name = f"trajectory_pca_{args.color_by}.html"
    else:
        html_name = f"trajectory_{args.color_by}.html"
    out_html = os.path.join(fig_dir, html_name)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved {out_html}")

    if args.reducer == "pca" and pca_ratios is not None:
        bar_path = os.path.join(fig_dir, "pca_explained_variance_bars.png")
        save_pca_explained_variance_bar_chart(
            bar_path,
            pca_ratios,
            max_components=args.pca_variance_bars,
        )
        print(f"Saved {bar_path}")


if __name__ == "__main__":
    main()
