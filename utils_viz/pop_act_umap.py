"""
Load export_pop_act output: pop_act.npy + labels.tsv (UMAP/PCA), or pop_act_dpca.npy (dPCA scatter).

Reduce (T, D) -> (T, 3) via UMAP or PCA; save Plotly HTML under ``<save_dir>/<run_tag>/``.

PCA mode also saves explained-variance bar chart. ``--reducer dpca`` saves one matplotlib PNG (1×2 panels) only.

Reducer implementations: utils_viz.dimred_reducer (UMAPReducer, PCAReducer); dPCA is numpy SVD (no extra package).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
        help="UMAP / PCA (3D Plotly) or dPCA (2D PNG from pop_act_dpca.npy in --pop_act_dir).",
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
        # default="./results/anal_data/pop_act/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model",
        # default="./results/anal_data/pop_act/rnn_sector_acc_h275_lr0.0005_wd0.0001_do0_model",
        # default="./results/anal_data/pop_act/lstm_sector_acc_h80_lr0.0005_wd0.0001_do0_model",
        default="./results/anal_data/pop_act/gru_sector_acc_h105_lr0.0005_wd0.0001_do0_model",
        help="Directory with pop_act.npy and labels.tsv (typically export_pop_act: <save_dir>/<run_tag>/).",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/pop_act_umap",
        help="Parent directory for figures; writes <save_dir>/<run_tag>/ (HTML + PCA bar PNG if applicable).",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Subfolder under --save_dir (default: basename of --pop_act_dir).",
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
            "Output HTML filename inside <save_dir>/<run_tag>/. "
            "Default: trajectory_<color_by>.html (umap) or trajectory_pca_<color_by>.html (pca)."
        ),
    )
    return p.parse_args()


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
        if r.fieldnames is None or "fg_char_x" not in r.fieldnames or "fg_char_y" not in r.fieldnames:
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


def run_dpca(X_dpca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    X_dpca: (D, 10, 9) condition-averaged pop act (may contain NaN for empty cells).

    Returns W_digit (D, 2) and W_sector (D, 2): first two left singular vectors of the
    digit- and sector-marginalized matrices (*condensed* dPCA / marginalization + SVD).
    """
    X = np.asarray(X_dpca, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (10, 9):
        raise ValueError(f"run_dpca expects (D, 10, 9), got {X.shape}")

    X_digit = np.nanmean(X, axis=2)
    X_digit -= X_digit.mean(axis=1, keepdims=True)
    U_d, _, _ = np.linalg.svd(X_digit, full_matrices=False)
    n_d = min(2, U_d.shape[1])
    W_digit = np.zeros((U_d.shape[0], 2), dtype=np.float64)
    W_digit[:, :n_d] = U_d[:, :n_d]

    X_sector = np.nanmean(X, axis=1)
    X_sector -= X_sector.mean(axis=1, keepdims=True)
    U_s, _, _ = np.linalg.svd(X_sector, full_matrices=False)
    n_s = min(2, U_s.shape[1])
    W_sector = np.zeros((U_s.shape[0], 2), dtype=np.float64)
    W_sector[:, :n_s] = U_s[:, :n_s]

    return W_digit.astype(np.float32), W_sector.astype(np.float32)


def save_dpca_scatter(
    coords_90x2: np.ndarray,
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

    c = np.asarray(coords_90x2, dtype=np.float64)
    if c.shape != (90, 2):
        raise ValueError(f"coords must be (90, 2), got {c.shape}")
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
    X_flat: np.ndarray,
    digit_labels: np.ndarray,
    sector_labels: np.ndarray,
    W_digit: np.ndarray,
    W_sector: np.ndarray,
    out_dir: str,
) -> None:
    """
    2×2 panels: digit-PC plane vs sector-PC plane × digit vs sector coloring (orthogonality check).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Xf = np.asarray(X_flat, dtype=np.float64)
    if Xf.shape[0] != 90:
        raise ValueError(f"X_flat must be (90, D), got {Xf.shape}")
    Wd = np.asarray(W_digit, dtype=np.float64)
    Ws = np.asarray(W_sector, dtype=np.float64)
    if Wd.shape[1] != 2 or Ws.shape[1] != 2:
        raise ValueError(f"W_digit / W_sector must have 2 columns, got {Wd.shape}, {Ws.shape}")

    digit_labels = np.asarray(digit_labels, dtype=np.int64).reshape(90)
    sector_labels = np.asarray(sector_labels, dtype=np.int64).reshape(90)

    coords_digit = Xf @ Wd
    coords_sector = Xf @ Ws

    def _sym_lim(c: np.ndarray) -> tuple[float, float]:
        m = float(np.max(np.abs(c))) if c.size else 0.0
        if m < 1e-30:
            m = 1e-12
        pad = m * 0.12
        return -m - pad, m + pad

    lo_d, hi_d = _sym_lim(coords_digit)
    lo_s, hi_s = _sym_lim(coords_sector)

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

    def _decorate(ax, xlabel: str, ylabel: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
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
    _draw_points(axes[0, 0], coords_digit, by_digit=True)
    _decorate(axes[0, 0], "dPC digit-1", "dPC digit-2", (lo_d, hi_d), (lo_d, hi_d))

    axes[0, 1].set_title("[expected: mixed if orthogonal]", fontsize=9)
    _draw_points(axes[0, 1], coords_digit, by_digit=False)
    _decorate(axes[0, 1], "dPC digit-1", "dPC digit-2", (lo_d, hi_d), (lo_d, hi_d))

    axes[1, 0].set_title("[expected: mixed if orthogonal]", fontsize=9)
    _draw_points(axes[1, 0], coords_sector, by_digit=True)
    _decorate(axes[1, 0], "dPC sector-1", "dPC sector-2", (lo_s, hi_s), (lo_s, hi_s))

    axes[1, 1].set_title("[expected: clustered]", fontsize=9)
    _draw_points(axes[1, 1], coords_sector, by_digit=False)
    _decorate(axes[1, 1], "dPC sector-1", "dPC sector-2", (lo_s, hi_s), (lo_s, hi_s))

    fig.suptitle("dPCA orthogonality check: digit vs sector axes", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    out_path = os.path.join(out_dir, "dpca_2x2_orthogonality.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    out_dir = os.path.join(args.save_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.reducer == "dpca":
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

        W_digit, W_sector = run_dpca(X_dpca)
        D = X_dpca.shape[0]
        X_flat = np.nan_to_num(X_dpca.reshape(D, 90).T.astype(np.float64), nan=0.0)
        W = np.stack([W_digit[:, 0].astype(np.float64), W_sector[:, 0].astype(np.float64)], axis=1)
        coords = (X_flat @ W).astype(np.float32)

        digit_labels = np.arange(90, dtype=np.int64) // 9
        sector_labels = np.arange(90, dtype=np.int64) % 9
        save_dpca_scatter(coords, digit_labels, sector_labels, out_dir)
        save_dpca_2x2_orthogonality(
            X_flat, digit_labels, sector_labels, W_digit, W_sector, out_dir
        )
        print(f"Saved {os.path.join(out_dir, 'dpca_scatter.png')}")
        print(f"Saved {os.path.join(out_dir, 'dpca_2x2_orthogonality.png')}")
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
    out_html = os.path.join(out_dir, html_name)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved {out_html}")

    if args.reducer == "pca" and pca_ratios is not None:
        bar_path = os.path.join(out_dir, "pca_explained_variance_bars.png")
        save_pca_explained_variance_bar_chart(bar_path, pca_ratios, max_components=args.pca_variance_bars)
        print(f"Saved {bar_path}")


if __name__ == "__main__":
    main()
