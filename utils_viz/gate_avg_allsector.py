"""
Visualize all-component decomposition for sector/digit conditioned trans_ih.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all-component decomposition figure.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/gate_avg_allsector",
        help="Directory containing exported avg_outer/trans files.",
    )
    parser.add_argument(
        "--conn_dir",
        type=str,
        default="./results/anal_data/whh",
        help="Directory containing sorted_npz_order.npy.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/gate_avg_allsector",
        help="Output directory for figures.",
    )
    parser.add_argument("--sector", type=int, default=None, choices=list(range(9)), help="Sector mode: target sector index (0-8).")
    parser.add_argument("--digit", type=int, default=None, choices=list(range(10)), help="Digit mode: target foreground digit (0-9).")
    parser.add_argument(
        "--agg",
        type=str,
        default="space",
        choices=["space", "feature"],
        help="Aggregation mode (space or feature).",
    )
    parser.add_argument(
        "--unit_tick_step",
        type=int,
        default=0,
        help="Tick step for hidden-unit axis; 0=auto.",
    )
    parser.add_argument(
        "--vmax_components",
        type=float,
        default=None,
        help="Symmetric color limit for 9 components + sum panel (default: auto abs-max).",
    )
    parser.add_argument(
        "--vmax_full",
        type=float,
        default=None,
        help="Symmetric color limit for full trans_ih panel (default: auto abs-max).",
    )
    parser.add_argument(
        "--use_cnn_channel_order",
        action="store_true",
        default=True,
        help=(
            "When set and --agg feature, reorder the 32 feature-channel rows "
            "using --channel_order_path (digit or sector mode)."
        ),
    )
    parser.add_argument(
        "--channel_order_path",
        type=str,
        default="./results/anal_data/cnn_channel/channel_order_by_cosine_similarity.npy",
        help="Path to channel_order_by_cosine_similarity.npy.",
    )
    return parser.parse_args()


def _draw_sector_hlines(ax, sector: int) -> None:
    sr, sc = sector // 3, sector % 3
    g1_start = sr * 12 + sc * 2
    g2_start = g1_start + 6
    groups = [(g1_start, g1_start + 1), (g2_start, g2_start + 1)]
    kw = dict(color="red", linewidth=0.7, linestyle="-", alpha=0.9)
    for first, last in groups:
        ax.axhline(y=first - 0.5, **kw)
        ax.axhline(y=last + 0.5, **kw)


def _draw_boundaries(ax, boundaries: np.ndarray) -> None:
    n_internal = len(boundaries) - 1
    for i, pos in enumerate(boundaries[1:], start=1):
        if pos == 0 or pos == boundaries[-1]:
            continue
        is_tuned_boundary = i == n_internal
        lw = 1.2 if is_tuned_boundary else 0.7
        ls = "--" if is_tuned_boundary else "-"
        ax.axvline(x=pos - 0.5, color="red", linewidth=lw, linestyle=ls, alpha=0.9)


def _load_channel_order(path: str, num_channels: int) -> np.ndarray:
    default_order = np.arange(num_channels, dtype=np.int64)
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        print(f"[viz][warn] channel order file not found: {abs_path}; use default order.")
        return default_order
    try:
        order = np.load(abs_path).astype(np.int64)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz][warn] failed to load channel order ({exc}); use default order.")
        return default_order
    if order.ndim != 1 or order.size != num_channels:
        print(
            f"[viz][warn] invalid channel order shape {order.shape} for C={num_channels}; "
            "use default order."
        )
        return default_order
    return order


def main() -> None:
    args = parse_args()
    if (args.sector is None) == (args.digit is None):
        raise ValueError("Specify exactly one of --sector or --digit.")
    if args.sector is not None:
        mode = "sector"
        selected_idx = int(args.sector)
        comp_label = "Sector component"
    else:
        mode = "digit"
        selected_idx = int(args.digit)
        comp_label = "Digit component"

    data_dir = os.path.join(os.path.abspath(args.data_dir), mode)
    conn_dir = os.path.abspath(args.conn_dir)
    save_dir = os.path.join(os.path.abspath(args.save_dir), mode)
    os.makedirs(save_dir, exist_ok=True)

    tag = f"{mode}{selected_idx}_{args.agg}"
    all_path = os.path.join(data_dir, f"avg_outer_ih_allcomp_{tag}.npy")
    sum_path = os.path.join(data_dir, f"avg_outer_ih_sumcomp_{tag}.npy")
    full_path = os.path.join(data_dir, f"avg_trans_ih_full_{tag}.npy")
    meta_path = os.path.join(data_dir, f"avg_gate_meta_allcomp_{tag}.json")
    ord_path = os.path.join(conn_dir, "sorted_npz_order.npy")
    bounds_path = os.path.join(data_dir, "digit_boundaries.npy")

    for p in (all_path, sum_path, full_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    outer_all = np.load(all_path).astype(np.float32)   # (9, rows, H)
    outer_sum = np.load(sum_path).astype(np.float32)   # (rows, H)
    trans_full = np.load(full_path).astype(np.float32) # (rows, H)

    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    sorted_npz_order = None
    if os.path.isfile(ord_path):
        sorted_npz_order = np.load(ord_path).astype(np.int64)

    digit_boundaries = None
    if os.path.isfile(bounds_path):
        digit_boundaries = np.load(bounds_path).astype(np.int64)

    if sorted_npz_order is not None:
        idx = sorted_npz_order
        outer_all = outer_all[:, :, idx]
        outer_sum = outer_sum[:, idx]
        trans_full = trans_full[:, idx]

    # Optional row reorder: feature agg → 32 channel rows; cosine-sim channel order.
    if args.agg == "feature" and args.use_cnn_channel_order:
        row_order = _load_channel_order(args.channel_order_path, outer_sum.shape[0])
        outer_all = outer_all[:, row_order, :]
        outer_sum = outer_sum[row_order, :]
        trans_full = trans_full[row_order, :]
    else:
        row_order = None

    n_rows, H = outer_sum.shape
    if args.unit_tick_step <= 0:
        unit_tick_step = max(1, min(32, H // 16))
    else:
        unit_tick_step = args.unit_tick_step
    h_ticks = list(range(0, H, unit_tick_step))
    if H - 1 not in h_ticks:
        h_ticks.append(H - 1)
    if sorted_npz_order is not None:
        h_tick_labels = [str(int(sorted_npz_order[i])) for i in h_ticks]
    else:
        h_tick_labels = [str(i) for i in h_ticks]

    row_step = max(1, n_rows // 8)
    r_ticks = list(range(0, n_rows, row_step))
    if n_rows - 1 not in r_ticks:
        r_ticks.append(n_rows - 1)
    if row_order is not None:
        r_tick_labels = [str(int(row_order[i])) for i in r_ticks]
    else:
        r_tick_labels = [str(i) for i in r_ticks]

    if args.vmax_components is None:
        vmax_components = float(
            max(
                np.abs(outer_all).max(),
                np.abs(outer_sum).max(),
                1e-8,
            )
        )
    else:
        vmax_components = float(args.vmax_components)

    if args.vmax_full is None:
        vmax_full = float(max(np.abs(trans_full).max(), 1e-8))
    else:
        vmax_full = float(args.vmax_full)

    row_label = "Spatial position (row×col)" if args.agg == "space" else "Feature channel"
    agg_desc = (
        f"mean over feature channels -> {n_rows} spatial"
        if args.agg == "space"
        else f"mean over 6x6 spatial -> {n_rows} feature channels"
    )
    n_frames = meta.get("n_frames", "?")
    n_samples = meta.get("n_samples", "?")

    fig_w = max(6.0, min(10.0, 6.0 * (H / 256.0)))
    fig_h = max(3.0, min(6.0, 3.0 * (n_rows / 36.0)))
    n_comp = int(outer_all.shape[0])
    n_panels = n_comp + 2
    n_cols = 3
    n_rows_fig = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(fig_w * n_cols + 2.5, fig_h * n_rows_fig + 1.5))
    axes = np.atleast_2d(axes)

    _cbar_kw = {"pad": 0.02, "fraction": 0.046}
    _imshow_kw = dict(origin="upper", interpolation="nearest", aspect="auto")

    def _set_axes(ax):
        ax.set_xticks(h_ticks)
        ax.set_xticklabels(h_tick_labels, rotation=45, ha="right")
        ax.set_yticks(r_ticks)
        ax.set_yticklabels(r_tick_labels)
        ax.set_xlabel("Hidden unit (npz row index)")
        ax.set_ylabel(row_label)

    for s in range(n_comp):
        r, c = divmod(s, n_cols)
        ax = axes[r, c]
        im = ax.imshow(
            outer_all[s],
            **_imshow_kw,
            cmap="RdBu_r",
            vmin=-vmax_components,
            vmax=vmax_components,
        )
        ax.set_title(f"{comp_label} {s}")
        _set_axes(ax)
        if digit_boundaries is not None:
            _draw_boundaries(ax, digit_boundaries)
        if args.agg == "space" and mode == "sector":
            _draw_sector_hlines(ax, selected_idx)
        fig.colorbar(im, ax=ax, **_cbar_kw)

    # Next panel: sum over selected components
    r, c = divmod(n_comp, n_cols)
    ax = axes[r, c]
    im = ax.imshow(
        outer_sum,
        **_imshow_kw,
        cmap="RdBu_r",
        vmin=-vmax_components,
        vmax=vmax_components,
    )
    ax.set_title(f"Sum over {n_comp} {mode} components")
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries)
    if args.agg == "space" and mode == "sector":
        _draw_sector_hlines(ax, selected_idx)
    fig.colorbar(im, ax=ax, **_cbar_kw)

    # Next panel: full trans_ih
    r, c = divmod(n_comp + 1, n_cols)
    ax = axes[r, c]
    im = ax.imshow(
        trans_full,
        **_imshow_kw,
        cmap="RdBu_r",
        vmin=-vmax_full,
        vmax=vmax_full,
    )
    ax.set_title("Full trans_ih (all feedback dims)")
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries)
    if args.agg == "space" and mode == "sector":
        _draw_sector_hlines(ax, selected_idx)
    fig.colorbar(im, ax=ax, **_cbar_kw)

    for k in range(n_panels, n_rows_fig * n_cols):
        rr, cc = divmod(k, n_cols)
        axes[rr, cc].axis("off")

    fig.suptitle(
        f"GaWF all-component decomposition (selected {mode}={selected_idx}, agg={args.agg})\n"
        f"{agg_desc}  |  n_frames={n_frames}, n_samples={n_samples}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(save_dir, f"{mode}{selected_idx}_{args.agg}_avg_gate_allcomp.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

