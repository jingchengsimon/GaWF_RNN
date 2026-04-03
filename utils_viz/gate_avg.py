"""
Gate-average visualization — digit mode and sector mode.

Digit + hh  (--digit D  without --agg):
  4 panels, all (H × H), units ordered by panel-4 (digit groups 0–9 + untuned tail):
    Panel 1 — avg U[:,D]·fb[D]·V_hh[D,:]          rank-1 outer product (no sigmoid)
    Panel 2 — avg gate_hh                           sigmoid gate, colorbar [0, vmax_gate]
    Panel 3 — avg gate_hh ⊙ W_hh                   gate-modulated connection
    Panel 4 — W_hh                                  raw static connection

Digit + ih  (--digit D  --agg {space|feature}):
  Same layout as sector (input_agg × H); data from avg_gate_ih_d{D}_{agg}.npy.
  For agg=feature, pass --use_cnn_channel_order to reorder rows like utils_viz/V_basis.py
  (channel_order_by_cosine_similarity.npy); default is encoder channel index order.

Sector mode  (--sector S  --agg {space|feature}, --agg required):
  4 panels, all (input_agg × H); agg=feature + optional --use_cnn_channel_order as above.
    input_agg = 36 spatial positions  (agg=space,   mean over 32 feature channels)
              = 32 feature channels   (agg=feature, mean over 6×6 spatial grid)
    Panel 1 — avg U[:,nc+S]·fb[nc+S]·V_ih[nc+S,:]  rank-1 outer (aggregated, no sigmoid)
    Panel 2 — avg gate_ih (aggregated)               sigmoid gate, colorbar [0, vmax_gate]
    Panel 3 — avg gate_ih_agg ⊙ W_ih_agg            gate-modulated
    Panel 4 — W_ih_agg                               raw static weights
  Hidden-unit axis (columns) optionally reordered by sorted_npz_order from conn_dir.
  Digit-group boundaries drawn as vertical lines on the hidden axis.
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
    parser = argparse.ArgumentParser(
        description=(
            "Plot avg gate panels for a fg digit or sector.\n"
            "Exactly one of --digit or --sector.  --agg required with --sector; "
            "with --digit, omit for hh or set space|feature for ih."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/gate_avg",
        help="Directory containing avg_gate_* files (from export_gate_avg.py).",
    )
    parser.add_argument(
        "--conn_dir",
        type=str,
        default="./results/anal_data/whh",
        help="Directory containing weight_hh.npy and sorted_npz_order.npy.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/gate_avg",
        help=(
            "Output root. Digit: <save_dir>/digit/hh (no --agg) or .../digit/ih (--agg). "
            "Sector: <save_dir>/sector/."
        ),
    )
    # --- mode selection ---
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=list(range(10)),
        help=(
            "Digit mode: fg digit. Without --agg: hh panels (avg_gate_hh_{digit}.npy). "
            "With --agg: ih panels (avg_gate_ih_d{digit}_{agg}.npy)."
        ),
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        choices=list(range(9)),
        help="Sector mode: index 0-8 (requires --agg; avg_gate_ih_s{S}_{agg}.npy).",
    )
    parser.add_argument(
        "--agg",
        type=str,
        default=None,
        choices=["space", "feature"],
        help=(
            "Required with --sector. With --digit: omit for hh; set for ih aggregation (space|feature)."
        ),
    )
    parser.add_argument(
        "--unit_tick_step",
        type=int,
        default=0,
        help="Tick step for hidden-unit axis; 0 = auto.",
    )
    parser.add_argument(
        "--vmax_gate",
        type=float,
        default=1.0,
        help="Upper color limit for gate panel (default: 1.0).",
    )
    parser.add_argument(
        "--vmax_w",
        type=float,
        default=None,
        help="Symmetric color limit for modulated/raw weight panels (default: shared abs-max).",
    )
    parser.add_argument(
        "--tuned_only",
        action="store_true",
        help="Digit mode only — hh or ih: show only tuned hidden units (uses n_tuned.npy).",
    )
    parser.add_argument(
        "--use_cnn_channel_order",
        action="store_true",
        default=True,
        help=(
            "ih + agg=feature only: reorder matrix rows by CNN activation channel order "
            "from --channel_order_path (same idea as utils_viz/V_basis.py --use_cnn_channel_order). "
            "Default: encoder channel order."
        ),
    )
    parser.add_argument(
        "--channel_order_path",
        type=str,
        default="./results/anal_data/cnn_channel/channel_order_by_cosine_similarity.npy",
        help="Used when --use_cnn_channel_order is set; if missing, encoder row order is kept.",
    )
    parser.add_argument(
        "--align_outer_cbar_with_allcomp",
        action="store_true",
        default=True,
        help=(
            "ih panel-1 only: align outer colorbar range with all-component decomposition "
            "(same vmax rule as utils_viz/gate_avg_allsector.py)."
        ),
    )
    parser.add_argument(
        "--allcomp_data_dir",
        type=str,
        default="./results/anal_data/gate_avg_allsector",
        help="Data dir used with --align_outer_cbar_with_allcomp.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Digit-mode data loading
# ---------------------------------------------------------------------------

def load_digit_data(
    avg_gate_dir: str,
    connection_matrix_dir: str,
    fg_digit: int,
    tuned_only: bool,
):
    gate_path  = os.path.join(avg_gate_dir, f"avg_gate_hh_{fg_digit}.npy")
    outer_path = os.path.join(avg_gate_dir, f"avg_outer_hh_{fg_digit}.npy")
    whh_path   = os.path.join(connection_matrix_dir, "weight_hh.npy")
    ord_path   = os.path.join(connection_matrix_dir, "sorted_npz_order.npy")

    for p in (gate_path, whh_path, ord_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run export_gate_avg.py / export_whh.py first."
            )

    avg_gate_hh      = np.load(gate_path).astype(np.float32)   # (H, H)
    W_hh             = np.load(whh_path).astype(np.float32)    # (H, H)
    sorted_npz_order = np.load(ord_path).astype(np.int64)      # (H,)

    avg_outer_hh: np.ndarray | None = None
    if os.path.isfile(outer_path):
        avg_outer_hh = np.load(outer_path).astype(np.float32)
    else:
        print(f"[viz][warn] avg_outer_hh_{fg_digit}.npy not found; outer panel skipped.")

    meta: dict = {}
    meta_path = os.path.join(avg_gate_dir, f"avg_gate_meta_{fg_digit}.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    digit_boundaries: np.ndarray | None = None
    bounds_path = os.path.join(avg_gate_dir, "digit_boundaries.npy")
    if os.path.isfile(bounds_path):
        digit_boundaries = np.load(bounds_path).astype(np.int64)
    else:
        print(f"[viz][warn] digit_boundaries.npy not found; boundary lines skipped.")

    if tuned_only:
        ntuned_path = os.path.join(connection_matrix_dir, "n_tuned.npy")
        if not os.path.isfile(ntuned_path):
            raise FileNotFoundError(f"n_tuned.npy not found in {connection_matrix_dir}.")
        n_tuned = int(np.load(ntuned_path))
        sorted_npz_order = sorted_npz_order[:n_tuned]
        if digit_boundaries is not None:
            digit_boundaries = np.clip(digit_boundaries[:11], 0, n_tuned)
        print(f"tuned_only: using {n_tuned} tuned units.")

    return avg_gate_hh, avg_outer_hh, W_hh, sorted_npz_order, meta, digit_boundaries


# ---------------------------------------------------------------------------
# Sector-mode data loading
# ---------------------------------------------------------------------------

def load_sector_data(
    avg_gate_dir: str,
    connection_matrix_dir: str,
    sector: int,
    agg: str,
):
    tag        = f"s{sector}_{agg}"
    gate_path  = os.path.join(avg_gate_dir, f"avg_gate_ih_{tag}.npy")
    outer_path = os.path.join(avg_gate_dir, f"avg_outer_ih_{tag}.npy")
    wih_path   = os.path.join(avg_gate_dir, f"weight_ih_{agg}.npy")
    ord_path   = os.path.join(connection_matrix_dir, "sorted_npz_order.npy")

    for p in (gate_path, wih_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run export_gate_avg.py --sector first."
            )

    # Shape: (input_agg, H)  e.g. (36, 256) or (32, 256)
    avg_gate_ih = np.load(gate_path).astype(np.float32)
    W_ih_agg    = np.load(wih_path).astype(np.float32)

    avg_outer_ih: np.ndarray | None = None
    if os.path.isfile(outer_path):
        avg_outer_ih = np.load(outer_path).astype(np.float32)
    else:
        print(f"[viz][warn] avg_outer_ih_{tag}.npy not found; outer panel skipped.")

    # sorted_npz_order for reordering the hidden axis (columns)
    sorted_npz_order: np.ndarray | None = None
    if os.path.isfile(ord_path):
        sorted_npz_order = np.load(ord_path).astype(np.int64)
    else:
        print(f"[viz][warn] sorted_npz_order.npy not found in '{connection_matrix_dir}'; "
              "hidden units will not be reordered.")

    meta: dict = {}
    meta_path = os.path.join(avg_gate_dir, f"avg_gate_meta_{tag}.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    digit_boundaries: np.ndarray | None = None
    bounds_path = os.path.join(avg_gate_dir, "digit_boundaries.npy")
    if os.path.isfile(bounds_path):
        digit_boundaries = np.load(bounds_path).astype(np.int64)
    else:
        print("[viz][warn] digit_boundaries.npy not found; boundary lines on hidden axis skipped.")

    return avg_gate_ih, avg_outer_ih, W_ih_agg, sorted_npz_order, meta, digit_boundaries


def load_digit_ih_data(
    avg_gate_dir: str,
    connection_matrix_dir: str,
    fg_digit: int,
    agg: str,
    tuned_only: bool,
):
    """Load digit-conditioned averaged ih gate (aggregated), same tensor layout as sector ih."""
    tag        = f"d{fg_digit}_{agg}"
    gate_path  = os.path.join(avg_gate_dir, f"avg_gate_ih_{tag}.npy")
    outer_path = os.path.join(avg_gate_dir, f"avg_outer_ih_{tag}.npy")
    wih_path   = os.path.join(avg_gate_dir, f"weight_ih_{agg}.npy")
    ord_path   = os.path.join(connection_matrix_dir, "sorted_npz_order.npy")

    for p in (gate_path, wih_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run export_gate_avg.py --digit D --agg {space|feature} first."
            )

    avg_gate_ih = np.load(gate_path).astype(np.float32)
    W_ih_agg    = np.load(wih_path).astype(np.float32)

    avg_outer_ih: np.ndarray | None = None
    if os.path.isfile(outer_path):
        avg_outer_ih = np.load(outer_path).astype(np.float32)
    else:
        print(f"[viz][warn] avg_outer_ih_{tag}.npy not found; outer panel skipped.")

    sorted_npz_order: np.ndarray | None = None
    if os.path.isfile(ord_path):
        sorted_npz_order = np.load(ord_path).astype(np.int64)
    else:
        print(f"[viz][warn] sorted_npz_order.npy not found in '{connection_matrix_dir}'; "
              "hidden units will not be reordered.")

    meta: dict = {}
    meta_path = os.path.join(avg_gate_dir, f"avg_gate_meta_{tag}.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    digit_boundaries: np.ndarray | None = None
    bounds_path = os.path.join(avg_gate_dir, "digit_boundaries.npy")
    if os.path.isfile(bounds_path):
        digit_boundaries = np.load(bounds_path).astype(np.int64)
    else:
        print("[viz][warn] digit_boundaries.npy not found; boundary lines on hidden axis skipped.")

    if tuned_only:
        if sorted_npz_order is None:
            raise FileNotFoundError(
                f"tuned_only requires sorted_npz_order.npy in {connection_matrix_dir}."
            )
        ntuned_path = os.path.join(connection_matrix_dir, "n_tuned.npy")
        if not os.path.isfile(ntuned_path):
            raise FileNotFoundError(f"n_tuned.npy not found in {connection_matrix_dir}.")
        n_tuned = int(np.load(ntuned_path))
        sorted_npz_order = sorted_npz_order[:n_tuned]
        if digit_boundaries is not None:
            digit_boundaries = np.clip(digit_boundaries[:11], 0, n_tuned)
        print(f"tuned_only: using {n_tuned} tuned units.")

    return avg_gate_ih, avg_outer_ih, W_ih_agg, sorted_npz_order, meta, digit_boundaries


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_cnn_feature_channel_order(path: str, num_channels: int) -> np.ndarray | None:
    """
    Load a permutation of [0..C-1] from .npy (same file as V_basis.py).
    Returns None if file missing or shape mismatch — caller keeps encoder row order.
    """
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        print(
            f"[viz][warn] CNN channel order file not found at '{abs_path}'; "
            "keeping encoder feature-channel row order."
        )
        return None
    try:
        order = np.load(abs_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz][warn] failed to load channel order ({exc}); keeping encoder order.")
        return None
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or order.size != num_channels:
        print(
            f"[viz][warn] channel order shape {order.shape} != ({num_channels},); "
            "keeping encoder order."
        )
        return None
    return order


def maybe_reorder_ih_feature_rows_for_viz(
    agg: str,
    use_cnn_order: bool,
    channel_order_path: str,
    gate_agg: np.ndarray,
    W_agg: np.ndarray,
    gated_W_agg: np.ndarray,
    outer_agg: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool, np.ndarray | None]:
    """
    For agg=='feature' and use_cnn_order, reorder matrix rows with the same rule
    used by gate_avg_allsector.py: apply_order = channel_order (no extra reverse).
    Returns (gate, W, gated_W, outer, applied, row_order_shown).
    """
    if agg != "feature" or not use_cnn_order:
        return gate_agg, W_agg, gated_W_agg, outer_agg, False, None
    n_row = gate_agg.shape[0]
    ch_order = _load_cnn_feature_channel_order(channel_order_path, n_row)
    if ch_order is None:
        return gate_agg, W_agg, gated_W_agg, outer_agg, False, None
    print("[viz] ih feature rows: CNN activation channel order (see --channel_order_path).")
    apply_order = ch_order
    g = gate_agg[apply_order].copy()
    w = W_agg[apply_order].copy()
    gw = gated_W_agg[apply_order].copy()
    o = outer_agg[apply_order].copy() if outer_agg is not None else None
    return g, w, gw, o, True, apply_order


def _load_outer_vmax_from_allcomp(
    allcomp_data_dir: str,
    mode: str,
    selected_idx: int,
    agg: str,
) -> float | None:
    tag = f"{mode}{selected_idx}_{agg}"
    all_path = os.path.join(allcomp_data_dir, f"avg_outer_ih_allcomp_{tag}.npy")
    sum_path = os.path.join(allcomp_data_dir, f"avg_outer_ih_sumcomp_{tag}.npy")
    if not (os.path.isfile(all_path) and os.path.isfile(sum_path)):
        return None
    try:
        outer_all = np.load(all_path).astype(np.float32)
        outer_sum = np.load(sum_path).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz][warn] failed to load allcomp outer files for cbar alignment ({exc}).")
        return None
    return float(max(np.abs(outer_all).max(), np.abs(outer_sum).max(), 1e-8))


def reorder(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Reorder a (H, H) matrix from (target, source) PyTorch convention into
    (source, target) display orientation, then select and reorder by idx.
    """
    return mat.T[idx, :][:, idx].astype(np.float32)


def _draw_sector_hlines(ax, sector: int) -> None:
    """Draw horizontal lines above/below the two 2-row groups that correspond to
    the given sector in the flattened 36-index (6×6) spatial axis (space agg mode).

    Each sector covers a 2×2 block in the 6×6 grid, which flattens into two
    non-contiguous pairs of rows:
      group 1: rows  sr*12 + sc*2,  sr*12 + sc*2 + 1
      group 2: rows  sr*12 + sc*2 + 6,  sr*12 + sc*2 + 7
    where sr = sector // 3, sc = sector % 3.

    Four lines are drawn (top + bottom of each group), matching the digit-group
    line style used for vertical boundaries.
    """
    sr, sc = sector // 3, sector % 3
    g1_start = sr * 12 + sc * 2
    g2_start = g1_start + 6          # one full spatial row (6 cols) later
    groups   = [(g1_start, g1_start + 1), (g2_start, g2_start + 1)]
    kw = dict(color="red", linewidth=0.7, linestyle="-", alpha=0.9)
    for first, last in groups:
        ax.axhline(y=first - 0.5, **kw)
        ax.axhline(y=last  + 0.5, **kw)


def _draw_boundaries(
    ax,
    boundaries: np.ndarray,
    vlines_only: bool = False,
) -> None:
    """Draw digit-group separator lines.

    boundaries: shape (12,) — [0, end_d0, ..., end_d9, H].
    vlines_only=True: draw only vertical lines (for sector panels where rows ≠ hidden units).
    """
    n_internal = len(boundaries) - 1
    for i, pos in enumerate(boundaries[1:], start=1):
        if pos == 0 or pos == boundaries[-1]:
            continue
        is_tuned_boundary = (i == n_internal)
        lw = 1.2 if is_tuned_boundary else 0.7
        ls = "--" if is_tuned_boundary else "-"
        kw = dict(color="red", linewidth=lw, linestyle=ls, alpha=0.9)
        ax.axvline(x=pos - 0.5, **kw)
        if not vlines_only:
            ax.axhline(y=pos - 0.5, **kw)


# ---------------------------------------------------------------------------
# Digit-mode plotting
# ---------------------------------------------------------------------------

def plot_panels(
    outer_sorted: np.ndarray | None,
    gate_sorted: np.ndarray,
    gated_W_sorted: np.ndarray,
    W_sorted: np.ndarray,
    sorted_npz_order: np.ndarray,
    fg_digit: int,
    meta: dict,
    out_path: str,
    unit_tick_step: int,
    vmax_gate: float,
    vmax_w: float | None,
    digit_boundaries: np.ndarray | None = None,
) -> None:
    H = gate_sorted.shape[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if unit_tick_step <= 0:
        unit_tick_step = max(1, min(32, H // 16))
    ticks = list(range(0, H, unit_tick_step))
    if H - 1 not in ticks:
        ticks.append(H - 1)
    tick_labels = [str(int(sorted_npz_order[i])) for i in ticks]

    if vmax_w is None:
        vmax_w = float(max(np.abs(gated_W_sorted).max(), np.abs(W_sorted).max(), 1e-8))

    n_frames  = meta.get("n_frames", "?")
    n_samples = meta.get("n_samples", "?")
    tau       = meta.get("tau", "?")

    n_panels = 4 if outer_sorted is not None else 3
    fig_side  = max(7.0, min(12.0, 7.0 * (H / 256.0)))
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_side * n_panels + 2.5, fig_side))

    _cbar_kw    = {"pad": 0.02, "fraction": 0.046}
    _imshow_kw  = dict(origin="upper", interpolation="nearest", aspect="auto")

    def _set_axes(ax):
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Target unit (npz row index)")
        ax.set_ylabel("Source unit (npz row index)")

    panel_idx = 0

    # Panel 1: rank-1 outer product (no sigmoid)
    if outer_sorted is not None:
        ax = axes[panel_idx]
        vmax_outer = float(max(np.abs(outer_sorted).max(), 1e-8))
        im = ax.imshow(outer_sorted, **_imshow_kw, cmap="RdBu_r",
                       vmin=-vmax_outer, vmax=vmax_outer)
        ax.set_title(
            f"Avg  U[:,{fg_digit}]·fb[{fg_digit}]·V_hh[{fg_digit},:]  (digit={fg_digit})\n"
            f"Rank-1 outer product (no sigmoid)  n_frames={n_frames}"
        )
        _set_axes(ax)
        if digit_boundaries is not None:
            _draw_boundaries(ax, digit_boundaries)
        fig.colorbar(im, ax=ax, **_cbar_kw)
        panel_idx += 1

    # Panel 2: avg gate_hh (sigmoid)
    ax = axes[panel_idx]
    im = ax.imshow(
        gate_sorted, **_imshow_kw, cmap="viridis", vmin=0.0, vmax=float(vmax_gate)
    )
    ax.set_title(
        f"Avg gate_hh  (digit={fg_digit})\n"
        f"n_frames={n_frames}, n_samples={n_samples}, tau={tau}"
    )
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries)
    fig.colorbar(im, ax=ax, **_cbar_kw)
    panel_idx += 1

    # Panel 3: avg gate_hh ⊙ W_hh
    ax = axes[panel_idx]
    im = ax.imshow(gated_W_sorted, **_imshow_kw, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    ax.set_title(f"avg gate_hh ⊙ W_hh  (digit={fg_digit})\nGate-modulated connection matrix")
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries)
    fig.colorbar(im, ax=ax, **_cbar_kw)
    panel_idx += 1

    # Panel 4: raw W_hh
    ax = axes[panel_idx]
    im = ax.imshow(W_sorted, **_imshow_kw, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    ax.set_title("W_hh — Raw static connection\n(for comparison)")
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries)
    fig.colorbar(im, ax=ax, **_cbar_kw)

    fig.suptitle(
        f"GaWF Gate-Modulated Connection Matrix  (fg_digit={fg_digit})\n"
        "Units ordered by panel-4 (FDR + effect filtered, digit groups 0–9 + untuned tail)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


# ---------------------------------------------------------------------------
# Sector-mode plotting
# ---------------------------------------------------------------------------

def plot_sector_panels(
    outer_agg: np.ndarray | None,
    gate_agg: np.ndarray,
    gated_W_agg: np.ndarray,
    W_agg: np.ndarray,
    sorted_npz_order: np.ndarray | None,
    agg: str,
    meta: dict,
    out_path: str,
    unit_tick_step: int,
    vmax_w: float | None,
    sector: int | None = None,
    fg_digit: int | None = None,
    vmax_gate: float = 1.0,
    digit_boundaries: np.ndarray | None = None,
    feature_rows_cnn_ordered: bool = False,
    row_order_shown: np.ndarray | None = None,
    vmax_outer_override: float | None = None,
) -> None:
    """
    4-panel ih figure (aggregated input × hidden).  Pass exactly one of sector or fg_digit.
    """
    if (sector is None) == (fg_digit is None):
        raise ValueError("plot_sector_panels: set exactly one of sector= or fg_digit=.")

    n_rows, H = gate_agg.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Hidden-unit axis ticks (columns)
    if unit_tick_step <= 0:
        unit_tick_step = max(1, min(32, H // 16))
    h_ticks = list(range(0, H, unit_tick_step))
    if H - 1 not in h_ticks:
        h_ticks.append(H - 1)
    if sorted_npz_order is not None:
        h_tick_labels = [str(int(sorted_npz_order[i])) for i in h_ticks]
    else:
        h_tick_labels = [str(i) for i in h_ticks]

    # Input-aggregation axis ticks (rows)
    row_step = max(1, n_rows // 8)
    r_ticks = list(range(0, n_rows, row_step))
    if n_rows - 1 not in r_ticks:
        r_ticks.append(n_rows - 1)
    if row_order_shown is not None:
        r_tick_labels = [str(int(row_order_shown[i])) for i in r_ticks]
    else:
        r_tick_labels = [str(i) for i in r_ticks]

    if vmax_w is None:
        vmax_w = float(max(np.abs(gated_W_agg).max(), np.abs(W_agg).max(), 1e-8))

    n_frames  = meta.get("n_frames", "?")
    n_samples = meta.get("n_samples", "?")
    tau       = meta.get("tau", "?")
    row_label = "Spatial position (row×col)" if agg == "space" else "Feature channel"
    agg_desc  = f"mean over feature channels → {n_rows} spatial" if agg == "space" \
                else f"mean over 6×6 spatial → {n_rows} feature channels"

    # 2×2 grid layout
    fig_w = max(6.0, min(10.0, 6.0 * (H / 256.0)))
    fig_h = max(3.0, min(6.0, 3.0 * (n_rows / 36.0)))
    fig, axes = plt.subplots(2, 2, figsize=(fig_w * 2 + 2.5, fig_h * 2 + 1.5))

    _cbar_kw   = {"pad": 0.02, "fraction": 0.046}
    _imshow_kw = dict(origin="upper", interpolation="nearest", aspect="auto")

    def _set_axes(ax):
        ax.set_xticks(h_ticks)
        ax.set_xticklabels(h_tick_labels, rotation=45, ha="right")
        ax.set_yticks(r_ticks)
        ax.set_yticklabels(r_tick_labels)
        ax.set_xlabel("Hidden unit (npz row index)")
        ax.set_ylabel(row_label)

    # Panel positions in row-major order: (0,0) (0,1) (1,0) (1,1)
    panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    panel_idx = 0

    cond = f"digit={fg_digit}" if fg_digit is not None else f"sector={sector}"

    def _maybe_sector_hlines(ax):
        if agg == "space" and sector is not None:
            _draw_sector_hlines(ax, sector)

    # Panel 1 (top-left): rank-1 outer product (no sigmoid)
    if outer_agg is not None:
        ax = axes[panel_positions[panel_idx]]
        vmax_outer = (
            float(vmax_outer_override)
            if vmax_outer_override is not None
            else float(max(np.abs(outer_agg).max(), 1e-8))
        )
        im = ax.imshow(outer_agg, **_imshow_kw, cmap="RdBu_r",
                       vmin=-vmax_outer, vmax=vmax_outer)
        if fg_digit is not None:
            d = fg_digit
            outer_title = (
                f"Avg  U[:,{d}]·fb[{d}]·V_ih[{d},:]  (digit={d})\n"
                f"Rank-1 outer, {agg_desc},  no sigmoid  n_frames={n_frames}"
            )
        else:
            s = sector
            outer_title = (
                f"Avg  U[:,nc+{s}]·fb[nc+{s}]·V_ih[nc+{s},:]  (sector={s})\n"
                f"Rank-1 outer, {agg_desc},  no sigmoid  n_frames={n_frames}"
            )
        ax.set_title(outer_title)
        _set_axes(ax)
        if digit_boundaries is not None:
            _draw_boundaries(ax, digit_boundaries, vlines_only=True)
        _maybe_sector_hlines(ax)
        fig.colorbar(im, ax=ax, **_cbar_kw)
        panel_idx += 1

    # Panel 2: avg gate_ih (sigmoid)
    ax = axes[panel_positions[panel_idx]]
    im = ax.imshow(
        gate_agg, **_imshow_kw, cmap="viridis", vmin=0.0, vmax=float(vmax_gate)
    )
    ax.set_title(
        f"Avg gate_ih ({cond}, agg={agg})\n"
        f"n_frames={n_frames}, n_samples={n_samples}, tau={tau}"
    )
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries, vlines_only=True)
    _maybe_sector_hlines(ax)
    fig.colorbar(im, ax=ax, **_cbar_kw)
    panel_idx += 1

    # Panel 3: avg gate_ih_agg ⊙ W_ih_agg
    ax = axes[panel_positions[panel_idx]]
    im = ax.imshow(gated_W_agg, **_imshow_kw, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    ax.set_title(
        f"avg gate_ih_agg ⊙ W_ih_agg  ({cond})\nGate-modulated input connection"
    )
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries, vlines_only=True)
    _maybe_sector_hlines(ax)
    fig.colorbar(im, ax=ax, **_cbar_kw)
    panel_idx += 1

    # Panel 4: raw W_ih_agg
    ax = axes[panel_positions[panel_idx]]
    im = ax.imshow(W_agg, **_imshow_kw, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
    ax.set_title(f"W_ih_agg ({agg}) — Raw static input connection\n(for comparison)")
    _set_axes(ax)
    if digit_boundaries is not None:
        _draw_boundaries(ax, digit_boundaries, vlines_only=True)
    _maybe_sector_hlines(ax)
    fig.colorbar(im, ax=ax, **_cbar_kw)

    fig.suptitle(
        f"GaWF Gate-Modulated Input Connection  ({cond}, agg={agg})\n"
        "Hidden units ordered by digit groups 0–9 + untuned tail  |  "
        f"{agg_desc}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.digit is None and args.sector is None:
        raise ValueError("Specify exactly one of --digit or --sector.")
    if args.digit is not None and args.sector is not None:
        raise ValueError("Specify exactly one of --digit or --sector, not both.")
    if args.sector is not None and args.agg is None:
        raise ValueError("Sector mode requires --agg {space|feature}.")

    mode         = "digit" if args.digit is not None else "sector"
    avg_gate_dir = os.path.join(os.path.abspath(args.data_dir), mode)
    conn_dir     = os.path.abspath(args.conn_dir)
    save_dir     = os.path.join(os.path.abspath(args.save_dir), mode)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ digit
    if args.digit is not None:
        fg_digit = int(args.digit)
        # Under gate_avg/digit: hh = W_hh panels (no --agg); ih = aggregated input (--agg)
        digit_save_dir = os.path.join(save_dir, "ih" if args.agg is not None else "hh")
        if args.agg is None:
            out_path = os.path.join(
                digit_save_dir,
                f"digit{fg_digit}_avg_gate{'_tuned' if args.tuned_only else ''}.png",
            )

            avg_gate_hh, avg_outer_hh, W_hh, sorted_npz_order, meta, digit_boundaries = \
                load_digit_data(avg_gate_dir, conn_dir, fg_digit, args.tuned_only)

            idx            = sorted_npz_order
            gate_sorted    = reorder(avg_gate_hh, idx)
            W_sorted       = reorder(W_hh, idx)
            gated_W_sorted = gate_sorted * W_sorted
            outer_sorted   = reorder(avg_outer_hh, idx) if avg_outer_hh is not None else None

            plot_panels(
                outer_sorted=outer_sorted,
                gate_sorted=gate_sorted,
                gated_W_sorted=gated_W_sorted,
                W_sorted=W_sorted,
                sorted_npz_order=sorted_npz_order,
                fg_digit=fg_digit,
                meta=meta,
                out_path=out_path,
                unit_tick_step=args.unit_tick_step,
                vmax_gate=args.vmax_gate,
                vmax_w=args.vmax_w,
                digit_boundaries=digit_boundaries,
            )
        else:
            agg = args.agg
            out_path = os.path.join(
                digit_save_dir,
                f"digit{fg_digit}_{agg}_avg_gate{'_tuned' if args.tuned_only else ''}.png",
            )

            avg_gate_ih, avg_outer_ih, W_ih_agg, sorted_npz_order, meta, digit_boundaries = \
                load_digit_ih_data(avg_gate_dir, conn_dir, fg_digit, agg, args.tuned_only)

            if sorted_npz_order is not None:
                idx       = sorted_npz_order
                gate_agg  = avg_gate_ih[:, idx]
                W_agg     = W_ih_agg[:, idx]
                outer_agg = avg_outer_ih[:, idx] if avg_outer_ih is not None else None
            else:
                gate_agg  = avg_gate_ih
                W_agg     = W_ih_agg
                outer_agg = avg_outer_ih

            gated_W_agg = gate_agg * W_agg

            gate_agg, W_agg, gated_W_agg, outer_agg, feat_cnn_ord, row_order_shown = (
                maybe_reorder_ih_feature_rows_for_viz(
                    agg,
                    args.use_cnn_channel_order,
                    args.channel_order_path,
                    gate_agg,
                    W_agg,
                    gated_W_agg,
                    outer_agg,
                )
            )
            vmax_outer_override = None
            if args.align_outer_cbar_with_allcomp:
                vmax_outer_override = _load_outer_vmax_from_allcomp(
                    os.path.join(os.path.abspath(args.allcomp_data_dir), "digit"),
                    mode="digit",
                    selected_idx=fg_digit,
                    agg=agg,
                )

            plot_sector_panels(
                outer_agg=outer_agg,
                gate_agg=gate_agg,
                gated_W_agg=gated_W_agg,
                W_agg=W_agg,
                sorted_npz_order=sorted_npz_order,
                agg=agg,
                meta=meta,
                out_path=out_path,
                unit_tick_step=args.unit_tick_step,
                vmax_w=args.vmax_w,
                sector=None,
                fg_digit=fg_digit,
                vmax_gate=args.vmax_gate,
                digit_boundaries=digit_boundaries,
                feature_rows_cnn_ordered=feat_cnn_ord,
                row_order_shown=row_order_shown,
                vmax_outer_override=vmax_outer_override,
            )

    # ---------------------------------------------------------------- sector
    else:
        sector   = int(args.sector)
        agg      = args.agg  # validated non-None
        out_path = os.path.join(save_dir, f"sector{sector}_{agg}_avg_gate.png")

        avg_gate_ih, avg_outer_ih, W_ih_agg, sorted_npz_order, meta, digit_boundaries = \
            load_sector_data(avg_gate_dir, conn_dir, sector, agg)

        # Reorder hidden-unit columns by sorted_npz_order (if available)
        if sorted_npz_order is not None:
            idx         = sorted_npz_order
            gate_agg    = avg_gate_ih[:, idx]
            W_agg       = W_ih_agg[:, idx]
            outer_agg   = avg_outer_ih[:, idx] if avg_outer_ih is not None else None
        else:
            gate_agg  = avg_gate_ih
            W_agg     = W_ih_agg
            outer_agg = avg_outer_ih

        gated_W_agg = gate_agg * W_agg

        gate_agg, W_agg, gated_W_agg, outer_agg, feat_cnn_ord, row_order_shown = (
            maybe_reorder_ih_feature_rows_for_viz(
                agg,
                args.use_cnn_channel_order,
                args.channel_order_path,
                gate_agg,
                W_agg,
                gated_W_agg,
                outer_agg,
            )
        )
        vmax_outer_override = None
        if args.align_outer_cbar_with_allcomp:
            vmax_outer_override = _load_outer_vmax_from_allcomp(
                os.path.join(os.path.abspath(args.allcomp_data_dir), "sector"),
                mode="sector",
                selected_idx=sector,
                agg=agg,
            )

        plot_sector_panels(
            outer_agg=outer_agg,
            gate_agg=gate_agg,
            gated_W_agg=gated_W_agg,
            W_agg=W_agg,
            sorted_npz_order=sorted_npz_order,
            agg=agg,
            meta=meta,
            out_path=out_path,
            unit_tick_step=args.unit_tick_step,
            vmax_w=args.vmax_w,
            sector=sector,
            fg_digit=None,
            vmax_gate=args.vmax_gate,
            digit_boundaries=digit_boundaries,
            feature_rows_cnn_ordered=feat_cnn_ord,
            row_order_shown=row_order_shown,
            vmax_outer_override=vmax_outer_override,
        )


if __name__ == "__main__":
    main()
