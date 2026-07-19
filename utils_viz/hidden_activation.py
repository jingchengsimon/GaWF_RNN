"""
Visualize GaWF hidden-unit activation statistics.

Reads gawf_hidden_activation_stats.npz (from analyze_gawf_hidden_activation.py)
and plots heatmaps: x = hidden units, y = digits (0–9), raw mean, z-score,
and (row-wise mode only) z-score with units grouped by FDR + effect filtering
(third panel uses tuned_display_order.npy from analyze by default).

Z-score modes (same semantics as viz_cnn_channel_activation.py):
    row-wise — per unit across digits
    col-wise — per digit across units
    global   — over the full matrix

When z_mode is row-wise, the third panel uses statistical grouping from analyze;
use --tuning_fallback_argmax to restore legacy argmax-only sorting.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GaWF digit × hidden-unit mean activation (units on x, digits on y)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(output_dir("E_relevance_alignment", "hidden_unit_tuning", "data")),
        help=(
            "Path to gawf_hidden_activation_stats.npz, or a directory "
            "(auto-completes filename)."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(output_dir("E_relevance_alignment", "hidden_activation", "figs")),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--unit_order_path",
        type=str,
        default=str(
            output_dir(
                "E_relevance_alignment",
                "hidden_unit_tuning",
                "data",
            ) / "unit_order_by_cosine_similarity.npy"
        ),
        help="Optional .npy reordering of hidden units (column order after transpose).",
    )
    parser.add_argument(
        "--tuning_stats_path",
        type=str,
        default="",
        help=(
            "gawf_hidden_tuning_stats.npz (for sanity check / future use). "
            "Empty = <stats_dir>/gawf_hidden_tuning_stats.npz."
        ),
    )
    parser.add_argument(
        "--tuned_order_path",
        type=str,
        default="",
        help=(
            "tuned_display_order.npy for row-wise third panel. "
            "Empty = <stats_dir>/tuned_display_order.npy."
        ),
    )
    parser.add_argument(
        "--argmax_order_path",
        type=str,
        default="",
        help=(
            "argmax_display_order.npy for row-wise argmax-grouped panel. "
            "Empty = <stats_dir>/argmax_display_order.npy."
        ),
    )
    parser.add_argument(
        "--tuning_fallback_argmax",
        action="store_true",
        default=False,
        help=(
            "If set, third panel uses legacy argmax grouping when tuned order is missing "
            "(default: off; row-wise requires tuned_display_order.npy)."
        ),
    )
    parser.add_argument(
        "--z_mode",
        type=str,
        default="row-wise",
        choices=["row-wise", "col-wise", "global"],
        help="Z-score mode for the right panel (default: row-wise).",
    )
    parser.add_argument(
        "--unit_tick_step",
        type=int,
        default=0,
        help=(
            "X-axis tick step for hidden unit index; 0 = auto "
            "(max(1, H//16) with cap 32)."
        ),
    )
    parser.add_argument(
        "--ytick_step",
        type=int,
        default=None,
        help="Deprecated alias for --unit_tick_step (when units were on y-axis).",
    )
    return parser.parse_args()


def load_stats(stats_path: str):
    obj = np.load(stats_path)
    mean_activation = np.asarray(obj["mean_activation"], dtype=np.float32)
    std_activation = np.asarray(obj["std_activation"], dtype=np.float32)
    digit_sample_count = np.asarray(obj["digit_sample_count"])

    if mean_activation.ndim != 2 or mean_activation.shape[1] != 10:
        raise ValueError(
            f"Expected mean_activation shape (H, 10), got {mean_activation.shape}"
        )
    if std_activation.shape != mean_activation.shape:
        raise ValueError(
            f"std_activation {std_activation.shape} != mean {mean_activation.shape}"
        )
    if digit_sample_count.shape != (10,):
        raise ValueError(
            f"Expected digit_sample_count shape (10,), got {digit_sample_count.shape}"
        )
    return mean_activation, std_activation, digit_sample_count


def load_unit_order(path: str, num_units: int) -> np.ndarray:
    default_order = np.arange(num_units, dtype=np.int64)
    if path is None or path == "":
        return default_order

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        print(
            f"[viz][warn] unit order file not found at '{abs_path}'; "
            "using default order."
        )
        return default_order

    try:
        order = np.load(abs_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[viz][warn] failed to load unit order from '{abs_path}' ({exc}); "
            "using default order."
        )
        return default_order

    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or order.size != num_units:
        print(
            "[viz][warn] unit order has incompatible shape "
            f"{order.shape} for num_units={num_units}; using default order."
        )
        return default_order

    return order[::-1]


def load_tuning_thresholds_json(stats_dir: str) -> dict:
    for name in ("gawf_hidden_tuning_meta.json", "gawf_hidden_tuning_thresholds.json"):
        p = os.path.join(stats_dir, name)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                print(f"[viz][warn] could not read {p}: {exc}")
    return {}


def verify_tuning_npz_shape(path: str, expected_h: int) -> None:
    if not os.path.isfile(path):
        return
    try:
        z = np.load(path)
        ma = np.asarray(z["mean_activation"])
        if ma.shape != (expected_h, 10):
            print(
                f"[viz][warn] tuning stats mean_activation shape {ma.shape} "
                f"!= ({expected_h}, 10)"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[viz][warn] could not verify tuning npz '{path}': {exc}")


def compute_zscore(mean_activation: np.ndarray, mode: str) -> np.ndarray:
    H, D = mean_activation.shape
    if D != 10:
        raise ValueError(f"Expected 10 columns (digits), got {mean_activation.shape}")

    if mode == "row-wise":
        z = np.zeros_like(mean_activation, dtype=np.float32)
        for i in range(H):
            row = mean_activation[i]
            mu = float(row.mean())
            sigma = float(row.std())
            if sigma < 1e-8:
                sigma = 1e-8
            z[i] = (row - mu) / sigma
        return z

    if mode == "col-wise":
        z = np.zeros_like(mean_activation, dtype=np.float32)
        for d in range(D):
            col = mean_activation[:, d]
            mu = float(col.mean())
            sigma = float(col.std())
            if sigma < 1e-8:
                sigma = 1e-8
            z[:, d] = (col - mu) / sigma
        return z

    if mode == "global":
        mu = float(mean_activation.mean())
        sigma = float(mean_activation.std())
        if sigma < 1e-8:
            sigma = 1e-8
        return (mean_activation - mu) / sigma

    raise ValueError(f"Unknown z-score mode: {mode}")


def _unit_order_by_argmax_digit(z_rowwise: np.ndarray) -> np.ndarray:
    """Sort unit indices: primary argmax digit 0…9 (left→right), then winning z (desc)."""
    h = z_rowwise.shape[0]
    win = np.argmax(z_rowwise, axis=1)
    win_z = z_rowwise[np.arange(h), win]
    return np.lexsort((-win_z, win))


def _is_perm_of_range(arr: np.ndarray, h: int) -> bool:
    a = np.asarray(arr, dtype=np.int64).ravel()
    if a.size != h:
        return False
    return bool(np.array_equal(np.sort(a), np.arange(h, dtype=np.int64)))


def plot_activation_matrices(
    mean_activation: np.ndarray,
    z_scores: np.ndarray,
    digit_sample_count: np.ndarray,
    out_path: str,
    z_mode: str,
    unit_tick_step: int,
    unit_order: np.ndarray | None = None,
    argmax_col_perm: np.ndarray | None = None,
    tuned_col_perm: np.ndarray | None = None,
    use_argmax_fallback: bool = False,
    panel3_caption: str = "",
) -> None:
    H, D = mean_activation.shape
    if z_scores.shape != (H, D):
        raise ValueError(
            f"z_scores shape {z_scores.shape} does not match mean {mean_activation.shape}"
        )

    mean_plot = mean_activation.T
    z_plot = z_scores.T

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rowwise_extra = z_mode == "row-wise"
    if unit_order is None:
        unit_order = np.arange(H, dtype=np.int64)

    fig_w = max(12.0, min(24.0, 12.0 * (H / 32.0)))
    panel_h = 5.0
    if not rowwise_extra:
        nrows = 2
    elif argmax_col_perm is not None:
        nrows = 4
    else:
        nrows = 3
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(fig_w, panel_h * nrows + 0.5),
        sharex=False,
        constrained_layout=False,
    )
    if nrows == 2:
        ax0, ax1 = axes
        ax1.sharex(ax0)
        ax2 = ax3 = None
    elif nrows == 3:
        ax0, ax1, ax2 = axes
        ax1.sharex(ax0)
        ax3 = None
    else:
        ax0, ax1, ax2, ax3 = axes
        ax1.sharex(ax0)

    im0 = ax0.imshow(
        mean_plot,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax0.set_title("Raw Mean Activation\n(Digits × Hidden units)")
    ax0.set_ylabel("Digit label (0–9)")
    ax0.set_yticks(range(D))
    ax0.set_yticklabels([str(d) for d in range(D)])

    if unit_tick_step <= 0:
        unit_tick_step = max(1, min(32, H // 16))
    xticks = list(range(0, H, unit_tick_step))
    if H - 1 not in xticks and H > 0:
        xticks.append(H - 1)
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([str(int(unit_order[i])) for i in xticks], rotation=45, ha="right")
    ax0.set_xlabel("Hidden unit (npz row index)")
    _cbar_kw = {"pad": 0.012, "fraction": 0.035}
    fig.colorbar(im0, ax=ax0, **_cbar_kw)

    im1 = ax1.imshow(
        z_plot,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        vmin=-3,
        vmax=3,
        cmap="RdBu_r",
    )
    if z_mode == "row-wise":
        title = "Row-wise Z-score Across Digits"
    elif z_mode == "col-wise":
        title = "Column-wise Z-score Across Units"
    else:
        title = "Global Z-score (All Entries)"
    ax1.set_title(f"{title}\n(Digits × Hidden units)")
    ax1.set_ylabel("Digit label (0–9)")
    ax1.set_yticks(range(D))
    ax1.set_yticklabels([str(d) for d in range(D)])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(int(unit_order[i])) for i in xticks], rotation=45, ha="right")
    ax1.set_xlabel("Hidden unit (npz row index)")
    fig.colorbar(im1, ax=ax1, **_cbar_kw)

    if rowwise_extra:
        # ax_last is the bottom panel (FDR-grouped or fallback)
        ax_last = ax3 if nrows == 4 else ax2

        # Panel 3 (new, only when argmax_col_perm is available): argmax-grouped z-score
        if nrows == 4:
            col_perm_argmax = np.asarray(argmax_col_perm, dtype=np.int64)
            if not _is_perm_of_range(col_perm_argmax, H):
                raise ValueError(
                    f"argmax_display_order must be a permutation of 0..{H-1}, "
                    f"got shape {col_perm_argmax.shape}"
                )
            z_argmax = z_scores[col_perm_argmax, :].T
            im2 = ax2.imshow(
                z_argmax,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                vmin=-3,
                vmax=3,
                cmap="RdBu_r",
            )
            ax2.set_title(
                "Row-wise Z-score — units grouped by argmax digit (0…9, all units)\n"
                "(Digits × Hidden units)"
            )
            ax2.set_ylabel("Digit label (0–9)")
            ax2.set_yticks(range(D))
            ax2.set_yticklabels([str(d) for d in range(D)])
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([str(int(unit_order[col_perm_argmax[i]])) for i in xticks], rotation=45, ha="right")
            ax2.set_xlabel("Hidden unit (npz row index, argmax digit order)")
            fig.colorbar(im2, ax=ax2, **_cbar_kw)

        # Last panel: FDR + effect filtered grouping (or fallback)
        if tuned_col_perm is not None:
            col_perm = np.asarray(tuned_col_perm, dtype=np.int64)
            if not _is_perm_of_range(col_perm, H):
                raise ValueError(
                    f"tuned_display_order must be a permutation of 0..{H-1}, got shape {col_perm.shape}"
                )
            p_last_title = (
                "Row-wise Z-score — FDR + effect filtered grouping "
                "(digits 0…9 blocks + untuned tail)\n"
                "(Digits × Hidden units)"
            )
            if panel3_caption:
                p_last_title = p_last_title + "\n" + panel3_caption
            write_argmax_sidecar = False
        elif use_argmax_fallback:
            col_perm = _unit_order_by_argmax_digit(z_scores)
            p_last_title = (
                "Row-wise Z-score — units sorted by argmax digit (0…9, left → right) "
                "[fallback]\n"
                "(Digits × Hidden units)"
            )
            write_argmax_sidecar = True
        else:
            raise ValueError(
                "row-wise third panel requires tuned_col_perm or use_argmax_fallback=True"
            )

        z_sorted = z_scores[col_perm, :].T
        im_last = ax_last.imshow(
            z_sorted,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=-3,
            vmax=3,
            cmap="RdBu_r",
        )
        ax_last.set_title(p_last_title)
        ax_last.set_xlabel("Hidden unit (npz row index)")
        ax_last.set_ylabel("Digit label (0–9)")
        ax_last.set_yticks(range(D))
        ax_last.set_yticklabels([str(d) for d in range(D)])
        ax_last.set_xticks(xticks)
        ax_last.set_xticklabels(
            [str(int(unit_order[col_perm[i]])) for i in xticks],
            rotation=45,
            ha="right",
        )
        fig.colorbar(im_last, ax=ax_last, **_cbar_kw)

        if write_argmax_sidecar:
            sidecar_path = os.path.join(
                os.path.dirname(out_path) or ".",
                "gawf_hidden_activation_rowwise_argmax_per_digit.txt",
            )
            win_digit = np.argmax(z_scores, axis=1)
            lines = [
                "GaWF row-wise z-score: units grouped by argmax digit across 0–9 "
                "(each unit appears in exactly one list).\n",
                "Within digit d: only units with argmax_u z[u,:] == d; order by z[u,d] "
                "(high → low).\n",
                "npz_row_index: row in gawf_hidden_activation_stats.npz "
                "(original unit id, used as x-axis label in all panels).\n",
                "\n",
            ]
            for d in range(D):
                u_in_d = np.nonzero(win_digit == d)[0]
                if u_in_d.size:
                    sub = np.argsort(-z_scores[u_in_d, d], kind="stable")
                    ranked_u = u_in_d[sub]
                else:
                    ranked_u = np.array([], dtype=np.int64)
                npz_list = ", ".join(str(int(unit_order[u])) for u in ranked_u)
                lines.append(f"digit {d} (argmax == {d}, n={len(ranked_u)}):\n")
                lines.append(f"  npz_row_index: {npz_list}\n")
                lines.append("\n")
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"Saved argmax summary to: {sidecar_path}")

    fig.suptitle(
        "GaWF Hidden Unit Activation Statistics\n"
        + "Per-digit sample counts: "
        + ", ".join(f"{d}:{int(n)}" for d, n in enumerate(digit_sample_count)),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94], h_pad=1.2, pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


def main() -> None:
    args = parse_args()

    raw_in_path = args.data_dir
    if raw_in_path is None or raw_in_path == "":
        raw_in_path = str(output_dir("E_relevance_alignment", "hidden_unit_tuning", "data"))
    if os.path.isdir(raw_in_path) or not os.path.splitext(raw_in_path)[1]:
        raw_in_path = os.path.join(raw_in_path, "gawf_hidden_activation_stats.npz")

    stats_file = os.path.abspath(raw_in_path)
    stats_dir = os.path.dirname(stats_file)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    suffix = {
        "row-wise": "rowwise",
        "col-wise": "colwise",
        "global": "global",
    }[args.z_mode]
    out_path = os.path.join(save_dir, f"gawf_hidden_activation_matrix_{suffix}.png")

    mean_activation, _, digit_sample_count = load_stats(stats_file)

    H, _ = mean_activation.shape
    unit_order = load_unit_order(args.unit_order_path, num_units=H)
    mean_activation = mean_activation[unit_order]

    z_scores = compute_zscore(mean_activation, mode=args.z_mode)

    unit_tick_step = args.unit_tick_step
    if args.ytick_step is not None:
        unit_tick_step = args.ytick_step

    tuning_stats_path = args.tuning_stats_path or os.path.join(
        stats_dir, "gawf_hidden_tuning_stats.npz"
    )
    tuned_order_path = args.tuned_order_path or os.path.join(
        stats_dir, "tuned_display_order.npy"
    )

    tuned_col_perm: np.ndarray | None = None
    argmax_col_perm: np.ndarray | None = None
    use_argmax_fallback = bool(args.tuning_fallback_argmax)
    panel3_caption = ""

    if args.z_mode == "row-wise":
        if os.path.isfile(tuned_order_path):
            tuned_col_perm = np.load(tuned_order_path)
            meta = load_tuning_thresholds_json(stats_dir)
            q_thr = meta.get("q_thr", "?")
            em = meta.get("effect_metric", "?")
            eth = meta.get("effect_thr", "?")
            fdr_m = meta.get("fdr_method", "bh")
            panel3_caption = (
                f"FDR ({fdr_m}): q < {q_thr}; effect: {em} ≥ {eth}"
            )
            verify_tuning_npz_shape(tuning_stats_path, H)
        elif use_argmax_fallback:
            tuned_col_perm = None
            print(
                "[viz][warn] tuned_display_order not found; using --tuning_fallback_argmax."
            )
        else:
            raise FileNotFoundError(
                f"Row-wise mode requires '{tuned_order_path}'. "
                "Run utils_anal/analyze_gawf_hidden_activation.py on this dataset, "
                "or pass --tuning_fallback_argmax."
            )

        argmax_order_path = args.argmax_order_path or os.path.join(
            stats_dir, "argmax_display_order.npy"
        )
        if os.path.isfile(argmax_order_path):
            argmax_col_perm = np.load(argmax_order_path)
        else:
            print(
                f"[viz][warn] argmax_display_order.npy not found at '{argmax_order_path}'; "
                "argmax panel will be skipped."
            )

    plot_activation_matrices(
        mean_activation=mean_activation,
        z_scores=z_scores,
        digit_sample_count=digit_sample_count,
        out_path=out_path,
        z_mode=args.z_mode,
        unit_tick_step=unit_tick_step,
        unit_order=unit_order,
        argmax_col_perm=argmax_col_perm,
        tuned_col_perm=tuned_col_perm,
        use_argmax_fallback=use_argmax_fallback,
        panel3_caption=panel3_caption,
    )


if __name__ == "__main__":
    main()
