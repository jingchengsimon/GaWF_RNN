"""
Visualize GaWF sector-specific modulation basis maps exported by export_gawf_sector_basis.py.

Generates two heatmaps:
  1) abs mean map across channels
  2) signed mean map across channels (symmetric color range around 0)
"""

import argparse
import os
import shutil
from typing import Tuple

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GaWF sector-specific basis maps (abs mean / signed mean)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./results/anal_data/V_basis_exports",
        help=(
            "Path to exported .pt file from export_gawf_sector_basis.py. "
            "If a directory is given, file name will be auto-completed as "
            "'sector_{sector}_basis.pt'."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_figs/V_basis",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        help="Sector index k whose basis file will be visualized. Mutually exclusive with --digit.",
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=list(range(10)),
        help="Digit index d (0-9) whose basis file will be visualized. Mutually exclusive with --sector.",
    )
    parser.add_argument(
        "--use_cnn_channel_order",
        action="store_true",
        default=False,
        help=(
            "When set, and in digit mode, reorder feature channels according to the "
            "CNN activation channel order loaded from --channel_order_path. "
            "By default this is disabled and the legacy channel order is used."
        ),
    )
    parser.add_argument(
        "--channel_order_path",
        type=str,
        default="./results/anal_data/cnn_channel_data/channel_order_by_cosine_similarity.npy",
        help=(
            "Path to a NumPy .npy file containing the CNN feature-channel order "
            "computed by analyze_cnn_channel_activation.py. Only used when "
            "--use_cnn_channel_order is True."
        ),
    )
    parser.add_argument(
        "--cnn_stats_path",
        type=str,
        default="./results/anal_data/cnn_channel_data",
        help=(
            "Path to cnn_channel_activation_stats.npz (or its containing directory) "
            "used in viz_cnn_channel_activation.py. In digit mode, this is used to "
            "draw a narrow row-wise z-score column for the selected digit next to "
            "the UxV input-part matrix."
        ),
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        default=False,
        help=(
            "When set: in digit mode, only save one UxV signed-mean figure per digit "
            "into save_dir/digit/ (skip other digit visualizations). In sector mode, "
            "additionally save a simplified copy under save_dir/sector/ for quick browsing."
        ),
    )
    parser.add_argument(
        "--sector_summary",
        action="store_true",
        default=False,
        help=(
            "When set, generate a 3x3 summary RF gallery across sectors 0-8 using the "
            "outer UV input-part matrices from exported sector_{k}_basis.pt files. "
            "Each subplot corresponds to its sector index (0-8) arranged bottom-up, "
            "left-to-right. For each sector, pick the recurrent unit with the largest "
            "abs-mean RF magnitude whose signed-mean RF is positive; if none are "
            "positive, fall back to the global top-1 by abs mean."
        ),
    )
    return parser.parse_args()


def _to_numpy_2d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape={x.shape}")
    return x


def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={x.shape}")
    return x


def _read_feature_shape(obj) -> Tuple[int, int, int]:
    fs = obj.get("feature_shape", None)
    if fs is None:
        raise KeyError("Missing 'feature_shape' in input file.")
    if isinstance(fs, torch.Tensor):
        fs = tuple(int(v) for v in fs.detach().cpu().tolist())
    else:
        fs = tuple(int(v) for v in fs)
    if len(fs) != 3:
        raise ValueError(f"Invalid feature_shape={fs} (expected (C,H,W))")
    return fs[0], fs[1], fs[2]


def _load_channel_order(path: str, num_channels: int) -> np.ndarray:
    """
    Load channel order from a .npy file.

    Returns a permutation of [0..C-1]. This函数本身只负责产生索引序列，
    不对任意矩阵做数值运算，确保之后的重排序仅仅是行交换。
    """
    default_order = np.arange(num_channels, dtype=np.int64)

    if not path:
        return default_order

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        print(
            f"[viz][warn] channel order file not found at '{abs_path}'; "
            "using default channel order."
        )
        return default_order

    try:
        order = np.load(abs_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[viz][warn] failed to load channel order from '{abs_path}' "
            f"({exc}); using default channel order."
        )
        return default_order

    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or order.size != num_channels:
        print(
            "[viz][warn] channel order has incompatible shape "
            f"{order.shape} for num_channels={num_channels}; "
            "using default channel order."
        )
        return default_order

    # 根据约定：order 中更早的 index 映射到更靠上的行（更大的 y 轴 index）。
    # 这里仅返回索引序列本身，具体如何应用交给调用方。
    return order


def _load_cnn_rowwise_column(
    stats_path: str,
    digit: int,
    channel_order: np.ndarray | None,
) -> np.ndarray:
    """
    Load the row-wise z-score matrix from cnn_channel_activation_stats.npz and
    return the column for the specified digit as a 1D array of length C.

    The computation mirrors the 'row-wise' z-score mode used in
    viz_cnn_channel_activation.py, and then applies the same optional
    channel-order reindexing logic as used for the digit-mode UxV matrices
    (so that rows align visually when --use_cnn_channel_order is enabled).
    """
    # Resolve directory vs file, mirroring viz_cnn_channel_activation.py.
    raw_in_path = stats_path
    if raw_in_path is None or raw_in_path == "":
        raw_in_path = "./results/anal_data/cnn_channel_data"
    if os.path.isdir(raw_in_path) or not os.path.splitext(raw_in_path)[1]:
        raw_in_path = os.path.join(raw_in_path, "cnn_channel_activation_stats.npz")
    stats_file = os.path.abspath(raw_in_path)

    obj = np.load(stats_file)
    mean_activation = np.asarray(obj["mean_activation"], dtype=np.float32)
    if mean_activation.shape != (32, 10):
        raise ValueError(
            f"Expected mean_activation shape (32, 10), got {mean_activation.shape}"
        )
    C, D = mean_activation.shape
    if not (0 <= digit < D):
        raise ValueError(f"digit index {digit} out of range for D={D}")

    # Row-wise z-score across digits (per-channel), identical to the
    # 'row-wise' branch of compute_zscore(...) in viz_cnn_channel_activation.py.
    z = np.zeros_like(mean_activation, dtype=np.float32)
    for c in range(C):
        row = mean_activation[c]
        mu = float(row.mean())
        sigma = float(row.std())
        if sigma < 1e-8:
            sigma = 1e-8
        z[c] = (row - mu) / sigma

    col = z[:, digit]  # (C,)

    # Apply the same row-reordering convention as digit-mode UxV matrices.
    if channel_order is not None:
        apply_order = channel_order[::-1]
        if apply_order.shape[0] != C:
            raise ValueError(
                f"channel_order length {apply_order.shape[0]} does not match C={C}"
            )
        col = col[apply_order]

    return col


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # Sector summary mode: one 3x3 RF gallery across sectors 0-8.
    # This is an optional extra visualization and does not modify the
    # existing per-sector/digit outputs.
    # ---------------------------------------------------------------------
    if bool(args.sector_summary):
        raw_in_path = args.data_dir
        if raw_in_path is None or raw_in_path == "":
            raw_in_path = "./results/anal_data/V_basis_exports"
        in_dir = os.path.abspath(raw_in_path)
        if os.path.isfile(in_dir):
            in_dir = os.path.dirname(in_dir)
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        summary_dir = os.path.join(save_dir, "sector_summary")
        os.makedirs(summary_dir, exist_ok=True)

        rf_h, rf_w = 6, 6
        selected = []
        for sector_id in range(9):
            pt_path = os.path.join(in_dir, f"sector_{sector_id}_basis.pt")
            obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "uxv_input_signed_mean" not in obj:
                raise KeyError(
                    f"Missing 'uxv_input_signed_mean' in {pt_path}. "
                    "Please re-run export_gawf_sector_basis.py with --export_uxv."
                )
            signed_mat = _to_numpy_2d(obj["uxv_input_signed_mean"])
            if signed_mat.shape[0] != rf_h * rf_w:
                raise ValueError(
                    f"Expected outer UV input-part shape ({rf_h*rf_w}, rec), got {signed_mat.shape}"
                )
            rec = int(signed_mat.shape[1])
            rf_all = signed_mat.reshape(rf_h, rf_w, rec).transpose(2, 0, 1)  # (rec,6,6)

            # Step 1: rank units by overall abs-mean (desc).
            abs_mean = np.mean(np.abs(rf_all), axis=(1, 2))  # (rec,)
            order = np.argsort(-abs_mean)  # desc

            # Step 2: within that ordering, prefer units whose PEAK block (among 36 blocks)
            # has a positive signed value; among positives, rank by the peak block magnitude.
            # This matches the requirement "pick the unit with the largest positive value at
            # its strongest (abs-largest) block", rather than using the global RF mean sign.
            rf_flat = rf_all.reshape(rec, rf_h * rf_w)  # (rec,36)
            abs_flat = np.abs(rf_flat)
            peak_idx = np.argmax(abs_flat, axis=1)  # (rec,)
            row_ids = np.arange(rec)
            peak_signed = rf_flat[row_ids, peak_idx]  # (rec,)
            peak_abs = np.abs(peak_signed)  # (rec,)

            pos_units_in_order = [int(u) for u in order if float(peak_signed[u]) > 0.0]
            if len(pos_units_in_order) > 0:
                pos_units = np.asarray(pos_units_in_order, dtype=np.int64)
                # Primary: peak_abs desc; Secondary: abs_mean desc (already mostly enforced by order)
                sort_idx = np.lexsort((-abs_mean[pos_units], -peak_abs[pos_units]))
                pick = int(pos_units[sort_idx[0]])
            else:
                pick = int(order[0])

            selected.append(
                {
                    "sector": int(sector_id),
                    "unit": int(pick),
                    "abs_mean": float(abs_mean[pick]),
                    "rf": rf_all[pick],
                }
            )

        vmax = float(np.max([np.max(np.abs(s["rf"])) for s in selected]))
        if vmax == 0.0:
            vmax = 1e-8
        vmin = -vmax

        fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.5))
        im0 = None
        for s in selected:
            k = int(s["sector"])
            r = 2 - (k // 3)  # bottom-up
            c = k % 3
            ax = axes[r, c]
            im0 = ax.imshow(
                s["rf"],
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"Sector={k} | unit={int(s['unit'])}"#\nabs_mean={float(s['abs_mean']):.4f}"
            )

        fig.suptitle(
            "Feedback-dependent gating on input-to-hidden connections in GaWF RNN model",
            # "Outer UV input-part sector summary | "
            # "sectors=0-8 (bottom-up) | top-1 unit by abs mean with positive signed mean | RF=6x6",
            y=0.985, size=13, weight="bold",
        )
        fig.subplots_adjust(
            left=0.05,
            right=0.88,
            top=0.93,
            bottom=0.05,
            wspace=0.15,
            hspace=0.25,
        )
        if im0 is not None:
            # Align colorbar height to the 3x3 grid bounding box.
            from matplotlib.transforms import Bbox  # local import to avoid global dependency

            grid_bbox = Bbox.union([ax.get_position() for ax in axes.ravel()])
            cbar_width = 0.03
            pad = 0.03
            x0 = min(float(grid_bbox.x1) + pad, 0.98 - cbar_width)
            cax = fig.add_axes([x0, float(grid_bbox.y0), cbar_width, float(grid_bbox.height)])
            cb = fig.colorbar(im0, cax=cax)
            cb.set_ticks(np.linspace(-0.5, 0.5, 5))

        out_path = os.path.join(summary_dir, "outer_uv_input_sector_summary_top1_possign.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
        return

    # Determine mode: sector vs digit
    use_sector = args.sector is not None
    use_digit = args.digit is not None

    if use_sector and use_digit:
        raise ValueError(
            "Both --sector and --digit were provided. "
            "Please specify only one of them."
        )
    if not use_sector and not use_digit:
        # Default behavior: sector=0 (backward-compatible)
        mode = "sector"
        idx = 0
    elif use_sector:
        mode = "sector"
        idx = int(args.sector)
    else:
        mode = "digit"
        idx = int(args.digit)

    label = "sector" if mode == "sector" else "digit"
    prefix = "sector" if mode == "sector" else "digit"

    # If input_path points to a directory, auto-complete file name using sector or digit.
    raw_in_path = args.data_dir
    if raw_in_path is None or raw_in_path == "":
        raw_in_path = "./results/anal_data/V_basis_exports"
    if os.path.isdir(raw_in_path) or not os.path.splitext(raw_in_path)[1]:
        raw_in_path = os.path.join(raw_in_path, f"{prefix}_{idx}_basis.pt")

    in_path = os.path.abspath(raw_in_path)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Output layout:
    # - Default (legacy): one subdir per sector/digit: save_dir/sector_k or save_dir/digit_k
    # - Digit + --simple: write only one UxV figure per digit into save_dir/digit/
    simple_digit_uv_only = (mode == "digit") and bool(args.simple)
    if simple_digit_uv_only:
        out_dir = os.path.join(save_dir, prefix)
    else:
        out_dir = os.path.join(save_dir, f"{prefix}_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    # PyTorch 2.6+ defaults torch.load(weights_only=True), which can reject
    # non-tensor objects (e.g., NumPy arrays) in our exported .pt. We trust
    # locally-exported analysis files, so explicitly disable weights-only mode.
    obj = torch.load(in_path, map_location="cpu", weights_only=False)

    C, H, W = _read_feature_shape(obj)

    # Decide channel order for digit-mode visualizations.
    if mode == "digit" and args.use_cnn_channel_order:
        channel_order = _load_channel_order(args.channel_order_path, num_channels=C)
    else:
        channel_order = None

    if mode == "sector":
        # Sector: V[k] input-part averaged over channels -> spatial (H, W)
        abs_mean_map = _to_numpy_2d(obj["abs_mean_map"])
        signed_mean_map = _to_numpy_2d(obj["signed_mean_map"])

        abs_out = os.path.join(out_dir, "basis_abs_mean.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(abs_mean_map, origin="lower", interpolation="nearest", aspect="auto")
        ax.set_title(
            "V[k] input-part abs mean across channels\n"
            f"{label}={idx}\n"
            f"feature shape=({C},{H},{W})"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(abs_out, dpi=150)
        plt.close(fig)
        print(f"Saved: {abs_out}")

        signed_out = os.path.join(out_dir, "basis_signed_mean.png")
        mn = float(np.min(signed_mean_map))
        mx = float(np.max(signed_mean_map))
        m = float(max(abs(mn), abs(mx)))
        if m == 0.0:
            m = 1e-8
        vmin, vmax = -m, m
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(
            signed_mean_map,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax.set_title(
            "V[k] input-part signed mean across channels\n"
            f"{label}={idx}\n"
            f"feature shape=({C},{H},{W})"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(signed_out, dpi=150)
        plt.close(fig)
        print(f"Saved: {signed_out}")
    else:
        if simple_digit_uv_only:
            # In digit+simple mode, we skip all other digit visualizations and only
            # save the UxV (U_k ⊗ V_k) input-part figure into save_dir/digit/.
            pass
        else:
            # Digit: V[k] input-part averaged over (H,W) -> 32 channels; display as 4x8 (no spatial meaning)
            if "channel_abs" in obj and "channel_signed" in obj:
                channel_abs = _to_numpy_1d(obj["channel_abs"])
                channel_signed = _to_numpy_1d(obj["channel_signed"])
            else:
                # Backward compat: compute from basis_input
                basis_input = obj["basis_input"]
                if isinstance(basis_input, torch.Tensor):
                    basis_input = basis_input.detach().cpu()
                basis_input_map = basis_input.view(C, H, W)
                channel_abs = basis_input_map.abs().mean(dim=(1, 2)).numpy()
                channel_signed = basis_input_map.mean(dim=(1, 2)).numpy()
            if channel_abs.size != C:
                raise ValueError(f"channel_abs size {channel_abs.size} != C={C}")

            # Optional: reorder feature channels using shared CNN activation order.
            # 这里仅做纯索引重排，不做任何数值运算；如果不开启开关，则保持旧行为。
            if channel_order is not None:
                # 我们希望 order 中更早的 index 在图中更靠上，因此在 0..C-1 这一轴上
                # 使用反向索引：较早的通道排到更大的行 index。
                apply_order = channel_order[::-1]
                channel_abs = channel_abs[apply_order]
                channel_signed = channel_signed[apply_order]

            # Layout 4x8 for 32 channels
            display_h, display_w = 4, 8
            if C != display_h * display_w:
                raise ValueError(f"Expected C=32 for 4x8 layout, got C={C}")
            channel_abs_2d = channel_abs.reshape(display_h, display_w)
            channel_signed_2d = channel_signed.reshape(display_h, display_w)

            abs_out = os.path.join(out_dir, "basis_abs_mean_4x8.png")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            im = ax.imshow(
                channel_abs_2d, origin="lower", interpolation="nearest", aspect="auto"
            )
            ax.set_xlabel("Channel index (layout)")
            ax.set_ylabel("Channel index (layout)")
            ax.set_title(
                "V[k] input-part mean over (H,W), 32 feature channels (4x8, no spatial meaning)\n"
                f"{label}={idx}\n"
                f"feature shape=({C},{H},{W})"
            )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(abs_out, dpi=150)
            plt.close(fig)
            print(f"Saved: {abs_out}")

            signed_out = os.path.join(out_dir, "basis_signed_mean_4x8.png")
            mn = float(np.min(channel_signed_2d))
            mx = float(np.max(channel_signed_2d))
            m = float(max(abs(mn), abs(mx)))
            if m == 0.0:
                m = 1e-8
            vmin, vmax = -m, m
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            im = ax.imshow(
                channel_signed_2d,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
            )
            ax.set_xlabel("Channel index (layout)")
            ax.set_ylabel("Channel index (layout)")
            ax.set_title(
                "V[k] input-part signed mean over (H,W), 32 channels (4x8, no spatial meaning)\n"
                f"{label}={idx}\n"
                f"feature shape=({C},{H},{W})"
            )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(signed_out, dpi=150)
            plt.close(fig)
            print(f"Saved: {signed_out}")

    # -------------------------------------------------------------------------
    # Visualize U_k ⊗ V_k input-part aggregated over channels
    #
    # NOTE: This script assumes export_gawf_sector_basis.py was run with --export_uxv,
    # and therefore the input .pt already contains the aggregated UxV matrices.
    # -------------------------------------------------------------------------
    if "uxv_input_abs_mean" not in obj or "uxv_input_signed_mean" not in obj:
        raise KeyError(
            "Missing UxV matrices in input .pt. Please re-run "
            "export_gawf_sector_basis.py with --export_uxv to include "
            "'uxv_input_abs_mean' and 'uxv_input_signed_mean'."
        )

    abs_mat = _to_numpy_2d(obj["uxv_input_abs_mean"])
    signed_mat = _to_numpy_2d(obj["uxv_input_signed_mean"])
    rec = int(abs_mat.shape[1])

    # Apply optional channel reordering to digit-mode matrices.
    if mode == "digit" and channel_order is not None:
        apply_order = channel_order[::-1]
        abs_mat = abs_mat[apply_order]
        signed_mat = signed_mat[apply_order]

    mat_h, mat_w = int(abs_mat.shape[0]), int(abs_mat.shape[1])
    if mode == "sector":
        ylabel = "Spatial blocks (flattened H*W)"
    else:
        ylabel = "Feature channel"
    shape_str = f"({mat_h},{mat_w})"

    print(f"[viz] final abs_mat.shape={abs_mat.shape} (expected ({mat_h}, {mat_w}))")
    print(f"[viz] final signed_mat.shape={signed_mat.shape} (expected ({mat_h}, {mat_w}))")

    abs_uv_out = os.path.join(out_dir, f"UxV_input_abs_mean_{mat_h}x{mat_w}.png")
    signed_uv_out = os.path.join(out_dir, f"UxV_input_signed_mean_{mat_h}x{mat_w}.png")
    if simple_digit_uv_only:
        # For quick browsing: one UV figure per digit under save_dir/digit/.
        signed_uv_out = os.path.join(
            out_dir, f"{prefix}_{idx}_UxV_input_signed_mean_{mat_h}x{mat_w}.png"
        )
    else:
        # Abs 版本：保持原来的单子图，不做拓展。
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = ax.imshow(
            abs_mat,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlabel("Recurrent units")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"U_k ⊗ V_k input-part signed mean {label}={idx}\n"
            f"matrix shape={shape_str}"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(abs_uv_out, dpi=150)
        plt.close(fig)
        print(f"Saved: {abs_uv_out}")

    # Signed 版本：在 digit 模式下，右侧额外加上一列 CNN row-wise z-score。
    mn = float(np.min(signed_mat))
    mx = float(np.max(signed_mat))
    m = float(max(abs(mn), abs(mx)))
    if m == 0.0:
        m = 1e-8
    vmin, vmax = -m, m

    if mode == "digit":
        cnn_col = _load_cnn_rowwise_column(
            stats_path=args.cnn_stats_path,
            digit=idx,
            channel_order=channel_order,
        )

        fig, (ax_main, ax_cnn) = plt.subplots(
            1, 2,
            figsize=(9.5, 5),
            gridspec_kw={"width_ratios": [4.0, 0.7]},
        )

        # 左：原 UxV signed matrix。
        im_main = ax_main.imshow(
            signed_mat,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax_main.set_xlabel("Recurrent units")
        ax_main.set_ylabel(ylabel)
        if mat_h == 32:
            yticks = list(range(0, 32, 5))
            ax_main.set_yticks(yticks)
            ax_main.set_yticklabels([str(c) for c in yticks])
        ax_main.set_title(
            f"U_k ⊗ V_k input-part signed mean {label}={idx}\n"
            f"matrix shape={shape_str}"
        )
        fig.colorbar(im_main, ax=ax_main)

        # 右：CNN row-wise z-score 单列，colorbar 独立。
        col_img = cnn_col.reshape(-1, 1)
        im_cnn = ax_cnn.imshow(
            col_img,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=-3.0,
            vmax=3.0,
            cmap="RdBu_r",
        )
        ax_cnn.set_xlabel("row-wise z")
        ax_cnn.set_xticks([])
        if mat_h == 32:
            ax_cnn.set_yticks(yticks)
            ax_cnn.set_yticklabels([str(c) for c in yticks])
        # ax_cnn.set_ylabel("Feature channel")
        ax_cnn.set_title(
            "CNN row-wise z-score\n"
            f"digit={idx}"
        )
        fig.colorbar(im_cnn, ax=ax_cnn)

        fig.tight_layout()
        fig.savefig(signed_uv_out, dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        im = ax.imshow(
            signed_mat,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax.set_xlabel("Recurrent units")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"U_k ⊗ V_k input-part signed mean {label}={idx}\n"
            f"matrix shape={shape_str}"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(signed_uv_out, dpi=150)
        plt.close(fig)
    print(f"Saved: {signed_uv_out}")

    # -------------------------------------------------------------------------
    # Additional sector-only figure:
    # Outer UV input-part RF gallery (top-9 recurrent units by abs-mean magnitude).
    #
    # Interpretation:
    # - In sector mode, signed_mat has shape (H*W, rec) where H=W=6 => (36, 256).
    # - For each recurrent unit j, reshape signed_mat[:, j] to a signed 6x6 RF.
    # - Rank by abs_mean_j = mean(abs(RF_j)), but plot the signed RF.
    # -------------------------------------------------------------------------
    if mode == "sector":
        if mat_h != 36:
            print(
                f"[viz][warn] skip outer UV RF gallery: expected mat_h=36 (6x6), got mat_h={mat_h}"
            )
        else:
            rf_h, rf_w = 6, 6
            rf_all = signed_mat.reshape(rf_h, rf_w, mat_w).transpose(2, 0, 1)  # (rec,6,6)
            abs_mean = np.mean(np.abs(rf_all), axis=(1, 2))  # (rec,)
            order = np.argsort(-abs_mean)  # desc
            topk = 9
            top_idx = order[:topk]
            rf_top = rf_all[top_idx]  # (9,6,6) signed
            vmax = float(np.max(np.abs(rf_top)))
            if vmax == 0.0:
                vmax = 1e-8
            vmin = -vmax

            fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.0))
            axes = np.asarray(axes).reshape(-1)
            im0 = None
            for i, (unit_idx, rf) in enumerate(zip(top_idx, rf_top, strict=True)):
                ax = axes[i]
                im0 = ax.imshow(
                    rf,
                    origin="lower",
                    interpolation="nearest",
                    aspect="equal",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="RdBu_r",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"unit={int(unit_idx)}\nabs_mean={float(abs_mean[unit_idx]):.4f}")

            fig.suptitle(
                "Outer UV input-part RF gallery | "
                f"sector={idx} | top-9 units by abs mean",
                y=0.995,
            )

            # Shared colorbar (dedicated axis to avoid overlapping subplots)
            fig.subplots_adjust(
                left=0.05,
                right=0.88,
                top=0.92,
                bottom=0.05,
                wspace=0.15,
                hspace=0.25,
            )
            if im0 is not None:
                cax = fig.add_axes([0.9, 0.18, 0.03, 0.64])
                fig.colorbar(im0, cax=cax)

            gallery_out = os.path.join(out_dir, "outer_uv_input_top9_rf_gallery.png")
            fig.savefig(gallery_out, dpi=150)
            plt.close(fig)
            print(f"Saved: {gallery_out}")

    # Optionally save a simplified copy of the UxV signed-mean matrix
    # under save_dir/sector/ or save_dir/digit/ for quick browsing.
    # In digit+simple mode, we already wrote the only desired output into save_dir/digit/.
    if args.simple and (not simple_digit_uv_only):
        simple_dir = os.path.join(save_dir, prefix)
        os.makedirs(simple_dir, exist_ok=True)
        simple_name = f"{prefix}_{idx}_UxV_input_signed_mean_{mat_h}x{mat_w}.png"
        simple_out = os.path.join(simple_dir, simple_name)
        shutil.copyfile(signed_uv_out, simple_out)
        print(f"Saved (simple): {simple_out}")


if __name__ == "__main__":
    main()
