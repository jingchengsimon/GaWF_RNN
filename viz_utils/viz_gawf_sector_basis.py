"""
Visualize GaWF sector-specific modulation basis maps exported by export_gawf_sector_basis.py.

Generates two heatmaps:
  1) abs mean map across channels
  2) signed mean map across channels (symmetric color range around 0)
"""

import argparse
import os
from typing import Tuple

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.train_gawf_core import GaWFRNNConv  # noqa: E402
from viz_utils.viz_single_result import parse_hparams_from_filename  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GaWF sector-specific basis maps (abs mean / signed mean)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./gawf_sector_basis_exports",
        help=(
            "Path to exported .pt file from export_gawf_sector_basis.py. "
            "If a directory is given, file name will be auto-completed as "
            "'sector_{sector_idx}_basis.pt'."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./gawf_sector_basis_figs",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--sector_idx",
        type=int,
        default=None,
        help="Sector index k whose basis file will be visualized. Mutually exclusive with --digit.",
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=list(range(10)),
        help="Digit index d (0-9) whose basis file will be visualized. Mutually exclusive with --sector_idx.",
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


def _build_model_from_ckpt(ckpt_path: str, device: torch.device) -> GaWFRNNConv:
    """
    Rebuild GaWFRNNConv from checkpoint, mirroring export_gawf_sector_basis.py.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    dropout_rate = hparams.get("dropout", 0.3)

    num_classes = 10
    num_pos = 9

    model = GaWFRNNConv(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=5,
        device=str(device),
        dropout_rate=dropout_rate,
        hidden_size=hidden_size,
        max_chars=15,
        predict_all_chars=False,
    )

    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[viz][load_state_dict] missing_keys:", load_result.missing_keys)
    print("[viz][load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    # Determine mode: sector vs digit
    use_sector = args.sector_idx is not None
    use_digit = args.digit is not None

    if use_sector and use_digit:
        raise ValueError(
            "Both --sector_idx and --digit were provided. "
            "Please specify only one of them."
        )
    if not use_sector and not use_digit:
        # Default behavior: sector_idx=0 (backward-compatible)
        mode = "sector"
        idx = 0
    elif use_sector:
        mode = "sector"
        idx = int(args.sector_idx)
    else:
        mode = "digit"
        idx = int(args.digit)

    label = "sector" if mode == "sector" else "digit"

    # If input_path points to a directory, auto-complete file name using sector_idx or digit.
    raw_in_path = args.input_path
    if raw_in_path is None or raw_in_path == "":
        raw_in_path = "./gawf_sector_basis_exports"
    if os.path.isdir(raw_in_path) or not os.path.splitext(raw_in_path)[1]:
        prefix = "sector" if mode == "sector" else "digit"
        raw_in_path = os.path.join(raw_in_path, f"{prefix}_{idx}_basis.pt")

    in_path = os.path.abspath(raw_in_path)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # One subdir per sector/digit: sector_k or digit_k
    out_dir = os.path.join(save_dir, f"{prefix}_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    obj = torch.load(in_path, map_location="cpu")

    C, H, W = _read_feature_shape(obj)
    prefix = "sector" if mode == "sector" else "digit"

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

        # Layout 4x8 for 32 channels
        display_h, display_w = 4, 8
        if C != display_h * display_w:
            raise ValueError(f"Expected C=32 for 4x8 layout, got C={C}")
        channel_abs_2d = channel_abs.reshape(display_h, display_w)
        channel_signed_2d = channel_signed.reshape(display_h, display_w)

        abs_out = os.path.join(out_dir, "basis_abs_mean_4x8.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(channel_abs_2d, origin="lower", interpolation="nearest", aspect="auto")
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
    # NEW: Visualize U_k ⊗ V_k input-part aggregated over channels
    # -------------------------------------------------------------------------
    ckpt_path = obj.get("ckpt_path", None)
    if ckpt_path is None:
        print(
            "[viz][warn] ckpt_path not found in input .pt; "
            "skip U_k ⊗ V_k visualization."
        )
        return

    ckpt_path = os.path.abspath(str(ckpt_path))
    device = torch.device("cpu")
    model = _build_model_from_ckpt(ckpt_path, device=device)

    # Shapes and consistency checks
    V = model.V.detach().cpu()
    U = model.U.detach().cpu()
    input_size_model = int(model.encoder_flatten_size)
    rec = int(model.rnn.hidden_size)
    input_size_from_feature = int(C * H * W)
    if input_size_model != input_size_from_feature:
        raise RuntimeError(
            f"Inconsistent input size: encoder_flatten_size={input_size_model}, "
            f"but C*H*W={input_size_from_feature} (C,H,W=({C},{H},{W}))."
        )

    feedback_dim, combined = V.shape
    expected_combined = input_size_model + rec
    if combined != expected_combined:
        raise RuntimeError(
            f"Unexpected V.shape={tuple(V.shape)}; expected "
            f"(feedback_dim, input_size+recurrent_size)=(*, {expected_combined})."
        )
    if U.shape != (rec, feedback_dim):
        raise RuntimeError(
            f"Unexpected U.shape={tuple(U.shape)}; expected (rec, feedback_dim)=({rec}, {feedback_dim})."
        )

    num_classes = int(model.num_classes)
    num_pos = int(model.num_pos)
    if feedback_dim != num_classes + num_pos:
        raise RuntimeError(
            f"Inconsistent feedback_dim: V.shape[0]={feedback_dim}, "
            f"but num_classes+num_pos={num_classes + num_pos}."
        )

    if mode == "sector":
        if not (0 <= idx < num_pos):
            raise IndexError(
                f"sector_idx={idx} out of valid range [0, num_pos={num_pos}). "
                f"(V.shape[0]={feedback_dim}, first {num_classes} rows are char logits, "
                f"last {num_pos} rows are sector feedback.)"
            )
        row_idx = num_classes + idx
    else:
        if not (0 <= idx < num_classes):
            raise IndexError(
                f"digit={idx} out of valid range [0, num_classes={num_classes}). "
                f"(V.shape[0]={feedback_dim}, first {num_classes} rows are char logits, "
                f"last {num_pos} rows are sector feedback.)"
            )
        row_idx = idx

    U_k = U[:, row_idx]  # (rec,)
    V_k = V[row_idx]  # (input_size + rec,)

    # Shape logging / asserts
    print(f"[viz] U_k.shape={tuple(U_k.shape)} (expected ({rec},))")
    print(
        f"[viz] V_k.shape={tuple(V_k.shape)} (expected ({input_size_model + rec},))"
    )

    if U_k.shape[0] != rec:
        raise RuntimeError(
            f"U_k has wrong shape {tuple(U_k.shape)}; expected ({rec},)."
        )
    if V_k.shape[0] != input_size_model + rec:
        raise RuntimeError(
            f"V_k has wrong shape {tuple(V_k.shape)}; expected ({input_size_model + rec},)."
        )

    # Outer product: (input+rec, rec) with rows from V_k, cols from U_k
    gate_k = torch.outer(V_k, U_k)  # (input_size+rec, rec)
    print(
        f"[viz] gate_k.shape={tuple(gate_k.shape)} "
        f"(expected ({input_size_model + rec}, {rec}))"
    )

    # Input part only: first input_size rows
    gate_input = gate_k[:input_size_model, :]  # (input_size, rec)
    print(
        f"[viz] gate_input.shape={tuple(gate_input.shape)} "
        f"(expected ({input_size_model}, {rec}))"
    )

    # Reshape to (C,H,W,rec)
    gate_input_4d = gate_input.view(C, H, W, rec)
    print(
        f"[viz] gate_input_4d.shape={tuple(gate_input_4d.shape)} "
        f"(expected ({C}, {H}, {W}, {rec}))"
    )

    if mode == "sector":
        # Sector: average over channels (dim=0) -> (H,W,rec), flatten to (HW, rec)
        abs_mean_3d = gate_input_4d.abs().mean(dim=0)  # (H,W,rec)
        signed_mean_3d = gate_input_4d.mean(dim=0)  # (H,W,rec)
        HW = int(H * W)
        abs_mat = abs_mean_3d.view(HW, rec).detach().cpu().numpy()
        signed_mat = signed_mean_3d.view(HW, rec).detach().cpu().numpy()
        mat_h, mat_w = HW, rec
        ylabel = "Spatial blocks (flattened H*W)"
        shape_str = f"({HW},{rec})"
    else:
        # Digit: average over (H,W) (dim 1,2) -> (C, rec); each row = one feature channel
        abs_mat_2d = gate_input_4d.abs().mean(dim=(1, 2))  # (C, rec)
        signed_mat_2d = gate_input_4d.mean(dim=(1, 2))  # (C, rec)
        abs_mat = abs_mat_2d.detach().cpu().numpy()
        signed_mat = signed_mat_2d.detach().cpu().numpy()
        mat_h, mat_w = C, rec
        ylabel = "Feature channel"
        shape_str = f"({C},{rec})"

    print(f"[viz] final abs_mat.shape={abs_mat.shape} (expected ({mat_h}, {mat_w}))")
    print(f"[viz] final signed_mat.shape={signed_mat.shape} (expected ({mat_h}, {mat_w}))")

    abs_uv_out = os.path.join(
        out_dir, f"UxV_input_abs_mean_{mat_h}x{mat_w}.png"
    )
    signed_uv_out = os.path.join(
        out_dir, f"UxV_input_signed_mean_{mat_h}x{mat_w}.png"
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(
        abs_mat,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("Recurrent units")
    ax.set_ylabel(ylabel)
    # For digit mode with 32 feature channels, align y-ticks with CNN feature matrices.
    if mode == "digit" and mat_h == 32:
        yticks = list(range(0, 32, 4))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(c) for c in yticks])
    ax.set_title(
        "U_k ⊗ V_k input-part abs mean"
        + (" across channels" if mode == "sector" else " over (H,W), rows=channels")
        + "\n"
        f"{label}={idx}\n"
        f"feature shape=({C},{H},{W})\n"
        f"matrix shape={shape_str}"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(abs_uv_out, dpi=150)
    plt.close(fig)
    print(f"Saved: {abs_uv_out}")

    mn = float(np.min(signed_mat))
    mx = float(np.max(signed_mat))
    m = float(max(abs(mn), abs(mx)))
    if m == 0.0:
        m = 1e-8
    vmin, vmax = -m, m

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
        "U_k ⊗ V_k input-part signed mean"
        + (" across channels" if mode == "sector" else " over (H,W), rows=channels")
        + "\n"
        f"{label}={idx}\n"
        f"feature shape=({C},{H},{W})\n"
        f"matrix shape={shape_str}"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(signed_uv_out, dpi=150)
    plt.close(fig)
    print(f"Saved: {signed_uv_out}")


if __name__ == "__main__":
    main()
