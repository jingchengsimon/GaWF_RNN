"""
Export GaWF sector-specific modulation basis from matrix V.

This script extracts a single sector (row) V[k] from a trained GaWFRNNConv checkpoint,
then analyzes only the input-part modulation basis (first input_size elements).

Output:
  - A .pt file containing basis vectors, reshaped feature maps, summary maps, and stats.

Important constraints (per project request):
  - Do NOT use U.
  - Analyze only V[k] for one sector (default 0).
  - Do NOT visualize or export the whole V.
  - Reshape must satisfy: assert C * H * W == input_size.
  - The (C,H,W) feature shape must match the CNN encoder output to RNN feature shape.
"""

import argparse
import os
from typing import Dict, Tuple

import torch

from utils.train_gawf_core import GaWFRNNConv
from viz_utils.viz_single_result import parse_hparams_from_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GaWF sector-specific modulation basis from V[k] (input-part only)."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/models/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        help="Path to trained GaWFRNNConv checkpoint (e.g. *_model.pth).",
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        help=(
            "Sector index k to analyze (sector feedback rows in V). "
            "If neither --sector nor --digit is provided, defaults to sector=0."
        ),
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=list(range(10)),
        help=(
            "Digit index d (0-9) to analyze using the character-feedback rows in V. "
            "If provided without --sector, digit mode is used and sector mode is skipped."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/gawf_sector_basis_exports",
        help="Directory to save exported .pt results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device: auto / cpu / cuda (default: auto).",
    )
    return parser.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --device cuda but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> GaWFRNNConv:
    """
    Instantiate GaWFRNNConv with hyperparameters parsed from the checkpoint filename,
    then load the checkpoint state_dict.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    dropout_rate = hparams.get("dropout", 0.3)

    # Project convention: sector mode => num_pos=9. (We only need V shape; keep consistent.)
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

    model.to(device)
    model.eval()
    return model


def infer_feature_shape(model: GaWFRNNConv, device: torch.device) -> Tuple[int, int, int]:
    """
    Infer (C,H,W) by running the model encoder once on a dummy input.
    This avoids loading dataset and guarantees alignment with encoder->RNN feature shape.
    """
    # BaseConvSequenceModel is built around a fixed "large" config expecting conv1 input channels=2
    # and MP1 output 48x48, implying input spatial size 96x96.
    dummy = torch.zeros(1, 2, 96, 96, device=device, dtype=torch.float32)
    with torch.no_grad():
        feat = model.encoder(dummy)
    if feat.ndim != 4:
        raise RuntimeError(f"Unexpected encoder output shape: {tuple(feat.shape)} (expect 4D)")
    _, C, H, W = feat.shape
    return int(C), int(H), int(W)


def stats_dict(x: torch.Tensor) -> Dict[str, float]:
    x_f = x.detach().to(dtype=torch.float32).reshape(-1)
    return {
        "min": float(x_f.min().item()) if x_f.numel() else float("nan"),
        "max": float(x_f.max().item()) if x_f.numel() else float("nan"),
        "mean": float(x_f.mean().item()) if x_f.numel() else float("nan"),
        "std": float(x_f.std(unbiased=False).item()) if x_f.numel() else float("nan"),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = os.path.abspath(args.ckpt)
    sector_arg = args.sector
    digit_arg = args.digit

    # Determine mode: sector vs digit
    use_sector = sector_arg is not None
    use_digit = digit_arg is not None

    if use_sector and use_digit:
        raise ValueError(
            "Both --sector and --digit were provided. "
            "Please specify only one of them."
        )
    if not use_sector and not use_digit:
        # Default behavior: sector=0 (backward-compatible)
        use_sector = True
        sector = 0
    elif use_sector:
        sector = int(sector_arg)
    else:
        # digit-only mode
        digit = int(digit_arg)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Load model and V
    model = build_model_from_ckpt(ckpt_path, device=device)
    V = model.V.detach().cpu()

    # Infer sizes
    input_size = int(model.encoder_flatten_size)
    recurrent_size = int(model.rnn.hidden_size)
    C, H, W = infer_feature_shape(model, device=device)
    assert C * H * W == input_size, (
        f"Feature shape mismatch: C*H*W={C*H*W} but input_size={input_size}. "
        "Encoder output shape must match the CNN->RNN feature shape."
    )

    # Extract basis (sector or digit mode)
    if V.ndim != 2:
        raise RuntimeError(f"Unexpected V.ndim={V.ndim}, V.shape={tuple(V.shape)}")
    feedback_dim, combined = V.shape
    expected_combined = input_size + recurrent_size
    if combined != expected_combined:
        raise RuntimeError(
            f"Unexpected V.shape={tuple(V.shape)}; expected (feedback_dim, input_size+recurrent_size)="
            f"(*, {expected_combined}). Got combined={combined}."
        )

    num_classes = int(model.num_classes)
    num_pos = int(model.num_pos)
    if feedback_dim != num_classes + num_pos:
        raise RuntimeError(
            f"Inconsistent feedback_dim: V.shape[0]={feedback_dim}, "
            f"but num_classes+num_pos={num_classes + num_pos}."
        )

    mode = "sector" if use_sector else "digit"
    if mode == "sector":
        if not (0 <= sector < num_pos):
            raise IndexError(
                f"sector={sector} out of valid range [0, num_pos={num_pos}). "
                f"(V.shape[0]={feedback_dim}, first {num_classes} rows are char logits, "
                f"last {num_pos} rows are sector feedback.)"
            )
        # 实际在 V 中对应的行索引：跳过前 num_classes 行，落在 sector 部分。
        row_idx = num_classes + sector
    else:
        if not (0 <= digit < num_classes):
            raise IndexError(
                f"digit={digit} out of valid range [0, num_classes={num_classes}). "
                f"(V.shape[0]={feedback_dim}, first {num_classes} rows are char logits, "
                f"last {num_pos} rows are sector feedback.)"
            )
        # Digit rows are in the first num_classes rows.
        row_idx = digit

    basis_vec = V[row_idx].clone()  # (input_size + recurrent_size,)
    basis_input = basis_vec[:input_size].clone()  # (input_size,)

    basis_input_map = basis_input.view(C, H, W)

    if mode == "sector":
        # Sector: average over feature channels -> spatial map (H, W)
        abs_mean_map = basis_input_map.abs().mean(dim=0)  # (H, W)
        signed_mean_map = basis_input_map.mean(dim=0)  # (H, W)
        channel_abs = None
        channel_signed = None
    else:
        # Digit: no spatial meaning; average over H, W -> one value per channel (C,)
        channel_abs = basis_input_map.abs().mean(dim=(1, 2))  # (C,)
        channel_signed = basis_input_map.mean(dim=(1, 2))  # (C,)
        abs_mean_map = basis_input_map.abs().mean(dim=0)  # (H,W) kept for file compat
        signed_mean_map = basis_input_map.mean(dim=0)  # (H,W) kept for file compat

    if mode == "sector":
        out_path = os.path.join(save_dir, f"sector_{sector}_basis.pt")
    else:
        out_path = os.path.join(save_dir, f"digit_{digit}_basis.pt")

    save_obj = {
        "mode": mode,
        "sector": int(sector) if mode == "sector" else None,
        "digit": int(digit) if mode == "digit" else None,
        "row_idx": int(row_idx),
        "V_shape": tuple(V.shape),
        "feature_shape": (int(C), int(H), int(W)),
        "basis_vec": basis_vec,
        "basis_input": basis_input,
        "abs_mean_map": abs_mean_map,
        "signed_mean_map": signed_mean_map,
        "ckpt_path": ckpt_path,
    }
    if mode == "digit":
        save_obj["channel_abs"] = channel_abs
        save_obj["channel_signed"] = channel_signed

    summary_stats = {
        "basis_vec": stats_dict(basis_vec),
        "basis_input": stats_dict(basis_input),
        "abs_mean_map": stats_dict(abs_mean_map),
        "signed_mean_map": stats_dict(signed_mean_map),
    }
    if mode == "digit":
        summary_stats["channel_abs"] = stats_dict(channel_abs)
        summary_stats["channel_signed"] = stats_dict(channel_signed)
    save_obj["summary_stats"] = summary_stats
    torch.save(save_obj, out_path)

    # Required logs
    print(f"checkpoint path: {ckpt_path}")
    print(f"V shape: {tuple(V.shape)}")
    if mode == "sector":
        print(f"mode: sector, sector: {sector}, row_idx: {row_idx}")
    else:
        print(f"mode: digit, digit: {digit}, row_idx: {row_idx}")
    print(f"input_size: {input_size}")
    print(f"recurrent_size: {recurrent_size}")
    print(f"feature shape: {(C, H, W)}")
    print(f"save path: {out_path}")


if __name__ == "__main__":
    main()

