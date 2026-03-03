"""
Export GaWF gate matrices (gate_ih / gate_hh) for a single sample.

The computation strictly follows the existing GaWFRNNConv implementation:
- Use the same encoder as in train_rnn_updated.py to obtain x_t.
- Run a single GaWF step at t=0 with zero feedback to obtain char_t / pos_t.
- Build feedback fb = cat([char_t, pos_t]) (logits, no softmax), clamp/unsqueeze.
- Recompute gate_ih / gate_hh (and optionally gated_weight_ih / gated_weight_hh)
  using the same formulas as middle_gawf but with the updated feedback.

Uses the test dataset only (splits=("test",)); train/valid are not loaded.

Example commands:

  # 导出 gate 矩阵（使用 test 集）
  python export_gawf_gates.py \\
      --ckpt /path/to/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_model.pth \\
      --split test \\
      --index 0 \\
      --out ./gawf_gates.pt

  # 使用 GPU
  python export_gawf_gates.py \\
      --ckpt /path/to/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_model.pth \\
      --split test \\
      --index 0 \\
      --out ./gawf_gates.pt \\
      --device cuda
"""

import argparse
import os
from typing import Tuple

import torch

from train_rnn_updated import MC_RNN_Dataset
from utils.train_gawf_core import GaWFRNNConv
from utils.train_helpers import (
    create_datasets,
    get_base_path,
    load_raw_data,
    prepare_data_paths,
    set_seed,
)
from viz_single_result import parse_hparams_from_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GaWF gate matrices (gate_ih / gate_hh) for a single sample."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/G/MIMOlab/Codes/aim3_RNN/results/models/sector_40h_uint8_5/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0.3_fb70_model.pth",
        help="Path to trained GaWFRNNConv checkpoint (e.g. *_model.pth).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
        help="Dataset split to sample from (analysis uses test set only).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the chosen split (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./gawf_gates.pt",
        help="Output path for gate dictionary (default: ./gawf_gates.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device: auto / cpu / cuda (default: auto).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Temperature tau used in gate computation (default: 2.0, same as middle_gawf).",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Also save gated_weight_ih / gated_weight_hh.",
    )

    # Dataset-related options (mirroring train_rnn_updated defaults)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Base directory for stimuli/labels. If empty, uses ENV (AIM3_STIMULI_PATH / "
            "FAW_RNN_DATA_PATH) or <repo>/stimuli (same resolution logic as training)."
        ),
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for stimulus_reg-* files (e.g. '40h'). "
            "Same semantics as train_rnn_updated.py."
        ),
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=True,
        help="Use sector mode (3x3 sectors) for position labels (default: True, matches training default).",
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters instead of only foreground (default: False).",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help="Load stimuli with numpy mmap_mode='r' (default: True, matches training default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling order (default: 42).",
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


def build_test_dataset(args: argparse.Namespace) -> Tuple[MC_RNN_Dataset, int]:
    """
    Build test dataset only, using the same helpers as training (splits=("test",)).
    """
    base_path = get_base_path(override=args.data_dir or None)
    paths = prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("test",)
    )
    stims_test, lbls_test = load_raw_data(
        None, None, None, None,
        use_mmap=args.use_mmap,
        paths_tuple=paths,
    )

    test_ds, num_pos = create_datasets(
        None, None, None, None,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    return test_ds, num_pos


def pick_sample(test_ds: MC_RNN_Dataset, index: int):
    """Select a single sample from the test dataset by index."""
    if index < 0 or index >= len(test_ds):
        raise IndexError(f"Index {index} out of range for test set (len={len(test_ds)}).")
    frames, labels = test_ds[index]
    return frames, labels, test_ds


def build_model_from_ckpt(
    ckpt_path: str,
    num_pos: int,
    device: torch.device,
) -> GaWFRNNConv:
    """
    Instantiate GaWFRNNConv with hyperparameters parsed from the checkpoint filename,
    then load the checkpoint state_dict.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    dropout_rate = hparams.get("dropout", 0.3)

    num_classes = 10  # fixed in training script

    model = GaWFRNNConv(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=5,
        device=str(device),
        dropout_rate=dropout_rate,
        hidden_size=hidden_size,
        max_chars=15,
        predict_all_chars=(num_pos == 0),
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    # prev_feedback is a runtime buffer; skip it so loading does not fail (we reset feedback in this script)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing_keys:", load_result.missing_keys)
    print("[load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # 1) Build test dataset only (same helpers as training, splits=("test",)).
    test_ds, num_pos = build_test_dataset(args)

    # 2) Select a single sample from test set.
    frames, labels, ds = pick_sample(test_ds, args.index)
    print(f"Selected sample index {args.index} from split test, frames shape={getattr(frames, 'shape', None)}")

    # 3) Prepare input tensor: (B=1, T=1, C, H, W)
    if isinstance(frames, torch.Tensor):
        frames_t = frames
    else:
        frames_t = torch.as_tensor(frames)

    if frames_t.ndim == 4:
        # Dataset returns (T, C, H, W). We only need the first time step (t=0).
        first_frame = frames_t[1:2]  # (1, C, H, W)
        x_input = first_frame.unsqueeze(0)  # (1, 1, C, H, W)
    elif frames_t.ndim == 3:
        # Single frame (C, H, W) -> treat as T=1.
        x_input = frames_t.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    else:
        raise ValueError(f"Unexpected frames tensor shape: {frames_t.shape}")

    x_input = x_input.to(device=device, dtype=torch.float32)

    # 4) Build and load GaWFRNNConv model from checkpoint.
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    print(
        f"Loaded GaWFRNNConv from '{args.ckpt}' "
        f"(hidden_size={model.hidden_size}, num_pos={model.num_pos})"
    )

    # 5) Single-step forward at t=0 to obtain prev_feedback using fb=0.
    with torch.no_grad():
        # Encoder path must exactly match GaWFRNNConv.forward.
        batch_size, frame_num, channels, height, width = x_input.size()
        x_flat = x_input.view(batch_size * frame_num, channels, height, width)
        x_encoded = model.encoder(x_flat)
        x_seq = x_encoded.view(batch_size, frame_num, -1)

        # t=0 step
        x_t = x_seq[:, 0, :]  # (B, input_size)
        fb_dim = model.num_classes + model.num_pos
        fb = torch.zeros(batch_size, fb_dim, device=device, dtype=torch.float32)

        hidden_size = model.rnn.hidden_size
        h_prev = torch.zeros(batch_size, hidden_size, device=device, dtype=x_t.dtype)

        # First middle_gawf call with zero feedback (not used for gate export,
        # only to obtain logits and thus prev_feedback).
        fb_t0 = fb.clamp(-10, 10).unsqueeze(2)
        gated_output0 = model.middle_gawf(x_t, h_prev, fb_t0)
        char_t, pos_t = model.classifier(gated_output0)

        # Feedback uses raw logits, exactly as in forward().
        prev_fb = torch.cat([char_t, pos_t], dim=-1)

        # 6) Recompute gates using updated feedback and the same x_t / h_prev.
        fb_t1 = prev_fb.clamp(-10, 10).unsqueeze(2)

        input_size = x_t.size(-1)
        weight_ih = model.rnn.weight_ih_l0  # (hidden_size, input_size)
        weight_hh = model.rnn.weight_hh_l0  # (hidden_size, hidden_size)

        V_ih = model.V[:, :input_size].unsqueeze(0)
        V_hh = model.V[:, input_size:].unsqueeze(0)

        # --- diagnostics (gate 计算前后) ---
        U, V = model.U, model.V
        print("[U] shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            tuple(U.shape), U.min().item(), U.max().item(), U.mean().item(), U.std().item()))
        print("[V] shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f" % (
            tuple(V.shape), V.min().item(), V.max().item(), V.mean().item(), V.std().item()))
        ft1 = fb_t1.squeeze()
        print("[fb_t1] shape=%s min=%.4f max=%.4f std=%.4f" % (
            tuple(fb_t1.shape), ft1.min().item(), ft1.max().item(), ft1.std().item()))
        fb_V_ih = fb_t1 * V_ih
        print("[fb_t1 * V_ih] mean=%.4f max=%.4f" % (fb_V_ih.mean().item(), fb_V_ih.abs().max().item()))

        trans_ih = torch.matmul(model.U, fb_V_ih)
        trans_hh = torch.matmul(model.U, fb_t1 * V_hh)

        print("[trans_ih] mean=%.4f max=%.4f" % (trans_ih.mean().item(), trans_ih.abs().max().item()))

        tau = float(args.tau)
        gate_ih = torch.sigmoid(trans_ih / tau)
        gate_hh = torch.sigmoid(trans_hh / tau)

        gated_weight_ih = gate_ih * weight_ih.unsqueeze(0)
        gated_weight_hh = gate_hh * weight_hh.unsqueeze(0)

    # 7) Prepare tensors for saving (remove batch dimension).
    gate_ih_2d = gate_ih.squeeze(0).detach().cpu()
    gate_hh_2d = gate_hh.squeeze(0).detach().cpu()
    prev_fb_vec = prev_fb.squeeze(0).detach().cpu()

    save_dict = {
        "gate_ih": gate_ih_2d,
        "gate_hh": gate_hh_2d,
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "fb_dim": int(prev_fb_vec.numel()),
        "tau": float(args.tau),
        "prev_fb": prev_fb_vec,
        "sample_index": int(args.index),
        "split": args.split,
        "ckpt_path": os.path.abspath(args.ckpt),
    }

    if args.save_weights:
        save_dict["gated_weight_ih"] = gated_weight_ih.squeeze(0).detach().cpu()
        save_dict["gated_weight_hh"] = gated_weight_hh.squeeze(0).detach().cpu()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(save_dict, args.out)

    # 8) Required diagnostics / prints.
    print(f"Saved gate dictionary to: {args.out}")
    print(f"gate_ih shape: {tuple(gate_ih_2d.shape)}")
    print(f"gate_hh shape: {tuple(gate_hh_2d.shape)}")

    prev_fb_min = prev_fb_vec.min().item()
    prev_fb_max = prev_fb_vec.max().item()
    print(f"prev_fb min: {prev_fb_min:.6f}, max: {prev_fb_max:.6f}")

    gate_ih_min = gate_ih_2d.min().item()
    gate_ih_max = gate_ih_2d.max().item()
    gate_ih_mean = gate_ih_2d.mean().item()
    print(f"gate_ih min: {gate_ih_min:.6f}, max: {gate_ih_max:.6f}, mean: {gate_ih_mean:.6f}")

    gate_hh_min = gate_hh_2d.min().item()
    gate_hh_max = gate_hh_2d.max().item()
    gate_hh_mean = gate_hh_2d.mean().item()
    print(f"gate_hh min: {gate_hh_min:.6f}, max: {gate_hh_max:.6f}, mean: {gate_hh_mean:.6f}")


if __name__ == "__main__":
    main()

