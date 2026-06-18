"""
Export a single contiguous test-segment population activity (T, hidden_size) from a
trained sequence model (GaWF / RNN / GRU / LSTM).

- Inputs match MC_RNN_Dataset channel stacking (default chan_num=2).
- Hidden state (and GaWF feedback) carry across all T steps; eval mode (no dropout).
- Saves under --save_dir/<run_tag>/: pop_act.npy, labels.tsv (same layout idea as other utils_anal exports).
- model_type defaults from checkpoint basename (gawf_/rnn_/lstm_/gru_); optional --model_type override.
- Optional --write_pop_act_dpca: same folder, mean over digit×sector → pop_act_dpca.npy (N,10,9).

Label columns match GenerateMovies TSV header (9 columns).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from train_model import MC_RNN_Dataset
from utils.train_gawf_core import GaWFRNNConv, MultiLayerGaWFRNNConv
from utils.train_rnn_core import GRUConv, LSTMConv, RNNConv
from utils.train_helpers import PathHelper, create_datasets, set_seed
from utils_viz.model_train_single_result import parse_hparams_from_filename

# parse_hparams_from_filename uses title case; cls_map keys are lowercase
_HPARAM_MODEL_TO_KEY = {
    "RNN": "rnn",
    "LSTM": "lstm",
    "GRU": "gru",
    "GaWF": "gawf",
    "GaWFMulti": "gawf_multi",
}

LABEL_COLUMNS = [
    "frame",
    "fg_char_id",
    "fg_char_x",
    "fg_char_y",
    "bg_char_ids",
    "fg_speed",
    "bg_mean_speed",
    "fg_switch",
    "bg_switch",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export contiguous pop_act (T, H) + labels.tsv from test split.")
    p.add_argument(
        "--ckpt",
        type=str,
        # default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth",
        # default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/rnn_sector_acc_h275_lr0.0005_wd0.0001_do0_model.pth",
        # default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/lstm_sector_acc_h80_lr0.0005_wd0.0001_do0_model.pth",
        default="/G/MIMOlab/Codes/aim3_RNN/results/train_data/sector_40h_adamw/gru_sector_acc_h105_lr0.0005_wd0.0001_do0_model.pth",
        help="Path to trained checkpoint (*_model.pth).",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["gawf", "gawf_multi", "rnn", "lstm", "gru"],
        help=(
            "Override architecture. Default: inferred from checkpoint basename prefix "
            "(gawf_ / gawf_multi_ / rnn_ / lstm_ / gru_), same convention as training artifact names."
        ),
    )
    p.add_argument("--T", type=int, default=None, help="Segment length in frames (overrides duration_sec*fps if set).")
    p.add_argument("--duration_sec", type=float, default=2400-1, help="Segment duration in seconds if --T not set.")
    p.add_argument("--fps", type=float, default=24.0, help="Frames per second for --duration_sec (default: 30).")
    p.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/pop_act",
        help="Parent directory; writes <save_dir>/<run_tag>/{pop_act.npy, labels.tsv}.",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Subfolder under --save_dir; default: checkpoint filename without extension.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    # Dataset args (aligned with export_gate_avg_allsector.py / export_gate_avg.py)
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--data_suffix", type=str, default="")
    p.add_argument("--use_sector_mode", action="store_true", default=True)
    p.add_argument("--predict_all_chars", action="store_true", default=False)
    p.add_argument("--use_mmap", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42, help="Random seed (segment start + NumPy).")
    p.add_argument(
        "--chan_num",
        type=int,
        default=2,
        help="Input channels (must match training; default 2, same as MC_RNN_Dataset).",
    )
    p.add_argument(
        "--write_pop_act_dpca",
        action="store_true",
        help=(
            "After export, aggregate pop_act.npy + labels.tsv by fg digit (0–9) × fg sector (0–8); "
            "save (N,10,9) mean as pop_act_dpca.npy in the same run directory."
        ),
    )
    p.add_argument(
        "--dpca_frame_height",
        type=int,
        default=96,
        help="Stimulus height for sector mapping when --write_pop_act_dpca (match stimuli).",
    )
    p.add_argument(
        "--dpca_frame_width",
        type=int,
        default=96,
        help="Stimulus width for sector mapping when --write_pop_act_dpca.",
    )
    p.add_argument(
        "--dpca_num_sectors",
        type=int,
        default=9,
        help="Number of sectors (default 9 = 3×3) when --write_pop_act_dpca.",
    )
    return p.parse_args()


def infer_model_type_from_ckpt(ckpt_path: str) -> str:
    """
    Resolve model class key from checkpoint filename (parse_hparams_from_filename).
    """
    base = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(base)
    raw = hparams.get("model_type")
    if raw is None:
        raise ValueError(
            f"Cannot infer model_type from filename {base!r}. "
            "Expected basename to start with gawf_, gawf_multi_, rnn_, lstm_, or gru_ "
            "(see parse_hparams_from_filename in utils_viz/model_train_single_result.py). "
            "Pass --model_type explicitly."
        )
    key = _HPARAM_MODEL_TO_KEY.get(raw)
    if key is None:
        raise ValueError(f"Unknown model_type {raw!r} from filename {base!r}")
    return key


def resolve_T(args: argparse.Namespace) -> int:
    if args.T is not None:
        return int(args.T)
    return int(round(args.duration_sec * args.fps))


def pick_segment_start(rng: np.random.Generator, n_frames: int, seg_len: int, chan_num: int) -> int:
    """
    First frame index in the segment (global stimulus index g for timestep 0).

    Requires g >= chan_num - 1 so each stacked input has valid look-back rows.
    """
    min_g = chan_num - 1
    max_start = n_frames - seg_len
    if max_start < min_g:
        raise ValueError(
            f"Not enough frames for segment: n_frames={n_frames}, T={seg_len}, "
            f"need n_frames >= T + (chan_num-1) = {seg_len + min_g}"
        )
    return int(rng.integers(min_g, max_start + 1))


def stacked_frame_at(
    data: np.ndarray,
    global_frame_idx: int,
    chan_num: int,
) -> np.ndarray:
    """One model input: (chan_num, H, W), float32, same stacking as MC_RNN_Dataset."""
    block = data[global_frame_idx - (chan_num - 1) : global_frame_idx + 1]
    if block.shape[0] != chan_num:
        raise ValueError(f"Bad block shape {block.shape} for global_frame_idx={global_frame_idx}, C={chan_num}")
    return np.asarray(block, dtype=np.float32)


def load_test_stimuli_and_num_pos(args: argparse.Namespace):
    base_path = PathHelper.get_base_path(override=args.data_dir or None)
    paths = PathHelper.prepare_data_paths(base_path, data_suffix=args.data_suffix, splits=("test",))
    stims_test, lbls_test = PathHelper.load_raw_data(
        None,
        None,
        None,
        None,
        use_mmap=args.use_mmap,
        paths_tuple=paths,
    )
    _, num_pos = create_datasets(
        None,
        None,
        None,
        None,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    return stims_test, lbls_test, num_pos


def build_model(
    model_type: str,
    ckpt_path: str,
    num_pos: int,
    predict_all_chars: bool,
    device: torch.device,
) -> torch.nn.Module:
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)
    hidden_size = int(hparams.get("hidden_size", 256))
    cnn_dropout = float(hparams.get("cnn_dropout", 0.0))
    rnn_dropout = float(hparams.get("rnn_dropout", hparams.get("dropout", 0.5)))
    num_classes = 10
    kernel_size = 5

    cls_map = {
        "gawf": GaWFRNNConv,
        "gawf_multi": MultiLayerGaWFRNNConv,
        "rnn": RNNConv,
        "lstm": LSTMConv,
        "gru": GRUConv,
    }
    ModelClass = cls_map[model_type]
    model_kwargs = {}
    if model_type in ("gawf", "gawf_multi"):
        parsed_feedback_dim = hparams.get("feedback_dim")
        if parsed_feedback_dim is not None:
            model_kwargs["feedback_dim"] = int(parsed_feedback_dim)
        if model_type == "gawf_multi":
            model_kwargs["num_layers"] = int(hparams.get("gawf_layers", 2))

    mdl = ModelClass(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=kernel_size,
        device=str(device),
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        hidden_size=hidden_size,
        max_chars=15,
        predict_all_chars=predict_all_chars,
        **model_kwargs,
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = mdl.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing_keys:", load_result.missing_keys)
    print("[load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    mdl.to(device)
    mdl.eval()
    if hasattr(mdl, "prev_feedback"):
        mdl.prev_feedback = None
    return mdl


def run_stream_pop_act(
    model: torch.nn.Module,
    x_tc_hw: np.ndarray,
    device: torch.device,
    model_type: str,
) -> np.ndarray:
    """
    x_tc_hw: (T, C, H, W) float32 on CPU; returns (T, hidden_size).
    """
    T, C, H, W = x_tc_hw.shape
    x = torch.from_numpy(x_tc_hw).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,T,C,H,W)
    enc_in = x.view(T, C, H, W)
    enc = model.encoder(enc_in)
    seq = enc.view(1, T, -1)

    # GaWF variants set self.hidden_size; BaseRNNConv subclasses expose rnn.hidden_size.
    if hasattr(model, "hidden_size"):
        Hdim = int(model.hidden_size)
    else:
        Hdim = int(model.rnn.hidden_size)
    out_np = np.zeros((T, Hdim), dtype=np.float32)

    with torch.no_grad():
        if model_type == "gawf":
            fb = torch.zeros(1, model.feedback_dim, device=device, dtype=torch.float32)
            h = torch.zeros(1, Hdim, device=device, dtype=seq.dtype)
            for t in range(T):
                x_t = seq[:, t, :]
                fb_t = fb.clamp(-10, 10).unsqueeze(2)
                gated_output = model.middle_gawf(x_t, h, fb_t)
                char_t, pos_t = model.classifier(gated_output)
                fb = model._compute_feedback(char_t, pos_t)
                h = gated_output
                out_np[t] = gated_output.squeeze(0).cpu().numpy()
        elif model_type == "gawf_multi":
            fb_top = torch.zeros(
                1, model.top_feedback_dim, device=device, dtype=torch.float32
            )
            h_states = [
                torch.zeros(1, Hdim, device=device, dtype=seq.dtype)
                for _ in range(model.num_layers)
            ]
            for t in range(T):
                layer_input = seq[:, t, :]
                next_h_states = []
                for layer_idx in range(model.num_layers):
                    if layer_idx == model.num_layers - 1:
                        fb = fb_top
                    elif model.use_feedback_projector:
                        fb = model.hidden_projectors[layer_idx](h_states[layer_idx + 1])
                    else:
                        fb = model._compute_hidden_feedback(h_states[layer_idx + 1])
                    fb_t = fb.clamp(-10, 10).unsqueeze(2)
                    h_t = model.middle_gawf_layer(
                        layer_idx,
                        layer_input,
                        h_states[layer_idx],
                        fb_t,
                    )
                    next_h_states.append(h_t)
                    layer_input = h_t
                char_t, pos_t = model.classifier(layer_input)
                fb_top = model._compute_output_feedback(char_t, pos_t)
                h_states = next_h_states
                out_np[t] = layer_input.squeeze(0).cpu().numpy()
        else:
            h_rnn: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            if model_type == "lstm":
                h0 = torch.zeros(1, 1, Hdim, device=device, dtype=seq.dtype)
                c0 = torch.zeros(1, 1, Hdim, device=device, dtype=seq.dtype)
                h_rnn = (h0, c0)
            else:
                h_rnn = torch.zeros(1, 1, Hdim, device=device, dtype=seq.dtype)

            for t in range(T):
                x_t = seq[:, t : t + 1, :]
                if model_type == "lstm":
                    hx, cx = h_rnn
                    out, (hx, cx) = model.rnn(x_t, (hx, cx))
                    h_rnn = (hx, cx)
                else:
                    out, h_rnn = model.rnn(x_t, h_rnn)
                z = out.squeeze(1)
                z = model.LNormRNN(z)
                z = F.relu(z)
                out_np[t] = z.squeeze(0).cpu().numpy()

    return out_np


def format_label_row(lbls_df: Any, iloc_idx: int, frame_id: int) -> list:
    """One TSV row: 9 fields matching GenerateMovies order."""
    row = lbls_df.iloc[iloc_idx]
    def _get(name: str):
        if name in row.index:
            return row[name]
        raise KeyError(f"Label column '{name}' missing (have {list(row.index)})")

    fg_id = _get("fg_char_id")
    fg_x = _get("fg_char_x")
    fg_y = _get("fg_char_y")
    raw_bg = _get("bg_char_ids")
    if raw_bg is None or (isinstance(raw_bg, float) and np.isnan(raw_bg)):
        bg_str = ""
    else:
        s = str(raw_bg).strip()
        if not s or s.lower() == "nan":
            bg_str = ""
        elif isinstance(raw_bg, str):
            bg_str = raw_bg
        elif isinstance(raw_bg, (list, tuple, np.ndarray)):
            bg_str = ",".join(str(int(x)) for x in raw_bg)
        else:
            bg_str = str(raw_bg)

    return [
        int(frame_id),
        int(fg_id) if not (isinstance(fg_id, float) and np.isnan(fg_id)) else fg_id,
        fg_x,
        fg_y,
        bg_str,
        _get("fg_speed"),
        _get("bg_mean_speed"),
        int(_get("fg_switch")),
        int(_get("bg_switch")),
    ]


def main() -> None:
    args = parse_args()
    model_type = args.model_type if args.model_type is not None else infer_model_type_from_ckpt(args.ckpt)
    print(f"model_type={model_type}" + (" (from --model_type)" if args.model_type is not None else " (inferred from ckpt basename)"))

    set_seed(args.seed)
    device = torch.device(args.device)
    T_seg = resolve_T(args)
    rng = np.random.default_rng(args.seed)

    if model_type == "gawf" and args.predict_all_chars:
        raise ValueError(
            "export_pop_act: GaWF checkpoints here expect sector/coordinate heads (fcchar+fcpos). "
            "Use --predict_all_chars with rnn/lstm/gru only."
        )

    stims_test, lbls_test, num_pos = load_test_stimuli_and_num_pos(args)
    predict_all_chars = args.predict_all_chars
    if predict_all_chars:
        num_pos = 0

    abs_start = pick_segment_start(rng, stims_test.shape[0], T_seg, args.chan_num)
    print(f"Segment: frames [{abs_start}, {abs_start + T_seg}) (T={T_seg}), chan_num={args.chan_num}")

    x_stack = np.empty((T_seg, args.chan_num, stims_test.shape[1], stims_test.shape[2]), dtype=np.float32)
    for t in range(T_seg):
        g = abs_start + t
        x_stack[t] = stacked_frame_at(stims_test, g, args.chan_num)

    model = build_model(model_type, args.ckpt, num_pos, predict_all_chars, device)
    pop = run_stream_pop_act(model, x_stack, device, model_type)

    run_tag = args.run_tag.strip() or os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = os.path.join(args.save_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pop_act.npy"), pop)

    tsv_path = os.path.join(out_dir, "labels.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(LABEL_COLUMNS)
        for t in range(T_seg):
            g = abs_start + t
            w.writerow(format_label_row(lbls_test, g, g))

    print(f"Wrote {pop.shape} pop_act.npy and labels.tsv under {out_dir}")

    if args.write_pop_act_dpca:
        from utils_anal.pop_act_dpca import aggregate_from_dir

        aggregate_from_dir(
            out_dir,
            frame_height=args.dpca_frame_height,
            frame_width=args.dpca_frame_width,
            num_sectors=args.dpca_num_sectors,
        )


if __name__ == "__main__":
    main()
