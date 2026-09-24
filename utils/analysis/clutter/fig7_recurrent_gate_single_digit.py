"""Collect real digit-0 recurrent-gate + activation data for the single-digit diagnostic.

Torch/dataset-dependent by design, kept separate from the numpy-only
``gawf_recurrent_gate_single_digit_diagnostic.py`` so that script stays fast and
dependency-light. Reruns the exact checkpoint / selectivity file used by
``gawf_recurrent_group_gate_distributions.py`` (sha256-verified) over the full test split,
keeps only digit==0 frames, and caches:

  W      : (H, H) frozen recurrent weight (model.rnn.weight_hh_l0), row=dst/col=src.
  gate   : (1, n_samples, 1, H, H) raw per-sample recurrent gate for digit 0.
  act    : (1, n_samples, 1, H) pre-update hidden state h_{t-1} paired with that gate
           (the state actually being gated at that step).
  T_new  : (H,) naive top-10% by mean activation -- a proxy for a coarser "new pipeline"
           tuned set (this repo has no separate new-pipeline tuning artifact to read).
  T_old  : (H,) FDR-selective top-10% among eligible units, exactly reproducing the
           ~0.238 result from gawf_recurrent_group_gate_distributions.

into an .npz consumed by gawf_recurrent_gate_single_digit_diagnostic.load_data().
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.analysis.anal_paths import output_dir

CATEGORY = "E_relevance_alignment"
SCRIPT_NAME = Path(__file__).stem
DIGIT = 0
TOP_FRACTION = 0.10

CKPT_PATH = PROJECT_ROOT / (
    "results/data/clutter/runs/best_6model_param_matched_40h/"
    "gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
)
SELECTIVITY_PATH = PROJECT_ROOT / (
    "results/data/analysis/D_variance_decomposition/"
    "gawf_symmetric_relevance_timing/part1_selectivity.npz"
)
DATA_DIR = PROJECT_ROOT / "source" / "clutter" / "stimuli"
DATA_SUFFIX = "40h-uint8"
EXPECTED_SHA256 = "5b109ff973bc54726be5a58cb50699d3e150da9d521da0a1126c089a2aeefc25"


def main() -> None:
    import hashlib

    import torch
    from torch.utils.data import DataLoader

    from utils.analysis.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils.analysis.clutter.fig7_relevance_timing import _gate_tensors
    from utils.analysis.clutter.fig7_relevance_stats import relevance_masks

    checksum = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise RuntimeError(f"checkpoint sha256 mismatch: {checksum} != {EXPECTED_SHA256}")

    args = argparse.Namespace(
        data_dir=str(DATA_DIR), data_suffix=DATA_SUFFIX, use_mmap=True,
        chan_num=2, use_sector_mode=True, predict_all_chars=False,
    )
    device = resolve_device("cpu")
    dataset, num_pos = build_eval_dataset(args, "test")
    model = build_model_from_ckpt(str(CKPT_PATH), num_pos, device, chan_num=2)
    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.rnn.input_size)
    print(f"dataset sequences={len(dataset)}, hidden_size={hidden_size}", flush=True)

    W = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)

    with np.load(SELECTIVITY_PATH, allow_pickle=False) as sel:
        tuning = np.asarray(sel["primary_hidden_tuning_digit"], dtype=np.float64)
        passed = np.asarray(sel["primary_hidden_passed_digit"], dtype=bool)
        dominant = np.asarray(sel["primary_hidden_interaction_dominant"], dtype=bool)
    eligible = passed & ~dominant
    T_old = relevance_masks(tuning, eligible, TOP_FRACTION)[DIGIT]
    print(f"eligible={int(eligible.sum())} T_old top-10%={int(T_old.sum())}", flush=True)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    gate_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    collected = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            frames = frames.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.int64)
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            batch_size_actual, frame_num, _ = encoded.shape
            hidden = torch.zeros(batch_size_actual, hidden_size, dtype=encoded.dtype)
            feedback = torch.zeros(batch_size_actual, model.feedback_dim, dtype=torch.float32)
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                digit_labels = labels[:, time_idx, 0]
                mask = digit_labels == DIGIT
                if mask.any():
                    gate_chunks.append(gate_recurrent[mask].detach().cpu().numpy().astype(np.float32))
                    act_chunks.append(hidden[mask].detach().cpu().numpy().astype(np.float32))
                    collected += int(mask.sum())
                input_term = torch.einsum(
                    "bi,bhi,hi->bh", encoded[:, time_idx], gate_input, model.rnn.weight_ih_l0
                )
                recurrent_term = torch.einsum(
                    "bi,bhi,hi->bh", hidden, gate_recurrent, model.rnn.weight_hh_l0
                )
                preactivation = input_term + recurrent_term
                if model.rnn.bias_ih_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_ih_l0.unsqueeze(0)
                if model.rnn.bias_hh_l0 is not None:
                    preactivation = preactivation + model.rnn.bias_hh_l0.unsqueeze(0)
                hidden = torch.relu(model.LNormRNN(preactivation))
                char_logits, sector_logits = model.classifier(hidden)
                feedback = torch.cat([char_logits, sector_logits], dim=-1).to(torch.float32)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(
                    f"  batch {batch_idx + 1}/{len(loader)}  digit-{DIGIT} collected={collected}  "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    G_flat = np.concatenate(gate_chunks, axis=0)
    act_flat = np.concatenate(act_chunks, axis=0)
    n_samples = G_flat.shape[0]
    print(f"total digit-{DIGIT} samples collected: {n_samples}", flush=True)

    top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
    T_new = np.zeros(hidden_size, dtype=bool)
    T_new[np.argsort(-act_flat.mean(axis=0))[:top_k]] = True

    gate = G_flat.reshape(1, n_samples, 1, hidden_size, hidden_size)
    act = act_flat.reshape(1, n_samples, 1, hidden_size)

    data_dir_out = output_dir(CATEGORY, SCRIPT_NAME, "data")
    cache_path = data_dir_out / f"digit{DIGIT}_gate_act_cache.npz"
    np.savez(
        cache_path,
        W=W.astype(np.float32),
        gate=gate,
        act=act,
        T_new=T_new,
        T_old=T_old,
    )
    print(f"Saved {cache_path}", flush=True)


if __name__ == "__main__":
    main()
