"""Collect equal-n recurrent gate/activation caches for all ten digit contexts.

One complete test-split pass reconstructs every frame's recurrent gate and hidden activation,
then retains a reproducible random subset from each digit.  The retained count is the minimum
observed digit-frame count, matching Figure 6's condition-wise equal-n protocol.  Each output
``digit{d}_gate_act_cache.npz`` retains the existing analysis schema plus sampling metadata.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIGITS = tuple(range(10))
TOP_FRACTION = 0.10
DIGIT_LABEL_COL = 0


def parse_args() -> argparse.Namespace:
    """Parse one checkpoint's selectivity artifact and isolated output cache."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    tuning_source = parser.add_mutually_exclusive_group(required=True)
    tuning_source.add_argument("--selectivity", type=Path)
    tuning_source.add_argument("--reference_cache_dir", type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "source" / "clutter" / "stimuli"))
    parser.add_argument("--data_suffix", default="40h-uint8")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--selection_seed", type=int, default=260718)
    return parser.parse_args()


def _reference_tuned_masks(cache_dir: Path) -> np.ndarray:
    """Load the previously validated digit-specific ``T_old`` masks before replacement."""

    masks: list[np.ndarray] = []
    for digit in DIGITS:
        path = cache_dir / f"digit{digit}_gate_act_cache.npz"
        with np.load(path, allow_pickle=False) as cached:
            masks.append(np.asarray(cached["T_old"], dtype=bool).copy())
    return np.stack(masks)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from utils.analysis.anal_helpers import build_eval_dataset, build_model_from_ckpt, resolve_device
    from utils.analysis.clutter.fig6_encoder_sector_patterns import _equal_n_condition_mask
    from utils.analysis.clutter.fig7_relevance_timing import _gate_tensors
    from utils.analysis.clutter.fig7_relevance_stats import relevance_masks

    cli = parse_args()
    dataset_args = argparse.Namespace(
        data_dir=str(cli.data_dir), data_suffix=cli.data_suffix, use_mmap=True,
        chan_num=2, use_sector_mode=True, predict_all_chars=False,
    )
    cli.cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cli.device, require_cuda_if_requested=True)
    dataset, num_pos = build_eval_dataset(dataset_args, "test")
    model = build_model_from_ckpt(str(cli.ckpt), num_pos, device, chan_num=2)
    hidden_size = int(model.rnn.hidden_size)
    input_size = int(model.rnn.input_size)
    total_frames = len(dataset) * int(dataset.frame_num)
    all_labels = np.asarray(
        dataset.labels_sector[2 : 2 + total_frames], dtype=np.int64
    ).reshape(total_frames, 2)
    selected, target, original_counts = _equal_n_condition_mask(
        all_labels[:, DIGIT_LABEL_COL], len(DIGITS), cli.selection_seed
    )
    print(f"dataset sequences={len(dataset)}, hidden_size={hidden_size}, "
          f"digits={DIGITS}, equal_n/digit={target}, "
          f"original_counts={original_counts.tolist()}", flush=True)

    W = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)

    if cli.selectivity is not None:
        with np.load(cli.selectivity, allow_pickle=False) as sel:
            tuning = np.asarray(sel["primary_hidden_tuning_digit"], dtype=np.float64)
            passed = np.asarray(sel["primary_hidden_passed_digit"], dtype=bool)
            dominant = np.asarray(sel["primary_hidden_interaction_dominant"], dtype=bool)
        eligible = passed & ~dominant
        T_old_all = relevance_masks(tuning, eligible, TOP_FRACTION)  # (10, H)
        tuning_source_name = str(cli.selectivity)
    else:
        assert cli.reference_cache_dir is not None
        T_old_all = _reference_tuned_masks(cli.reference_cache_dir)
        tuning_source_name = f"T_old from {cli.reference_cache_dir}"
    if T_old_all.shape != (len(DIGITS), hidden_size):
        raise RuntimeError(f"Unexpected digit tuned-mask shape: {T_old_all.shape}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    gate_chunks: dict[int, list[np.ndarray]] = {d: [] for d in DIGITS}
    act_chunks: dict[int, list[np.ndarray]] = {d: [] for d in DIGITS}
    counts: dict[int, int] = {d: 0 for d in DIGITS}
    started = time.perf_counter()
    frame_offset = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            frames, batch_labels = batch[0], batch[1]
            label_array = np.asarray(batch_labels, dtype=np.int64)
            batch_size_actual, frame_num = label_array.shape[:2]
            frame_count = batch_size_actual * frame_num
            expected = all_labels[frame_offset : frame_offset + frame_count]
            if not np.array_equal(label_array.reshape(-1, 2), expected):
                raise RuntimeError("DataLoader label order differs from equal-n selection order.")
            batch_selected = torch.as_tensor(
                selected[frame_offset : frame_offset + frame_count].reshape(
                    batch_size_actual, frame_num
                ),
                device=device,
                dtype=torch.bool,
            )
            frames = frames.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            labels = batch_labels.to(
                device=device, dtype=torch.int64, non_blocking=device.type == "cuda"
            )
            encoded_maps = model.encoder_module(frames.reshape(-1, *frames.shape[2:]))
            encoded = encoded_maps.reshape(frames.shape[0], frames.shape[1], -1)
            hidden = torch.zeros(batch_size_actual, hidden_size, dtype=encoded.dtype, device=device)
            feedback = torch.zeros(batch_size_actual, model.feedback_dim, dtype=torch.float32, device=device)
            for time_idx in range(frame_num):
                gate_input, gate_recurrent = _gate_tensors(feedback, model, input_size)
                digit_labels = labels[:, time_idx, DIGIT_LABEL_COL]
                for d in DIGITS:
                    mask = batch_selected[:, time_idx] & (digit_labels == d)
                    if mask.any():
                        gate_chunks[d].append(gate_recurrent[mask].detach().cpu().numpy().astype(np.float32))
                        act_chunks[d].append(hidden[mask].detach().cpu().numpy().astype(np.float32))
                        counts[d] += int(mask.sum())
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
            frame_offset += frame_count
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"  batch {batch_idx + 1}/{len(loader)}  counts={counts}  "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)

    if frame_offset != total_frames or any(counts[d] != target for d in DIGITS):
        raise RuntimeError(
            f"Equal-n digit accumulation mismatch: expected {target}, got {counts}."
        )

    for d in DIGITS:
        if not gate_chunks[d]:
            raise RuntimeError(f"digit {d}: no frames collected")
        G_flat = np.concatenate(gate_chunks[d], axis=0)
        act_flat = np.concatenate(act_chunks[d], axis=0)
        n_samples = G_flat.shape[0]
        print(f"digit {d}: collected {n_samples} samples", flush=True)

        top_k = max(1, int(round(TOP_FRACTION * hidden_size)))
        T_new = np.zeros(hidden_size, dtype=bool)
        T_new[np.argsort(-act_flat.mean(axis=0))[:top_k]] = True

        gate = G_flat.reshape(1, n_samples, 1, hidden_size, hidden_size)
        act = act_flat.reshape(1, n_samples, 1, hidden_size)

        cache_path = cli.cache_dir / f"digit{d}_gate_act_cache.npz"
        temporary = cache_path.with_name(f".{cache_path.stem}.tmp.npz")
        np.savez(
            temporary,
            W=W.astype(np.float32),
            gate=gate,
            act=act,
            T_new=T_new,
            T_old=T_old_all[d],
            original_frames_by_context=original_counts.astype(np.int64),
            selected_frames_per_context=np.int64(target),
            selection_seed=np.int64(cli.selection_seed),
            tuning_source=np.asarray(tuning_source_name),
        )
        temporary.replace(cache_path)
        print(f"Saved {cache_path}", flush=True)


if __name__ == "__main__":
    main()
