"""Cross-decode GaWF identity modulation against CNN activation channels.

This script tests whether feedback-driven GaWF input modulation shares the same
32-channel digit-discriminative structure as the CNN activation code. It splits the
test dataset into two non-overlapping sequence halves: the first half provides
activation samples and the second half provides per-frame modulation samples.

Outputs (in --save_dir):
- activation_per_sample.npy  (N_a, 32), float32  — z-score source activation samples
- activation_labels.npy      (N_a,),    int64    — foreground digit labels
- modulation_per_sample.npy  (N_m, 32), float32  — per-frame GaWF modulation samples
- modulation_labels.npy      (N_m,),    int64    — foreground digit labels
- modulation_pattern_V.npy   (10, 32),  float32  — digit rows of V_ih, spatial mean pooled
- activation_classifier_weights.npy  (10, 32), float32 — activation-trained class weights
- modulation_classifier_weights.npy  (10, 32), float32 — modulation-trained class weights
- confusion_A2M.npy          (10, 10),  int64    — rows true digit, columns predicted digit
- confusion_M2A.npy          (10, 10),  int64    — rows true digit, columns predicted digit
- align_matrix.npy           (10, 10),  float32  — cosine(m_d^V, w_d')
- results.json               scalar metrics and run metadata
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.clutter_train_helpers import set_seed
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset


@dataclass
class LinearClassifier:
    """Minimal adapter for sklearn or torch linear softmax classifiers."""

    backend: str
    model: Any
    weights: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "sklearn":
            return self.model.predict(x).astype(np.int64, copy=False)

        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32)
            logits = self.model(xt)
            return logits.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GaWF identity cross-decoding and channel-pattern alignment analysis."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to GaWF checkpoint.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/anal_data/crossdecode_gawf_identity",
        help="Directory for analysis arrays and results.json.",
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--data_suffix", type=str, default="40h-uint8")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=256,
        help="Number of flattened frames per model chunk.",
    )
    parser.add_argument("--use_mmap", action="store_true", default=True)
    parser.add_argument("--use_sector_mode", action="store_true", default=True)
    parser.add_argument("--predict_all_chars", action="store_true", default=False)
    parser.add_argument(
        "--modulation_mode",
        type=str,
        default="trans",
        choices=["trans", "gate"],
        help="'trans' uses pre-sigmoid trans_ih; 'gate' uses gate_ih - 0.5.",
    )
    parser.add_argument(
        "--max_frames_per_half",
        type=int,
        default=0,
        help="Optional cap per half for quick smoke runs; 0 uses every frame.",
    )
    parser.add_argument(
        "--max_samples_per_digit",
        type=int,
        default=0,
        help="Optional balanced post-collection cap per digit; 0 keeps all samples.",
    )
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--permutes", type=int, default=1000)
    return parser.parse_args()


def _zscore_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each sample along its 32 channel dimensions."""
    x64 = x.astype(np.float64, copy=False)
    mu = x64.mean(axis=1, keepdims=True)
    sigma = x64.std(axis=1, keepdims=True)
    sigma = np.maximum(sigma, eps)
    return ((x64 - mu) / sigma).astype(np.float32)


def _confusion_matrix(true: np.ndarray, pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true.astype(np.int64), pred.astype(np.int64)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def _permutation_p(
    true: np.ndarray,
    pred: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_perm: int,
) -> Tuple[float, np.ndarray]:
    null = np.empty((n_perm,), dtype=np.float32)
    for i in range(n_perm):
        shuffled = rng.permutation(true)
        null[i] = float(np.mean(pred == shuffled))
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_perm + 1))
    return p_value, null


def _stratified_split(
    labels: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    for digit in range(10):
        idx = np.flatnonzero(labels == digit)
        if idx.size < 2:
            raise RuntimeError(
                f"Need at least two activation samples for digit {digit}, got {idx.size}."
            )
        idx = rng.permutation(idx)
        n_test = max(1, int(round(idx.size * test_fraction)))
        n_test = min(n_test, idx.size - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def _balanced_cap(
    x: np.ndarray,
    y: np.ndarray,
    max_per_digit: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_per_digit <= 0:
        return x, y

    keep = []
    for digit in range(10):
        idx = np.flatnonzero(y == digit)
        if idx.size > max_per_digit:
            idx = rng.choice(idx, size=max_per_digit, replace=False)
        keep.append(idx)
    keep_idx = np.concatenate(keep)
    keep_idx = rng.permutation(keep_idx)
    return x[keep_idx], y[keep_idx]


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> LinearClassifier:
    """Fit multinomial logistic regression, preferring sklearn when available."""
    classes = np.unique(y)
    if classes.size != 10:
        raise RuntimeError(f"Classifier training needs all 10 digits, got {classes.tolist()}.")

    try:
        from sklearn.linear_model import LogisticRegression

        import inspect

        kwargs = {
            "C": c_value,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": max_iter,
            "random_state": seed,
        }
        if "multi_class" in inspect.signature(LogisticRegression).parameters:
            kwargs["multi_class"] = "multinomial"
        clf = LogisticRegression(**kwargs)
        clf.fit(x, y)
        return LinearClassifier(
            backend="sklearn",
            model=clf,
            weights=clf.coef_.astype(np.float32, copy=False),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] sklearn logistic regression unavailable/failed ({exc}); using torch.")

    torch.manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], 10)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    l2 = 1.0 / max(c_value, 1e-8)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(xt)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss = loss + 0.5 * l2 * torch.sum(model.weight.square()) / max(1, xt.shape[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    weights = model.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    return LinearClassifier(backend="torch", model=model, weights=weights)


def _iter_frame_chunks(frames: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    flat = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    for start in range(0, flat.shape[0], chunk_size):
        yield flat[start : start + chunk_size]


def _collect_activation_samples(
    dataset,
    seq_indices: np.ndarray,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, seq_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.use_mmap else 4,
        pin_memory=False,
    )
    acts = []
    labels_out = []
    n_frames = 0
    model.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            labels_np = labels[..., 0].cpu().numpy().reshape(-1).astype(np.int64)
            frames = frames.to(device=device, dtype=torch.float32)
            batch_acts = []
            for chunk in _iter_frame_chunks(frames, args.frame_batch_size):
                feats = model.encoder(chunk)
                batch_acts.append(feats.mean(dim=(2, 3)).detach().cpu().numpy())
            acts_np = np.concatenate(batch_acts, axis=0).astype(np.float32, copy=False)
            acts.append(acts_np)
            labels_out.append(labels_np)
            n_frames += int(labels_np.size)
            if (bidx + 1) % 10 == 0:
                print(f"[activation] batches={bidx + 1} frames={n_frames}")
            if args.max_frames_per_half > 0 and n_frames >= args.max_frames_per_half:
                break

    x = np.concatenate(acts, axis=0)
    y = np.concatenate(labels_out, axis=0)
    if args.max_frames_per_half > 0:
        x = x[: args.max_frames_per_half]
        y = y[: args.max_frames_per_half]
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def _feedback_from_logits(model, char_t: torch.Tensor, pos_t: torch.Tensor) -> torch.Tensor:
    if getattr(model, "proj_out", None) is None:
        return torch.cat([char_t, pos_t], dim=-1)
    return model._compute_feedback(char_t, pos_t)


def _collect_modulation_samples(
    dataset,
    seq_indices: np.ndarray,
    model,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, seq_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.use_mmap else 4,
        pin_memory=False,
    )
    n_feat = int(model.conv_reduce.out_channels)
    h_sp, w_sp = model.pool_reduce.output_size
    input_size = int(n_feat * h_sp * w_sp)
    hidden_size = int(model.rnn.hidden_size)
    feedback_dim = int(model.V.shape[0])

    u_mean = model.U.detach().to(device=device, dtype=torch.float32).mean(dim=0)
    v_ih = model.V[:, :input_size].detach().to(device=device, dtype=torch.float32)
    v_ch = v_ih.reshape(feedback_dim, n_feat, h_sp, w_sp).mean(dim=(2, 3))
    trans_channel_basis = u_mean[:, None] * v_ch

    mods = []
    labels_out = []
    n_frames = 0
    model.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            frames, labels = batch[0], batch[1]
            labels_np = labels[..., 0].cpu().numpy().reshape(-1).astype(np.int64)
            frames = frames.to(device=device, dtype=torch.float32)
            batch_mods = []
            for chunk in _iter_frame_chunks(frames, args.frame_batch_size):
                feats = model.encoder(chunk).reshape(chunk.shape[0], -1)
                h0 = torch.zeros(
                    feats.shape[0],
                    hidden_size,
                    device=device,
                    dtype=feats.dtype,
                )
                fb0 = torch.zeros(
                    feats.shape[0],
                    feedback_dim,
                    device=device,
                    dtype=feats.dtype,
                )
                gated = model.middle_gawf(feats, h0, fb0.clamp(-10, 10).unsqueeze(2))
                char_t, pos_t = model.classifier(gated)
                fb1 = _feedback_from_logits(model, char_t, pos_t).clamp(-10, 10)

                if args.modulation_mode == "trans":
                    m = torch.matmul(fb1, trans_channel_basis)
                else:
                    v_full = v_ih.unsqueeze(0)
                    trans_ih = torch.matmul(model.U, fb1.unsqueeze(2) * v_full)
                    gate_delta = torch.sigmoid(trans_ih / float(model.gate_tau)) - 0.5
                    m = gate_delta.reshape(feats.shape[0], hidden_size, n_feat, h_sp, w_sp)
                    m = m.mean(dim=(1, 3, 4))
                batch_mods.append(m.detach().cpu().numpy())

            mods_np = np.concatenate(batch_mods, axis=0).astype(np.float32, copy=False)
            mods.append(mods_np)
            labels_out.append(labels_np)
            n_frames += int(labels_np.size)
            if (bidx + 1) % 10 == 0:
                print(f"[modulation] batches={bidx + 1} frames={n_frames}")
            if args.max_frames_per_half > 0 and n_frames >= args.max_frames_per_half:
                break

    x = np.concatenate(mods, axis=0)
    y = np.concatenate(labels_out, axis=0)
    if args.max_frames_per_half > 0:
        x = x[: args.max_frames_per_half]
        y = y[: args.max_frames_per_half]
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def _modulation_pattern_v(model) -> np.ndarray:
    n_feat = int(model.conv_reduce.out_channels)
    h_sp, w_sp = model.pool_reduce.output_size
    input_size = int(n_feat * h_sp * w_sp)
    if int(model.V.shape[0]) < 10:
        raise RuntimeError(f"Need at least 10 feedback rows in V, got {model.V.shape[0]}.")
    v_ih = model.V[:10, :input_size].detach().cpu().numpy()
    return v_ih.reshape(10, n_feat, h_sp, w_sp).mean(axis=(2, 3)).astype(np.float32)


def _cosine_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    a_norm = np.maximum(np.linalg.norm(a64, axis=1, keepdims=True), eps)
    b_norm = np.maximum(np.linalg.norm(b64, axis=1, keepdims=True), eps)
    return ((a64 / a_norm) @ (b64 / b_norm).T).astype(np.float32)


def _diag_minus_offdiag(mat: np.ndarray) -> float:
    diag = np.diag(mat)
    off_mask = ~np.eye(mat.shape[0], dtype=bool)
    return float(diag.mean() - mat[off_mask].mean())


def _align_permutation_p(
    align: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_perm: int,
) -> Tuple[float, np.ndarray]:
    null = np.empty((n_perm,), dtype=np.float32)
    for i in range(n_perm):
        perm = rng.permutation(align.shape[1])
        null[i] = _diag_minus_offdiag(align[:, perm])
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_perm + 1))
    return p_value, null


def _domain_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[LinearClassifier, np.ndarray, float, np.ndarray]:
    clf = _fit_classifier(
        train_x,
        train_y,
        c_value=args.logreg_c,
        max_iter=args.max_iter,
        seed=seed,
    )
    pred = clf.predict(test_x)
    acc = float(np.mean(pred == test_y))
    cm = _confusion_matrix(test_y, pred)
    return clf, pred, acc, cm


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Building test dataset...")
    test_ds, num_pos = build_test_dataset(args)
    n_seq = len(test_ds)
    if n_seq < 2:
        raise RuntimeError(f"Need at least two test sequences for split hygiene, got {n_seq}.")
    split = n_seq // 2
    act_seq = np.arange(0, split, dtype=np.int64)
    mod_seq = np.arange(split, n_seq, dtype=np.int64)
    print(f"Split test sequences: activation A={act_seq.size}, modulation B={mod_seq.size}")

    print(f"Loading model from: {args.ckpt}")
    model = build_model_from_ckpt(args.ckpt, num_pos=num_pos, device=device)
    if not hasattr(model, "U") or not hasattr(model, "V") or not hasattr(model, "middle_gawf"):
        raise RuntimeError("This analysis requires a single-layer GaWF-like model with U/V.")

    print("Collecting activation samples from first non-overlapping half...")
    activation, activation_labels = _collect_activation_samples(
        test_ds, act_seq, model, device, args
    )
    print(f"activation: {activation.shape}, labels: {activation_labels.shape}")

    print("Collecting modulation samples from second non-overlapping half...")
    modulation, modulation_labels = _collect_modulation_samples(
        test_ds, mod_seq, model, device, args
    )
    print(f"modulation: {modulation.shape}, labels: {modulation_labels.shape}")

    activation, activation_labels = _balanced_cap(
        activation, activation_labels, args.max_samples_per_digit, rng
    )
    modulation, modulation_labels = _balanced_cap(
        modulation, modulation_labels, args.max_samples_per_digit, rng
    )

    act_z = _zscore_rows(activation)
    mod_z = _zscore_rows(modulation)

    ceiling_train_idx, ceiling_test_idx = _stratified_split(
        activation_labels, args.test_fraction, rng
    )
    _ceiling_clf, _ceiling_pred, ceiling_acc, _ceiling_cm = _domain_metrics(
        act_z[ceiling_train_idx],
        activation_labels[ceiling_train_idx],
        act_z[ceiling_test_idx],
        activation_labels[ceiling_test_idx],
        args,
        seed=args.seed + 11,
    )

    clf_a, pred_a2m, acc_a2m, cm_a2m = _domain_metrics(
        act_z,
        activation_labels,
        mod_z,
        modulation_labels,
        args,
        seed=args.seed + 23,
    )
    clf_m, pred_m2a, acc_m2a, cm_m2a = _domain_metrics(
        mod_z,
        modulation_labels,
        act_z,
        activation_labels,
        args,
        seed=args.seed + 37,
    )

    p_a2m, null_a2m = _permutation_p(
        modulation_labels, pred_a2m, acc_a2m, rng, args.permutes
    )
    p_m2a, null_m2a = _permutation_p(
        activation_labels, pred_m2a, acc_m2a, rng, args.permutes
    )

    m_v = _modulation_pattern_v(model)
    w_d = clf_a.weights.astype(np.float32, copy=False)
    align = _cosine_matrix(m_v, w_d)
    align_scalar = _diag_minus_offdiag(align)
    align_p, align_null = _align_permutation_p(align, align_scalar, rng, args.permutes)

    np.save(
        os.path.join(args.save_dir, "activation_per_sample.npy"),
        activation.astype(np.float32),
    )
    np.save(
        os.path.join(args.save_dir, "activation_labels.npy"),
        activation_labels.astype(np.int64),
    )
    np.save(
        os.path.join(args.save_dir, "modulation_per_sample.npy"),
        modulation.astype(np.float32),
    )
    np.save(
        os.path.join(args.save_dir, "modulation_labels.npy"),
        modulation_labels.astype(np.int64),
    )
    np.save(os.path.join(args.save_dir, "modulation_pattern_V.npy"), m_v.astype(np.float32))
    np.save(
        os.path.join(args.save_dir, "activation_classifier_weights.npy"),
        w_d.astype(np.float32),
    )
    np.save(
        os.path.join(args.save_dir, "modulation_classifier_weights.npy"),
        clf_m.weights.astype(np.float32),
    )
    np.save(os.path.join(args.save_dir, "confusion_A2M.npy"), cm_a2m.astype(np.int64))
    np.save(os.path.join(args.save_dir, "confusion_M2A.npy"), cm_m2a.astype(np.int64))
    np.save(os.path.join(args.save_dir, "align_matrix.npy"), align.astype(np.float32))
    np.save(os.path.join(args.save_dir, "null_transfer_A2M.npy"), null_a2m.astype(np.float32))
    np.save(os.path.join(args.save_dir, "null_transfer_M2A.npy"), null_m2a.astype(np.float32))
    np.save(os.path.join(args.save_dir, "null_align_diag_minus_offdiag.npy"), align_null)

    results: Dict[str, Any] = {
        "transfer_acc_A2M": acc_a2m,
        "transfer_acc_M2A": acc_m2a,
        "ceiling_acc": ceiling_acc,
        "perm_p_A2M": p_a2m,
        "perm_p_M2A": p_m2a,
        "align_diag_minus_offdiag": align_scalar,
        "align_perm_p": align_p,
        "chance_acc": 0.1,
        "classifier_backend_A": clf_a.backend,
        "classifier_backend_M": clf_m.backend,
        "modulation_mode": args.modulation_mode,
        "n_activation_frames": int(activation_labels.size),
        "n_modulation_frames": int(modulation_labels.size),
        "activation_sequence_range": [int(act_seq[0]), int(act_seq[-1])],
        "modulation_sequence_range": [int(mod_seq[0]), int(mod_seq[-1])],
        "test_sequences_total": int(n_seq),
        "num_pos": int(num_pos),
        "hidden_size": int(model.rnn.hidden_size),
        "input_size": int(model.encoder_flatten_size),
        "n_feat": int(model.conv_reduce.out_channels),
        "h_sp": int(model.pool_reduce.output_size[0]),
        "w_sp": int(model.pool_reduce.output_size[1]),
        "ckpt": os.path.abspath(args.ckpt),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
