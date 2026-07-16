#!/usr/bin/env python3
"""One-time IMDB sentiment data preparation (login-node, internet-enabled).

Downloads the Large Movie Review Dataset (aclImdb_v1), builds a vocabulary from the
training split only, tokenizes + pads every review to a fixed length, carves a
stratified validation split out of train, and saves plain tensors so that Amarel
compute jobs (no internet) can load everything offline. This mirrors the existing
``.npy`` stimuli pipeline used by the vision task.

Outputs under ``<data_dir>/imdb/`` (data_dir resolved like the vision pipeline:
``--data_dir`` CLI -> ``AIM3_STIMULI_PATH`` / ``FAW_RNN_DATA_PATH`` env -> ``<repo>/stimuli``):

    vocab.json                         # {token: id}, with <pad>=0 and <unk>=1
    imdb_meta.json                     # vocab_size, max_len, split sizes, config echo
    imdb_train_ids.pt   (N_tr, max_len) int64
    imdb_train_len.pt   (N_tr,)         int64  true (pre-pad) length, for last-token pooling
    imdb_train_label.pt (N_tr,)         int64  0=neg, 1=pos
    imdb_val_ids.pt / _len.pt / _label.pt
    imdb_test_ids.pt / _len.pt / _label.pt

Pure stdlib parsing (tarfile + regex) plus torch for tensor saving; no torchtext /
HuggingFace ``datasets`` dependency is added to the environment.

Usage (run once on a login node)::

    python source/text/prepare_imdb_data.py --data_dir /scratch/$USER/stimuli
    # or rely on AIM3_STIMULI_PATH / FAW_RNN_DATA_PATH
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_NAME = "aclImdb_v1.tar.gz"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

# Simple, deterministic tokenizer: lowercase, strip basic HTML breaks, keep word-ish
# tokens. Kept intentionally dependency-free and reproducible.
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_data_dir(cli_data_dir: str | None) -> str:
    """Resolve the base data dir the same way the vision pipeline does."""
    if cli_data_dir:
        return os.path.abspath(cli_data_dir)
    env_path = os.environ.get("AIM3_STIMULI_PATH") or os.environ.get("FAW_RNN_DATA_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(repo_root(), "stimuli")


def tokenize(text: str) -> List[str]:
    text = _BR_RE.sub(" ", text)
    return _TOKEN_RE.findall(text.lower())


def download_archive(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ARCHIVE_NAME
    if archive_path.is_file() and archive_path.stat().st_size > 0:
        print(f"[prepare_imdb] archive already present: {archive_path}")
        return archive_path
    print(f"[prepare_imdb] downloading {IMDB_URL} -> {archive_path}")
    urllib.request.urlretrieve(IMDB_URL, archive_path)  # noqa: S310 (trusted Stanford URL)
    return archive_path


def extract_archive(archive_path: Path, work_dir: Path) -> Path:
    root = work_dir / "aclImdb"
    if root.is_dir() and (root / "train").is_dir() and (root / "test").is_dir():
        print(f"[prepare_imdb] already extracted: {root}")
        return root
    print(f"[prepare_imdb] extracting {archive_path} -> {work_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        # Defensive extraction: skip any member that escapes work_dir.
        safe_members = []
        work_dir_abs = os.path.abspath(work_dir)
        for member in tar.getmembers():
            target = os.path.abspath(os.path.join(work_dir, member.name))
            if target.startswith(work_dir_abs + os.sep) or target == work_dir_abs:
                safe_members.append(member)
        tar.extractall(work_dir, members=safe_members)  # noqa: S202 (filtered above)
    return root


def read_split(root: Path, split: str) -> Tuple[List[str], List[int]]:
    """Read raw texts and labels for a split ('train' or 'test'). pos=1, neg=0."""
    texts: List[str] = []
    labels: List[int] = []
    for label_name, label_id in (("pos", 1), ("neg", 0)):
        sub = root / split / label_name
        if not sub.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {sub}")
        files = sorted(sub.glob("*.txt"))
        if not files:
            raise RuntimeError(f"No .txt review files under {sub}")
        for fp in files:
            texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label_id)
    return texts, labels


def build_vocab(train_texts: List[str], vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))
    # Most-common, tie-broken by token for determinism.
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in ranked:
        if freq < min_freq:
            break
        if len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(texts: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(texts)
    ids = torch.full((n, max_len), PAD_ID, dtype=torch.int64)
    lengths = torch.zeros(n, dtype=torch.int64)
    for i, text in enumerate(texts):
        toks = tokenize(text)[:max_len]
        if not toks:
            # Guard against empty reviews: keep a single <unk> so length>=1 for pooling.
            toks = [UNK_TOKEN]
        for j, tok in enumerate(toks):
            ids[i, j] = vocab.get(tok, UNK_ID)
        lengths[i] = len(toks)
    return ids, lengths


def stratified_val_split(
    labels: List[int], val_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Return (train_idx, val_idx) with class balance preserved."""
    g = torch.Generator().manual_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    label_tensor = torch.tensor(labels)
    for cls in label_tensor.unique().tolist():
        cls_idx = (label_tensor == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
        n_val = int(round(val_frac * perm.numel()))
        val_idx.extend(perm[:n_val].tolist())
        train_idx.extend(perm[n_val:].tolist())
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def save_split(out_dir: Path, split: str, ids: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor) -> None:
    torch.save(ids, out_dir / f"imdb_{split}_ids.pt")
    torch.save(lengths, out_dir / f"imdb_{split}_len.pt")
    torch.save(labels, out_dir / f"imdb_{split}_label.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=None, help="Base data dir; defaults to env or <repo>/stimuli.")
    parser.add_argument("--vocab_size", type=int, default=25000, help="Max vocab incl. <pad>,<unk>.")
    parser.add_argument("--min_freq", type=int, default=1, help="Drop tokens below this train frequency.")
    parser.add_argument("--max_len", type=int, default=400, help="Truncate/pad reviews to this length.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Stratified val fraction carved from train.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default=None, help="Where to download/extract; defaults to <data_dir>/imdb/_raw.")
    args = parser.parse_args()

    base = Path(resolve_data_dir(args.data_dir))
    out_dir = base / "imdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else (out_dir / "_raw")

    archive = download_archive(work_dir)
    root = extract_archive(archive, work_dir)

    print("[prepare_imdb] reading raw splits ...")
    train_texts_full, train_labels_full = read_split(root, "train")
    test_texts, test_labels = read_split(root, "test")
    print(f"[prepare_imdb] raw train={len(train_texts_full)} test={len(test_texts)}")

    print("[prepare_imdb] building vocab (train only) ...")
    vocab = build_vocab(train_texts_full, args.vocab_size, args.min_freq)
    print(f"[prepare_imdb] vocab_size={len(vocab)} (requested<= {args.vocab_size})")

    tr_idx, val_idx = stratified_val_split(train_labels_full, args.val_frac, args.seed)
    train_texts = [train_texts_full[i] for i in tr_idx]
    train_labels = [train_labels_full[i] for i in tr_idx]
    val_texts = [train_texts_full[i] for i in val_idx]
    val_labels = [train_labels_full[i] for i in val_idx]
    print(f"[prepare_imdb] split: train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}")

    print("[prepare_imdb] encoding ...")
    for split, texts, labels in (
        ("train", train_texts, train_labels),
        ("val", val_texts, val_labels),
        ("test", test_texts, test_labels),
    ):
        ids, lengths = encode(texts, vocab, args.max_len)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        save_split(out_dir, split, ids, lengths, label_tensor)
        pos_frac = float(label_tensor.float().mean().item())
        print(
            f"[prepare_imdb]   {split}: ids={tuple(ids.shape)} "
            f"pos_frac={pos_frac:.3f} mean_len={float(lengths.float().mean()):.1f}"
        )

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    meta = {
        "vocab_size": len(vocab),
        "max_len": args.max_len,
        "min_freq": args.min_freq,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "source_url": IMDB_URL,
    }
    with open(out_dir / "imdb_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[prepare_imdb] done. Tensors + vocab.json + imdb_meta.json under {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
