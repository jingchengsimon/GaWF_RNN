"""IMDB sentiment dataset + dataloader utilities (text task).

Loads the pre-tokenized tensors produced by ``scripts/prepare_imdb_data.py`` and
exposes them to ``train_imdb.py``. Compute jobs load everything offline; no
torchtext / HuggingFace ``datasets`` dependency.

Tensor layout (per split) under ``<data_dir>/imdb/``::

    imdb_{split}_ids.pt   (N, max_len) int64
    imdb_{split}_len.pt   (N,)         int64  true (pre-pad) length
    imdb_{split}_label.pt (N,)         int64  0=neg, 1=pos
    vocab.json, imdb_meta.json
"""
from __future__ import annotations

import json
import os
from functools import partial
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .train_helpers import worker_init_fn

PAD_ID = 0
UNK_ID = 1


def resolve_data_dir(cli_data_dir: Optional[str]) -> str:
    """Resolve the base data dir like the vision pipeline (CLI -> env -> <repo>/stimuli)."""
    if cli_data_dir:
        return os.path.abspath(cli_data_dir)
    env_path = os.environ.get("AIM3_STIMULI_PATH") or os.environ.get("FAW_RNN_DATA_PATH")
    if env_path:
        return os.path.abspath(env_path)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo, "stimuli")


def imdb_dir(cli_data_dir: Optional[str]) -> str:
    return os.path.join(resolve_data_dir(cli_data_dir), "imdb")


def load_meta(cli_data_dir: Optional[str]) -> Dict:
    meta_path = os.path.join(imdb_dir(cli_data_dir), "imdb_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"IMDB metadata not found at {meta_path}. "
            "Run scripts/prepare_imdb_data.py first (login node)."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(cli_data_dir: Optional[str]) -> Dict[str, int]:
    vocab_path = os.path.join(imdb_dir(cli_data_dir), "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


class IMDBDataset(Dataset):
    """Returns (ids, length, label) for one review.

    ``ids``   : LongTensor (max_len,)
    ``length``: LongTensor scalar, true pre-pad token count (>=1) for last-token pooling
    ``label`` : LongTensor scalar in {0, 1}
    """

    def __init__(self, data_dir: Optional[str], split: str):
        base = imdb_dir(data_dir)
        self.split = split
        self.ids = torch.load(os.path.join(base, f"imdb_{split}_ids.pt"))
        self.lengths = torch.load(os.path.join(base, f"imdb_{split}_len.pt"))
        self.labels = torch.load(os.path.join(base, f"imdb_{split}_label.pt"))
        if not (len(self.ids) == len(self.lengths) == len(self.labels)):
            raise ValueError(
                f"Length mismatch for split={split}: ids={len(self.ids)} "
                f"len={len(self.lengths)} label={len(self.labels)}"
            )

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, idx: int):
        return self.ids[idx], self.lengths[idx], self.labels[idx]


def build_imdb_loaders(
    data_dir: Optional[str],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    """Build DataLoaders for the requested splits.

    Mirrors ``utils/train_acceleration.build_loaders`` conventions: train shuffles
    and drops the last partial batch; eval splits keep every example in order.
    """
    worker_init = partial(worker_init_fn, seed=seed) if num_workers > 0 else None
    generator = torch.Generator().manual_seed(seed)
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = IMDBDataset(data_dir, split)
        is_train = split == "train"
        kw = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory and num_workers > 0,
            persistent_workers=(num_workers > 0),
            drop_last=is_train,
            worker_init_fn=worker_init,
        )
        if is_train:
            kw["generator"] = generator
        loaders[split] = DataLoader(**kw)
    return loaders
