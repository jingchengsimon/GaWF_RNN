"""SentiHood query dataset and dataloader utilities.

The raw benchmark stores one sentence with zero or more opinions. This module
expands each sentence into query examples of the form:

    sentence1 + <sep> + "location - {1|2} - {aspect}"

Each query receives one label in ``None / Positive / Negative``. The saved tensor
layout mirrors ``utils.text_imdb_data`` so Amarel compute jobs can train offline.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .text_imdb_data import resolve_data_dir
from .common_train_helpers import worker_init_fn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"
PAD_ID = 0
UNK_ID = 1
SEP_ID = 2

PAPER_ASPECTS = ("general", "price", "transit-location", "safety")
LABELS = ("None", "Positive", "Negative")
TARGETS = ("LOCATION1", "LOCATION2")

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_HF_DATASET = "bhavnicksm/sentihood"
_HF_SPLITS = {"train": "train", "validation": "val", "test": "test"}
_LOCAL_SPLITS = {
    "sentihood-train.json": "train",
    "sentihood-dev.json": "val",
    "sentihood-test.json": "test",
}


def sentihood_dir(cli_data_dir: Optional[str]) -> str:
    return os.path.join(resolve_data_dir(cli_data_dir), "sentihood")


def tokenize(text: str) -> List[str]:
    """Lowercase word-level tokenizer shared by SentiHood preprocessing."""
    text = text.replace("LOCATION1", "location 1").replace("LOCATION2", "location 2")
    text = text.replace("transit-location", "transit location")
    return _TOKEN_RE.findall(text.lower())


def query_text(target: str, aspect: str) -> str:
    target_text = "location - 1" if target == "LOCATION1" else "location - 2"
    aspect_text = aspect.replace("-", " ")
    return f"{target_text} - {aspect_text}"


def combined_tokens(text: str, target: str, aspect: str) -> List[str]:
    return tokenize(text) + [SEP_TOKEN] + tokenize(query_text(target, aspect))


def _hf_fetch_rows(split: str) -> List[Dict]:
    rows: List[Dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": _HF_DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            data = json.load(response)
        page = data.get("rows", [])
        if not page:
            break
        rows.extend(item["row"] for item in page)
        offset += len(page)
        if offset >= int(data.get("num_rows_total", offset)):
            break
    return rows


def load_raw_splits(source_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Load raw SentiHood rows from local JSON files or the HF Dataset Viewer API."""
    if source_dir:
        base = Path(source_dir)
        splits: Dict[str, List[Dict]] = {}
        for filename, split in _LOCAL_SPLITS.items():
            path = base / filename
            with open(path, "r", encoding="utf-8") as f:
                splits[split] = json.load(f)
        return splits
    return {out_split: _hf_fetch_rows(hf_split) for hf_split, out_split in _HF_SPLITS.items()}


def iter_query_examples(rows: Sequence[Dict], aspects: Sequence[str]) -> Iterable[Dict]:
    """Yield flattened ``(sentence, target, aspect)`` query examples."""
    aspect_set = set(aspects)
    for row in rows:
        text = row["text"]
        sent_id = int(row["id"])
        opinions = row.get("opinions") or []
        labels_by_pair: Dict[Tuple[str, str], str] = {}
        for opinion in opinions:
            aspect = opinion["aspect"]
            if aspect not in aspect_set:
                continue
            target = opinion["target_entity"]
            sentiment = opinion["sentiment"]
            key = (target, aspect)
            if key in labels_by_pair and labels_by_pair[key] != sentiment:
                raise ValueError(f"Conflicting labels for sentence {sent_id}: {key}")
            labels_by_pair[key] = sentiment

        present_targets = ["LOCATION1"]
        if "LOCATION2" in text:
            present_targets.append("LOCATION2")

        for target in present_targets:
            for aspect in aspects:
                label = labels_by_pair.get((target, aspect), "None")
                yield {
                    "sentence_id": sent_id,
                    "text": text,
                    "target": target,
                    "aspect": aspect,
                    "label": label,
                    "tokens": combined_tokens(text, target, aspect),
                }


def build_vocab(examples: Sequence[Dict], vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for ex in examples:
        counter.update(ex["tokens"])
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, SEP_TOKEN: SEP_ID}
    for token, freq in ranked:
        if token in vocab:
            continue
        if freq < min_freq:
            break
        if len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_examples(
    examples: Sequence[Dict],
    vocab: Dict[str, int],
    aspects: Sequence[str],
    max_len: int,
) -> Dict[str, torch.Tensor]:
    n = len(examples)
    ids = torch.full((n, max_len), PAD_ID, dtype=torch.int64)
    lengths = torch.zeros(n, dtype=torch.int64)
    labels = torch.zeros(n, dtype=torch.int64)
    aspect_ids = torch.zeros(n, dtype=torch.int64)
    target_ids = torch.zeros(n, dtype=torch.int64)
    sentence_ids = torch.zeros(n, dtype=torch.int64)
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    aspect_to_id = {aspect: idx for idx, aspect in enumerate(aspects)}
    target_to_id = {target: idx for idx, target in enumerate(TARGETS)}

    for i, ex in enumerate(examples):
        toks = ex["tokens"][:max_len]
        if not toks:
            toks = [UNK_TOKEN]
        for j, tok in enumerate(toks):
            ids[i, j] = vocab.get(tok, UNK_ID)
        lengths[i] = len(toks)
        labels[i] = label_to_id[ex["label"]]
        aspect_ids[i] = aspect_to_id[ex["aspect"]]
        target_ids[i] = target_to_id[ex["target"]]
        sentence_ids[i] = int(ex["sentence_id"])
    return {
        "ids": ids,
        "lengths": lengths,
        "labels": labels,
        "aspect_ids": aspect_ids,
        "target_ids": target_ids,
        "sentence_ids": sentence_ids,
    }


def prepare_sentihood_tensors(
    data_dir: Optional[str] = None,
    source_dir: Optional[str] = None,
    aspects: Sequence[str] = PAPER_ASPECTS,
    vocab_size: int = 12000,
    min_freq: int = 1,
    max_len: int = 80,
) -> Dict:
    """Create offline tensors under ``<data_dir>/sentihood``."""
    out_dir = Path(sentihood_dir(data_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_splits = load_raw_splits(source_dir)
    query_splits = {
        split: list(iter_query_examples(rows, aspects)) for split, rows in raw_splits.items()
    }
    vocab = build_vocab(query_splits["train"], vocab_size=vocab_size, min_freq=min_freq)
    for split, examples in query_splits.items():
        encoded = encode_examples(examples, vocab=vocab, aspects=aspects, max_len=max_len)
        for name, tensor in encoded.items():
            torch.save(tensor, out_dir / f"sentihood_{split}_{name}.pt")

    meta = {
        "dataset": "sentihood",
        "source": source_dir or f"hf://datasets/{_HF_DATASET}",
        "vocab_size": len(vocab),
        "max_len": max_len,
        "min_freq": min_freq,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "sep_id": SEP_ID,
        "labels": list(LABELS),
        "aspects": list(aspects),
        "targets": list(TARGETS),
        "n_raw": {split: len(rows) for split, rows in raw_splits.items()},
        "n_query": {split: len(examples) for split, examples in query_splits.items()},
    }
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open(out_dir / "sentihood_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(cli_data_dir: Optional[str]) -> Dict:
    meta_path = os.path.join(sentihood_dir(cli_data_dir), "sentihood_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"SentiHood metadata not found at {meta_path}. "
            "Run source/text/prepare_sentihood_data.py first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


class SentiHoodDataset(Dataset):
    """Returns one flattened query example."""

    def __init__(self, data_dir: Optional[str], split: str):
        base = sentihood_dir(data_dir)
        self.split = split
        self.ids = torch.load(os.path.join(base, f"sentihood_{split}_ids.pt"))
        self.lengths = torch.load(os.path.join(base, f"sentihood_{split}_lengths.pt"))
        self.labels = torch.load(os.path.join(base, f"sentihood_{split}_labels.pt"))
        self.aspect_ids = torch.load(os.path.join(base, f"sentihood_{split}_aspect_ids.pt"))
        self.target_ids = torch.load(os.path.join(base, f"sentihood_{split}_target_ids.pt"))
        self.sentence_ids = torch.load(os.path.join(base, f"sentihood_{split}_sentence_ids.pt"))
        sizes = {
            len(self.ids),
            len(self.lengths),
            len(self.labels),
            len(self.aspect_ids),
            len(self.target_ids),
            len(self.sentence_ids),
        }
        if len(sizes) != 1:
            raise ValueError(f"Length mismatch for SentiHood split={split}: {sizes}")

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.ids[idx],
            self.lengths[idx],
            self.labels[idx],
            self.aspect_ids[idx],
            self.target_ids[idx],
            self.sentence_ids[idx],
        )


def build_sentihood_loaders(
    data_dir: Optional[str],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    balance_labels: bool = False,
) -> Dict[str, DataLoader]:
    """Build DataLoaders over flattened SentiHood query examples."""
    worker_init = partial(worker_init_fn, seed=seed) if num_workers > 0 else None
    generator = torch.Generator().manual_seed(seed)
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        dataset = SentiHoodDataset(data_dir, split)
        is_train = split == "train"
        sampler = None
        shuffle = is_train
        if is_train and balance_labels:
            class_counts = torch.bincount(dataset.labels, minlength=len(LABELS)).float()
            sample_weights = 1.0 / class_counts.clamp_min(1.0)[dataset.labels]
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        kw = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and num_workers > 0,
            persistent_workers=(num_workers > 0),
            drop_last=is_train,
            worker_init_fn=worker_init,
        )
        if is_train and sampler is None:
            kw["generator"] = generator
        loaders[split] = DataLoader(**kw)
    return loaders
