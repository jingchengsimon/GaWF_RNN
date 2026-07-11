"""Smoke tests for SentiHood query-pair preparation and training."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.text_sentihood_data import (  # noqa: E402
    PAPER_ASPECTS,
    build_sentihood_loaders,
    iter_query_examples,
    prepare_sentihood_tensors,
)
from utils.text_sentihood_metrics import compute_sentihood_metrics  # noqa: E402
from utils.text_task_models import TextLSTM  # noqa: E402
import train_sentihood  # noqa: E402


def _raw_rows():
    return [
        {
            "id": 1,
            "text": "LOCATION1 is very safe",
            "opinions": [
                {"target_entity": "LOCATION1", "aspect": "safety", "sentiment": "Positive"}
            ],
        },
        {
            "id": 2,
            "text": "LOCATION1 is expensive but LOCATION2 is safe",
            "opinions": [
                {"target_entity": "LOCATION1", "aspect": "price", "sentiment": "Negative"},
                {"target_entity": "LOCATION2", "aspect": "safety", "sentiment": "Positive"},
            ],
        },
    ]


def _write_raw_source(base_dir: str) -> str:
    source = os.path.join(base_dir, "raw")
    os.makedirs(source, exist_ok=True)
    rows = _raw_rows()
    for filename in ("sentihood-train.json", "sentihood-dev.json", "sentihood-test.json"):
        with open(os.path.join(source, filename), "w", encoding="utf-8") as f:
            json.dump(rows, f)
    return source


def _make_args(tmp: str):
    return SimpleNamespace(
        seed=42,
        embed_dim=8,
        embed_dropout=0.0,
        rnn_dropout=0.0,
        pooling="last",
        optim="adam",
        use_acceleration=False,
        num_epochs=1,
        patience=3,
        selection_metric="aspect_f1",
        batch_size=4,
        balance_train_labels=True,
        result_dir=tmp,
        result_suffix="sentihood_smoke",
    )


class TestSentiHoodSmoke(unittest.TestCase):
    def test_query_expansion_uses_actual_locations_and_all_paper_aspects(self):
        examples = list(iter_query_examples(_raw_rows(), PAPER_ASPECTS))
        by_id = {}
        for ex in examples:
            by_id.setdefault(ex["sentence_id"], []).append(ex)
        self.assertEqual(len(by_id[1]), 4)
        self.assertEqual(len(by_id[2]), 8)
        self.assertEqual({ex["target"] for ex in by_id[1]}, {"LOCATION1"})
        self.assertEqual({ex["target"] for ex in by_id[2]}, {"LOCATION1", "LOCATION2"})
        labels = {(ex["sentence_id"], ex["target"], ex["aspect"]): ex["label"] for ex in examples}
        self.assertEqual(labels[(1, "LOCATION1", "safety")], "Positive")
        self.assertEqual(labels[(1, "LOCATION1", "price")], "None")
        self.assertEqual(labels[(2, "LOCATION1", "price")], "Negative")
        self.assertEqual(labels[(2, "LOCATION2", "safety")], "Positive")

    def test_metrics_are_perfect_for_perfect_logits(self):
        labels = torch.tensor([1, 2, 0] * 4, dtype=torch.long)
        aspect_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)
        target_ids = torch.zeros(12, dtype=torch.long)
        sentence_ids = torch.arange(12, dtype=torch.long)
        logits = torch.full((12, 3), -5.0)
        logits[torch.arange(12), labels] = 5.0
        metrics = compute_sentihood_metrics(labels, logits, aspect_ids, target_ids, sentence_ids)
        self.assertAlmostEqual(metrics["query_acc"], 1.0)
        self.assertAlmostEqual(metrics["aspect_f1"], 1.0)
        self.assertAlmostEqual(metrics["sentiment_acc"], 1.0)
        self.assertAlmostEqual(metrics["aspect_auc"], 1.0)
        self.assertAlmostEqual(metrics["sentiment_auc"], 1.0)

    def test_prepare_loader_and_lstm_train_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_raw_source(tmp)
            meta = prepare_sentihood_tensors(
                data_dir=tmp,
                source_dir=source,
                aspects=PAPER_ASPECTS,
                vocab_size=200,
                max_len=32,
            )
            self.assertEqual(meta["n_query"]["train"], 12)
            loaders = build_sentihood_loaders(
                tmp,
                batch_size=4,
                num_workers=0,
                seed=42,
                balance_labels=True,
            )
            batch = next(iter(loaders["train"]))
            ids, lengths, labels, aspect_ids, target_ids, sentence_ids = batch
            self.assertEqual(tuple(ids.shape), (4, 32))
            self.assertEqual(tuple(labels.shape), (4,))
            model = TextLSTM(
                vocab_size=meta["vocab_size"],
                embed_dim=8,
                hidden_size=8,
                num_classes=3,
                device="cpu",
            )
            logits = model(ids, lengths)
            self.assertEqual(tuple(logits.shape), (4, 3))
            args = _make_args(tmp)
            cfg = {"model": "lstm", "hidden": 8, "lr": 1e-2, "wd": 0.0}
            out = train_sentihood.train_one_config(args, cfg, loaders, "cpu", meta)
            for key in (
                "test_aspect_f1_at_best",
                "test_sentiment_acc_at_best",
                "test_aspect_auc_at_best",
                "test_sentiment_auc_at_best",
            ):
                self.assertIn(key, out["metrics"])
            metrics_path = train_sentihood.save_outputs(args, cfg, out)
            self.assertTrue(os.path.isfile(metrics_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
