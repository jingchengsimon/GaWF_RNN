"""Smoke test for the IMDB text subsystem (lstm + gawf).

Builds a tiny *synthetic* pre-tokenized dataset (no download needed) in the layout
``source/text/prepare_imdb_data.py`` produces, then trains both models for a few epochs
on CPU and asserts they run end-to-end, fit a learnable signal, and write a metrics
JSON with the expected schema. Mirrors the role of the Mamba/S5 optimizer smoke.

Run::

    python -m pytest tests/test_imdb_smoke.py -q
    # or
    python tests/test_imdb_smoke.py
"""

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

from utils.text_imdb_data import build_imdb_loaders  # noqa: E402
from utils.text_task_models import (
    TextGaWF,
    TextGaWFLogits,
    TextLSTM,
    get_text_model_classes,
)  # noqa: E402
import train_imdb  # noqa: E402

VOCAB_SIZE = 50
MAX_LEN = 20
POS_TOKEN = 5  # frequent in positive reviews
NEG_TOKEN = 6  # frequent in negative reviews


def _make_split(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, VOCAB_SIZE, (n, MAX_LEN), generator=g)
    labels = torch.randint(0, 2, (n,), generator=g)
    lengths = torch.randint(MAX_LEN // 2, MAX_LEN + 1, (n,), generator=g)
    # Inject a separable signal at the END of each review (where last-token pooling
    # reads), so both models can learn it quickly without long-range memory.
    for i in range(n):
        marker = POS_TOKEN if labels[i].item() == 1 else NEG_TOKEN
        end = int(lengths[i].item())
        ids[i, max(0, end - 6) : end] = marker
    return ids, lengths.long(), labels.long()


def _write_synthetic_dataset(base_dir: str):
    out = os.path.join(base_dir, "imdb")
    os.makedirs(out, exist_ok=True)
    sizes = {"train": 128, "val": 48, "test": 48}
    for split, n in sizes.items():
        ids, lengths, labels = _make_split(n, seed=hash(split) % 1000)
        torch.save(ids, os.path.join(out, f"imdb_{split}_ids.pt"))
        torch.save(lengths, os.path.join(out, f"imdb_{split}_len.pt"))
        torch.save(labels, os.path.join(out, f"imdb_{split}_label.pt"))
    vocab = {f"tok{i}": i for i in range(VOCAB_SIZE)}
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    meta = {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN, "n_train": 128, "n_val": 48, "n_test": 48}
    with open(os.path.join(out, "imdb_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def _make_args(seed=42):
    return SimpleNamespace(
        seed=seed,
        embed_dim=16,
        embed_dropout=0.0,
        rnn_dropout=0.5,
        pooling="last",
        optim="adam",
        use_acceleration=False,
        num_epochs=6,
        patience=10,
        num_layers=2,
        gawf_feedback_lr_scale=0.1,
        batch_size=32,
    )


class TestIMDBSmoke(unittest.TestCase):
    def test_models_instantiate_and_forward(self):
        for cls in (TextLSTM, TextGaWF, TextGaWFLogits):
            model = cls(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_size=24, device="cpu")
            ids = torch.randint(0, VOCAB_SIZE, (4, MAX_LEN))
            lengths = torch.randint(1, MAX_LEN + 1, (4,))
            logits = model(ids, lengths)
            self.assertEqual(tuple(logits.shape), (4, 2))

    def test_gawf_feedback_is_hidden_sized_no_proj(self):
        model = TextGaWF(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_size=24, device="cpu")
        # Feedback = hidden state directly: feedback_dim == hidden_size, U is (hidden, hidden).
        self.assertEqual(model.feedback_dim, 24)
        self.assertEqual(tuple(model.U.shape), (24, 24))
        self.assertEqual(tuple(model.V.shape), (24, 16 + 24))
        self.assertFalse(hasattr(model, "proj_out") and model.__dict__.get("proj_out") is not None)

    def test_gawf_logits_feedback_is_output_sized(self):
        model = TextGaWFLogits(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_size=24, device="cpu")
        self.assertEqual(model.feedback_dim, 2)
        self.assertEqual(tuple(model.U.shape), (24, 2))
        self.assertEqual(tuple(model.V.shape), (2, 16 + 24))
        self.assertTrue(model.include_fc_in_core_params)
        self.assertEqual(
            train_imdb.count_core_params(model), 24 * 24 + 24 * 16 + 10 * 24 + 2 * 16 + 2
        )

    def test_gawf_logits_multilayer_direct_feedback_dims(self):
        model = TextGaWFLogits(
            vocab_size=VOCAB_SIZE,
            embed_dim=16,
            hidden_size=24,
            num_layers=2,
            device="cpu",
        )
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.feedback_dim, 2)
        self.assertEqual(model.core.layer_feedback_dims, [24, 2])
        self.assertEqual(tuple(model.core.U_layers[0].shape), (24, 24))
        self.assertEqual(tuple(model.core.V_layers[0].shape), (24, 16 + 24))
        self.assertEqual(tuple(model.core.U_layers[1].shape), (24, 2))
        self.assertEqual(tuple(model.core.V_layers[1].shape), (2, 24 + 24))
        self.assertTrue(model.include_fc_in_core_params)

    def test_gawf_hidden_multilayer_feedback_dims(self):
        model = TextGaWF(
            vocab_size=VOCAB_SIZE,
            embed_dim=16,
            hidden_size=24,
            num_layers=3,
            device="cpu",
        )
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(model.feedback_dim, 24)
        self.assertEqual(model.core.layer_feedback_dims, [24, 24, 24])

    def test_optimizer_excludes_UV_from_weight_decay(self):
        model = TextGaWF(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_size=24, device="cpu")
        optim = train_imdb.build_optimizer(model, lr=1e-3, weight_decay=1e-4, optim_name="adamw")
        decay_group, no_decay_group = optim.param_groups[0], optim.param_groups[1]
        self.assertEqual(decay_group["weight_decay"], 1e-4)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        no_decay_ids = {id(p) for p in no_decay_group["params"]}
        self.assertIn(id(model.U), no_decay_ids)
        self.assertIn(id(model.V), no_decay_ids)

    def test_optimizer_scales_multilayer_gawf_feedback_lr(self):
        model = TextGaWFLogits(
            vocab_size=VOCAB_SIZE,
            embed_dim=16,
            hidden_size=24,
            num_layers=2,
            device="cpu",
        )
        optim = train_imdb.build_optimizer(
            model,
            lr=1e-3,
            weight_decay=1e-4,
            optim_name="adamw",
            gawf_feedback_lr_scale=0.25,
        )
        self.assertEqual(optim.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optim.param_groups[2]["lr"], 2.5e-4)
        self.assertEqual(optim.param_groups[2]["weight_decay"], 0.0)
        gate_ids = {id(p) for p in optim.param_groups[2]["params"]}
        self.assertIn(id(model.core.U_layers[0]), gate_ids)
        self.assertIn(id(model.core.V_layers[1]), gate_ids)

    def test_end_to_end_train_both_models(self):
        self.assertEqual(
            set(get_text_model_classes()),
            {"rnn", "lstm", "gru", "gawf", "gawf_logits"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            meta = _write_synthetic_dataset(tmp)
            loaders = build_imdb_loaders(tmp, batch_size=32, num_workers=0, seed=42)
            args = _make_args()
            for model_name in ("lstm", "gawf", "gawf_logits"):
                cfg = {"model": model_name, "hidden": 24, "lr": 1e-2, "wd": 0.0}
                out = train_imdb.train_one_config(args, cfg, loaders, "cpu", meta)
                m = out["metrics"]
                # Schema present.
                for key in (
                    "model_type",
                    "val_acc",
                    "best_val_acc",
                    "val_acc_at_best",
                    "test_acc_at_best",
                    "core_param_count",
                    "actual_epochs",
                ):
                    self.assertIn(key, m, f"{model_name}: missing metrics key {key}")
                # Loss decreases over training.
                self.assertLess(
                    m["train_loss"][-1],
                    m["train_loss"][0] + 1e-6,
                    f"{model_name}: train loss did not decrease",
                )
                # Learns the separable signal above chance.
                self.assertGreater(
                    m["best_val_acc"], 0.6, f"{model_name}: val acc not above chance"
                )

    def test_unified_gawf_depth_train_path_records_layer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = _write_synthetic_dataset(tmp)
            loaders = build_imdb_loaders(tmp, batch_size=32, num_workers=0, seed=42)
            args = _make_args()
            args.num_epochs = 1
            args.rnn_dropout = 0.0
            cfg = {
                "model": "gawf_logits",
                "hidden": 12,
                "lr": 1e-2,
                "wd": 0.0,
                "num_layers": 2,
            }
            out = train_imdb.train_one_config(args, cfg, loaders, "cpu", meta)
            metrics = out["metrics"]
            self.assertEqual(metrics["model_type"], "gawf_logits")
            self.assertEqual(metrics["num_layers"], 2)
            self.assertEqual(metrics["feedback_mode"], "logits")
            self.assertEqual(metrics["feedback_dim"], 2)
            self.assertEqual(metrics["layer_feedback_dims"], [12, 2])
            args.result_dir = tmp
            args.result_suffix = "imdb_gawf_depth_smoke"
            metrics_path = train_imdb.save_outputs(args, cfg, out)
            self.assertTrue(os.path.isfile(metrics_path))
            self.assertIn("_h12_L2_emb", metrics_path)

    def test_save_outputs_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = _write_synthetic_dataset(tmp)
            loaders = build_imdb_loaders(tmp, batch_size=32, num_workers=0, seed=42)
            args = _make_args()
            args.result_dir = tmp
            args.result_suffix = "imdb_smoke"
            cfg = {"model": "lstm", "hidden": 24, "lr": 1e-2, "wd": 0.0}
            out = train_imdb.train_one_config(args, cfg, loaders, "cpu", meta)
            metrics_path = train_imdb.save_outputs(args, cfg, out)
            self.assertTrue(os.path.isfile(metrics_path))
            with open(metrics_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["model_type"], "lstm")
            self.assertEqual(saved["dataset"], "imdb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
