"""Tests for condition-level population representation similarity analysis."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from utils_anal.pop_act_representation_similarity import (
    analyze_representations,
    compute_euclidean_rdm,
    compute_linear_cka,
    compute_rdm_spearman,
    normalize_representation,
)


class PopActRepresentationSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_rdm_is_symmetric_with_zero_diagonal(self) -> None:
        representation = self.rng.normal(size=(90, 17))
        centered = normalize_representation(representation, "centered")
        rdm = compute_euclidean_rdm(centered)

        self.assertEqual(rdm.shape, (90, 90))
        self.assertEqual(rdm.dtype, np.float32)
        np.testing.assert_allclose(rdm, rdm.T, atol=1e-6)
        np.testing.assert_array_equal(np.diag(rdm), np.zeros(90, dtype=np.float32))

    def test_similarity_supports_different_hidden_widths(self) -> None:
        base = self.rng.normal(size=(90, 5))
        orthogonal_columns, _ = np.linalg.qr(self.rng.normal(size=(8, 5)))
        wider = base @ orthogonal_columns.T
        base = normalize_representation(base, "centered")
        wider = normalize_representation(wider, "centered")

        cka = compute_linear_cka(base, wider)
        rsa = compute_rdm_spearman(
            compute_euclidean_rdm(base),
            compute_euclidean_rdm(wider),
        )
        self.assertAlmostEqual(cka, 1.0, places=10)
        self.assertAlmostEqual(rsa, 1.0, places=10)

    def test_analysis_writes_data_only_outputs(self) -> None:
        models = ["model_a", "model_b"]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = os.path.join(tmpdir, "pop_act")
            output_dir = os.path.join(tmpdir, "5_pop_act_umap")
            for model, hidden in zip(models, (7, 11)):
                model_dir = os.path.join(input_root, model)
                os.makedirs(model_dir)
                array = self.rng.normal(size=(hidden, 10, 9)).astype(np.float32)
                np.save(os.path.join(model_dir, "pop_act_dpca.npy"), array)

            result = analyze_representations(input_root, output_dir, models)

            self.assertEqual(result["rsa"].shape, (2, 2))
            self.assertEqual(result["linear_cka"].shape, (2, 2))
            for model in models:
                rdm_path = os.path.join(output_dir, model, "condition_rdm.npy")
                meta_path = os.path.join(
                    output_dir,
                    model,
                    "representation_similarity_meta.json",
                )
                self.assertEqual(np.load(rdm_path).shape, (90, 90))
                with open(meta_path) as file:
                    metadata = json.load(file)
                self.assertEqual(
                    metadata["condition_order"], "digit-major: flat_index = digit * 9 + sector"
                )

            extensions = {
                os.path.splitext(filename)[1]
                for _, _, filenames in os.walk(output_dir)
                for filename in filenames
            }
            self.assertTrue(extensions <= {".npy", ".csv", ".json"})


if __name__ == "__main__":
    unittest.main()
