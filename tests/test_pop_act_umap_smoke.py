"""Smoke tests for reusable 3D dPCA coordinates and compatibility with 2D diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from utils_viz.pop_act_umap import run_dpca_condensed, save_dpca_3d_coordinates


class PopActDpca3DTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.pop_act = rng.normal(size=(16, 10, 9)).astype(np.float32)
        self.digit_labels = np.arange(90, dtype=np.int64) // 9
        self.sector_labels = np.arange(90, dtype=np.int64) % 9

    def test_condensed_dpca_keeps_first_three_components(self) -> None:
        result = run_dpca_condensed(self.pop_act, n_components=6)

        self.assertEqual(result["coords_digit"].shape, (90, 3))
        self.assertEqual(result["coords_sector"].shape, (90, 3))
        self.assertEqual(result["coords_digit"].dtype, np.float32)
        self.assertEqual(result["coords_sector"].dtype, np.float32)
        self.assertEqual(result["digit_axes"].shape, (16, 3))
        self.assertEqual(result["sector_axes"].shape, (16, 3))
        self.assertGreater(float(np.linalg.norm(result["coords_digit"][:, 2])), 0.0)
        self.assertGreater(float(np.linalg.norm(result["coords_sector"][:, 2])), 0.0)

    def test_coordinate_export_uses_stable_dtypes_and_metadata(self) -> None:
        result = run_dpca_condensed(self.pop_act)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path, meta_path = save_dpca_3d_coordinates(
                result["coords_digit"],
                result["coords_sector"],
                self.digit_labels,
                self.sector_labels,
                tmpdir,
                method="condensed",
                source_path="synthetic/pop_act_dpca.npy",
            )

            self.assertTrue(os.path.isfile(data_path))
            self.assertTrue(os.path.isfile(meta_path))
            with np.load(data_path) as payload:
                self.assertEqual(payload["coords_digit"].shape, (90, 3))
                self.assertEqual(payload["coords_sector"].shape, (90, 3))
                self.assertEqual(payload["coords_digit"].dtype, np.float32)
                self.assertEqual(payload["coords_sector"].dtype, np.float32)
                self.assertEqual(payload["digit_labels"].dtype, np.int64)
                self.assertEqual(payload["sector_labels"].dtype, np.int64)
            with open(meta_path) as f:
                metadata = json.load(f)
            self.assertEqual(metadata["coords_digit"]["axes"][2], "digit dPC3")
            self.assertEqual(metadata["coords_sector"]["axes"][2], "sector dPC3")
            self.assertEqual(metadata["interactive_html"], "dpca_3d_interactive_condensed.html")


if __name__ == "__main__":
    unittest.main()
