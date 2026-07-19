"""Smoke tests for reusable 3D dPCA coordinates and compatibility with 2D diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from utils_viz.pop_act_umap import (
    build_dpca_3d_plot_coordinates,
    resolve_dpca_output_dirs,
    run_dpca_condensed,
    save_dpca_3d_coordinates,
)


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
            fig_dir = os.path.join(
                tmpdir, "D_variance_decomposition", "pop_act_umap", "figs", "run"
            )
            data_dir = os.path.join(
                tmpdir, "D_variance_decomposition", "pop_act_umap", "data", "run"
            )
            data_path, meta_path = save_dpca_3d_coordinates(
                result["coords_digit"],
                result["coords_sector"],
                self.digit_labels,
                self.sector_labels,
                data_dir,
                fig_dir,
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
                self.assertEqual(payload["coords_digit_plot"].shape, (90, 3))
                self.assertEqual(payload["coords_sector_plot"].shape, (90, 3))
                self.assertEqual(payload["digit_labels"].dtype, np.int64)
                self.assertEqual(payload["sector_labels"].dtype, np.int64)
            with open(meta_path) as f:
                metadata = json.load(f)
            self.assertEqual(metadata["coords_digit"]["axes"][2], "digit dPC3")
            self.assertEqual(metadata["coords_sector"]["axes"][2], "sector dPC3")
            self.assertTrue(metadata["interactive_html"].endswith("dpca_3d_interactive.html"))
            self.assertTrue(metadata["plot_coordinates"]["visual_dodge_only"])
            self.assertEqual(os.path.basename(data_path), "dpca_3d_coordinates.npz")
            self.assertEqual(os.path.basename(meta_path), "dpca_3d_coordinates_meta.json")

    def test_90_point_dodge_separates_collapsed_conditions(self) -> None:
        zeros = np.zeros((90, 3), dtype=np.float32)
        raw_digit, raw_sector, digit_plot, sector_plot, _, _ = build_dpca_3d_plot_coordinates(
            zeros,
            zeros,
            self.digit_labels,
            self.sector_labels,
        )

        np.testing.assert_array_equal(raw_digit, zeros)
        np.testing.assert_array_equal(raw_sector, zeros)
        self.assertEqual(np.unique(digit_plot[:9, :2], axis=0).shape[0], 9)
        same_sector = np.arange(0, 90, 9)
        self.assertEqual(np.unique(sector_plot[same_sector, :2], axis=0).shape[0], 10)
        np.testing.assert_array_equal(digit_plot[:, 2], zeros[:, 2])
        np.testing.assert_array_equal(sector_plot[:, 2], zeros[:, 2])

    def test_output_dirs_keep_figures_and_data_parallel(self) -> None:
        fig_dir, data_dir = resolve_dpca_output_dirs(
            "results/anal_index/D_variance_decomposition/pop_act_umap/figs",
            "",
            "model_run",
        )
        self.assertEqual(
            fig_dir,
            "results/anal_index/D_variance_decomposition/pop_act_umap/figs/model_run",
        )
        self.assertEqual(
            data_dir,
            "results/anal_index/D_variance_decomposition/pop_act_umap/data/model_run",
        )


if __name__ == "__main__":
    unittest.main()
