"""Tests for exact frame-major export of unified GaWF decomposition sources."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from utils.recurrent_cores.gawf import GaWFCore
from utils_anal.export_unified_variance_sources import (
    _array_shapes,
    _compact_paths,
    _finalize_compact_artifacts,
    _input_manifest_payload,
    _open_partial_arrays,
    _required_bytes,
    export_encoded_batch,
)
from utils_anal.run_unified_variance_decomposition import _load_inputs


class TinyModel(nn.Module):
    """Small canonical GaWF core plus a two-element feedback head."""

    def __init__(self) -> None:
        super().__init__()
        self.core = GaWFCore(input_size=3, hidden_size=2, feedback_dim=2, dropout=0.0)
        self.classifier_layer = nn.Linear(2, 2)

    @property
    def U(self) -> torch.Tensor:
        return self.core.U

    @property
    def V(self) -> torch.Tensor:
        return self.core.V

    @property
    def gate_tau(self) -> float:
        return self.core.gate_tau

    @property
    def feedback_dim(self) -> int:
        return self.core.feedback_dim

    def classifier(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.classifier_layer(hidden)
        return values[:, :1], values[:, 1:]

    def _compute_feedback(
        self, char_logits: torch.Tensor, sector_logits: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([char_logits, sector_logits], dim=-1)


def test_export_encoded_batch_is_frame_major_and_matches_canonical_steps() -> None:
    torch.manual_seed(4)
    model = TinyModel().eval()
    encoded = torch.randn(2, 3, 3)
    labels = torch.tensor(
        [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 0], [1, 2]]], dtype=torch.int64
    )
    arrays = {
        "encoder_activation": np.full((12, 3), np.nan, dtype=np.float32),
        "input_gate": np.full((12, 6), np.nan, dtype=np.float32),
        "hidden_state": np.full((12, 2), np.nan, dtype=np.float32),
        "recurrent_gate": np.full((12, 4), np.nan, dtype=np.float32),
    }
    feedback = np.full((12, 2), np.nan, dtype=np.float32)
    saved_labels = np.full((12, 2), -1, dtype=np.int64)
    export_encoded_batch(encoded, labels, model, arrays, feedback, saved_labels, 1)

    np.testing.assert_array_equal(
        arrays["encoder_activation"][3:9], encoded.numpy().reshape(6, 3)
    )
    np.testing.assert_array_equal(saved_labels[3:9], labels.numpy().reshape(6, 2))
    np.testing.assert_array_equal(feedback[[3, 6]], np.zeros((2, 2), dtype=np.float32))
    assert np.isnan(arrays["encoder_activation"][:3]).all()
    assert np.isnan(arrays["encoder_activation"][9:]).all()

    state = model.core.initial_state(2, encoded.device, encoded.dtype)
    current_feedback = torch.zeros(2, 2)
    expected_hidden = np.empty((12, 2), dtype=np.float32)
    for time_idx in range(3):
        fb_t = current_feedback.clamp(-10, 10).unsqueeze(2)
        scaled_u = model.U.unsqueeze(0) * fb_t.transpose(1, 2)
        direct_gate = torch.sigmoid(torch.matmul(scaled_u, model.V) / model.gate_tau)
        rows = np.asarray([3 + time_idx, 6 + time_idx])
        np.testing.assert_allclose(
            arrays["input_gate"][rows], direct_gate[..., :3].detach().numpy().reshape(2, -1)
        )
        np.testing.assert_allclose(
            arrays["recurrent_gate"][rows], direct_gate[..., 3:].detach().numpy().reshape(2, -1)
        )
        state = model.core.step(encoded[:, time_idx], state, current_feedback)
        expected_hidden[rows] = state.detach().numpy()
        char_logits, sector_logits = model.classifier(state)
        current_feedback = model._compute_feedback(char_logits, sector_logits)
    np.testing.assert_allclose(
        arrays["hidden_state"][3:9], expected_hidden[3:9], rtol=1e-6, atol=1e-6
    )


def test_required_bytes_accounts_for_all_four_float32_arrays() -> None:
    shapes = _array_shapes(n_frames=5, input_size=3, hidden_size=2)
    assert shapes == {
        "encoder_activation": (5, 3),
        "input_gate": (5, 6),
        "hidden_state": (5, 2),
        "recurrent_gate": (5, 4),
    }
    assert _required_bytes(shapes) == (15 + 30 + 10 + 20) * 4


def test_export_rejects_multilayer_state() -> None:
    model = TinyModel()
    model.core = GaWFCore(
        input_size=3,
        hidden_size=2,
        feedback_dim=2,
        num_layers=2,
        layer_feedback_dims=[2, 2],
    )
    arrays = {
        "encoder_activation": np.empty((1, 3), dtype=np.float32),
        "input_gate": np.empty((1, 6), dtype=np.float32),
        "hidden_state": np.empty((1, 2), dtype=np.float32),
        "recurrent_gate": np.empty((1, 4), dtype=np.float32),
    }
    with pytest.raises(RuntimeError, match="single-layer"):
        export_encoded_batch(
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 2, dtype=torch.int64),
            model,
            arrays,
            np.empty((1, 2), dtype=np.float32),
            np.empty((1, 2), dtype=np.int64),
            0,
        )


@pytest.mark.parametrize(
    "occupied_name",
    [
        "encoder_activation.npy",
        "input_gate.partial.npy",
        "gawf_gate_trajectory.npz",
        "input_manifest.partial.json",
        "source_provenance.json",
    ],
)
def test_source_export_refuses_to_overwrite_any_artifact(tmp_path, occupied_name: str) -> None:
    (tmp_path / occupied_name).write_bytes(b"sentinel")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _open_partial_arrays(tmp_path, _array_shapes(1, 3, 2))
    assert (tmp_path / occupied_name).read_bytes() == b"sentinel"


def test_compact_artifacts_are_published_from_partial_paths(tmp_path) -> None:
    final_paths, partial_paths = _compact_paths(tmp_path)
    for name, path in partial_paths.items():
        path.write_bytes(name.encode("utf-8"))
    _finalize_compact_artifacts(final_paths, partial_paths)
    for name, path in final_paths.items():
        assert path.read_bytes() == name.encode("utf-8")
        assert not partial_paths[name].exists()


def test_exported_manifest_schema_is_consumed_by_unified_runner(tmp_path) -> None:
    n_trials = 1
    shapes = {
        "encoder_activation": (n_trials, 1152),
        "input_gate": (n_trials, 256 * 1152),
        "hidden_state": (n_trials, 256),
        "recurrent_gate": (n_trials, 256 * 256),
    }
    paths = {}
    for name, shape in shapes.items():
        path = tmp_path / f"{name}.npy"
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape).flush()
        paths[name] = path
    trajectory = tmp_path / "gawf_gate_trajectory.npz"
    np.savez_compressed(
        trajectory,
        feedback=np.zeros((n_trials, 19), dtype=np.float32),
        labels=np.zeros((n_trials, 2), dtype=np.int64),
        weight_ih=np.zeros((256, 1152), dtype=np.float32),
        weight_hh=np.zeros((256, 256), dtype=np.float32),
    )
    payload = _input_manifest_payload(trajectory, paths, {"test": True})
    manifest = tmp_path / "input_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    labels, sources, loaded = _load_inputs(manifest)
    assert labels.shape == (1, 2)
    assert set(sources) == {
        "encoder_activation",
        "input_gate",
        "effective_input_weight",
        "hidden_state",
        "recurrent_gate",
        "effective_recurrent_weight",
        "feedback_vector",
    }
    assert loaded["source"] == {"test": True}
