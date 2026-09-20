"""Meaningful unit checks for continuous switch rollout and PCA event selection."""

from __future__ import annotations

import numpy as np
import torch

from utils.analysis.clutter.continuous_switch_pca import (
    _event_windows,
    _heldout_capture,
    _pca,
    _stacked_frames,
    clean_joint_switch_frames,
    continuous_rollout,
)
from utils.training.clutter.clutter_task_models import GaWFRNNConv, RNNConv


def test_stacked_frames_preserve_boundary_history_and_tail() -> None:
    data = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
    frames = _stacked_frames(data, 2, 12, 2)
    assert frames.shape == (10, 2, 1, 1)
    np.testing.assert_array_equal(frames[:, :, 0, 0], np.column_stack((np.arange(1, 11), np.arange(2, 12))))


def test_clean_joint_windows_exclude_second_event_and_boundary() -> None:
    fg = np.zeros(20, dtype=np.int64)
    bg = np.zeros(20, dtype=np.int64)
    fg[[2, 8, 13]] = 1
    bg[[2, 8]] = 1
    assert clean_joint_switch_frames(fg, bg, radius=3, first_frame=2, stop_frame=20).tolist() == [8]


def test_event_window_tail_and_pca_are_runnable() -> None:
    frames = np.arange(2, 10, dtype=np.int64)
    values = np.arange(8, dtype=np.float32)[:, None]
    selected = _event_windows(values, frames, np.asarray([8], dtype=np.int64), radius=2)
    np.testing.assert_array_equal(selected[:, :, 0], [[4, 5, 6, 7]])
    values = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    mean, basis, explained, _singular = _pca(values, components=2)
    assert mean.shape == (2,)
    assert basis.shape == (2, 2)
    np.testing.assert_allclose(explained.sum(), 1.0)


def test_heldout_pca_removes_training_events_sharing_heldout_frames() -> None:
    values = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    frame_ids = np.asarray(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [15, 16, 17, 18]],
        dtype=np.int64,
    )
    captured, provenance = _heldout_capture(values, frame_ids, components=2)
    assert np.isfinite(captured)
    assert provenance["train_event_indices_after_frame_disjoint_filter"] == [0, 1, 2]
    assert provenance["removed_overlapping_train_event_indices"] == [3]
    assert provenance["shared_raw_frames_after_filter"] == 0


@torch.no_grad()
def test_rnn_and_gawf_chunked_rollouts_match_single_chunk() -> None:
    torch.manual_seed(4)
    data = np.random.default_rng(9).normal(size=(9, 96, 96)).astype(np.float32)
    for model_name, model in (
        ("rnn", RNNConv(10, 9, hidden_size=4, kernel_size=5, device="cpu", rnn_dropout=0.0)),
        ("gawf", GaWFRNNConv(10, 9, hidden_size=4, kernel_size=5, device="cpu", rnn_dropout=0.0)),
    ):
        model.eval()
        whole = continuous_rollout(model, data, model_name=model_name, chan_num=2, chunk_size=32,
                                   device=torch.device("cpu"))
        chunked = continuous_rollout(model, data, model_name=model_name, chan_num=2, chunk_size=3,
                                     device=torch.device("cpu"))
        for key in whole:
            np.testing.assert_allclose(whole[key], chunked[key], rtol=1e-5, atol=1e-6)
        frames = torch.from_numpy(_stacked_frames(data, 2, len(data), 2)).unsqueeze(0)
        expected_char, expected_sector = model(frames)
        np.testing.assert_allclose(
            whole["char_logits"], expected_char.squeeze(0).numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            whole["sector_logits"], expected_sector.squeeze(0).numpy(), rtol=1e-5, atol=1e-6
        )
