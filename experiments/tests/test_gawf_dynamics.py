"""Checks for the GaWF task-driven dynamics analysis."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import torch

from utils.analysis.clutter.gawf_dynamics import (
    _event_measurements,
    _event_candidates,
    gawf_jacobian_objects,
    plot,
    select_balanced_events,
)
from utils.training.clutter.clutter_task_models import GaWFRNNConv


def test_closed_loop_jacobian_matches_autograd() -> None:
    torch.manual_seed(7)
    model = GaWFRNNConv(
        num_classes=10,
        num_pos=9,
        hidden_size=4,
        kernel_size=5,
        device="cpu",
        rnn_dropout=0.0,
    ).eval()
    encoded = torch.randn(1, model.encoder_flatten_size) * 0.05
    hidden = torch.rand(1, 4)
    char_logits, sector_logits = model.classifier(hidden)
    feedback = model._compute_feedback(char_logits, sector_logits)
    objects = gawf_jacobian_objects(model, encoded, hidden, feedback)

    def frozen_map(hidden_vector: torch.Tensor) -> torch.Tensor:
        return model.core.step(encoded, hidden_vector.unsqueeze(0), feedback).squeeze(0)

    def closed_map(hidden_vector: torch.Tensor) -> torch.Tensor:
        current = hidden_vector.unsqueeze(0)
        char, sector = model.classifier(current)
        current_feedback = model._compute_feedback(char, sector)
        return model.core.step(encoded, current, current_feedback).squeeze(0)

    closed = torch.autograd.functional.jacobian(closed_map, hidden.squeeze(0))
    torch.testing.assert_close(objects["hidden_next"], frozen_map(hidden.squeeze(0)).unsqueeze(0))
    torch.testing.assert_close(objects["closed_loop_jacobian"][0], closed, atol=2e-5, rtol=2e-4)


def test_event_selection_is_balanced_and_uses_widest_eligible_window() -> None:
    frame_num = 64
    chan_num = 2
    total_frames = chan_num + 90 * frame_num
    fg_switch = np.zeros(total_frames, dtype=np.int64)
    bg_switch = np.zeros(total_frames, dtype=np.int64)
    labels = np.zeros((total_frames, 2), dtype=np.int64)
    for condition_index, (digit, sector) in enumerate(
        (digit, sector) for digit in range(10) for sector in range(9)
    ):
        raw_frame = chan_num + condition_index * frame_num + 32
        fg_switch[raw_frame] = 1
        bg_switch[raw_frame] = 1
        labels[raw_frame] = (digit, sector)
    dataset = SimpleNamespace(
        fg_switch=fg_switch,
        bg_switch=bg_switch,
        labels_sector=labels,
        frame_num=frame_num,
        chan_num=chan_num,
    )
    radius, events, audit = select_balanced_events(dataset, [10, 20], 1, 1, 4)
    assert radius == 20
    assert len(events) == 90
    assert audit["events_per_cell"] == 1
    assert len({(event["digit"], event["sector"]) for event in events}) == 90


def test_event_selection_excludes_nonjoint_target_switches() -> None:
    frame_num = 64
    chan_num = 2
    total_frames = chan_num + 90 * frame_num
    fg_switch = np.zeros(total_frames, dtype=np.int64)
    bg_switch = np.zeros(total_frames, dtype=np.int64)
    labels = np.zeros((total_frames, 2), dtype=np.int64)
    for condition_index, (digit, sector) in enumerate(
        (digit, sector) for digit in range(10) for sector in range(9)
    ):
        raw_frame = chan_num + condition_index * frame_num + 32
        fg_switch[raw_frame] = 1
        bg_switch[raw_frame] = condition_index != 0
        labels[raw_frame] = (digit, sector)
    dataset = SimpleNamespace(
        fg_switch=fg_switch,
        bg_switch=bg_switch,
        labels_sector=labels,
        frame_num=frame_num,
        chan_num=chan_num,
    )
    with np.testing.assert_raises(RuntimeError):
        select_balanced_events(dataset, [10], 1, 1, 4)


def test_event_candidates_exclude_trailing_incomplete_rollout() -> None:
    frame_num = 64
    chan_num = 2
    total_frames = chan_num + frame_num + 32
    fg_switch = np.zeros(total_frames, dtype=np.int64)
    bg_switch = np.zeros(total_frames, dtype=np.int64)
    labels = np.zeros((total_frames, 2), dtype=np.int64)
    trailing_event = chan_num + frame_num + 16
    fg_switch[trailing_event] = 1
    bg_switch[trailing_event] = 1
    dataset = SimpleNamespace(
        fg_switch=fg_switch,
        bg_switch=bg_switch,
        labels_sector=labels,
        frame_num=frame_num,
        chan_num=chan_num,
    )
    assert _event_candidates(dataset, radius=10) == []


def test_event_measurements_accept_numpy_frames() -> None:
    class NumpyFrameDataset:
        def __getitem__(self, _index):
            frames = np.zeros((32, 2, 96, 96), dtype=np.float32)
            labels = np.zeros((32, 2), dtype=np.int64)
            return frames, labels

    model = GaWFRNNConv(
        num_classes=10,
        num_pos=9,
        hidden_size=4,
        kernel_size=5,
        device="cpu",
        rnn_dropout=0.0,
    ).eval()
    event = {
        "sequence_id": 0,
        "center": 16,
        "cell_rank": 0,
        "event_id": 0,
        "raw_frame": 18,
        "digit": 0,
        "sector": 0,
    }
    rows, eigen_rows, propagator_rows = _event_measurements(
        NumpyFrameDataset(), model, torch.device("cpu"), event, 10, torch.float32, 1
    )
    assert len(rows) == 60
    assert len(eigen_rows) == 60
    assert len(propagator_rows) == 4


def _write_rows(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_smoke(tmp_path) -> None:
    objects = ("static_weight", "effective_weight", "closed_loop_jacobian")
    offsets = (-1, 1, 3, 4, 10)
    landmarks = ("post1", "post3", "post4", "post10", "post_extended")
    windows = (
        "post1_to_post3",
        "post1_to_post4",
        "post1_to_post10",
        "post1_to_post_extended",
    )
    for seed in range(1, 11):
        seed_dir = tmp_path / f"gawf-seed{seed:02d}"
        seed_dir.mkdir()
        matrix_rows = []
        for offset in offsets:
            for digit in range(10):
                for sector in range(9):
                    for object_index, object_name in enumerate(objects):
                        value = 0.8 + object_index / 10 + digit / 100 + sector / 1000
                        matrix_rows.append(
                            {
                                "seed": seed,
                                "offset": offset,
                                "digit": digit,
                                "sector": sector,
                                "object": object_name,
                                "spectral_radius": value,
                                "sigma_max": value + 0.2,
                                "frobenius_norm": value + 1.0,
                                "expansive_fraction": 0.1,
                            }
                        )
        eigen_rows = [
            {
                "seed": seed,
                "landmark": landmark,
                "object": object_name,
                "real": value,
                "imag": value / 2,
            }
            for landmark in landmarks
            for object_name in objects
            for value in (-0.5, 0.0, 0.5)
        ]
        finite_rows = [
            {
                "seed": seed,
                "window": window,
                "object": object_name,
                "maximum_log_gain": 0.01 * seed,
            }
            for window in windows
            for object_name in ("closed_loop_jacobian",)
        ]
        _write_rows(seed_dir / "event_matrix_metrics.csv", matrix_rows)
        _write_rows(seed_dir / "landmark_eigenvalues.csv", eigen_rows)
        _write_rows(seed_dir / "finite_time_gain.csv", finite_rows)
        np.savez_compressed(
            seed_dir / "static_recurrent_spectrum.npz",
            eigenvalues=np.asarray([-0.5 + 0.1j, 0.5 - 0.1j]),
        )
        (seed_dir / ".complete").touch()
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "unrelated.pdf").touch()
    plot(SimpleNamespace(input_root=tmp_path, figure_dir=figure_dir, expected_seeds=10))
    assert len(list(figure_dir.glob("gawf_*.pdf"))) == 5
    assert (tmp_path / "figure_manifest.json").is_file()
