"""Tests for configurable current-frame versus previous-plus-current-frame inputs."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiments.generalization.clutter_best6_chan1_multiseed import (
    build_sjc_manifest as build_multiseed_sjc_manifest,
)
from experiments.generalization.clutter_best6_chan1_seed42 import (
    build_sjc_manifest as build_seed42_sjc_manifest,
)
from train_model import MC_RNN_Dataset
from utils.clutter_task_models import RNNConv


def _labels(num_frames: int) -> pd.DataFrame:
    """Return minimal single-character sector labels for a synthetic sequence."""

    return pd.DataFrame(
        {
            "fg_char_id": np.arange(num_frames, dtype=np.int64) % 10,
            "fg_char_x": np.zeros(num_frames, dtype=np.int64),
            "fg_char_y": np.zeros(num_frames, dtype=np.int64),
        }
    )


def test_chan1_dataset_exposes_only_the_current_frame() -> None:
    """chan=1 must not leak the preceding image into the model input."""

    data = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    dataset = MC_RNN_Dataset(
        data,
        _labels(6),
        frame_num=2,
        chan_num=1,
        use_sector=True,
    )
    batch = dataset[0]
    frames, labels = batch[0], batch[1]
    assert frames.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(frames[:, 0], data[1:3])
    np.testing.assert_array_equal(labels[:, 0], np.array([1, 2], dtype=np.int64))


def test_chan1_rnn_encoder_and_forward_shapes() -> None:
    """The shared CNN/RNN model must accept one-channel sequences end to end."""

    model = RNNConv(
        num_classes=10,
        num_pos=9,
        kernel_size=5,
        device="cpu",
        input_channels=1,
        hidden_size=8,
        cnn_dropout=0.0,
        rnn_dropout=0.0,
    )
    assert model.encoder_module.conv1.in_channels == 1
    char_out, pos_out = model(torch.randn(2, 3, 1, 96, 96))
    assert char_out.shape == (2, 3, 10)
    assert pos_out.shape == (2, 3, 9)


def test_chan1_protocol_emits_isolated_paths_and_channel_count() -> None:
    """The chan=1 wrapper must not reuse chan=2 result or artifact namespaces."""

    result = subprocess.run(
        [
            sys.executable,
            "experiments/generalization/clutter_best6_chan1_multiseed.py",
            "emit-task",
            "--task-id",
            "0",
            "--root",
            "/remote/root",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "CHAN_NUM=1" in result.stdout
    assert "clutter_best6_multiseed_40h_chan1_ep150" in result.stdout
    assert "clutter_best6_10seed_ep150/" not in result.stdout


def test_sjc_manifest_tracks_balanced_analysis_before_unit_completion() -> None:
    """JOBS.md evidence must include the isolated balanced result for every unit."""

    manifest = build_multiseed_sjc_manifest("run-123", "/remote/root", "/conda/init.sh")
    assert manifest["host"] == "sjc-remote"
    assert manifest["scheduler"]["tmux_session"] == "run-123"
    assert manifest["tracking"]["expected_units"] == 60
    first = manifest["tracking"]["units"][0]
    assert first["expected"]["chan_num"] == 1
    assert first["analysis_result_dir"].endswith("/gawf-seed01")
    assert first["required_analysis_globs"] == [
        "fg_switch_offset_acc_*.npz",
        "fg_switch_offset_meta_*.json",
    ]


def test_chan1_seed42_protocol_contains_only_six_units() -> None:
    """The reduced Amarel submission must map six models at seed 42 and nothing else."""

    result = subprocess.run(
        [
            sys.executable,
            "experiments/generalization/clutter_best6_chan1_seed42.py",
            "emit-task",
            "--task-id",
            "5",
            "--root",
            "/remote/root",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "UNIT_ID=s5-seed42" in result.stdout
    assert "SEED=42" in result.stdout
    assert "CHAN_NUM=1" in result.stdout
    assert "clutter_best6_chan1_seed42_40h_ep150" in result.stdout


def test_chan1_seed42_sjc_manifest_tracks_training_only() -> None:
    """The reduced sjc run must track six training outputs without analysis evidence."""

    manifest = build_seed42_sjc_manifest("run-s42", "/remote/root", "/conda/init.sh")
    assert manifest["host"] == "sjc-remote"
    assert manifest["status"] == "running"
    assert manifest["scheduler"]["tmux_session"] == "run-s42"
    assert manifest["tracking"]["expected_units"] == 6
    assert manifest["tracking"]["units"][0]["id"] == "gawf-seed42"
    assert all("analysis_result_dir" not in unit for unit in manifest["tracking"]["units"])
