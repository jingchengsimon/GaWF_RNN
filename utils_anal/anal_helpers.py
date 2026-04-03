"""
Shared helpers for analysis scripts: test split dataset construction and GaWF checkpoint loading.

Imports: train_model (MC_RNN_Dataset), utils.train_helpers (paths/datasets), utils_viz (hparam parsing).
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch

from train_model import MC_RNN_Dataset
from utils.train_gawf_core import GaWFRNNConv
from utils.train_helpers import PathHelper, create_datasets
from utils_viz.model_train_single_result import parse_hparams_from_filename


def resolve_device(
    device_flag: str,
    *,
    require_cuda_if_requested: bool = False,
) -> torch.device:
    """
    Map CLI device string to ``torch.device``.

    When ``require_cuda_if_requested`` is True and ``device_flag == "cuda"``,
    raises if CUDA is not available (stricter scripts).
    """
    if require_cuda_if_requested and device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested via --device cuda but no GPU is available."
            )
    return torch.device(device_flag)


def build_test_dataset(args: argparse.Namespace) -> Tuple[MC_RNN_Dataset, int]:
    """
    Build test dataset only, using the same helpers as training (splits=("test",)).
    """
    base_path = PathHelper.get_base_path(override=args.data_dir or None)
    paths = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("test",)
    )
    stims_test, lbls_test = PathHelper.load_raw_data(
        None, None, None, None,
        use_mmap=args.use_mmap,
        paths_tuple=paths,
    )

    test_ds, num_pos = create_datasets(
        None, None, None, None,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    return test_ds, num_pos


def build_model_from_ckpt(
    ckpt_path: str,
    num_pos: int,
    device: torch.device,
) -> GaWFRNNConv:
    """
    Instantiate GaWFRNNConv with hyperparameters parsed from the checkpoint filename,
    then load the checkpoint state_dict.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    dropout = float(hparams.get("dropout", 0.0))

    num_classes = 10

    model = GaWFRNNConv(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=5,
        device=str(device),
        dropout=dropout,
        hidden_size=hidden_size,
        max_chars=15,
        predict_all_chars=(num_pos == 0),
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing_keys:", load_result.missing_keys)
    print("[load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    model.to(device)
    model.eval()
    return model
