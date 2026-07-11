"""
Shared helpers for analysis scripts: test split dataset construction and GaWF checkpoint loading.

Imports: train_model (MC_RNN_Dataset), utils.clutter_train_helpers (paths/datasets), utils_viz (hparam parsing).
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch

from train_model import MC_RNN_Dataset
from utils.clutter_task_models import (
    GaWFRNNConv,
    GRUConv,
    LSTMConv,
    MambaConv,
    MultiLayerGaWFRNNConv,
    RNNConv,
    S5Conv,
)
from utils.clutter_train_helpers import PathHelper, create_datasets
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

    If ``args.switch_target`` is ``"fg"`` or ``"bg"`` (export script): sector single-char
    test split; timing windows use ``fg_switch`` vs ``bg_switch`` only in the export script.
    Otherwise uses ``args.use_sector_mode`` and ``args.predict_all_chars`` (legacy callers).
    """
    switch = getattr(args, "switch_target", None)
    if switch in ("fg", "bg"):
        use_sector_mode, predict_all_chars = True, False
    else:
        use_sector_mode = getattr(args, "use_sector_mode", True)
        predict_all_chars = getattr(args, "predict_all_chars", False)

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
        use_sector_mode=use_sector_mode,
        predict_all_chars=predict_all_chars,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("test",),
        stims_test=stims_test,
        lbls_test=lbls_test,
    )
    return test_ds, num_pos


def build_train_dataset_allchars(args: argparse.Namespace) -> MC_RNN_Dataset:
    """Training split only, predict_all_chars labels (for bg head finetune)."""
    base_path = PathHelper.get_base_path(override=args.data_dir or None)
    paths = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("train",)
    )
    stims_train, lbls_train = PathHelper.load_raw_data(
        None, None, None, None,
        use_mmap=args.use_mmap,
        paths_tuple=paths,
    )
    train_ds, _num_pos = create_datasets(
        stims_train,
        lbls_train,
        None,
        None,
        use_sector_mode=True,
        predict_all_chars=True,
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=("train",),
    )
    return train_ds


_HPARAM_MODEL_TO_KEY: Dict[str, str] = {
    "GaWF": "gawf",
    "GaWFMulti": "gawf_multi",
    "RNN": "rnn",
    "LSTM": "lstm",
    "GRU": "gru",
    "MAMBA": "mamba",
    "S5": "s5",
}


def build_model_from_ckpt(
    ckpt_path: str,
    num_pos: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Instantiate a sequence model with hyperparameters parsed from the checkpoint filename,
    then load the checkpoint state_dict.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    cnn_dropout = float(hparams.get("cnn_dropout", 0.0))
    rnn_dropout = float(
        hparams.get("rnn_dropout", hparams.get("dropout", 0.5))
    )

    num_classes = 10
    model_name = hparams.get("model_type")
    model_key = _HPARAM_MODEL_TO_KEY.get(model_name, "gawf")
    model_class_map = {
        "gawf": GaWFRNNConv,
        "gawf_multi": MultiLayerGaWFRNNConv,
        "rnn": RNNConv,
        "lstm": LSTMConv,
        "gru": GRUConv,
        "mamba": MambaConv,
        "s5": S5Conv,
    }
    model_cls = model_class_map[model_key]
    model_kwargs = {}
    if model_key in ("gawf", "gawf_multi"):
        parsed_feedback_dim = hparams.get("feedback_dim")
        if parsed_feedback_dim is not None:
            model_kwargs["feedback_dim"] = int(parsed_feedback_dim)
        if model_key == "gawf_multi":
            model_kwargs["num_layers"] = int(hparams.get("gawf_layers", 2))
    elif model_key == "mamba":
        model_kwargs["mamba_d_model"] = int(hparams.get("d_model", 170))
    elif model_key == "s5":
        model_kwargs["s5_d_model"] = int(hparams.get("d_model", 256))
        model_kwargs["s5_state_size"] = int(hparams.get("state_size", 128))

    model = model_cls(
        num_classes=num_classes,
        num_pos=num_pos,
        kernel_size=5,
        device=str(device),
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        **({} if model_key in ("mamba", "s5") else {"hidden_size": hidden_size}),
        max_chars=15,
        predict_all_chars=(num_pos == 0),
        **model_kwargs,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict] missing_keys:", load_result.missing_keys)
    print("[load_state_dict] unexpected_keys:", load_result.unexpected_keys)
    model.to(device)
    model.eval()
    return model


def build_rnn_allchars_model_from_sector_ckpt(
    ckpt_path: str,
    device: torch.device,
    *,
    max_chars: int = 15,
    train_mode: bool = False,
) -> torch.nn.Module:
    """
    RNN/LSTM/GRU only: ``predict_all_chars=True`` (fcchars), load backbone weights from a
    sector checkpoint; ``fcchar``/``fcpos`` keys are skipped, ``fcchars`` stays randomly init.
    GaWF raises ``RuntimeError``.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    cnn_dropout = float(hparams.get("cnn_dropout", 0.0))
    rnn_dropout = float(
        hparams.get("rnn_dropout", hparams.get("dropout", 0.5))
    )

    num_classes = 10
    model_name = hparams.get("model_type")
    model_key = _HPARAM_MODEL_TO_KEY.get(model_name, "gawf")
    if model_key in ("gawf", "gawf_multi"):
        raise RuntimeError(
            "BG switch offset analysis does not support GaWF checkpoints "
            "(use RNN, LSTM, or GRU sector checkpoints only)."
        )
    if model_key not in ("rnn", "lstm", "gru"):
        raise RuntimeError(
            f"BG mode requires RNN/LSTM/GRU checkpoint; got model_type={model_name!r}."
        )
    model_class_map = {
        "rnn": RNNConv,
        "lstm": LSTMConv,
        "gru": GRUConv,
    }
    model_cls = model_class_map[model_key]

    model = model_cls(
        num_classes=num_classes,
        num_pos=9,
        kernel_size=5,
        device=str(device),
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        hidden_size=hidden_size,
        max_chars=max_chars,
        predict_all_chars=True,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict allchars head] missing_keys:", load_result.missing_keys)
    print("[load_state_dict allchars head] unexpected_keys:", load_result.unexpected_keys)

    for name, p in model.named_parameters():
        p.requires_grad = "fcchars" in name

    model.to(device)
    model.train(train_mode)
    return model
