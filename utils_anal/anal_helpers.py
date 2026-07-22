"""
Shared helpers for analysis scripts: test split dataset construction and GaWF checkpoint loading.

Imports: train_model (MC_RNN_Dataset), clutter helpers (paths/datasets), and the canonical
checkpoint-filename hyperparameter parser.
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
            raise RuntimeError("CUDA requested via --device cuda but no GPU is available.")
    return torch.device(device_flag)


def build_eval_dataset(
    args: argparse.Namespace,
    split: str,
) -> Tuple[MC_RNN_Dataset, int]:
    """Build one held-out Clutter split with the same helpers used by training.

    ``split`` must be ``"validation"``/``"valid"`` or ``"test"``.  This helper exists so
    analyses that
    estimate statistics on validation and test hypotheses on test cannot accidentally load the
    training split or silently reuse one held-out split for both roles.

    If ``args.switch_target`` is ``"fg"`` or ``"bg"`` (export script): sector single-char
    labels are used; timing windows use ``fg_switch`` vs ``bg_switch`` only in the caller.
    Otherwise uses ``args.use_sector_mode`` and ``args.predict_all_chars`` (legacy callers).
    """
    if split not in ("validation", "valid", "test"):
        raise ValueError(f"split must be 'validation', 'valid', or 'test', got {split!r}")
    canonical_split = "valid" if split == "validation" else split
    switch = getattr(args, "switch_target", None)
    if switch in ("fg", "bg"):
        use_sector_mode, predict_all_chars = True, False
    else:
        use_sector_mode = getattr(args, "use_sector_mode", True)
        predict_all_chars = getattr(args, "predict_all_chars", False)

    base_path = PathHelper.get_base_path(override=args.data_dir or None)
    paths = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=(canonical_split,)
    )
    loaded = PathHelper.load_raw_data(
        None, None, None, None, use_mmap=args.use_mmap, paths_tuple=paths
    )
    split_stims, split_labels = loaded[:2]
    if canonical_split == "test":
        dataset_args = (None, None, None, None)
        dataset_kwargs = {"stims_test": split_stims, "lbls_test": split_labels}
    else:
        dataset_args = (None, None, split_stims, split_labels)
        dataset_kwargs = {}

    dataset, num_pos = create_datasets(
        *dataset_args,
        use_sector_mode=use_sector_mode,
        predict_all_chars=predict_all_chars,
        chan_num=int(getattr(args, "chan_num", 2)),
        max_chars=15,
        dataset_class=MC_RNN_Dataset,
        splits=(canonical_split,),
        **dataset_kwargs,
    )
    return dataset, num_pos


def build_test_dataset(args: argparse.Namespace) -> Tuple[MC_RNN_Dataset, int]:
    """Build the test split; retained as the compatibility entry point for existing analyses."""

    return build_eval_dataset(args, "test")


def build_train_dataset_allchars(args: argparse.Namespace) -> MC_RNN_Dataset:
    """Training split only, predict_all_chars labels (for bg head finetune)."""
    base_path = PathHelper.get_base_path(override=args.data_dir or None)
    paths = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("train",)
    )
    stims_train, lbls_train = PathHelper.load_raw_data(
        None,
        None,
        None,
        None,
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
        chan_num=int(getattr(args, "chan_num", 2)),
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


def _remap_legacy_clutter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map pre-componentization clutter checkpoint keys to the current module layout."""

    target_keys = set(model.state_dict())
    if any(key.startswith("encoder_module.") for key in state_dict):
        return {key: value for key, value in state_dict.items() if key != "prev_feedback"}

    encoder_prefixes = ("conv1.", "LNorm1.", "conv2.", "LNorm2.", "conv_reduce.")
    head_prefixes = ("fcchar.", "fcchars.", "fcpos.")
    sequence_core_prefix = (
        "core." if getattr(model, "uses_mamba_core", False) or getattr(model, "uses_s5_core", False)
        else "core.rnn."
    )
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == "prev_feedback":
            continue
        if key in target_keys:
            new_key = key
        elif key.startswith(encoder_prefixes):
            new_key = f"encoder_module.{key}"
        elif key.startswith(head_prefixes):
            new_key = f"head.{key}"
        elif key.startswith("rnn."):
            new_key = f"{sequence_core_prefix}{key[len('rnn.') :]}"
        elif key.startswith("LNormRNN."):
            new_key = f"core.norm.{key[len('LNormRNN.') :]}"
        elif key in ("U", "V"):
            new_key = f"core.{key}"
        else:
            new_key = key
        if new_key in remapped:
            raise RuntimeError(f"Checkpoint keys collide after legacy remapping: {new_key}")
        remapped[new_key] = value
    return remapped


def build_model_from_ckpt(
    ckpt_path: str,
    num_pos: int,
    device: torch.device,
    chan_num: int = 2,
) -> torch.nn.Module:
    """
    Instantiate a sequence model with hyperparameters parsed from the checkpoint filename,
    then load the checkpoint state_dict.
    """
    ckpt_basename = os.path.basename(ckpt_path)
    hparams = parse_hparams_from_filename(ckpt_basename)

    hidden_size = hparams.get("hidden_size", 256)
    cnn_dropout = float(hparams.get("cnn_dropout", 0.0))
    rnn_dropout = float(hparams.get("rnn_dropout", hparams.get("dropout", 0.5)))

    num_classes = 10
    model_name = hparams.get("model_type")
    model_key = _HPARAM_MODEL_TO_KEY.get(model_name, "gawf")
    num_layers = int(hparams.get("num_layers", hparams.get("gawf_layers", 1)))
    if model_key == "gawf" and num_layers > 1:
        model_key = "gawf_multi"
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
            model_kwargs["num_layers"] = num_layers
    elif model_key in ("rnn", "lstm", "gru"):
        model_kwargs["num_layers"] = num_layers
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
        input_channels=chan_num,
        cnn_dropout=cnn_dropout,
        rnn_dropout=rnn_dropout,
        **({} if model_key in ("mamba", "s5") else {"hidden_size": hidden_size}),
        max_chars=15,
        predict_all_chars=(num_pos == 0),
        **model_kwargs,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _remap_legacy_clutter_state_dict(state_dict, model)
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
    rnn_dropout = float(hparams.get("rnn_dropout", hparams.get("dropout", 0.5)))

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
    state_dict = _remap_legacy_clutter_state_dict(state_dict, model)
    load_result = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict allchars head] missing_keys:", load_result.missing_keys)
    print("[load_state_dict allchars head] unexpected_keys:", load_result.unexpected_keys)

    for name, p in model.named_parameters():
        p.requires_grad = "fcchars" in name

    model.to(device)
    model.train(train_mode)
    return model
