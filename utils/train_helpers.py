"""
Helper functions for training script.
Contains utility functions for data loading, path management, result saving,
random seed setting, GPU memory management, model class mapping, logging,
and experiment metrics summaries (SummaryStatsHelper).
"""
import argparse
import logging
import os
from typing import Any, Dict, List, Optional
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# Logging (LoggingHelper): tqdm-safe console + optional file
# -----------------------------------------------------------------------------


class LoggingHelper:
    """Experiment logging helpers; console output uses tqdm.write when available."""

    BANNER_LEN = 60

    class TqdmStreamHandler(logging.StreamHandler):
        """Console handler that uses tqdm.write when available to avoid breaking progress bars."""

        def emit(self, record):
            try:
                msg = self.format(record)
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except Exception:
                    self.stream.write(msg + self.terminator)
                    self.flush()
            except Exception:
                self.handleError(record)

    @staticmethod
    def setup_logger(name="train", level=logging.INFO, log_file=None):
        """
        Lightweight logger setup using stdlib logging. Idempotent: repeated calls
        for the same name do not add duplicate handlers.

        Args:
            name: Logger name (e.g. "train").
            level: Log level (default INFO).
            log_file: If set, also append logs to this file.

        Returns:
            logging.Logger configured with console (and optional file) output.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.handlers:
            return logger
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        console = LoggingHelper.TqdmStreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        if log_file:
            try:
                fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except OSError:
                pass
        return logger

    @staticmethod
    def log_experiment_config(
        logger,
        total_experiments,
        model_types,
        hidden_sizes,
        lrs,
        weight_decays,
        cnn_dropout,
        rnn_dropout,
    ):
        sep = "=" * LoggingHelper.BANNER_LEN
        logger.info(sep)
        logger.info("Starting training loop: %s experiments", total_experiments)
        logger.info(
            "Models: %s | Hidden sizes: %s",
            model_types,
            hidden_sizes,
        )
        logger.info(
            "Learning rates: %s | Weight decays: %s | CNN dropout (grid): %s | RNN dropout: %s",
            lrs,
            weight_decays,
            cnn_dropout,
            rnn_dropout,
        )
        logger.info(sep)

    @staticmethod
    def log_experiment_start(
        logger,
        experiment_num,
        total_experiments,
        model_type,
        hidden_size,
        lr,
        weight_decay,
        cnn_dropout,
        rnn_dropout,
    ):
        sep = "=" * LoggingHelper.BANNER_LEN
        logger.info(
            "Experiment %s/%s: %s | hidden_size=%s | lr=%s | weight_decay=%s | cnn_dropout=%s | rnn_dropout=%s",
            experiment_num,
            total_experiments,
            model_type.upper(),
            hidden_size,
            lr,
            weight_decay,
            cnn_dropout,
            rnn_dropout,
        )
        logger.info(sep)

    @staticmethod
    def log_write(logger, msg, level=logging.INFO):
        logger.log(level, msg)

    @staticmethod
    def log_dataset_and_batch_info(
        logger,
        train_data,
        val_data,
        batch_size,
        accel_config,
        train_dl,
        num_workers,
        pin_memory,
        use_sector,
        predict_all_chars,
    ):
        logger.info("Dataset Information:")
        logger.info("  Training dataset size: %s samples", len(train_data))
        logger.info("  Validation dataset size: %s samples", len(val_data))
        logger.info("  Batch size per step: %s", batch_size)
        logger.info(
            "  Effective batch size (with grad accum): %s",
            batch_size * accel_config.grad_accum_steps,
        )
        logger.info("  Number of batches per epoch: %s", len(train_dl))
        logger.info("  use_sector: %s, predict_all_chars: %s", use_sector, predict_all_chars)
        logger.info(
            "  acceleration: %s, workers: %s, pin_memory: %s",
            accel_config.use_acceleration,
            num_workers,
            pin_memory and num_workers > 0,
        )
        if num_workers > 0:
            logger.info(
                "  DataLoader prefetch_factor: %s",
                accel_config.dataloader_prefetch_factor,
            )


BANNER_LEN = LoggingHelper.BANNER_LEN


# -----------------------------------------------------------------------------
# Paths, raw data load, and artifact saves (PathHelper)
# -----------------------------------------------------------------------------


class PathHelper:
    """Stimulus paths, mmap/numpy loads, training pickles, and metrics JSON."""

    @staticmethod
    def get_base_path(
        override: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> str:
        if override and override.strip():
            base_path = os.path.abspath(os.path.expanduser(override.strip()))
            if logger is not None:
                logger.info("Using base path (override): %s", base_path)
            return base_path
        env_path = os.environ.get("AIM3_STIMULI_PATH") or os.environ.get("FAW_RNN_DATA_PATH")
        if env_path and env_path.strip():
            base_path = os.path.abspath(os.path.expanduser(env_path.strip()))
            if logger is not None:
                logger.info("Using base path (env): %s", base_path)
            return base_path
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(_this_dir)
        base_path = os.path.join(_repo_root, "stimuli")
        if logger is not None:
            logger.info("Using base path (project-relative): %s", base_path)
        return base_path

    @staticmethod
    def get_results_root(
        override: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> str:
        """Resolve the root directory under which results are written.

        Mirrors get_base_path so results can live off the (quota-limited) home
        directory: an explicit override wins, then the AIM3_RESULTS_PATH /
        FAW_RNN_RESULTS_PATH env vars (set on Amarel to point at /scratch),
        else the project-relative ``results`` directory.
        """
        if override and override.strip():
            root = os.path.abspath(os.path.expanduser(override.strip()))
            if logger is not None:
                logger.info("Using results root (override): %s", root)
            return root
        env_path = os.environ.get("AIM3_RESULTS_PATH") or os.environ.get("FAW_RNN_RESULTS_PATH")
        if env_path and env_path.strip():
            root = os.path.abspath(os.path.expanduser(env_path.strip()))
            if logger is not None:
                logger.info("Using results root (env): %s", root)
            return root
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(_this_dir)
        root = os.path.join(_repo_root, "results")
        if logger is not None:
            logger.info("Using results root (project-relative): %s", root)
        return root

    @staticmethod
    def prepare_data_paths(
        base_path: str,
        data_suffix: str = "",
        splits: tuple = ("train", "valid"),
        logger: Optional[logging.Logger] = None,
    ):
        if data_suffix:
            if not data_suffix.startswith("-"):
                suffix = f"-{data_suffix}"
            else:
                suffix = data_suffix
        else:
            suffix = ""

        _path_spec = {
            "train": ("stimulus_reg-train", "train"),
            "valid": ("stimulus_reg-validation", "val"),
            "test": ("stimulus_reg-test", "test"),
        }
        out = []
        checks = []
        for name in splits:
            if name not in _path_spec:
                raise ValueError(f"Unknown split: {name}. Must be one of {list(_path_spec)}")
            base, label = _path_spec[name]
            stim_path = os.path.join(base_path, f"{base}{suffix}.npy")
            label_path = os.path.join(base_path, f"{base}{suffix}.tsv")
            out.extend([stim_path, label_path])
            checks.extend([(f"{label} stim", stim_path), (f"{label} label", label_path)])

        if logger is not None:
            logger.info("Checking data paths...")
            logger.info("Base path: %s", base_path)
        for path_name, path in checks:
            if not os.path.exists(path):
                msg = f"Data file not found: {path_name} -> {path}"
                if logger is not None:
                    logger.error(msg)
                raise FileNotFoundError(msg)
            if logger is not None:
                logger.info("  OK %s: %s", path_name, path)

        return tuple(out)

    @staticmethod
    def save_results(
        results,
        filepath,
        logger: Optional[logging.Logger] = None,
    ):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            if logger is not None:
                logger.info("Created directory: %s", directory)

        results_path = filepath + ".pkl"
        save_dict = {}
        for key, value in results.items():
            if key != "model":
                save_dict[key] = value

        model_path = results_path.replace(".pkl", "_model.pth")
        if "model" in results:
            if hasattr(results["model"], "fcpos") and results["model"].fcpos is not None:
                saved_num_pos = results["model"].fcpos.out_features
                if logger is not None:
                    logger.info(
                        "Model has position prediction layer (num_pos=%s)",
                        saved_num_pos,
                    )
            elif (
                hasattr(results["model"], "predict_all_chars")
                and results["model"].predict_all_chars
            ):
                if logger is not None:
                    logger.info("Model is in predict_all_chars mode (no position prediction)")
            else:
                if logger is not None:
                    logger.info("Model does not have position prediction layer")
            torch.save(results["model"].state_dict(), model_path)
            if logger is not None:
                logger.info("Model state dict saved to: %s", model_path)

        with open(results_path, "wb") as f:
            pickle.dump(save_dict, f)
        if logger is not None:
            logger.info("Results saved to: %s", results_path)

    @staticmethod
    def load_raw_data(
        stim_train_path: str,
        label_train_path: str,
        stim_val_path: str,
        label_val_path: str,
        use_mmap: bool = False,
        paths_tuple=None,
        logger=None,
    ):
        """Load raw stimulus arrays and label TSVs. Optional logger for progress."""
        if paths_tuple is not None:
            n = len(paths_tuple)
            if n == 2:
                stim_path, label_path = paths_tuple
                return PathHelper._load_single_split(
                    stim_path, label_path, "test", use_mmap, logger=logger
                )
            if n == 4:
                stim_train_path, label_train_path, stim_val_path, label_val_path = paths_tuple
                return PathHelper._load_train_val(
                    stim_train_path,
                    label_train_path,
                    stim_val_path,
                    label_val_path,
                    use_mmap,
                    logger=logger,
                )
            if n == 6:
                (
                    stim_train_path,
                    label_train_path,
                    stim_val_path,
                    label_val_path,
                    stim_test_path,
                    label_test_path,
                ) = paths_tuple
                train_val = PathHelper._load_train_val(
                    stim_train_path,
                    label_train_path,
                    stim_val_path,
                    label_val_path,
                    use_mmap,
                    logger=logger,
                )
                test_pair = PathHelper._load_single_split(
                    stim_test_path, label_test_path, "test", use_mmap, logger=logger
                )
                return train_val + test_pair
            raise ValueError(f"paths_tuple must have length 2, 4, or 6, got {n}")

        return PathHelper._load_train_val(
            stim_train_path,
            label_train_path,
            stim_val_path,
            label_val_path,
            use_mmap,
            logger=logger,
        )

    @staticmethod
    def _load_train_val(
        stim_train_path,
        label_train_path,
        stim_val_path,
        label_val_path,
        use_mmap,
        logger=None,
    ):
        if logger is not None:
            logger.info("Loading data (train, valid)...")
        if use_mmap:
            stims_train = np.load(stim_train_path, allow_pickle=True, mmap_mode="r")
            stims_val = np.load(stim_val_path, allow_pickle=True, mmap_mode="r")
            if logger is not None:
                logger.info("  Loaded training stimuli (mmap): %s", stims_train.shape)
                logger.info("  Loaded validation stimuli (mmap): %s", stims_val.shape)
        else:
            stims_train = np.load(stim_train_path, allow_pickle=True)
            stims_val = np.load(stim_val_path, allow_pickle=True)
            if logger is not None:
                logger.info("  Loaded training stimuli (ndarray): %s", stims_train.shape)
                logger.info("  Loaded validation stimuli (ndarray): %s", stims_val.shape)
        lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
        if logger is not None:
            logger.info("  Loaded training labels: %s", lbls_train.shape)
        lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
        if logger is not None:
            logger.info("  Loaded validation labels: %s", lbls_val.shape)
        return stims_train, lbls_train, stims_val, lbls_val

    @staticmethod
    def _load_single_split(stim_path, label_path, split_name: str, use_mmap: bool, logger=None):
        if logger is not None:
            logger.info("Loading data (%s)...", split_name)
        if use_mmap:
            stims = np.load(stim_path, allow_pickle=True, mmap_mode="r")
            if logger is not None:
                logger.info("  Loaded %s stimuli (mmap): %s", split_name, stims.shape)
        else:
            stims = np.load(stim_path, allow_pickle=True)
            if logger is not None:
                logger.info("  Loaded %s stimuli (ndarray): %s", split_name, stims.shape)
        lbls = pd.read_csv(label_path, sep="\t", index_col=0)
        if logger is not None:
            logger.info("  Loaded %s labels: %s", split_name, lbls.shape)
        return (stims, lbls)

    @staticmethod
    def save_metrics_summary(
        metric_summary: Dict[str, Any],
        metric_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        directory = os.path.dirname(metric_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metric_summary, f, indent=2)
        if logger is not None:
            logger.info("Wrote metrics summary to %s", metric_path)


# -----------------------------------------------------------------------------
# Per-epoch array summaries for metrics JSON (SummaryStatsHelper)
# -----------------------------------------------------------------------------


class SummaryStatsHelper:
    """Safe reductions over 1D numpy arrays and scalar helpers for experiment summaries."""

    @staticmethod
    def as_array_or_none(x) -> Optional[np.ndarray]:
        """Convert input to 1D numpy array or return None if not available."""
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def safe_max(x) -> Optional[float]:
        arr = SummaryStatsHelper.as_array_or_none(x)
        if arr is None:
            return None
        return float(np.nanmax(arr))

    @staticmethod
    def safe_min(x) -> Optional[float]:
        arr = SummaryStatsHelper.as_array_or_none(x)
        if arr is None:
            return None
        return float(np.nanmin(arr))

    @staticmethod
    def safe_last(x) -> Optional[float]:
        arr = SummaryStatsHelper.as_array_or_none(x)
        if arr is None:
            return None
        return float(arr[-1])

    @staticmethod
    def safe_best_epoch_1based(x) -> Optional[int]:
        """Return 1-based epoch index of minimum value in array (e.g., best val loss)."""
        arr = SummaryStatsHelper.as_array_or_none(x)
        if arr is None:
            return None
        idx = int(np.nanargmin(arr))
        return idx + 1

    @staticmethod
    def gap(train_acc: Optional[float], val_acc: Optional[float]) -> Optional[float]:
        """Generalization gap (train - val) in accuracy percentage."""
        if train_acc is None or val_acc is None:
            return None
        return float(train_acc - val_acc)

    @staticmethod
    def round2(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return float(round(x, 2))


def _enum_cuda_device_indices() -> Optional[List[int]]:
    """Return visible GPU index list, or None if nvidia-smi probe failed."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        indices = []
        for line in out.stdout.strip().splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                indices.append(int(s))
            except ValueError:
                continue
        return indices if indices else None
    except Exception:
        return None


def _gpu_has_train_rnn_process(gpu_index: int) -> bool:
    """
    Return True if the given GPU has a Python process running train_model (this script).
    Used to avoid placing a second training job on the same GPU.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return False
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        for line in lines:
            if not line.isdigit():
                continue
            pid = int(line)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode("utf-8", errors="ignore").replace("\x00", " ")
                if "train_model" in cmd:
                    return True
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        return False
    except Exception:
        return False


def _gpu_has_python_compute_process(gpu_index: int) -> bool:
    """
    True if any compute process on this GPU looks like Python (Linux ``/proc/<pid>/cmdline``).
    Used by analysis scripts to avoid a busy GPU when another card is free.
    """
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return False
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        for line in lines:
            if not line.isdigit():
                continue
            pid = int(line)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode("utf-8", errors="ignore").replace("\x00", " ")
                if "python" in cmd.lower():
                    return True
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        return False
    except Exception:
        return False


def pick_cuda_device_index() -> Optional[int]:
    """
    Choose a CUDA device index: prefer a GPU that has no Python train_model task.
    If multiple GPUs exist, returns the first index with no such process; otherwise 0.
    Call this before any torch.cuda init and set CUDA_VISIBLE_DEVICES to the returned index.
    Returns None if no NVIDIA GPU is visible (or nvidia-smi is unavailable).
    """
    indices = _enum_cuda_device_indices()
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    for idx in indices:
        if not _gpu_has_train_rnn_process(idx):
            return idx
    return indices[0]


def pick_cuda_device_index_prefer_no_python() -> Optional[int]:
    """
    Prefer a GPU with **no** Python process among its compute apps (see
    ``_gpu_has_python_compute_process``). If all visible GPUs run Python, fall back to
    the first index. Single-GPU systems always return that index.

    Call before ``torch`` allocates CUDA tensors; set ``CUDA_VISIBLE_DEVICES`` to the
    returned index (same pattern as ``train_model.py``).
    """
    indices = _enum_cuda_device_indices()
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    for idx in indices:
        if not _gpu_has_python_compute_process(idx):
            return idx
    return indices[0]


def create_datasets(
    stims_train,
    lbls_train,
    stims_val,
    lbls_val,
    use_sector_mode: bool,
    predict_all_chars: bool,
    max_chars: int = 10,
    dataset_class=None,
    splits: tuple = ("train", "valid"),
    stims_test=None,
    lbls_test=None,
    logger=None,
):
    """Create train/validation/test dataset(s) and return dataset objects and num_pos.

    Args:
        stims_train: Training stimuli numpy array (required when "train" in splits).
        lbls_train: Training labels DataFrame (required when "train" in splits).
        stims_val: Validation stimuli numpy array (required when "valid" in splits).
        lbls_val: Validation labels DataFrame (required when "valid" in splits).
        use_sector_mode: Whether to use sector mode (3x3 grid).
        predict_all_chars: Whether to predict all characters (fg+bg).
        max_chars: Maximum number of characters per frame (for predict_all_chars mode).
        dataset_class: Dataset class to use (MC_RNN_Dataset). If None, will raise error.
        splits: Which splits to create. ("train", "valid") | ("test",) | ("train", "valid", "test").
        stims_test: Test stimuli (required when "test" in splits).
        lbls_test: Test labels (required when "test" in splits).

    Returns:
        - splits=("train", "valid"): (train_ds, val_ds, num_pos)
        - splits=("test",): (test_ds, num_pos)
        - splits=("train", "valid", "test"): (train_ds, val_ds, test_ds, num_pos)
    """
    if dataset_class is None:
        raise ValueError("dataset_class must be provided (e.g., MC_RNN_Dataset)")

    if logger is not None:
        logger.info("Creating datasets...")
    if "test" in splits and (stims_test is None or lbls_test is None):
        raise ValueError("stims_test and lbls_test are required when 'test' is in splits")

    if predict_all_chars:
        num_pos = 0
        _kw = {"use_sector": False, "predict_all_chars": True, "max_chars": max_chars}
        if logger is not None:
            logger.info(
                "Using all-chars mode: predict all characters (fg+bg) per frame, max_chars=%s",
                max_chars,
            )
    elif use_sector_mode:
        num_pos = 9
        _kw = {"use_sector": True, "num_sectors": num_pos, "predict_all_chars": False}
        if logger is not None:
            logger.info("Using sector mode (3x3 grid, 9 sectors)")
    else:
        num_pos = 2
        _kw = {"use_sector": False, "predict_all_chars": False}
        if logger is not None:
            logger.info(
                "Using coordinate mode (directly predict x, y coordinates)"
            )

    def make_ds(stims, lbls):
        return dataset_class(stims, lbls, **_kw)

    # (stims, lbls) per split in fixed order
    _data_by_split = [
        ("train", stims_train, lbls_train),
        ("valid", stims_val, lbls_val),
        ("test", stims_test, lbls_test),
    ]
    out = [make_ds(stims, lbls) for name, stims, lbls in _data_by_split if name in splits]
    out.append(num_pos)
    return tuple(out)


def set_seed(seed: int):
    """Set random seed for reproducibility (Python, NumPy, PyTorch, CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed):
    """Worker initialization function for DataLoader workers.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        worker_id: Worker process ID
        seed: Base random seed
    """
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def get_model_classes(
    rnn_conv_class,
    lstm_conv_class,
    gru_conv_class,
    gawf_rnn_conv_class,
    feedforward_conv_class,
    dendritic_ann_conv_class,
    mamba_conv_class=None,
    diaglti_conv_class=None,
    s5_conv_class=None,
    gawf_multi_conv_class=None,
):
    """Return mapping from model type name to model class.
    
    Args:
        rnn_conv_class: RNNConv class
        lstm_conv_class: LSTMConv class
        gru_conv_class: GRUConv class
        gawf_rnn_conv_class: GaWFRNNConv class
        feedforward_conv_class: FeedForwardConv class
        dendritic_ann_conv_class: DendriticANNConv class
    
    Returns:
        Dictionary mapping model type names to model classes
    """
    model_classes = {
        "rnn": rnn_conv_class,
        "lstm": lstm_conv_class,
        "gru": gru_conv_class,
        "gawf": gawf_rnn_conv_class,
        "ffn": feedforward_conv_class,
        "dann": dendritic_ann_conv_class,
    }
    if gawf_multi_conv_class is not None:
        model_classes["gawf_multi"] = gawf_multi_conv_class
    if mamba_conv_class is not None:
        model_classes["mamba"] = mamba_conv_class
    if diaglti_conv_class is not None:
        model_classes["diaglti"] = diaglti_conv_class
    if s5_conv_class is not None:
        model_classes["ssm"] = s5_conv_class
        model_classes["s5"] = s5_conv_class
    return model_classes


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Train RNN models for sector classification")
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["rnn"],
        choices=[
            "rnn",
            "lstm",
            "gru",
            "gawf",
            "gawf_multi",
            "ffn",
            "dann",
            "mamba",
            "diaglti",
            "ssm",
            "s5",
        ],
        help='Model types to train (default: ["rnn"])',
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Hidden sizes to test (default: [256])",
    )
    parser.add_argument(
        "--mamba_d_models",
        type=int,
        nargs="+",
        default=[170],
        help="Mamba d_model values to test (default: [170]).",
    )
    parser.add_argument(
        "--diaglti_d_models",
        type=int,
        nargs="+",
        default=[256],
        help="DiagLTI sequence feature width d_model values to test (default: [256]).",
    )
    parser.add_argument(
        "--diaglti_state_sizes",
        type=int,
        nargs="+",
        default=[189],
        help="DiagLTI latent state sizes to test (default: [189]).",
    )
    parser.add_argument(
        "--s5_d_models",
        type=int,
        nargs="+",
        default=[256],
        help="S5 sequence feature width d_model values to test (default: [256]).",
    )
    parser.add_argument(
        "--s5_state_sizes",
        type=int,
        nargs="+",
        default=[128],
        help="S5 latent state sizes to test (default: [128], param-matched to GaWF h=256).",
    )
    parser.add_argument(
        "--s5_num_layers",
        type=int,
        default=1,
        help="Number of S5 layers (default: 1).",
    )
    parser.add_argument(
        "--s5_dropout",
        type=float,
        default=0.0,
        help="Dropout between stacked S5 layers (default: 0).",
    )
    parser.add_argument(
        "--s5_ssm_lr_scale",
        type=float,
        default=0.1,
        help="S5 SSM-core learning-rate multiplier (default: 0.1).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.001],
        help="Learning rates to search over (default: [0.001])",
    )
    parser.add_argument(
        "--wds",
        type=float,
        nargs="+",
        default=[0],
        help="Weight decay values to search over (default: [1e-4])",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "muon"],
        help="Optimizer (optim) to use: 'adam', 'adamw', or 'muon' (default: 'adam')",
    )
    parser.add_argument(
        "--cnn_dropout",
        dest="cnn_dropout",
        type=float,
        nargs="+",
        default=[0.0],
        help="Dropout p for CNN encoder (dropout2d); repeat for grid search (default: [0])",
    )
    parser.add_argument(
        "--rnn_dropout",
        type=float,
        default=0.5,
        help="Dropout p after RNN/GaWF/FFN middle (after ReLU); used in checkpoint suffix rdo (default: 0.5)",
    )
    parser.add_argument(
        "--feedback_dim",
        "--dz",
        type=int,
        default=None,
        help=(
            "GaWFRNN only: feedback context dimension dz. "
            "For model_type=gawf, omitting this keeps legacy feedback dim "
            "(num_classes + num_pos). For model_type=gawf_multi, omitting this "
            "or setting 0 disables feedback projectors; values > 0 enable projected dz."
        ),
    )
    parser.add_argument(
        "--gawf_layers",
        type=int,
        default=2,
        help=(
            "gawf_multi only: number of recurrent GaWF layers. "
            "Must be >= 2; default 2."
        ),
    )
    parser.add_argument(
        "--gawf_feedback_lr_scale",
        type=float,
        default=1.0,
        help=(
            "Single-layer gawf only: learning-rate scale for U/V parameter group. "
            "Default 1.0 preserves legacy behavior; gawf_multi uses "
            "--gawf_multi_feedback_lr_scale instead."
        ),
    )
    parser.add_argument(
        "--gawf_multi_feedback_lr_scale",
        type=float,
        default=0.1,
        help=(
            "gawf_multi only: learning-rate scale for U/V and feedback projector "
            "parameter groups, relative to the already scaled multi-layer base lr. "
            "Single-layer gawf is unchanged. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--gawf_diag",
        action="store_true",
        default=False,
        help=(
            "Enable opt-in GaWF diagnostics JSONL logging. Records gate logits, "
            "gate saturation, feedback norms, pre-clip gradient norms, and "
            "U/V/projector parameter norms. Default: disabled."
        ),
    )
    parser.add_argument(
        "--gawf_diag_every",
        type=int,
        default=1,
        help="Record one GaWF diagnostics step every N train batches when --gawf_diag is set.",
    )
    parser.add_argument(
        "--gawf_diag_gate_eps",
        type=float,
        default=0.01,
        help=(
            "Gate saturation threshold for --gawf_diag; a gate is saturated if "
            "gate <= eps or gate >= 1 - eps. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--gawf_diag_dir",
        type=str,
        default="",
        help=(
            "Directory for GaWF diagnostics JSONL files. Empty uses "
            "results/train_data/<result_suffix>/gawf_diagnostics/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_acceleration",
        action="store_true",
        default=True,
        help="Enable acceleration features for training (default: False)",
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=True,
        help="Use sector mode (3x3 grid, 9 sectors) instead of coordinate mode (default: False)",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help=(
            "Load stimuli with memory mapping (mmap_mode='r'). "
            "If not set, load as ndarray in memory so num_workers can be used (default: False)"
        ),
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters (fg+bg) per frame instead of only foreground character (default: False)",
    )
    parser.add_argument(
        "--nofb",
        action="store_true",
        default=False,
        help=(
            "GaWFRNN only: disable feedback. Behavior: "
            "(1) omit --nofb -> full feedback throughout. "
            "(2) use --nofb only -> no feedback throughout. "
            "(3) use --nofb and --fb_start_epoch N -> no feedback until epoch N, then feedback on"
        ),
    )
    parser.add_argument(
        "--fb_start_epoch",
        type=int,
        default=999999,
        help="GaWFRNN with --nofb: 0-based epoch at which to turn on feedback and unfreeze U,V.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Base directory containing stimulus_reg-* files. "
            "If not set, uses AIM3_STIMULI_PATH / FAW_RNN_DATA_PATH env, else <repo>/stimuli."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help=(
            "Root directory under which results/train_data is written. "
            "If not set, uses AIM3_RESULTS_PATH / FAW_RNN_RESULTS_PATH env, else <repo>/results."
        ),
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="40h-float32",
        help=(
            "Suffix appended to stimulus_reg-* file names. "
            "Example: 'cplx' -> 'stimulus_reg-train-cplx.npy'. "
            "Default: empty string (no suffix)."
        ),
    )
    parser.add_argument(
        "--eval_data_suffix",
        type=str,
        default="",
        help=(
            "Suffix for validation split only (stimulus_reg-validation-<suffix>). "
            "Empty means use the same suffix as --data_suffix."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help=(
            "Early stopping patience on validation character accuracy (fair eval). "
            "0 disables early stopping. Default: 15."
        ),
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="sector",
        help="Suffix to append to result file names for distinguishing different training runs (default: empty string)",
    )
    return parser


def summarize_experiment_metrics(
    results: Dict[str, Any],
    *,
    model_type: str,
    dataset_suffix: str,
    dataset_mode: str,
    num_epochs: int,
    hidden_size: int,
    lr: float,
    weight_decay: float,
    cnn_dropout: float,
    rnn_dropout: float,
    optimizer: str,
) -> Dict[str, Any]:
    """
    Build a compact metrics summary dict from a single training run.

    This mirrors the logic used in hparam-search style helpers:
    - best / final train & val accuracies
    - best / final losses
    - best epoch indices (1-based, by validation loss)
    """
    actual_epochs = int(results.get("actual_epochs", num_epochs))

    train_acc_char = results.get("train_acc_char")
    val_acc_char = results.get("val_acc_char")
    # 注意：这些可能是 numpy 数组，不能直接用 `or` 做布尔判断，否则会触发
    # "The truth value of an array with more than one element is ambiguous" 错误。
    train_acc_pos = results.get("train_acc_pos")
    if train_acc_pos is None:
        train_acc_pos = results.get("train_metric_pos")
    val_acc_pos = results.get("val_acc_pos")
    if val_acc_pos is None:
        val_acc_pos = results.get("val_metric_pos")

    train_loss_char = results.get("train_loss_char")
    val_loss_char = results.get("val_loss_char")
    train_loss_pos = results.get("train_loss_pos")
    val_loss_pos = results.get("val_loss_pos")

    best_train_acc_char = SummaryStatsHelper.safe_max(train_acc_char)
    best_train_acc_pos = SummaryStatsHelper.safe_max(train_acc_pos)
    best_train_loss_char = SummaryStatsHelper.safe_min(train_loss_char)
    best_train_loss_pos = SummaryStatsHelper.safe_min(train_loss_pos)
    final_train_acc_char = SummaryStatsHelper.safe_last(train_acc_char)
    final_train_acc_pos = SummaryStatsHelper.safe_last(train_acc_pos)

    best_val_acc_char = SummaryStatsHelper.safe_max(val_acc_char)
    best_val_acc_pos = SummaryStatsHelper.safe_max(val_acc_pos)
    best_val_loss_char = SummaryStatsHelper.safe_min(val_loss_char)
    best_val_loss_pos = SummaryStatsHelper.safe_min(val_loss_pos)
    final_val_acc_char = SummaryStatsHelper.safe_last(val_acc_char)
    final_val_acc_pos = SummaryStatsHelper.safe_last(val_acc_pos)

    best_epoch_char = SummaryStatsHelper.safe_best_epoch_1based(val_loss_char)
    best_epoch_pos = SummaryStatsHelper.safe_best_epoch_1based(val_loss_pos)

    final_train_loss_char = SummaryStatsHelper.safe_last(train_loss_char)
    final_val_loss_char = SummaryStatsHelper.safe_last(val_loss_char)
    final_train_loss_pos = SummaryStatsHelper.safe_last(train_loss_pos)
    final_val_loss_pos = SummaryStatsHelper.safe_last(val_loss_pos)

    gap_char = SummaryStatsHelper.gap(final_train_acc_char, final_val_acc_char)
    gap_pos = SummaryStatsHelper.gap(final_train_acc_pos, final_val_acc_pos)
    overfit_flag = bool(
        (gap_char is not None and gap_char > 10.0)
        or (gap_pos is not None and gap_pos > 10.0)
    )

    glob_train_acc_char = results.get("glob_train_acc_char")
    glob_val_acc_char = results.get("glob_val_acc_char")
    glob_train_acc_pos = results.get("glob_train_acc_pos")
    glob_val_acc_pos = results.get("glob_val_acc_pos")
    final_glob_train_char = SummaryStatsHelper.safe_last(glob_train_acc_char)
    final_glob_val_char = SummaryStatsHelper.safe_last(glob_val_acc_char)
    final_glob_train_pos = SummaryStatsHelper.safe_last(glob_train_acc_pos)
    final_glob_val_pos = SummaryStatsHelper.safe_last(glob_val_acc_pos)
    final_fg_pre5_train_char = SummaryStatsHelper.safe_last(
        results.get("fg_switch_pre5_train_acc_char")
    )
    final_fg_pre5_val_char = SummaryStatsHelper.safe_last(results.get("fg_switch_pre5_val_acc_char"))
    final_fg_post5_train_char = SummaryStatsHelper.safe_last(
        results.get("fg_switch_post5_train_acc_char")
    )
    final_fg_post5_val_char = SummaryStatsHelper.safe_last(
        results.get("fg_switch_post5_val_acc_char")
    )
    final_fg_pre5_train_pos = SummaryStatsHelper.safe_last(
        results.get("fg_switch_pre5_train_acc_pos")
    )
    final_fg_pre5_val_pos = SummaryStatsHelper.safe_last(results.get("fg_switch_pre5_val_acc_pos"))
    final_fg_post5_train_pos = SummaryStatsHelper.safe_last(
        results.get("fg_switch_post5_train_acc_pos")
    )
    final_fg_post5_val_pos = SummaryStatsHelper.safe_last(
        results.get("fg_switch_post5_val_acc_pos")
    )

    metric_summary: Dict[str, Any] = {
        "model_type": model_type,
        "dataset_suffix": dataset_suffix,
        "dataset_mode": dataset_mode,
        "num_epochs": num_epochs,
        "hidden_size": hidden_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "cnn_dropout": cnn_dropout,
        "rnn_dropout": rnn_dropout,
        "optimizer": optimizer,
        "actual_epochs": actual_epochs,
        "best_train_acc_char": best_train_acc_char,
        "best_train_acc_pos": best_train_acc_pos,
        "best_train_loss_char": best_train_loss_char,
        "best_train_loss_pos": best_train_loss_pos,
        "best_val_acc_char": best_val_acc_char,
        "best_val_acc_pos": best_val_acc_pos,
        "best_val_loss_char": best_val_loss_char,
        "best_val_loss_pos": best_val_loss_pos,
        "best_epoch_char": best_epoch_char,
        "best_epoch_pos": best_epoch_pos,
        "gap_char": SummaryStatsHelper.round2(gap_char),
        "gap_pos": SummaryStatsHelper.round2(gap_pos),
        "overfit_flag": overfit_flag,
        "final_train_loss_char": final_train_loss_char,
        "final_val_loss_char": final_val_loss_char,
        "final_train_loss_pos": final_train_loss_pos,
        "final_val_loss_pos": final_val_loss_pos,
        "final_train_acc_char": final_train_acc_char,
        "final_train_acc_pos": final_train_acc_pos,
        "final_val_acc_char": final_val_acc_char,
        "final_val_acc_pos": final_val_acc_pos,
        "final_glob_train_acc_char": final_glob_train_char,
        "final_glob_val_acc_char": final_glob_val_char,
        "final_glob_train_acc_pos": final_glob_train_pos,
        "final_glob_val_acc_pos": final_glob_val_pos,
        "final_fg_switch_pre5_train_acc_char": final_fg_pre5_train_char,
        "final_fg_switch_pre5_val_acc_char": final_fg_pre5_val_char,
        "final_fg_switch_post5_train_acc_char": final_fg_post5_train_char,
        "final_fg_switch_post5_val_acc_char": final_fg_post5_val_char,
        "final_fg_switch_pre5_train_acc_pos": final_fg_pre5_train_pos,
        "final_fg_switch_pre5_val_acc_pos": final_fg_pre5_val_pos,
        "final_fg_switch_post5_train_acc_pos": final_fg_post5_train_pos,
        "final_fg_switch_post5_val_acc_pos": final_fg_post5_val_pos,
    }

    for k in (
        "eval_dataset_suffix",
        "train_acc_at_best_val",
        "val_acc_at_best",
        "overfit_gap",
        "best_epoch_val_acc_1based",
        "train_acc_sector_at_best_val_sector",
        "val_acc_sector_at_best",
        "overfit_gap_sector",
        "best_epoch_val_acc_sector_1based",
        "early_stop_epoch_1based",
        "stopped_by_patience",
    ):
        if k in results and results[k] is not None:
            v = results[k]
            if k in ("overfit_gap", "overfit_gap_sector") and v is not None:
                metric_summary[k] = SummaryStatsHelper.round2(v)
            else:
                metric_summary[k] = v

    return metric_summary
