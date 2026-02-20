"""
Helper functions for training script.
Contains utility functions for data loading, path management, result saving,
random seed setting, GPU memory management, model class mapping, and logging.
"""
import logging
import os
from typing import Optional
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Logging: setup_logger, log_experiment_config, log_write (tqdm-safe)
# -----------------------------------------------------------------------------

BANNER_LEN = 60


class _TqdmStreamHandler(logging.StreamHandler):
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
    # Avoid duplicate handlers: if already configured, return as-is
    if logger.handlers:
        return logger
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    console = _TqdmStreamHandler()
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


def log_experiment_config(
    logger,
    total_experiments,
    model_types,
    hidden_sizes,
    lrs,
    weight_decays,
    dropout_rates,
    cnn_feature_sizes,
):
    """
    Log training loop banner and config summary (same content as original print block).
    Banner line length is 60. Uses logger.info only (no print).
    """
    sep = "=" * BANNER_LEN
    logger.info(sep)
    logger.info("Starting training loop: %s experiments", total_experiments)
    logger.info(
        "Models: %s | Hidden sizes: %s",
        model_types,
        hidden_sizes,
    )
    logger.info(
        "Learning rates: %s | Weight decays: %s | Dropout rates: %s",
        lrs,
        weight_decays,
        dropout_rates,
    )
    logger.info("CNN feature sizes: %s", cnn_feature_sizes)
    logger.info(sep)


def log_experiment_start(
    logger,
    experiment_num,
    total_experiments,
    model_type,
    hidden_size,
    lr,
    weight_decay,
    dropout_rate,
):
    """Log per-experiment banner (one block)."""
    sep = "=" * BANNER_LEN
    logger.info(
        "Experiment %s/%s: %s | hidden_size=%s | lr=%s | weight_decay=%s | dropout=%s",
        experiment_num,
        total_experiments,
        model_type.upper(),
        hidden_size,
        lr,
        weight_decay,
        dropout_rate,
    )
    logger.info(sep)


def log_write(logger, msg, level=logging.INFO):
    """
    Log message in a tqdm-safe way. When the console handler is TqdmStreamHandler,
    logger already writes via tqdm.write; use this for explicit log calls from
    inside tqdm loops. Safe when tqdm is not used (handler falls back to stream.write).
    """
    logger.log(level, msg)


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
    """
    Log dataset and batch information (content lives here so main training code stays minimal).
    """
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
        logger.info("  DataLoader prefetch_factor: %s", accel_config.dataloader_prefetch_factor)


def pick_device(requested: str) -> str:
    """
    Resolve device string: auto -> cuda > mps > cpu; or explicit cuda/mps/cpu.
    Raises if user requests cuda/mps but it is not available.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("--device mps requested but MPS is not available (macOS Apple Silicon required)")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    raise ValueError(f"Invalid --device: {requested}. Use one of: auto, cuda, mps, cpu")


def save_results(results, filepath):
    """
    Save training results to local file
    
    Args:
        results: Training results dictionary
        filepath: Save path (e.g., 'results_rnn' or 'results/rnn_sector')
                  If directory doesn't exist, it will be created
    """
    # Extract directory and filename
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create save dictionary (does not include model, because model is too large)
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # Save model state dict (can be saved separately if needed)
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # Check if model has fcpos (position prediction layer)
        # In predict_all_chars mode, fcpos is None
        if hasattr(results["model"], 'fcpos') and results["model"].fcpos is not None:
            saved_num_pos = results["model"].fcpos.out_features
            print(f"Model has position prediction layer (num_pos={saved_num_pos})")
        elif hasattr(results["model"], 'predict_all_chars') and results["model"].predict_all_chars:
            print("Model is in predict_all_chars mode (no position prediction)")
        else:
            print("Model does not have position prediction layer")
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # Save other results
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


def get_base_path(override: Optional[str] = None) -> str:
    """Get the base directory for stimulus/label files. Adapts to current environment.

    Resolution order:
    1. override: if provided (e.g. from --data_dir), use it after expanding user and resolving to absolute path.
    2. Environment: AIM3_STIMULI_PATH or FAW_RNN_DATA_PATH (first set wins).
    3. Project-relative: <repo_root>/stimuli, where repo_root is the directory containing the 'utils' package.
    """
    if override and override.strip():
        base_path = os.path.abspath(os.path.expanduser(override.strip()))
        print(f"Using base path (override): {base_path}")
        return base_path
    env_path = os.environ.get("AIM3_STIMULI_PATH") or os.environ.get("FAW_RNN_DATA_PATH")
    if env_path and env_path.strip():
        base_path = os.path.abspath(os.path.expanduser(env_path.strip()))
        print(f"Using base path (env): {base_path}")
        return base_path
    # Project-relative: this file is in <repo>/utils/, so repo root is parent of utils
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_this_dir)
    base_path = os.path.join(_repo_root, "stimuli")
    print(f"Using base path (project-relative): {base_path}")
    return base_path


def prepare_data_paths(base_path: str, data_suffix: str = ""):
    """Construct and validate stimulus / label file paths.

    Args:
        base_path: Base directory containing stimulus/label files.
        data_suffix: Optional suffix appended to base filenames.
            - Default ""  -> use filenames without any suffix, e.g. 'stimulus_reg-train.npy'
            - Non-empty (e.g. "cplx", "40h") -> a leading hyphen is added automatically,
              resulting in filenames like 'stimulus_reg-train-cplx.npy'.
    
    Returns:
        Tuple of (stim_train_path, label_train_path, stim_val_path, label_val_path)
    """
    # Normalize suffix: ensure a single leading hyphen when non-empty
    if data_suffix:
        if not data_suffix.startswith("-"):
            suffix = f"-{data_suffix}"
        else:
            suffix = data_suffix
    else:
        suffix = ""

    stim_train_path = os.path.join(base_path, f"stimulus_reg-train{suffix}.npy")
    label_train_path = os.path.join(base_path, f"stimulus_reg-train{suffix}.tsv")
    stim_val_path = os.path.join(base_path, f"stimulus_reg-validation{suffix}.npy")
    label_val_path = os.path.join(base_path, f"stimulus_reg-validation{suffix}.tsv")

    print("Checking data paths...")
    print(f"Base path: {base_path}")
    for path_name, path in [
        ("train stim", stim_train_path),
        ("train label", label_train_path),
        ("val stim", stim_val_path),
        ("val label", label_val_path),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {path_name} path does not exist: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        else:
            print(f"  ✓ {path_name}: {path}")

    return stim_train_path, label_train_path, stim_val_path, label_val_path


def load_raw_data(stim_train_path: str, label_train_path: str,
                  stim_val_path: str, label_val_path: str,
                  use_mmap: bool = False):
    """Load raw numpy and label data from disk.

    Args:
        stim_train_path: Path to training stimuli numpy file
        label_train_path: Path to training labels TSV file
        stim_val_path: Path to validation stimuli numpy file
        label_val_path: Path to validation labels TSV file
        use_mmap: If True, load stimuli with mmap_mode='r' (memory-mapped, use num_workers=0).
                  If False, load as ndarray in memory so DataLoader can use num_workers > 0.
    
    Returns:
        Tuple of (stims_train, lbls_train, stims_val, lbls_val)
    """
    print("\nLoading data...")
    if use_mmap:
        stims_train = np.load(stim_train_path, allow_pickle=True, mmap_mode="r")
        stims_val = np.load(stim_val_path, allow_pickle=True, mmap_mode="r")
        print(f"  ✓ Loaded training stimuli (mmap): {stims_train.shape}")
        print(f"  ✓ Loaded validation stimuli (mmap): {stims_val.shape}")
    else:
        stims_train = np.load(stim_train_path, allow_pickle=True)
        stims_val = np.load(stim_val_path, allow_pickle=True)
        print(f"  ✓ Loaded training stimuli (ndarray): {stims_train.shape}")
        print(f"  ✓ Loaded validation stimuli (ndarray): {stims_val.shape}")

    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded training labels: {lbls_train.shape}")
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded validation labels: {lbls_val.shape}")

    return stims_train, lbls_train, stims_val, lbls_val


def create_datasets(stims_train, lbls_train, stims_val, lbls_val,
                    use_sector_mode: bool, predict_all_chars: bool,
                    max_chars: int = 10, dataset_class=None):
    """Create training / validation datasets and return dataset objects and num_pos.
    
    Args:
        stims_train: Training stimuli numpy array
        lbls_train: Training labels DataFrame
        stims_val: Validation stimuli numpy array
        lbls_val: Validation labels DataFrame
        use_sector_mode: Whether to use sector mode (3x3 grid)
        predict_all_chars: Whether to predict all characters (fg+bg)
        max_chars: Maximum number of characters per frame (for predict_all_chars mode)
        dataset_class: Dataset class to use (MC_RNN_Dataset). If None, will raise error.
    
    Returns:
        Tuple of (train_ds, val_ds, num_pos)
    """
    if dataset_class is None:
        raise ValueError("dataset_class must be provided (e.g., MC_RNN_Dataset)")
    
    print("Creating datasets...")

    if predict_all_chars:
        train_ds = dataset_class(
            stims_train, lbls_train, use_sector=False,
            predict_all_chars=True, max_chars=max_chars,
        )
        val_ds = dataset_class(
            stims_val, lbls_val, use_sector=False,
            predict_all_chars=True, max_chars=max_chars,
        )
        num_pos = 0
        print(f"Using all-chars mode: predict all characters (fg+bg) per frame, max_chars={max_chars}")
    elif use_sector_mode:
        num_pos = 9
        train_ds = dataset_class(
            stims_train, lbls_train, use_sector=True, num_sectors=num_pos,
            predict_all_chars=False,
        )
        val_ds = dataset_class(
            stims_val, lbls_val, use_sector=True, num_sectors=num_pos,
            predict_all_chars=False,
        )
        print("Using sector mode (3x3 grid, 9 sectors)")
    else:
        num_pos = 2
        train_ds = dataset_class(
            stims_train, lbls_train, use_sector=False, predict_all_chars=False,
        )
        val_ds = dataset_class(
            stims_val, lbls_val, use_sector=False, predict_all_chars=False,
        )
        print("Using coordinate mode (directly predict x, y coordinates)")

    return train_ds, val_ds, num_pos


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


def get_gpu_memory_usage():
    """Get current GPU memory usage (only used in acceleration mode)"""
    if not torch.cuda.is_available():
        return 0.0
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    return (reserved / total) * 100.0 if total > 0 else 0.0


def find_optimal_batch_size(model, train_data, device='cuda', start_batch_size=32, max_batch_size=256,
                            enable_grad_accum=False, grad_accum_steps=4):
    # GPU memory search only for CUDA; MPS/CPU use default batch size
    if device != 'cuda':
        num_workers = 0 if device == 'cpu' else min(4, os.cpu_count() or 1)
        return start_batch_size, num_workers

    model.eval()
    optimal_batch_size = start_batch_size
    num_workers_adjusted = 0

    test_sizes = [start_batch_size, 64] if not enable_grad_accum else [start_batch_size, 32, 16, 8]

    for batch_size in test_sizes:
        if batch_size > max_batch_size:
            break

        try:
            torch.cuda.empty_cache()
            test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
            test_batch = next(iter(test_loader))

            # Support (inputs, labels) or (inputs, labels, idx)
            if isinstance(test_batch, (list, tuple)) and len(test_batch) == 3:
                inputs, labels, _ = test_batch
            else:
                inputs, labels = test_batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            # IMPORTANT: clear GaWFRNN state between different batch sizes
            if hasattr(model, "prev_feedback"):
                model.prev_feedback = None

            with torch.no_grad():
                # If model supports reset_feedback, use it; otherwise fallback
                try:
                    _ = model(inputs, use_feedback=True, reset_feedback=True)
                except TypeError:
                    _ = model(inputs)

            memory_usage = get_gpu_memory_usage()

            if memory_usage < 70.0:
                optimal_batch_size = batch_size
                num_workers_adjusted = 2 if enable_grad_accum else 4
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, usable (num_workers will be {num_workers_adjusted})")
            else:
                print(f"Testing batch_size={batch_size}: GPU memory usage {memory_usage:.1f}%, exceeds limit")
                break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} caused OOM, using batch_size={optimal_batch_size}")
                torch.cuda.empty_cache()
                break
            else:
                raise e
        finally:
            if 'test_loader' in locals():
                del test_loader
                import gc
                gc.collect()

    return optimal_batch_size, num_workers_adjusted


def worker_init_fn(worker_id, seed):
    """Worker initialization function for DataLoader workers.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        worker_id: Worker process ID
        seed: Base random seed
    """
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def get_model_classes(rnn_conv_class, lstm_conv_class, gru_conv_class, 
                      gawf_rnn_conv_class, feedforward_conv_class, dendritic_ann_conv_class):
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
    return {
        "rnn": rnn_conv_class,
        "lstm": lstm_conv_class,
        "gru": gru_conv_class,
        "gawf": gawf_rnn_conv_class,
        "ffn": feedforward_conv_class,
        "dann": dendritic_ann_conv_class,
    }
