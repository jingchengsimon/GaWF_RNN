"""
Standalone RNN Sector training script
Used to train RNN models and save results

Device support: CUDA (Linux/NVIDIA) as default, CPU as fallback.

Smoke-test (run 1 epoch, minimal config):
  CUDA:  python train_model.py --num_epochs 1
"""

import os
import sys
import time
from itertools import product
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.clutter_train_helpers import (
    LoggingHelper,
    PathHelper,
    create_datasets,
    set_seed,
    get_model_classes,
    build_arg_parser,
    pick_cuda_device_index,
    summarize_experiment_metrics,
)

from utils.clutter_train_sector import compute_fg_transition_masks

from utils.clutter_train_engine import (
    setup_training_components,
    cleanup_dataloaders,
    stop_requested,
    begin_epoch,
    train_batch,
    summarize_online_train,
    eval_train_subset,
    eval_valid,
)

from utils.clutter_task_models import (
    GaWFRNNConv,
    GRUConv,
    LSTMConv,
    MambaConv,
    MultiLayerGaWFRNNConv,
    RNNConv,
    S5Conv,
)


torch.set_num_threads(4)


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        frame_num=32,
        chan_num=2,
        use_sector=False,
        num_sectors=9,
        max_chars=15,
        predict_all_chars=False,
    ):
        """
        Args:
            data (np.ndarray): Array of shape (num_frames_total, height, width); supports float32 for faster loading.
            labels (DataFrame): columns include ``fg_char_id``, ``fg_char_x``, ``fg_char_y``, ``bg_char_ids``;
                optional ``fg_switch`` (0/1 per frame) for transition-window eval in sector mode.
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
            max_chars (int): Maximum number of characters per frame (for padding)
            predict_all_chars (bool): If True, predict all characters (fg+bg), else only fg
        """
        self.data = data
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        self.data_dtype = getattr(data, "dtype", np.uint8)

        if predict_all_chars:
            self.labels_df = labels
            self.fg_char_ids = labels["fg_char_id"].values
            raw_bg = labels["bg_char_ids"].values
            self.bg_char_ids_parsed = []
            for i in range(len(raw_bg)):
                s = str(raw_bg[i]) if raw_bg[i] is not None else ""
                if s and s != "nan" and s.strip():
                    try:
                        self.bg_char_ids_parsed.append([int(x) for x in s.split(",") if x.strip()])
                    except (ValueError, TypeError):
                        self.bg_char_ids_parsed.append([])
                else:
                    self.bg_char_ids_parsed.append([])
        else:
            self.labels = labels[["fg_char_id", "fg_char_x", "fg_char_y"]].values
            if use_sector:
                # Sector precomputation (one-time): full sequence sector labels so __getitem__ only slices.
                N = self.data.shape[0]
                height = self.data.shape[-2]
                width = self.data.shape[-1]
                grid_size = int(np.sqrt(self.num_sectors))
                if grid_size * grid_size != self.num_sectors:
                    raise ValueError(
                        f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid"
                    )
                x = self.labels[:, 1].astype(np.float64)
                y = self.labels[:, 2].astype(np.float64)
                col = np.clip((x / max(width - 1, 1)) * grid_size, 0, grid_size - 1).astype(
                    np.int64
                )
                row = np.clip((y / max(height - 1, 1)) * grid_size, 0, grid_size - 1).astype(
                    np.int64
                )
                sector = row * grid_size + col
                char_id = self.labels[:, 0].astype(np.int64)
                self.labels_sector = np.stack([char_id, sector], axis=1).astype(
                    np.int64, copy=False
                )
            else:
                # Coordinate mode: one-time float32 conversion for labels.
                self.labels_coord = self.labels.astype(np.float32, copy=False)

        # fg_switch + transition masks (sector, single-char only; used for fair eval metrics).
        if "fg_switch" in labels.columns:
            self.fg_switch = labels["fg_switch"].values.astype(np.int32, copy=False)
        else:
            self.fg_switch = np.zeros(self.data.shape[0], dtype=np.int32)
        if "bg_switch" in labels.columns:
            self.bg_switch = labels["bg_switch"].values.astype(np.int32, copy=False)
        else:
            self.bg_switch = np.zeros(self.data.shape[0], dtype=np.int32)
        if not predict_all_chars and use_sector:
            self.pre5_mask_global, self.post5_mask_global = compute_fg_transition_masks(
                self.fg_switch
            )
        else:
            self.pre5_mask_global = None
            self.post5_mask_global = None

    def __len__(self):
        return (self.data.shape[0] - self.chan_num) // self.frame_num

    def __getitem__(self, idx):
        start_idx = (idx * self.frame_num) + self.chan_num
        end_idx = start_idx + self.frame_num
        T, C = self.frame_num, self.chan_num

        # Frames: stride view (zero-copy when data is contiguous) instead of list + np.stack.
        block_start = start_idx - (C - 1)
        block_end = end_idx
        block = self.data[block_start:block_end]
        use_strided = (
            block.shape[0] == T + C - 1
            and len(block.shape) == 3
            and block.strides[0] >= 0
            and block.strides[1] >= 0
            and block.strides[2] >= 0
        )
        if use_strided:
            s0, s1, s2 = block.strides[0], block.strides[1], block.strides[2]
            H, W = block.shape[1], block.shape[2]
            stacked_frames = np.lib.stride_tricks.as_strided(
                block, shape=(T, C, H, W), strides=(s0, s0, s1, s2)
            )
        else:
            frames = [self.data[start_idx + i : end_idx + i] for i in range(-(C - 1), 1)]
            stacked_frames = np.stack(frames, axis=1)

        if stacked_frames.dtype != np.float32:
            stacked_frames = stacked_frames.astype(np.float32, copy=False)

        # When using np.memmap (use_mmap=True), stride views may be backed by
        # non-writable arrays. Torch's default collate uses torch.as_tensor,
        # which warns on non-writable NumPy arrays. Make a writable copy only
        # in that case to avoid the warning while keeping copies minimal.
        if not stacked_frames.flags.writeable:
            stacked_frames = np.array(stacked_frames, copy=True)

        if self.predict_all_chars:
            all_chars_per_frame = []
            for frame_idx in range(start_idx, end_idx):
                fg_char_id = int(self.fg_char_ids[frame_idx])
                bg_char_ids = (
                    self.bg_char_ids_parsed[frame_idx]
                    if frame_idx < len(self.bg_char_ids_parsed)
                    else []
                )
                all_chars = [fg_char_id] + bg_char_ids
                padded_chars = all_chars[: self.max_chars] + [-1] * max(
                    0, self.max_chars - len(all_chars)
                )
                all_chars_per_frame.append(padded_chars)
            labels = np.array(all_chars_per_frame, dtype=np.int64)
        else:
            if self.use_sector:
                labels = self.labels_sector[start_idx:end_idx]
            else:
                labels = self.labels_coord[start_idx:end_idx].copy()

        if self.pre5_mask_global is not None:
            pre5_win = self.pre5_mask_global[start_idx:end_idx]
            post5_win = self.post5_mask_global[start_idx:end_idx]
            return stacked_frames, labels, pre5_win, post5_win

        return stacked_frames, labels


# ==================== Dendritic ANN (dANN) with Global RFs ====================
# Aligned with opt.py get_model(): model type 1 (dend_ann_global_rfs).
# - Dend: linear only (no nonlinearity on dendrite outputs).
# - Soma: fixed aggregation (non-learnable), i.e. sum over dendrites per soma; then LeakyReLU on soma (match opt).
# ==================== Training Function ====================
def network_train(
    mdl,
    train_data,
    val_data,
    num_epochs=50,
    loss_weights=None,
    lr=0.001,
    use_acceleration=False,
    weight_decay=None,
    rnn_diag_lambda=1e-4,
    use_mmap=False,
    use_tqdm=True,
    nofb=False,
    fb_start_epoch=999999,
    seed=42,
    logger=None,
    optim: str = "adam",
    patience: int = 0,
    run_label: str = "",
    gawf_feedback_lr_scale: float = 1.0,
    gawf_diag_enabled: bool = False,
    gawf_diag_path: str | None = None,
    gawf_diag_every: int = 1,
    gawf_diag_gate_eps: float = 0.01,
    s5_ssm_lr_scale: float = 0.1,
):
    """
    Train model, supports sector mode and coordinate mode.
    Acceleration is config-driven; training loop is single-path (no AMP branches).
    """
    # === High-level skeleton: move model to device, build loss & optimizer, run training loop ===
    # 1) Move model to device and build all low-level components in utils.
    components = setup_training_components(
        mdl=mdl,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        loss_weights=loss_weights,
        lr=lr,
        use_acceleration=use_acceleration,
        weight_decay=weight_decay,
        rnn_diag_lambda=rnn_diag_lambda,
        use_mmap=use_mmap,
        use_tqdm=use_tqdm,
        seed=seed,
        logger=logger,
        optim_name=optim,
        run_label=run_label,
        gawf_feedback_lr_scale=gawf_feedback_lr_scale,
        gawf_diag_enabled=gawf_diag_enabled,
        gawf_diag_path=gawf_diag_path,
        gawf_diag_every=gawf_diag_every,
        gawf_diag_gate_eps=gawf_diag_gate_eps,
        s5_ssm_lr_scale=s5_ssm_lr_scale,
    )

    val_every = 1  # run full validation only every N epochs
    epoch = -1
    best_val_acc = float("-inf")
    best_epoch_idx = 0
    best_state = None
    epochs_without_improvement = 0
    stopped_by_patience = False
    try:
        for epoch in range(num_epochs):
            epoch_wall_start = time.perf_counter()
            mdl.train()

            epoch_ctx = begin_epoch(
                mdl=mdl,
                components=components,
                epoch=epoch,
                num_epochs=num_epochs,
                nofb=nofb,
                fb_start_epoch=fb_start_epoch,
            )
            train_pbar = epoch_ctx["train_pbar"]

            for batch_idx, batch in train_pbar:
                if stop_requested(components["stop_flag"]):
                    raise KeyboardInterrupt

                train_batch(epoch_ctx, batch_idx, batch)

            train_online_summary = summarize_online_train(epoch_ctx)

            # Fair eval: same protocol for train subset and valid (full-dataloader eval).
            train_eval_str = eval_train_subset(mdl, components, epoch_ctx)
            val_str = eval_valid(mdl, components, epoch_ctx, val_every=val_every)
            if logger is not None:
                msg = train_eval_str + val_str
                if train_online_summary:
                    msg += f" | Train(online): {train_online_summary}"
                logger.info(msg)
                wall = time.perf_counter() - epoch_wall_start
                lbl = (components.get("run_label") or "").strip()
                prefix = f"[{lbl}] " if lbl else ""
                logger.info(
                    "%sEpoch %s/%s wall_time_sec=%.2f (train + fair train-eval + fair val)",
                    prefix,
                    epoch + 1,
                    num_epochs,
                    wall,
                )

            val_acc = float(components["val_acc_char"][epoch])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch_idx = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in mdl.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                stopped_by_patience = True
                if logger is not None:
                    logger.info(
                        "Early stopping at epoch %s (1-based), best val epoch: %s, "
                        "best val acc (char): %.6f",
                        epoch + 1,
                        best_epoch_idx + 1,
                        best_val_acc,
                    )
                break

    except (KeyboardInterrupt, SystemExit):
        # Handle interruption gracefully
        if logger is not None:
            logger.info("Training interrupted, cleaning up resources...")
    finally:
        # Explicit resource cleanup for DataLoader so worker processes exit.
        # Use Ctrl+C (SIGINT) to stop training; kill -9 skips this and can leave workers running.
        cleanup_dataloaders(components)

    if components["device"] == "cuda" or (
        isinstance(components["device"], str) and components["device"].startswith("cuda:")
    ):
        torch.cuda.empty_cache()

    # Epochs actually completed (epoch is last finished epoch index, or -1 if none).
    actual_epochs = epoch + 1 if epoch >= 0 else 0

    if best_state is not None and actual_epochs > 0:
        load_dev = next(mdl.parameters()).device
        mdl.load_state_dict(
            {k: v.to(load_dev, non_blocking=True) for k, v in best_state.items()},
            strict=True,
        )

    if actual_epochs <= 0:
        train_acc_at_best = float("nan")
        val_acc_at_best = float("nan")
        overfit_gap = float("nan")
        train_acc_sector_at_best = float("nan")
        val_acc_sector_at_best = float("nan")
        overfit_gap_sector = float("nan")
        best_epoch_sector_idx = -1
    else:
        train_acc_at_best = float(components["train_acc_char"][best_epoch_idx])
        val_acc_at_best = float(components["val_acc_char"][best_epoch_idx])
        overfit_gap = train_acc_at_best - val_acc_at_best
        train_pos_arr = components["train_metric_pos"][:actual_epochs]
        val_pos_arr = components["val_metric_pos"][:actual_epochs]
        if len(val_pos_arr) and not np.all(np.isnan(val_pos_arr)):
            best_epoch_sector_idx = int(np.nanargmax(val_pos_arr))
            train_acc_sector_at_best = float(train_pos_arr[best_epoch_sector_idx])
            val_acc_sector_at_best = float(val_pos_arr[best_epoch_sector_idx])
            overfit_gap_sector = train_acc_sector_at_best - val_acc_sector_at_best
        else:
            train_acc_sector_at_best = float("nan")
            val_acc_sector_at_best = float("nan")
            overfit_gap_sector = float("nan")
            best_epoch_sector_idx = -1
    if logger is not None and actual_epochs > 0:
        logger.info(
            "Training summary: early_stop_epoch=%s, best_val_epoch=%s, "
            "train_acc@best=%.6f, val_acc@best=%.6f, overfit_gap=%.6f, "
            "stopped_by_patience=%s",
            actual_epochs,
            best_epoch_idx + 1,
            train_acc_at_best,
            val_acc_at_best,
            overfit_gap,
            stopped_by_patience,
        )

    # Slice metrics from components for the actually trained epochs
    train_acc_char = components["train_acc_char"]
    val_acc_char = components["val_acc_char"]
    metrics_mode = components["metrics_mode"]
    train_metric_pos = components["train_metric_pos"]
    val_metric_pos = components["val_metric_pos"]
    train_loss_pos = components["train_loss_pos"]
    val_loss_pos = components["val_loss_pos"]
    train_loss_char = components["train_loss_char"]
    val_loss_char = components["val_loss_char"]

    base = {
        "train_acc_char": train_acc_char[:actual_epochs],
        "val_acc_char": val_acc_char[:actual_epochs],
        "model": mdl.to("cpu"),
        "actual_epochs": actual_epochs,
        "train_acc_at_best_val": train_acc_at_best,
        "val_acc_at_best": val_acc_at_best,
        "overfit_gap": overfit_gap,
        "best_epoch_val_acc_1based": best_epoch_idx + 1 if actual_epochs > 0 else 0,
        "train_acc_sector_at_best_val_sector": train_acc_sector_at_best,
        "val_acc_sector_at_best": val_acc_sector_at_best,
        "overfit_gap_sector": overfit_gap_sector,
        "best_epoch_val_acc_sector_1based": (
            best_epoch_sector_idx + 1 if best_epoch_sector_idx >= 0 else 0
        ),
        "early_stop_epoch_1based": actual_epochs,
        "stopped_by_patience": stopped_by_patience,
    }

    out = metrics_mode.add_pos_to_result_dict(
        base,
        train_metric_pos,
        val_metric_pos,
        actual_epochs,
        train_loss_pos=train_loss_pos,
        val_loss_pos=val_loss_pos,
        train_loss_char=train_loss_char,
        val_loss_char=val_loss_char,
    )
    if components.get("glob_train_acc_char") is not None:
        out["glob_train_acc_char"] = components["glob_train_acc_char"][:actual_epochs]
        out["glob_val_acc_char"] = components["glob_val_acc_char"][:actual_epochs]
        out["glob_train_acc_pos"] = components["glob_train_acc_pos"][:actual_epochs]
        out["glob_val_acc_pos"] = components["glob_val_acc_pos"][:actual_epochs]
        out["fg_switch_pre5_train_acc_char"] = components["fg_switch_pre5_train_acc_char"][
            :actual_epochs
        ]
        out["fg_switch_pre5_val_acc_char"] = components["fg_switch_pre5_val_acc_char"][
            :actual_epochs
        ]
        out["fg_switch_pre5_train_acc_pos"] = components["fg_switch_pre5_train_acc_pos"][
            :actual_epochs
        ]
        out["fg_switch_pre5_val_acc_pos"] = components["fg_switch_pre5_val_acc_pos"][:actual_epochs]
        out["fg_switch_post5_train_acc_char"] = components["fg_switch_post5_train_acc_char"][
            :actual_epochs
        ]
        out["fg_switch_post5_val_acc_char"] = components["fg_switch_post5_val_acc_char"][
            :actual_epochs
        ]
        out["fg_switch_post5_train_acc_pos"] = components["fg_switch_post5_train_acc_pos"][
            :actual_epochs
        ]
        out["fg_switch_post5_val_acc_pos"] = components["fg_switch_post5_val_acc_pos"][
            :actual_epochs
        ]
    gawf_diagnostics = components.get("gawf_diagnostics")
    if gawf_diagnostics is not None:
        diag_result = gawf_diagnostics.to_result_dict()
        if diag_result is not None:
            out["gawf_diagnostics"] = diag_result
    return out


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_layers <= 0:
        parser.error("--num_layers must be >= 1")
    if args.gawf_feedback_lr_scale <= 0:
        parser.error("--gawf_feedback_lr_scale must be > 0")
    if args.feedback_dim is not None and args.feedback_dim < 0:
        parser.error("--feedback_dim/--dz must be >= 0")
    if args.gawf_diag_every <= 0:
        parser.error("--gawf_diag_every must be > 0")
    if args.gawf_diag_gate_eps <= 0 or args.gawf_diag_gate_eps >= 0.5:
        parser.error("--gawf_diag_gate_eps must be in (0, 0.5)")

    # GPU selection before any CUDA init. If the launcher already set
    # CUDA_VISIBLE_DEVICES, respect it (logical cuda:0 is that device).
    cuda_visible_preset = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    if cuda_visible_preset:
        cuda_index = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cuda_index = pick_cuda_device_index()
        if cuda_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)
            device = "cuda"
        else:
            device = "cpu"

    set_seed(args.seed)

    disable_tqdm_env = os.environ.get("DISABLE_TQDM", "").lower() in ["1", "true", "yes"]
    enable_tqdm_env = os.environ.get("ENABLE_TQDM", "").lower() in ["1", "true", "yes"]
    term_ok = os.environ.get("TERM", "").lower() not in ["", "dumb"]
    use_tqdm = enable_tqdm_env or (not disable_tqdm_env and sys.stdout.isatty() and term_ok)

    results_root = PathHelper.get_results_root(override=args.results_dir or None)
    results_dir = os.path.join(results_root, "train_data", args.result_suffix)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, "train.log")
    logger = LoggingHelper.setup_logger("train", log_file=log_file)

    if cuda_visible_preset:
        logger.info(
            "Using preset CUDA_VISIBLE_DEVICES=%s (device=%s)",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            device,
        )
    elif cuda_index is not None:
        logger.info("Using CUDA device: %s", cuda_index)
    logger.info("Random seed set to: %s", args.seed)
    logger.info("Created or using results directory: %s", results_dir)

    base_path = PathHelper.get_base_path(override=args.data_dir or None, logger=logger)
    eval_suffix = (args.eval_data_suffix or "").strip() or args.data_suffix
    stim_train_path, label_train_path = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("train",), logger=logger
    )
    stim_val_path, label_val_path = PathHelper.prepare_data_paths(
        base_path, data_suffix=eval_suffix, splits=("valid",), logger=logger
    )
    if eval_suffix != args.data_suffix:
        logger.info(
            "Train suffix=%s, validation suffix=%s (shared eval set across scales).",
            args.data_suffix,
            eval_suffix,
        )
    stims_train, lbls_train, stims_val, lbls_val = PathHelper.load_raw_data(
        stim_train_path,
        label_train_path,
        stim_val_path,
        label_val_path,
        use_mmap=args.use_mmap,
        logger=logger,
    )

    use_sector_mode = args.use_sector_mode
    predict_all_chars = args.predict_all_chars
    use_acceleration = args.use_acceleration
    max_chars = 15  # Num of bg digit in 40h is 12

    train_ds, val_ds, num_pos = create_datasets(
        stims_train,
        lbls_train,
        stims_val,
        lbls_val,
        use_sector_mode=args.use_sector_mode,
        predict_all_chars=args.predict_all_chars,
        max_chars=max_chars,
        dataset_class=MC_RNN_Dataset,
        splits=("train", "valid"),
        logger=logger,
    )

    model_classes = get_model_classes(
        RNNConv,
        LSTMConv,
        GRUConv,
        GaWFRNNConv,
        MambaConv,
        S5Conv,
    )

    model_types = args.model_types
    hidden_sizes = args.hidden_sizes
    mamba_d_models = args.mamba_d_models
    s5_d_models = args.s5_d_models
    s5_state_sizes = args.s5_state_sizes
    feedback_dim = args.feedback_dim
    lrs = args.lrs
    wds = args.wds
    cnn_dropout_grid = args.cnn_dropout
    rnn_dropout = args.rnn_dropout
    num_layers = args.num_layers

    # Build hyperparameter combinations with model-specific width/state names.
    experiment_configs = []
    for model_type in model_types:
        if model_type == "mamba":
            for mamba_d_model, lr, weight_decay, cnn_dropout in product(
                mamba_d_models, lrs, wds, cnn_dropout_grid
            ):
                experiment_configs.append(
                    {
                        "model_type": model_type,
                        "model_width": mamba_d_model,
                        "width_label": "dmodel",
                        "state_size": None,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "cnn_dropout": cnn_dropout,
                    }
                )
        elif model_type in ("ssm", "s5"):
            for s5_d_model, s5_state_size, lr, weight_decay, cnn_dropout in product(
                s5_d_models, s5_state_sizes, lrs, wds, cnn_dropout_grid
            ):
                experiment_configs.append(
                    {
                        "model_type": model_type,
                        "model_width": s5_d_model,
                        "width_label": "dmodel",
                        "state_size": s5_state_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "cnn_dropout": cnn_dropout,
                    }
                )
        else:
            for hidden_size, lr, weight_decay, cnn_dropout in product(
                hidden_sizes, lrs, wds, cnn_dropout_grid
            ):
                experiment_configs.append(
                    {
                        "model_type": model_type,
                        "model_width": hidden_size,
                        "width_label": "h",
                        "state_size": None,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "cnn_dropout": cnn_dropout,
                    }
                )

    # Training loop over all hyperparameter combinations
    total_experiments = len(experiment_configs)
    experiment_num = 0

    LoggingHelper.log_experiment_config(
        logger,
        total_experiments,
        model_types,
        hidden_sizes,
        lrs,
        wds,
        cnn_dropout_grid,
        rnn_dropout,
    )

    for config in experiment_configs:
        model_type = config["model_type"]
        model_width = config["model_width"]
        width_label = config["width_label"]
        state_size = config["state_size"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        cnn_dropout = config["cnn_dropout"]
        experiment_num += 1
        LoggingHelper.log_experiment_start(
            logger,
            experiment_num,
            total_experiments,
            model_type,
            model_width,
            lr,
            weight_decay,
            cnn_dropout,
            rnn_dropout,
        )

        if predict_all_chars:
            num_pos = 0

        # Create model
        if model_type not in model_classes:
            logger.warning("Unsupported model_type: %s, skipping...", model_type)
            continue

        ModelClass = (
            MultiLayerGaWFRNNConv
            if model_type == "gawf" and num_layers > 1
            else model_classes[model_type]
        )
        model_kwargs = {}
        width_kwarg = "hidden_size"
        if model_type == "mamba":
            width_kwarg = "mamba_d_model"
        elif model_type in ("ssm", "s5"):
            width_kwarg = "s5_d_model"
            model_kwargs["s5_state_size"] = state_size
            model_kwargs["s5_num_layers"] = args.s5_num_layers
            model_kwargs["s5_dropout"] = args.s5_dropout
        elif model_type == "gawf":
            model_kwargs["feedback_dim"] = feedback_dim
            if num_layers > 1:
                model_kwargs["num_layers"] = num_layers
        elif model_type in ("rnn", "gru", "lstm"):
            model_kwargs["num_layers"] = num_layers
        mdl = ModelClass(
            num_classes=10,
            num_pos=num_pos,
            kernel_size=5,
            device=device,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            **{width_kwarg: model_width},
            max_chars=max_chars,
            predict_all_chars=predict_all_chars,
            **model_kwargs,
        )

        width_desc = f"{width_label}={model_width}"
        if model_type in ("ssm", "s5"):
            width_desc = f"s5_d_model={model_width}, s5_state_size={state_size}"
        elif model_type == "gawf" and feedback_dim is not None:
            width_desc = f"{width_desc}, dz={feedback_dim}"
        elif model_type == "gawf" and num_layers > 1:
            if getattr(mdl, "use_feedback_projector", False):
                feedback_desc = f"dz={mdl.feedback_dim}"
            else:
                feedback_desc = "direct_feedback"
            width_desc = f"{width_desc}, layers={num_layers}, {feedback_desc}"
        logger.info(
            "Created %s model (predict_all_chars=%s, max_chars=%s, cnn_dropout=%s, rnn_dropout=%s, %s, cnn_feature_size=large)",
            model_type.upper(),
            predict_all_chars,
            max_chars,
            cnn_dropout,
            rnn_dropout,
            width_desc,
        )

        train_lr = lr
        gawf_feedback_lr_scale = args.gawf_feedback_lr_scale
        if model_type == "gawf":
            logger.info(
                "gawf optimizer lr: base_lr=%s, feedback_lr_scale=%s",
                lr,
                args.gawf_feedback_lr_scale,
            )

        mode_suffix = (
            "allchars" if predict_all_chars else ("sector" if use_sector_mode else "coord")
        )
        acc_suffix = "_acc" if use_acceleration else ""
        hp_suffix = f"_lr{train_lr}_wd{weight_decay}_cdo{cnn_dropout}_rdo{rnn_dropout}"
        # nofb/fb_start_epoch in result path: nofb only -> _nofb; nofb + fb_start_epoch -> _fb{N} only
        if args.nofb:
            if args.fb_start_epoch >= 999999:
                fb_path_suffix = "_nofb"
            else:
                fb_path_suffix = f"_fb{args.fb_start_epoch}"
        else:
            fb_path_suffix = ""
        width_suffix = f"_{width_label}{model_width}"
        if model_type in ("ssm", "s5"):
            width_suffix = f"_dmodel{model_width}_state{state_size}"
        layer_suffix = ""
        if model_type in ("rnn", "gru", "lstm", "gawf") and num_layers > 1:
            layer_suffix = f"_L{num_layers}"
        dz_suffix = ""
        if model_type == "gawf" and feedback_dim is not None:
            dz_suffix = f"_dz{feedback_dim}"
        elif (
            model_type == "gawf"
            and num_layers > 1
            and getattr(mdl, "use_feedback_projector", False)
        ):
            dz_suffix = f"_dz{mdl.feedback_dim}"
        results_stem = (
            f"{model_type}_{mode_suffix}{acc_suffix}{width_suffix}"
            f"{layer_suffix}{hp_suffix}{dz_suffix}{fb_path_suffix}"
        )
        results_path = os.path.join(results_dir, results_stem)

        gawf_diag_path = None
        if args.gawf_diag and model_type == "gawf":
            diag_dir = args.gawf_diag_dir.strip() or os.path.join(
                results_dir,
                "gawf_diagnostics",
            )
            gawf_diag_path = os.path.join(
                diag_dir,
                f"{results_stem}_gawf_diag.jsonl",
            )

        # # [COMPILE] compile model for speed (PyTorch 2.x)
        # try:
        #     mdl = torch.compile(mdl)  # 可选：torch.compile(mdl, mode="max-autotune")
        # except Exception as e:
        #     logger.warning("[COMPILE] torch.compile failed, fallback to eager: %s", e)

        # Train model
        logger.info("Starting training...")
        logger.info(
            "Acceleration training enabled"
            if use_acceleration
            else "Using standard training method"
        )

        run_label = f"{args.result_suffix}|e{experiment_num:03d}|{model_type}"
        results = network_train(
            mdl,
            train_ds,
            val_ds,
            num_epochs=args.num_epochs,
            lr=train_lr,
            use_acceleration=use_acceleration,
            weight_decay=weight_decay,
            rnn_diag_lambda=1e-4,
            use_mmap=args.use_mmap,
            use_tqdm=use_tqdm,
            nofb=args.nofb,
            fb_start_epoch=args.fb_start_epoch,
            seed=args.seed,
            logger=logger,
            optim=args.optim,
            patience=args.patience,
            run_label=run_label,
            gawf_feedback_lr_scale=gawf_feedback_lr_scale,
            gawf_diag_enabled=args.gawf_diag,
            gawf_diag_path=gawf_diag_path,
            gawf_diag_every=args.gawf_diag_every,
            gawf_diag_gate_eps=args.gawf_diag_gate_eps,
            s5_ssm_lr_scale=args.s5_ssm_lr_scale,
        )

        # Save training results
        logger.info(
            "Saving results for %s (%s)...",
            model_type.upper(),
            width_desc,
        )
        PathHelper.save_results(results, results_path, logger=logger)

        # Build and save concise metrics summary for this experiment
        dataset_mode = mode_suffix
        results["eval_dataset_suffix"] = eval_suffix
        metric_summary = summarize_experiment_metrics(
            results,
            model_type=model_type,
            dataset_suffix=args.data_suffix,
            dataset_mode=dataset_mode,
            num_epochs=args.num_epochs,
            hidden_size=model_width,
            lr=train_lr,
            weight_decay=weight_decay,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            optimizer=args.optim,
        )
        if train_lr != lr:
            metric_summary["requested_lr"] = lr
            metric_summary["effective_lr"] = train_lr
        if model_type in ("rnn", "gru", "lstm", "gawf"):
            metric_summary["num_layers"] = int(num_layers)
        metric_summary["core_param_count"] = int(sum(p.numel() for p in mdl.core.parameters()))
        metric_summary["total_param_count"] = int(sum(p.numel() for p in mdl.parameters()))
        if model_type == "mamba":
            metric_summary["mamba_d_model"] = model_width
        elif model_type in ("ssm", "s5"):
            metric_summary["s5_d_model"] = model_width
            metric_summary["s5_state_size"] = state_size
            metric_summary["s5_num_layers"] = args.s5_num_layers
            metric_summary["s5_dropout"] = args.s5_dropout
            metric_summary["s5_ssm_lr_scale"] = args.s5_ssm_lr_scale
        elif model_type == "gawf":
            metric_summary["feedback_dim"] = (
                int(mdl.feedback_dim) if hasattr(mdl, "feedback_dim") else None
            )
            metric_summary["num_layers"] = int(num_layers)
            metric_summary["gawf_feedback_lr_scale"] = args.gawf_feedback_lr_scale
            if num_layers > 1:
                metric_summary["use_feedback_projector"] = bool(
                    getattr(mdl, "use_feedback_projector", False)
                )
                metric_summary["layer_feedback_dims"] = [
                    int(dim) for dim in getattr(mdl, "layer_feedback_dims", [])
                ]
        if gawf_diag_path is not None:
            metric_summary["gawf_diag_path"] = gawf_diag_path
            metric_summary["gawf_diag_every"] = args.gawf_diag_every
            metric_summary["gawf_diag_gate_eps"] = args.gawf_diag_gate_eps
        metrics_path = os.path.join(results_dir, f"{results_stem}_metrics.json")
        PathHelper.save_metrics_summary(metric_summary, metrics_path, logger=logger)

        logger.info("Experiment %s/%s completed!", experiment_num, total_experiments)

    logger.info("=" * 60)
    logger.info(
        "All %s experiments completed! Results saved to: %s/", total_experiments, results_dir
    )
    logger.info("=" * 60)
