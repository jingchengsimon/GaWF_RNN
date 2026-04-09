"""
Standalone RNN Sector training script
Used to train RNN models and save results

Device support: CUDA (Linux/NVIDIA) as default, CPU as fallback.

Smoke-test (run 1 epoch, minimal config):
  CUDA:  python train_model.py --num_epochs 1
"""
import os
import sys
from itertools import product
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.train_helpers import (
    LoggingHelper,
    PathHelper,
    create_datasets,
    set_seed,
    get_model_classes,
    build_arg_parser,
    pick_cuda_device_index,
    summarize_experiment_metrics,
)

from utils.train_sector import compute_fg_transition_masks

from utils.train_rnn_engine import (
    setup_training_components,
    cleanup_dataloaders,
    stop_requested,
    begin_epoch,
    train_batch,
    summarize_online_train,
    eval_train_subset,
    eval_valid,
)

from utils.train_rnn_core import RNNConv, GRUConv, LSTMConv
from utils.train_gawf_core import GaWFRNNConv
from utils.train_ann_core import DendriticANNConv, FeedForwardConv


torch.set_num_threads(4)


# ==================== Dataset Class ====================
class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9,
                 max_chars=15, predict_all_chars=False):
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
            self.fg_char_ids = labels['fg_char_id'].values
            raw_bg = labels['bg_char_ids'].values
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
            self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values
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
                col = np.clip((x / max(width - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
                row = np.clip((y / max(height - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
                sector = row * grid_size + col
                char_id = self.labels[:, 0].astype(np.int64)
                self.labels_sector = np.stack([char_id, sector], axis=1).astype(np.int64, copy=False)
            else:
                # Coordinate mode: one-time float32 conversion for labels.
                self.labels_coord = self.labels.astype(np.float32, copy=False)

        # fg_switch + transition masks (sector, single-char only; used for fair eval metrics).
        if "fg_switch" in labels.columns:
            self.fg_switch = labels["fg_switch"].values.astype(np.int32, copy=False)
        else:
            self.fg_switch = np.zeros(self.data.shape[0], dtype=np.int32)
        if not predict_all_chars and use_sector:
            self.pre5_mask_global, self.post5_mask_global = compute_fg_transition_masks(self.fg_switch)
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
                bg_char_ids = self.bg_char_ids_parsed[frame_idx] if frame_idx < len(self.bg_char_ids_parsed) else []
                all_chars = [fg_char_id] + bg_char_ids
                padded_chars = all_chars[: self.max_chars] + [-1] * max(0, self.max_chars - len(all_chars))
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
    )

    val_every = 1  # run full validation only every N epochs
    try:
        for epoch in range(num_epochs):
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
  
    except (KeyboardInterrupt, SystemExit):
        # Handle interruption gracefully
        if logger is not None:
            logger.info("Training interrupted, cleaning up resources...")
    finally:
        # Explicit resource cleanup for DataLoader so worker processes exit.
        # Use Ctrl+C (SIGINT) to stop training; kill -9 skips this and can leave workers running.
        cleanup_dataloaders(components)

    if components["device"] == "cuda" or (isinstance(components["device"], str) and components["device"].startswith("cuda:")):
        torch.cuda.empty_cache()

    # If early stopping triggered, only return actual trained epochs (epoch starts from 0, so actually trained epoch+1 epochs)
    actual_epochs = epoch + 1

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
        out["fg_switch_pre5_val_acc_char"] = components["fg_switch_pre5_val_acc_char"][:actual_epochs]
        out["fg_switch_pre5_train_acc_pos"] = components["fg_switch_pre5_train_acc_pos"][
            :actual_epochs
        ]
        out["fg_switch_pre5_val_acc_pos"] = components["fg_switch_pre5_val_acc_pos"][:actual_epochs]
        out["fg_switch_post5_train_acc_char"] = components["fg_switch_post5_train_acc_char"][
            :actual_epochs
        ]
        out["fg_switch_post5_val_acc_char"] = components["fg_switch_post5_val_acc_char"][:actual_epochs]
        out["fg_switch_post5_train_acc_pos"] = components["fg_switch_post5_train_acc_pos"][
            :actual_epochs
        ]
        out["fg_switch_post5_val_acc_pos"] = components["fg_switch_post5_val_acc_pos"][:actual_epochs]
    return out

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # GPU selection before any CUDA init (matches previous module-level behavior)
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
    use_tqdm = enable_tqdm_env or (
        not disable_tqdm_env and sys.stdout.isatty() and term_ok
    )

    results_dir = f"results/train_data/{args.result_suffix}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, "train.log")
    logger = LoggingHelper.setup_logger("train", log_file=log_file)

    if cuda_index is not None:
        logger.info("Using CUDA device: %s", cuda_index)
    logger.info("Random seed set to: %s", args.seed)
    logger.info("Created or using results directory: %s", results_dir)

    base_path = PathHelper.get_base_path(override=args.data_dir or None, logger=logger)
    stim_train_path, label_train_path, stim_val_path, label_val_path = PathHelper.prepare_data_paths(
        base_path, data_suffix=args.data_suffix, splits=("train", "valid"), logger=logger
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
        FeedForwardConv,
        DendriticANNConv,
    )

    model_types = args.model_types
    hidden_sizes = args.hidden_sizes
    lrs = args.lrs
    wds = args.wds
    cnn_dropout_grid = args.cnn_dropout
    rnn_dropout = args.rnn_dropout

    # Build hyperparameter combinations: (model_type, hidden_size, lr, weight_decay, cnn_dropout)
    experiment_configs = list(
        product(model_types, hidden_sizes, lrs, wds, cnn_dropout_grid)
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

    for model_type, hidden_size, lr, weight_decay, cnn_dropout in experiment_configs:
        experiment_num += 1
        LoggingHelper.log_experiment_start(
            logger,
            experiment_num,
            total_experiments,
            model_type,
            hidden_size,
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

        ModelClass = model_classes[model_type]
        mdl = ModelClass(
                num_classes=10,
                num_pos=num_pos,
                kernel_size=5,
                device=device,
                cnn_dropout=cnn_dropout,
                rnn_dropout=rnn_dropout,
                hidden_size=hidden_size,
                max_chars=max_chars,
                predict_all_chars=predict_all_chars,
            )

        logger.info(
            "Created %s model (predict_all_chars=True, max_chars=%s, cnn_dropout=%s, rnn_dropout=%s, hidden_size=%s, cnn_feature_size=large)",
            model_type.upper(), max_chars, cnn_dropout, rnn_dropout, hidden_size,
        )
       
        # # [COMPILE] compile model for speed (PyTorch 2.x)
        # try:
        #     mdl = torch.compile(mdl)  # 可选：torch.compile(mdl, mode="max-autotune")
        # except Exception as e:
        #     logger.warning("[COMPILE] torch.compile failed, fallback to eager: %s", e)

        # Train model
        logger.info("Starting training...")
        logger.info("Acceleration training enabled" if use_acceleration else "Using standard training method")
        

        results = network_train(
            mdl,
            train_ds,
            val_ds,
            num_epochs=args.num_epochs,
            lr=lr,
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
        )

        # Save training results
        logger.info(
            "Saving results for %s (hidden_size=%s)...",
            model_type.upper(),
            hidden_size,
        )
        mode_suffix = "allchars" if predict_all_chars else ("sector" if use_sector_mode else "coord")
        acc_suffix = "_acc" if use_acceleration else ""
        hp_suffix = f"_lr{lr}_wd{weight_decay}_cdo{cnn_dropout}_rdo{rnn_dropout}"
        # nofb/fb_start_epoch in result path: nofb only -> _nofb; nofb + fb_start_epoch -> _fb{N} only
        if args.nofb:
            if args.fb_start_epoch >= 999999:
                fb_path_suffix = "_nofb"
            else:
                fb_path_suffix = f"_fb{args.fb_start_epoch}"
        else:
            fb_path_suffix = ""
        results_stem = f"{model_type}_{mode_suffix}{acc_suffix}_h{hidden_size}{hp_suffix}{fb_path_suffix}"
        results_path = os.path.join(results_dir, results_stem)

        PathHelper.save_results(results, results_path, logger=logger)

        # Build and save concise metrics summary for this experiment
        dataset_mode = mode_suffix
        metric_summary = summarize_experiment_metrics(
            results,
            model_type=model_type,
            dataset_suffix=args.data_suffix,
            dataset_mode=dataset_mode,
            num_epochs=args.num_epochs,
            hidden_size=hidden_size,
            lr=lr,
            weight_decay=weight_decay,
            cnn_dropout=cnn_dropout,
            rnn_dropout=rnn_dropout,
            optimizer=args.optim,
        )
        metrics_path = os.path.join(results_dir, f"{results_stem}_metrics.json")
        PathHelper.save_metrics_summary(metric_summary, metrics_path, logger=logger)

        logger.info("Experiment %s/%s completed!", experiment_num, total_experiments)

    logger.info("=" * 60)
    logger.info("All %s experiments completed! Results saved to: %s/", total_experiments, results_dir)
    logger.info("=" * 60)

