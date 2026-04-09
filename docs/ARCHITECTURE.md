# ARCHITECTURE.md — Module Responsibilities & Dependency Map

## 1. High-Level Package Map

```
aim3_RNN/
│
├── train_model.py     ← CLI entry-point: parse args, load data, loop over hparam combos
│                               Owns MC_RNN_Dataset and network_train()
│
├── utils/                   ← Training pipeline (imported by train_model only)
│   ├── train_rnn_core.py    ← Model base classes and standard RNN variants
│   ├── train_gawf_core.py   ← GaWF feedback model
│   ├── train_ann_core.py    ← ANN/Dendritic model variants
│   ├── train_helpers.py     ← I/O, logging, arg parsing, seeding, path resolution
│   ├── train_rnn_engine.py  ← Training step orchestration (setup, train, evaluate)
│   ├── train_acceleration.py← AMP, grad scaler, DataLoader builder, TrainStepper
│   ├── train_sector.py      ← Loss & metrics for single-char + sector/coordinate mode
│   └── train_predict_all_chars.py ← Loss & metrics for all-chars mode
│
├── utils_anal/              ← Post-training analysis (no matplotlib)
│   ├── export_gate_sample.py← CANONICAL: build_model_from_ckpt, build_test_dataset
│   ├── export_gate_avg.py   ← Per-digit/sector averaged gate matrices
│   ├── export_gate_avg_allsector.py ← All-component decomposition of trans_ih
│   ├── export_V_basis.py    ← V parameter basis maps per sector/digit
│   ├── export_whh.py        ← Hidden-hidden weight W_hh analysis
│   ├── export_pop_act.py    ← Population activity export
│   ├── cnn_channel_stats.py ← CNN channel activation statistics
│   ├── hidden_unit_tuning.py← Hidden unit digit tuning + FDR analysis
│   ├── model_param_counts.py← Parameter count utility
│   └── pop_act_dpca.py      ← dPCA dimensionality reduction
│
├── utils_viz/               ← Visualisation (matplotlib, no model loading)
│   ├── model_train_single_result.py ← Training curve plots + parse_hparams_from_filename
│   ├── gate_avg.py          ← Gate/outer matrix heatmaps (single digit/sector)
│   ├── gate_avg_allsector.py← Multi-panel all-component decomposition
│   ├── gate_avg_sector_summary.py ← Summary across all sectors
│   ├── gate_sample.py       ← Single-sample gate visualisation
│   ├── cnn_channel.py       ← CNN channel activation matrix heatmap
│   ├── hidden_activation.py ← Hidden unit tuning heatmaps
│   ├── V_basis.py           ← V parameter basis map heatmaps
│   ├── whh.py               ← W_hh connectivity matrix
│   ├── dimred_reducer.py    ← Shared dimensionality reduction wrapper
│   ├── pop_act_umap.py      ← UMAP population activity plots
│   ├── sample_fg+bg_frames_sample.py ← Stimulus frame examples
│   └── paper_figs/          ← Publication figures (fig1.py, metrics_best_acc_bars.py)
│
└── source/                  ← Stimulus generation (independent, rarely modified)
    ├── GenerateMovies.py
    └── GenerateMovies_cplx.py
```

---

## 2. Dependency Graph

```
stdlib / third-party (torch, numpy, matplotlib, scipy, tqdm)
        │
        ▼
utils/train_helpers.py ◄────────────────────┐
        │                                    │
        ▼                                    │
utils/train_acceleration.py                  │
        │                                    │
        ▼                                    │
utils/train_rnn_core.py                      │
        │                                    │
        ▼                                    │
utils/train_gawf_core.py    utils/train_ann_core.py
        │                          │
        └──────────┬───────────────┘
                   ▼
        utils/train_sector.py
        utils/train_predict_all_chars.py
                   │
                   ▼
        utils/train_rnn_engine.py
                   │
                   ▼
        train_model.py (MC_RNN_Dataset lives here)
                   │
                   ▼
        utils_anal/export_gate_sample.py  ◄── canonical shared helpers
                   │
                   ▼
        utils_anal/*.py  (all other analysis scripts)
                   │
                   ▼
        utils_viz/*.py   (reads files from results/anal_data/)
```

**Key rule:** dependency arrows are one-directional and downward only.

---

## 3. Module Responsibilities

### `utils/train_helpers.py`
Single source for: path resolution and I/O (`PathHelper`: `get_base_path`, `prepare_data_paths`,
`load_raw_data`, `save_results`, `save_metrics_summary`), dataset construction (`create_datasets`),
model class registry (`get_model_classes`), logging (`LoggingHelper`),
argument parser (`build_arg_parser`), seeding (`set_seed`), metrics helpers (`summarize_experiment_metrics`).

### `utils/train_rnn_engine.py`
Owns the training loop internals: `setup_training_components` (builds all
components dict), `begin_epoch`, `train_batch`, `summarize_online_train`,
`eval_train_subset`, `eval_valid` (both wrap `evaluate_epoch` for fair full-loader eval),
and core `evaluate_epoch`. The `network_train` skeleton in `train_model.py` only calls these.

### `utils/train_acceleration.py`
Owns: `AccelerationConfig` (all AMP/grad-accum flags), `setup_acceleration`
(returns autocast_fn, scaler, batch_size, …), `build_loaders` (train/val/eval
DataLoaders), `TrainStepper` (one-step forward+backward, no branches in
training loop), `run_forward_with_feedback`.

### `utils/train_sector.py`
Owns all metric and loss logic for single-char + sector/coordinate mode:
`loss_char_single`, `loss_pos_single`, `batch_metric_*`, `eval_accumulate_batch_*`,
`finalize_metrics_single`, `build_loss_fn_single`, `SingleCharMetricsMode`,
and fg-switch window helpers: `compute_fg_transition_masks`, `single_char_global_eval_*`.

### `utils/train_predict_all_chars.py`
Same pattern as `train_sector.py` but for all-chars mode (greedy matching).

### `utils_anal/export_gate_sample.py`
**Canonical model loader.** All analysis scripts must call:
```python
from utils_anal.export_gate_sample import build_model_from_ckpt, build_test_dataset
```
These two functions handle: hparam parsing from filename, GaWFRNNConv instantiation,
state_dict filtering (`prev_feedback`), test split dataset construction.

### `utils_viz/model_train_single_result.py`
Contains `parse_hparams_from_filename()` — used by analysis scripts to extract
`hidden_size`, `cnn_dropout`, `rnn_dropout` (and legacy `dropout` / `_do` for old stems), `lr`, `wd` from checkpoint filenames. Import when
rebuilding a model from a checkpoint filename alone.

---

## 4. Data Flow Diagram

```
source/GenerateMovies*.py
        │
        ▼
stimuli/  (stimulus_reg-*.npy  +  labels_reg-*.csv)
        │
        ▼  train_helpers.PathHelper.load_raw_data()
MC_RNN_Dataset  (train_model.py)
        │
        ▼  DataLoader
train_rnn_engine  →  GaWFRNNConv / RNNConv / …
        │
        ▼  PathHelper.save_results()
results/train_data/<suffix>/
  ├── *_model.pth          ← state dict
  ├── *.pkl                ← metrics curves
  └── *_metrics.json       ← summary

        │
        ▼  utils_anal/export_*.py
results/anal_data/<module>/
  ├── *.npy / *.npz        ← arrays
  └── *_meta_*.json        ← metadata

        │
        ▼  utils_viz/*.py
results/anal_figs/<module>/
  └── *.png
```

---

## 5. Key Shared Interfaces

### `MetricsMode` protocol (both implementations must satisfy)
```python
init_epoch_train()                     -> dict
update_train_batch(acc, out_char, labels, batch_idx, len_dl, out_pos) -> dict
finalize_train_epoch(acc, num_batches) -> (acc_char, metric_pos, ...)
init_eval()                            -> dict
update_eval_batch(acc, out_char, labels, out_pos) -> dict
finalize_eval(acc, num_batches)        -> (acc_char, metric_pos, ...)
format_train_str(epoch, num_epochs, acc_char, metric_pos, gpu_info) -> str
format_val_str(acc_char, metric_pos)   -> str
postfix_for_pbar(loss, out_char, out_pos, labels) -> dict
add_pos_to_result_dict(base, ...) -> dict
```

### `components` dict keys (returned by `setup_training_components`)
`device`, `train_dl`, `train_eval_dl`, `val_dl`, `stepper`, `metrics_mode`,
`train_acc_char`, `val_acc_char`, `train_metric_pos`, `val_metric_pos`,
`train_loss_pos`, `val_loss_pos`, `train_loss_char`, `val_loss_char`,
`glob_*`, `fg_switch_pre5_*`, `fg_switch_post5_*` (sector single-char only, when TSV has `fg_switch`),
`stop_flag`, `use_tqdm`, `logger`.
