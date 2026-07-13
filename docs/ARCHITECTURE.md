# ARCHITECTURE.md — Module Responsibilities & Dependency Map

## 1. High-Level Package Map

```
aim3_RNN/
│
├── train_model.py     ← CLI entry-point: parse args, load data, loop over hparam combos
│                               Owns MC_RNN_Dataset and network_train()
├── experiments/generalization/ ← Generalization launchers + hparam aggregation
├── experiments/amarel/ ← Amarel/Slurm probe, submit, status, and rerun helpers
│
├── utils/                   ← Task models, shared recurrent cores, train/data helpers
│   ├── recurrent_cores/     ← RNN/GRU/LSTM/GaWF/Mamba/S5 task-agnostic recurrent cores
│   ├── clutter_task_models.py
│   ├── clutter_train_helpers.py
│   ├── clutter_train_engine.py
│   ├── clutter_train_acceleration.py
│   ├── clutter_train_sector.py
│   ├── clutter_train_predict_all_chars.py
│   ├── text_task_models.py
│   ├── text_imdb_data.py
│   ├── text_sentihood_data.py
│   ├── text_sentihood_metrics.py
│   ├── text_train_utils.py
│   ├── atari_dqn_models.py
│   ├── atari_envs.py
│   ├── atari_replay.py
│   ├── atari_train_acceleration.py
│   └── common_train_helpers.py
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
│   ├── plot_generalization.py ← Phase-3 CSV → char/sector gap + train/val acc vs scale (PNG; optional PDF)
│   └── paper_figs/          ← Publication figures (fig1.py, metrics_best_acc_bars.py)
│
└── source/                  ← Data generation/preparation scripts by task family
    ├── clutter/generate_movies.py
    ├── clutter/generate_movies_cplx.py
    ├── text/prepare_imdb_data.py
    ├── text/prepare_sentihood_data.py
    └── atari/
```

---

## 2. Dependency Graph

```
stdlib / third-party (torch, numpy, matplotlib, scipy, tqdm)
        │
        ▼
utils/clutter_train_helpers.py ◄────────────┐
        │                                    │
        ▼                                    │
utils/clutter_train_acceleration.py          │
        │                                    │
        ▼                                    │
utils/recurrent_cores/                       │
        │                                    │
        ▼                                    │
utils/clutter_task_models.py                 │
        │                                    │
        └──────────┬─────────────────────────┘
                   ▼
        utils/clutter_train_sector.py
        utils/clutter_train_predict_all_chars.py
                   │
                   ▼
        utils/clutter_train_engine.py
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

### `utils/clutter_train_helpers.py`
Single source for: path resolution and I/O (`PathHelper`: `get_base_path`, `prepare_data_paths`,
`load_raw_data`, `save_results`, `save_metrics_summary`), dataset construction (`create_datasets`),
model class registry (`get_model_classes`), logging (`LoggingHelper`),
argument parser (`build_arg_parser`), seeding (`set_seed`), metrics helpers (`summarize_experiment_metrics`).

### `utils/clutter_train_engine.py`
Owns the training loop internals: `setup_training_components` (builds all
components dict), `begin_epoch`, `train_batch`, `summarize_online_train`,
`eval_train_subset`, `eval_valid` (both wrap `evaluate_epoch` for fair full-loader eval),
and core `evaluate_epoch`. The `network_train` skeleton in `train_model.py` only calls these.
GaWF handling uses one public `gawf --num_layers N` interface for feedback scheduling,
feedback freezing, and no-weight-decay U/V optimizer grouping.

### `utils/clutter_train_acceleration.py`
Owns: `AccelerationConfig` (all AMP/grad-accum flags), `setup_acceleration`
(returns autocast_fn, scaler, batch_size, …), `build_loaders` (train/val/eval
DataLoaders), `TrainStepper` (one-step forward+backward, no branches in
training loop), `run_forward_with_feedback`.
For complex-parameter models (e.g. S5), training keeps AMP autocast active but
disables GradScaler to avoid `ComplexFloat` unscale limitations in CUDA AMP.
Gradient clipping in `TrainStepper` is applied to real-valued gradients only;
complex gradients are skipped for foreach clip ops.

### Atari DQN training path
`train_atari_dqn.py` owns the DQN/DRQN loop and composes `utils/atari_dqn_models.py`,
`utils/atari_replay.py`, and `utils/atari_envs.py`. Pong defaults to one ALE frame and one
observed frame per environment step (`frame_skip=1`, `frame_stack=1`); new result suffixes use
`pong_fs1_stack1` so they cannot be confused with historical frame-skip-4 pilots.

`utils/atari_train_acceleration.py` owns CUDA autocast, TF32, gradient scaling, and optional
`torch.compile` configuration. `AtariQNetwork.forward_sequence` uses a semantics-equivalent
whole-sequence cuDNN fast path for RNN/GRU/LSTM windows without internal episode resets and
falls back to the reset-aware stepwise path otherwise. Acceleration must not change replay
sampling, update cadence, UTD, loss definitions, or recurrent/GaWF architecture.
Amarel Pong launchers compile ANN only because the installed PyTorch 2.3 Dynamo path cannot
reliably trace `AtariQNetworkState`; recurrent models still use BF16, TF32, and the fused scan.
The completed five-seed, 70-unit historical sweep is labeled `pong_fs4_stack1`: one pooled
observation was supplied per decision, while each chosen action advanced four ALE frames.
The `pong_fs1_stack1` label is reserved for strict one-decision/one-ALE-frame runs.

### `utils/clutter_train_sector.py`
Owns all metric and loss logic for single-char + sector/coordinate mode:
`loss_char_single`, `loss_pos_single`, `batch_metric_*`, `eval_accumulate_batch_*`,
`finalize_metrics_single`, `build_loss_fn_single`, `SingleCharMetricsMode`,
and fg-switch window helpers: `compute_fg_transition_masks`, `single_char_global_eval_*`.

### `utils/clutter_train_predict_all_chars.py`
Same pattern as `clutter_train_sector.py` but for all-chars mode (greedy matching).

### `utils_anal/export_gate_sample.py`
**Canonical model loader.** All analysis scripts must call:
```python
from utils_anal.export_gate_sample import build_model_from_ckpt, build_test_dataset
```
These two functions handle: hparam parsing from filename, unified single/multi-layer
GaWF instantiation, state_dict filtering (`prev_feedback`), GaWF feedback dim parsing (`_dz*`),
and test split dataset construction.

### `utils_viz/model_train_single_result.py`
Contains `parse_hparams_from_filename()` — used by analysis scripts to extract
`hidden_size`, `feedback_dim` (from `_dz*` when present), `num_layers` (from `_L*`),
`cnn_dropout`, `rnn_dropout` (and legacy `dropout` / `_do` for old stems), `lr`, `wd`
from checkpoint filenames. Import when
rebuilding a model from a checkpoint filename alone.

---

## 4. Data Flow Diagram

```
source/clutter/generate_movies*.py
        │
        ▼
stimuli/  (stimulus_reg-*.npy  +  labels_reg-*.csv)
        │
        ▼  clutter_train_helpers.PathHelper.load_raw_data()
MC_RNN_Dataset  (train_model.py)
        │
        ▼  DataLoader
clutter_train_engine  →  clutter_task_models → recurrent_cores
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
`stop_flag`, `use_tqdm`, `logger`, `run_label` (optional prefix for tqdm/log lines in multi-job runs).

---

## 6. Generalization experiment orchestration

**Location:** `experiments/generalization/` (shell + **`collect_results.py`** +
**`hparam_full_grid.py`**, stdlib only) and Amarel submission wrappers under
`experiments/amarel/`.

**Role:** Launch **`train_model.py`** for train-scale vs **fixed 40h validation**
protocols; aggregate `*_metrics.json` into **`experiments/generalization/artifacts/`**
(`phase1_best*.json`, `phase2_final_hparams*.json`, `hparam_best.*`,
`phase3_summary_*.csv` with char and sector columns); plot via
**`utils_viz/plot_generalization.py`** → **`results/anal_figs/generalization/`**
(char/sector 1x2 panels; default PNG; **`--save-pdf`** for PDF).

**Dependency rule:** No imports from `utils/` inside `collect_results.py` beyond what a normal script would use; training logic stays in `utils/` + `train_model.py`.

**Pipelines:** **`run_all_scales_2gpu.sh [short|full]`** (default **short**):
**short** = smaller Phase 1 (four scales, including 40h), no Phase 2, inlined
`collect_results` + `emit_hparams_shared`, `phase2_final_hparams_short.json`,
CSV tag **`_short_ep${NUM_EPOCHS}`**; **full** = larger Phase 1 → inlined
**`collect_results.py phase1`** → **`phase2_lr_check.sh`** per scale →
`phase2_final_hparams.json` → Phase 3. The single-stage full-grid search is
defined by **`hparam_full_grid.py`** and run on Amarel with
**`experiments/amarel/submit_hparam_full_grid_batches.sh`**. Per-phase training
launchers: **`experiments/generalization/phase1_gawf_search.sh`**,
**`phase2_lr_check.sh`**, **`phase3_train_scale.sh`**. Ad-hoc aggregate /
local-Phase3-only tools live under **`experiments/archive/`** (not used by the
default `run_all` flow). See **`AGENTS.md` section 8** and **`workflow.mdc`**.
Amarel logs for full and smoke-test submissions are written under
**`experiments/amarel/artifacts/`**; the 4h/5-epoch smoke test runs only the four
model families at fixed `hidden_size=256`, `lr=5e-4`, `wd=1e-4`.
Local two-GPU launchers live under **`experiments/local/`** and reuse the same
`hparam_full_grid.py` task-id mapping while replacing Slurm arrays with local
`CUDA_VISIBLE_DEVICES` process scheduling.

**Doc maintenance:** Human-requested edits to `.cursor/rules` should update **`AGENTS.md`** and this file in the same change unless scoped otherwise (`workflow.mdc` **Doc alignment**).
