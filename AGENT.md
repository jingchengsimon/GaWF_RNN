# AGENT.md — aim3_RNN Project Specification
> Authoritative constraint file for Claude Code, Codex, Cursor, and any LLM-assisted code generation.
> Read this file in full before generating, editing, or refactoring any code in this project.

**Import propagation:** Whenever you rename, move, split, or delete a module or public symbol, search the repository and update every affected `import` and call site in the **same** change (do not leave stale imports).

---

## 0. Project Overview

This project trains and analyses **CNN-RNN / GaWF (Gated-Weight-on-Feedback) neural network models**
for a temporal task with multi-character visual recognition. Stimuli are multi-frame grayscale image sequences;
labels encode foreground digit identity and 2-D position (sector or coordinate).

**Entry point:** `train_model.py`
**Package roots:** `utils/`, `utils_anal/`, `utils_viz/`
**Conda env:** `aim3_rnn`  |  **Python ≥ 3.10**  |  **PyTorch ≥ 2.0**

---

## 1. Repository Layout

```
.
├── source/                  # Stimulus generation scripts (do not modify during training)
├── utils/                   # Training pipeline (models, engine, helpers, acceleration)
│   ├── train_rnn_core.py    # BaseConvSequenceModel, RNNConv, GRUConv, LSTMConv
│   ├── train_gawf_core.py   # GaWFRNNConv (feedback gating)
│   ├── train_ann_core.py    # DendriticANNConv, FeedForwardConv
│   ├── train_helpers.py     # I/O, logging, arg parsing, seeding
│   ├── train_rnn_engine.py  # setup_training_components, begin_epoch, train_batch, eval_train_subset, eval_valid, evaluate_epoch
│   ├── train_acceleration.py# AccelerationConfig, TrainStepper, build_loaders
│   ├── train_sector.py      # Loss / metrics for sector (single-char) mode
│   └── train_predict_all_chars.py  # Loss / metrics for all-chars mode
├── utils_anal/              # Post-training analysis scripts (export_*.py, …)
├── utils_viz/               # Visualisation scripts (gate_avg*.py, …)
│   ├── plot_generalization.py  # Generalization figures (char/sector gap + train/val acc vs scale)
│   └── paper_figs/          # Publication-quality figures
├── results/
│   ├── train_data/<suffix>/ # Checkpoints (.pth) + metrics (.pkl, .json)
│   ├── anal_data/<module>/  # Analysis outputs (.npy, .npz, .json)
│   └── anal_figs/<module>/  # Figure outputs (.png)
├── train_model.py           # CLI training entry-point
├── experiments/generalization/  # Generalization launchers + hparam_full_grid.py (see §8)
├── experiments/amarel/          # Amarel/Slurm submission, probe, status, and rerun helpers
├── hparam_search.sh         # Hyperparameter sweep launcher (zsh)
└── visualize_batch.sh       # Batch visualisation launcher (zsh)
```

**Rule:** Never place new analysis or visualisation logic inside `utils/`.
Analysis → `utils_anal/`; visualisation → `utils_viz/`.

---

## 2. Architecture Constraints

Refactors that touch exported names must propagate: grep for old symbols and fix imports in `train_model.py`, `utils/`, `utils_anal/`, `utils_viz/`, notebooks, and shell launchers as needed.

### 2.1 Model Hierarchy
```
nn.Module
└── BaseConvSequenceModel          (utils/train_rnn_core.py)
    ├── BaseRNNConv → RNNConv / GRUConv / LSTMConv
    └── GaWFRNNConv                (utils/train_gawf_core.py)
        └── feedback loop via U, V parameters + middle_gawf()
BaseMergeConvModel                 (utils/train_ann_core.py)
└── DendriticANNConv / FeedForwardConv
```

### 2.2 CNN Encoder (fixed "large" config)
- `conv1`: 2→32 ch, same-padding  →  `MP1` 2×2  →  `LNorm1` [32,48,48]
- `conv2`: 32→64 ch              →  `MP2` 4×4  →  `LNorm2` [64,12,12]
- `conv_reduce`: 64→32 ch (1×1)  →  `pool_reduce` AdaptiveAvgPool2d((6,6))
- `encoder_flatten_size` = 32 × 6 × 6 = **1152**

**Do not change encoder output shape** without updating all downstream analysis scripts
that assume spatial dimensions (6,6) and 32 feature channels.

### 2.3 GaWF Gating
- Feedback vector: `fb ∈ ℝ^(fb_dim)`. For projected GaWF, `fb_dim = dz`
  (`--feedback_dim` / `--dz`).
- Legacy compatibility: for single-layer `gawf`, if `--feedback_dim` is omitted,
  `fb_dim = num_classes + num_pos`.
- Optional projector: `proj_out` maps output logits `y ∈ ℝ^(num_classes + num_pos)` to `fb ∈ ℝ^(dz)`.
- Gate: `sigmoid(U @ (fb * V) / gate_tau)`, `gate_tau = 0.5`
- U shape: `(hidden_size, fb_dim)`, V shape: `(fb_dim, input_size + hidden_size)`
- `prev_feedback` is a runtime buffer (not a parameter); **skip it** when loading state_dicts.
- Optimizer grouping: for single-layer `gawf`, U/V are placed in a no-weight-decay
  parameter group but use the same learning rate as the rest of the model. For
  `gawf_multi`, base parameters use the searched learning rate directly, while
  U/V are no-weight-decay and use the configured feedback learning-rate scale
  (`--gawf_multi_feedback_lr_scale`, default `0.1`).
- Multi-layer GaWF uses separate CLI model type `gawf_multi` and class
  `MultiLayerGaWFRNNConv`; default `--gawf_layers 2`. If `--dz` is omitted or set
  to `0`, multi-layer GaWF uses direct feedback: non-final layers receive the
  detached adjacent higher layer's previous hidden output (`hidden_size` dim), and
  the final layer receives detached previous output logits (`num_classes + num_pos`
  dim). If `--dz > 0`, each recurrent layer has its own U/V pair (`U_layers`,
  `V_layers`) and receives projected feedback with dimension `dz`.

### 2.4 Label Format
| Mode | `labels` shape | `labels[..., 0]` | `labels[..., 1]` |
|------|---------------|-------------------|------------------|
| sector (default) | `(B, T, 2)` int64 | fg digit 0–9 | sector 0–8 |
| coordinate | `(B, T, 3)` float32 | fg digit | x coord | y coord |
| predict_all_chars | `(B, T, max_chars)` int64 | chars in order; -1 = pad | — |

**Sector fair-eval extensions (single-char only):** label TSV may include `fg_switch` (0/1). The dataset derives per-frame **pre5** / **post5** masks (see `utils/train_sector.compute_fg_transition_masks`) and, when present, training pickles store strict **global** accuracies (`glob_*`) and fg-switch-window accuracies (`fg_switch_pre5_*`, `fg_switch_post5_*`) alongside the legacy batch-mean curves. `predict_all_chars` runs are unchanged.

---

## 3. Coding Conventions

### 3.1 General
- **Python 3.10+** type hints on all public functions. Use `from __future__ import annotations`.
- Every module starts with a **docstring** describing purpose, inputs, and outputs.
- No wildcard imports (`from x import *`).
- Line length: **100** characters (Black default).
- Use `np.float32` / `np.int64` for all saved arrays; cast explicitly before `np.save`.

### 3.2 Script Structure (Analysis & Visualisation)
Every script in `utils_anal/` and `utils_viz/` **must** follow this structure:

```python
"""<one-line summary>

<Extended description: what this script does, what it reads, what it writes>

Outputs (in --save_dir / --output_dir):
- <filename>  (<shape>), <dtype>  — <description>
"""
from __future__ import annotations
# stdlib
import argparse, os, sys
from typing import ...
# third-party
import numpy as np
import torch
# project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# local project imports ...


def parse_args() -> argparse.Namespace: ...
def <core_logic_function>(...) -> ...: ...
def main() -> None: ...

if __name__ == "__main__":
    main()
```

### 3.3 Argument Naming Conventions
| Argument | Type | Meaning |
|----------|------|---------|
| `--ckpt` | str | Path to `.pth` checkpoint |
| `--save_dir` | str | Output directory for analysis `.npy/.npz/.json` |
| `--data_dir` | str | Input data/analysis directory |
| `--device` | str | `"cuda"` or `"cpu"` |
| `--seed` | int | Random seed (default 42) |
| `--batch_size` | int | DataLoader batch size |
| `--use_mmap` | flag | Load stimuli with `mmap_mode='r'` |
| `--use_sector_mode` | flag | Use 3×3 sector position labels |
| `--predict_all_chars` | flag | Predict all characters (fg+bg) |
| `--sector` | int 0–8 | Target sector index |
| `--digit` | int 0–9 | Target foreground digit |
| `--agg` | str `space\|feature` | Input axis aggregation mode |
| `--cnn_dropout` | float+ | `train_model.py` only: CNN encoder `dropout2d` *p*; repeat for grid (default `[0]`) |
| `--rnn_dropout` | float | `train_model.py` only: middle-path dropout *p* after ReLU (RNN/GaWF/FFN); single value (default `0.5`) |
| `--mamba_d_models` | int+ | `train_model.py` only: Mamba sequence width `d_model`; repeat for grid (default `[170]`) |
| `--ssm_d_models` | int+ | `train_model.py` only: SSM sequence feature width `d_model`; repeat for grid (default `[256]`) |
| `--ssm_state_sizes` | int+ | `train_model.py` only: diagonal SSM latent state size; repeat for grid (default `[189]`) |
| `--feedback_dim` / `--dz` | int | `train_model.py` only, GaWF: feedback context dimension `dz`; default `None` keeps single-layer legacy `num_classes + num_pos`; for `gawf_multi`, `None` or `0` disables projectors and values `>0` enable projected feedback |
| `--gawf_layers` | int | `train_model.py` only, `gawf_multi`: recurrent layer count; default `2`, must be `>=2` |
| `--data_suffix` | str | Suffix for **train** (and default val): `stimulus_reg-train-<suffix>.npy` / `stimulus_reg-validation-<suffix>` |
| `--eval_data_suffix` | str | Suffix for **validation only**; empty → same as `--data_suffix` (use for train/val scale mismatch, e.g. 4h train + 40h val) |
| `--patience` | int | Early stopping on fair **`val_acc_char`** after each epoch; **`0` disables**; best weights restored before save (default `15`) |

**Do not rename existing analysis/visualisation arguments** when extending those scripts. For `train_model.py` hyperparameter search, use **`--cnn_dropout`** (grid) and **`--rnn_dropout`** (single), not legacy `--dropout` / `--dropouts`.

### 3.4 Device & dtype Handling
```python
# Always check MPS/CUDA availability; never assume float64 on MPS
if inputs.dtype == torch.float64:
    inputs = inputs.float()
inputs = inputs.to(device, non_blocking=pin_memory)
```

### 3.5 DataLoader for mmap Data
When `use_mmap=True`: always set `num_workers=0` and `pin_memory=False`.

### 3.6 Model Loading Pattern
```python
state_dict = torch.load(ckpt_path, map_location=device)
state_dict = {k: v for k, v in state_dict.items() if k != "prev_feedback"}
model.load_state_dict(state_dict, strict=False)
```
Always print `missing_keys` and `unexpected_keys` after loading.

### 3.7 Result File Naming
Analysis outputs follow: `<descriptor>_<tag>.npy` where `tag = f"{mode}{idx}_{agg}"`.
Figures follow: `<mode><idx>_<agg>_<descriptor>.png`.
Metadata JSON follows: `<descriptor>_meta_<tag>.json`.
Single-layer GaWF checkpoints may include optional `_dz{value}` when `--feedback_dim`
is explicitly set. Multi-layer GaWF checkpoints use prefix `gawf_multi_` and include
`_L{layers}`; projected multi-layer runs additionally include `_dz{value}`.

---

## 4. Analysis Module Conventions (`utils_anal/`)

- Each script exports **one logical result set** per run (keyed by `--sector` or `--digit`).
- Save arrays as `.npy` (single array) or `.npz` (multiple related arrays).
- Always save a companion `*_meta_*.json` with: `mode`, `selected_idx`, `n_frames`,
  `n_samples`, `hidden_size`, `input_size`, `agg`, `ckpt` (absolute path), `n_feat`,
  `h_sp`, `w_sp`, `agg_shape`.
- Accumulate in `float64`, cast to `float32` only before saving.
- Progress prints every 200 samples: `[{sidx+1}/{n_total}] qualifying: {n_samples} | frames: {n_frames}`.
- Raise `RuntimeError` (not silent return) when zero frames match the filter.

### Shared helpers to reuse (do not re-implement):
```python
from utils_anal.export_gate_sample import build_model_from_ckpt, build_test_dataset
```
These two functions are the canonical entry points for model loading and dataset construction
in all analysis scripts.

---

## 5. Visualisation Module Conventions (`utils_viz/`)

- Use `matplotlib.use("Agg")` before importing `pyplot` (headless rendering).
- Default figure DPI: **150**; save with `bbox_inches="tight"`, `pad_inches=0.06`.
- Color conventions:
  - Diverging heatmaps: `"RdBu_r"`, symmetric `vmin=-vmax / vmax=vmax`.
  - Sequential heatmaps: default `viridis` unless overridden by `--cmap`.
- Spatial axis labelling: x = "Hidden unit (npz row index)", y = spatial/feature label.
- Always call `plt.close(fig)` after saving to free memory.
- Channel/unit reordering:
  - Feature channels: load from `channel_order_by_cosine_similarity.npy` (fallback to natural order).
  - Hidden units: load from `sorted_npz_order.npy` in `--conn_dir`; apply `outer_all[:, :, idx]`.
- Boundary lines: red, `linewidth=0.7`; sector highlight: red `axhline` pairs.

### Figure layout rule:
When plotting N components + sum + full (N+2 panels), use `n_cols=3`, 
`n_rows = ceil((N+2)/3)`. Hide unused axes with `ax.axis("off")`.

---

## 6. Training Conventions (`utils/`)

- **No new model classes** should be added directly to `train_rnn_core.py` unless they
  extend `BaseConvSequenceModel` and use the same encoder architecture.
- All new model types must be registered in `get_model_classes()` in `train_helpers.py`.
- Loss functions must be constructed via `build_loss_fn_*()` factories in
  `train_sector.py` or `train_predict_all_chars.py`; do not inline loss logic in the engine.
- The training loop in `train_model.py` is a skeleton; heavy logic lives in `train_rnn_engine.py`.
- `AccelerationConfig` is the single source of truth for AMP/grad-accum settings;
  never add `if use_acceleration` branches inside the training loop.

---

## 7. Hyperparameter Search (`hparam_search.sh`)

- Shell: **zsh** with `setopt KSH_ARRAYS`.
- Combo format: `"model_type,hidden_size,lr,wd,cnn_dropout,stage_label,seed"` (7 fields, comma-separated). Set **`rnn_dropout`** separately in the launch command if it must differ from the default.
- Log files: `logs_hparam/job{N}_{model}_{stage}_s{seed}.log`.
- Always pass `--use_acceleration` and `--use_sector_mode` flags.
- Result suffix encodes the sweep stage, e.g. `lr_search_sector`.

---

## 8. Generalization experiment launchers (`experiments/generalization/`)

Orchestration for **train-scale vs fixed 40h validation** studies. Training always via **`train_model.py`**. Helper **`experiments/generalization/collect_results.py`** (stdlib only) aggregates `*_metrics.json`; figures via **`utils_viz/plot_generalization.py`** (char/sector 1x2 panels; default PNG; **`--save-pdf`** for PDF too).

### 8.1 Training flags used by these launchers

- **`--eval_data_suffix`** — e.g. `40h-float32` so validation uses `stimulus_reg-validation-40h-float32.*` while **`--data_suffix`** sets train hours (`4h-float32`, …).
- **`--patience`** — early stop on fair val char accuracy; **`0`** = run full **`--num_epochs`**.
- **Multi-job logs** — tqdm / logger lines are prefixed with `[result_suffix|eNNN|model_type]`.
- **GPU** — if **`CUDA_VISIBLE_DEVICES`** is already set, `train_model.py` does **not** override it (supports parallel launchers).

### 8.2 `run_all_scales_2gpu.sh [short|full]` (default **short**)

One entry: **`bash experiments/generalization/run_all_scales_2gpu.sh`**, **`… short`**, or **`… full`** (no separate `*_short.sh` wrapper).

- **short** — **Phase 1** — `phase1_gawf_search.sh <4h|10h|20h|40h> short` (reduced grid; **`gen_phase1_short_gawf_*`**; **40h** is trained and searched like the other scales, val **40h**). Then **`run_all` inlines** `collect_results.py phase1` (four short dirs) + **`emit_hparams_shared`**. **No Phase 2.** **Phase 3** — `phase3_train_scale.sh` each scale **`short`**; all four models share **`results/train_data/gen_phase3_short_<scale>_ep<N>`** (one folder per scale+epoch). **Plot** — **`--csv_tag`** = **`${CSV_TAG}_ep${NUM_EPOCHS}`** (default **`_short_ep100`**).

- **full** — **Phase 1** — `phase1_gawf_search.sh` … **`full`**. **Aggregate** in **`run_all`**: `collect_results.py phase1` → **`phase1_best.json`**. **Phase 2** — `phase2_lr_check.sh` (two GPUs) → **`phase2_final_hparams.json`**. **Phase 3** — `phase3_train_scale.sh` … **`full`** → shared **`results/train_data/gen_phase3_<scale>_ep<N>`**. **Plot** — **`--csv_tag _ep${NUM_EPOCHS}`** (default **`NUM_EPOCHS=100`**).

- **Legacy Phase3 layout** (separate folder per model) — run **`python experiments/generalization/merge_phase3_result_dirs.py --apply`** once to merge **`train_data`** and **`train_figs`** into the new naming.

- **Offline / ad-hoc** (not used by default orchestration) — `experiments/archive/run_phase1_aggregate.sh [short|full]`, `experiments/archive/run_local_phase3.sh [short|full|…]`.

- **`collect_results.py phase1_short`**: legacy **three** Phase-1 dirs + **`--preset_40h_dir`**; use if you do not have a local **`gen_phase1_short_gawf_40h`** (or `gen_phase1_gawf_40h`) run.

### 8.3 `collect_results.py` subcommands

| Command | Output |
|---------|--------|
| `phase1` | Four Phase-1 dirs in scale order; default **`--out`** `phase1_best.json`, or pass **`--out phase1_best_short.json`** for short runs |
| `phase1_short` | *Legacy:* three dirs + preset 40h → `phase1_best_short.json` |
| `emit_hparams_shared` | `phase2_final_hparams_short.json` |
| `phase2` | updates `phase2_final_hparams.json` |
| `phase3` | `phase3_summary_<scale>[out_tag].csv` with char aliases plus sector columns (`train_acc_sector`, `val_acc_sector`, `overfit_gap_sector`) |
| `phase3_import` | one scale CSV from a single metrics directory (optional; e.g. legacy 40h import) |

Legacy metrics without `train_acc_at_best_val` / `val_acc_at_best`: CSV uses `best_train_acc_char` / `best_val_acc_char` fallbacks; legacy Phase3 JSON without sector-at-best fields can be backfilled from `.pkl` via `experiments/archive/backfill_phase3_sector_metrics.py --apply`.

### 8.4 Single-stage full-grid hparam search

`experiments/generalization/hparam_full_grid.py` defines the 1024-run grid:
4 scales (`4h`, `10h`, `20h`, `40h`) × 4 models (`rnn`, `lstm`, `gru`, `gawf`) ×
hidden sizes (`64`, `128`, `256`, `512`) × LR (`1e-4`, `5e-4`, `1e-3`, `5e-3`) ×
WD (`0`, `1e-5`, `1e-4`, `1e-3`). All runs use sector mode, acceleration,
`cnn_dropout=0.0`, `rnn_dropout=0.5`, `num_epochs=100`, `patience=15`, `seed=42`,
and fixed **40h validation** via `--eval_data_suffix 40h-float32`. Each task writes
to `results/train_data/gen_hparam_full_grid/task_<id>/`.

Best hparams are selected per scale × model by highest **`val_acc_at_best`** (char);
sector metrics are reported from the same selected run. `hparam_full_grid.py summarize`
writes `hparam_best.{json,csv}`, `hparam_best_summary.md`, all-trial CSV, and
`phase3_summary_<scale>_hparam_full_grid.csv` so `utils_viz/plot_generalization.py`
can draw overfit gap, train acc, and validation acc for char and sector.

Amarel helpers live in `experiments/amarel/`: `probe_amarel_slurm_limits.sh`,
`submit_hparam_full_grid_batches.sh`, `run_hparam_full_grid_array.sh`,
`check_hparam_full_grid_status.sh`, `rerun_hparam_full_grid_failed.sh`, and
`summarize_hparam_full_grid.sh`. The default Amarel settings are `gpu-redhat`,
`account=general`, `gpu:1`, `cpus-per-task=4`, `mem=16G`, `time=72:00:00`, 200
tasks per batch, and array concurrency `%96`; batch submission waits for each
batch before submitting the next to respect submit limits. Failed or missing
tasks are recorded and rerun explicitly; there is no automatic retry loop.
When submitting any training job on Amarel through Slurm, explicitly export
`AIM3_NUM_WORKERS=12` and `AIM3_PIN_MEMORY=1` in the submission environment so
the run logs show `num_workers=12, pin_memory=True`. Do this at submission time
only; do not bake these values into local launchers or training scripts because
interactive/local remote runs can OOM with the same DataLoader settings. Always
pair these DataLoader acceleration settings with explicit Slurm resources:
`--cpus-per-task=16`, `--mem=64G`, `--gres=gpu:1`, and
`--constraint=adalovelace`, instead of relying on the default 4 CPU / 16G
allocation or mixed GPU architectures. Use `--nodelist=gpuk018` only when an
experiment specifically needs to reproduce the old p0 node-level execution
path. Amarel Slurm scripts submitted by Codex must activate the project conda
environment with
`source /home/js3269/enter/etc/profile.d/conda.sh` followed by
`conda activate aim3_rnn`; do not use `module` or a `faw_rnn_env` virtualenv.
After submitting from Codex, verify and report the requested resources and
environment settings in the user-facing update or final response.
Amarel Slurm stdout/stderr and submission logs live under
`experiments/amarel/artifacts/`. For launch validation before the full 1024-run
campaign, use `submit_hparam_4h_5epoch_test.sh` and
`check_hparam_4h_5epoch_test_status.sh`; this runs only four 4h/5-epoch jobs
(`rnn`, `lstm`, `gru`, `gawf`) at `hidden_size=256`, `lr=5e-4`, `wd=1e-4`.
For training-metrics visualizations, always use `visualize_batch.sh` from the
repo root so plots share the same style and output layout; do not create ad hoc
plotting scripts for saved metrics unless the user explicitly asks for a custom
figure. Activate `aim3_rnn` first on Amarel before running visualization so the
script uses the environment with `numpy`/matplotlib installed.
`submit_hparam_full_grid_batches.sh --scale <4|10|20|40|all ...>` submits only
the selected scale slices when requested, e.g. `--scale 10 20 40` skips 4h.
Local two-GPU debugging uses
`experiments/local/run_hparam_full_grid_2gpu.sh --scale <4|10|20|40|all ...>`,
which reuses the same `hparam_full_grid.py` task mapping but runs at most two
training processes concurrently via `CUDA_VISIBLE_DEVICES`.

---

## 9. Paths & Environment

```
# Data resolution order (train_helpers.PathHelper.get_base_path):
1. --data_dir CLI argument
2. $AIM3_STIMULI_PATH environment variable
3. $FAW_RNN_DATA_PATH environment variable
4. <repo_root>/stimuli/

# Default checkpoint root:
results/train_data/<result_suffix>/

# Default analysis root:
results/anal_data/<module>/

# Default figure root:
results/anal_figs/<module>/
```

Conda env: `aim3_rnn`  (activate before running any script)
Remote env rule: on both `amarel` and `sjc-remote`, do not run tests or training
with the default shell Python. Activate conda env `aim3_rnn` first, including
inside Slurm scripts and SSH diagnostics.
GPU allocation: if **`CUDA_VISIBLE_DEVICES`** is unset, `train_model.py` may set it via `pick_cuda_device_index()`; **preset `CUDA_VISIBLE_DEVICES` is preserved** (parallel launchers).

### 9.1 Amarel Interactive Command Format

For remote diagnostics, prefer running commands directly over the configured SSH
aliases instead of asking the user to paste commands manually:

- `ssh amarel` for Amarel (`js3269@amarel.rutgers.edu`)
- `ssh sjc-remote` for the user's remote terminal (`sjc@172.26.48.213`)

These hosts are expected to use SSH ControlMaster connection reuse. If the master
connection is active, run remote checks directly from Codex. If SSH requires
interactive authentication and Codex cannot proceed, then provide the user a
single pasteable heredoc block as a fallback.

When providing fallback commands intended to be pasted into an interactive
Amarel or remote shell, wrap the full command block in a single heredoc:

```bash
bash <<'EOF'
# commands go here
EOF
```

Do not split related Amarel diagnostics into separate command snippets. Provide
one combined heredoc block so it can be pasted and run as a single unit. This is
especially important for multi-line checks using loops, shell variables, or
quoted patterns; interactive Amarel/tmux sessions may otherwise execute pasted
commands line-by-line before the full block is entered.

---

## 10. Forbidden Patterns

- ❌ Do not import from `utils_viz/` inside `utils/` or `utils_anal/`.
- ❌ Do not import from `utils_anal/` inside `utils/`.
- ❌ Do not use `print()` for training progress; use `logger.info()`.
- ❌ Do not use `torch.no_grad()` as a persistent model state; scope it to inference blocks.
- ❌ Do not load the full dataset into GPU memory; keep stimuli on CPU / mmap.
- ❌ Do not add cross-batch state to models except via the explicit `prev_feedback` buffer.
- ❌ Do not change `encoder_flatten_size` without a migration comment and updating all
  downstream scripts.
- ❌ Do not use `strict=True` when loading checkpoints (prev_feedback buffer causes mismatches).

---

## 11. Quick Checklist for New Scripts

Before submitting any new `utils_anal/` or `utils_viz/` script, verify:

- [ ] If you touched shared APIs or paths, all repo imports/call sites are updated (no stale references)
- [ ] Module docstring present with Outputs section
- [ ] `from __future__ import annotations` at top
- [ ] `PROJECT_ROOT` sys.path injection block present
- [ ] `parse_args() -> argparse.Namespace` function exists
- [ ] `main() -> None` function exists and is called under `if __name__ == "__main__"`
- [ ] All saved arrays cast to `float32` / `int64` before `np.save`
- [ ] `plt.close(fig)` called after every `fig.savefig()`
- [ ] `os.makedirs(save_dir, exist_ok=True)` called before any file write
- [ ] Zero-frame case raises `RuntimeError`, not silent return
- [ ] Accumulation buffers use `float64`, final arrays cast to `float32`
