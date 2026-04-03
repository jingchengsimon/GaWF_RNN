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
│   └── paper_figs/          # Publication-quality figures
├── results/
│   ├── train_data/<suffix>/ # Checkpoints (.pth) + metrics (.pkl, .json)
│   ├── anal_data/<module>/  # Analysis outputs (.npy, .npz, .json)
│   └── anal_figs/<module>/  # Figure outputs (.png)
├── train_model.py     # CLI training entry-point
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
- Feedback vector: `fb ∈ ℝ^(num_classes + num_pos)` — first `num_classes` slots are digit logits,
  remaining `num_pos` slots are sector/position logits.
- Gate: `sigmoid(U @ (fb * V) / gate_tau)`, `gate_tau = 0.5`
- U shape: `(hidden_size, fb_dim)`,  V shape: `(fb_dim, input_size + hidden_size)`
- `prev_feedback` is a runtime buffer (not a parameter); **skip it** when loading state_dicts.

### 2.4 Label Format
| Mode | `labels` shape | `labels[..., 0]` | `labels[..., 1]` |
|------|---------------|-------------------|------------------|
| sector (default) | `(B, T, 2)` int64 | fg digit 0–9 | sector 0–8 |
| coordinate | `(B, T, 3)` float32 | fg digit | x coord | y coord |
| predict_all_chars | `(B, T, max_chars)` int64 | chars in order; -1 = pad | — |

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
| `--dropout` | float+ | `train_model.py` only: dropout *p* for CNN (`dropout2d`) and middle path; repeat for grid (default `[0]`) |

**Do not rename existing analysis/visualisation arguments** when extending those scripts. For `train_model.py` hyperparameter search, use **`--dropout`** (singular), not `--dropouts`.

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
- Combo format: `"model_type,hidden_size,lr,wd,dropout,stage_label,seed"` (7 fields, comma-separated).
- Log files: `logs_hparam/job{N}_{model}_{stage}_s{seed}.log`.
- Always pass `--use_acceleration` and `--use_sector_mode` flags.
- Result suffix encodes the sweep stage, e.g. `lr_search_sector`.

---

## 8. Paths & Environment

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
GPU allocation: `CUDA_VISIBLE_DEVICES` set by `pick_cuda_device_index()` at import time.

---

## 9. Forbidden Patterns

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

## 10. Quick Checklist for New Scripts

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
