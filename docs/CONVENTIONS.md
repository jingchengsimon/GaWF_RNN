# CONVENTIONS.md — Naming Conventions & Abbreviation Table

## 1. Abbreviation Dictionary

| Abbreviation | Full term | Context |
|---|---|---|
| `gawf` | Gated-with-Feedback | Model family name |
| `gawf` | Gated-with-Feedback | Unified single/multi-layer model selected by `--num_layers` |
| `rnn` | Recurrent Neural Network | Model type key in CLI |
| `gru` | Gated Recurrent Unit | Model type key |
| `lstm` | Long Short-Term Memory | Model type key |
| `ann` | Artificial Neural Network (feedforward) | Model type key |
| `dann` | Dendritic ANN | `DendriticANNConv` |
| `ffn` | Feedforward Network | `FeedForwardConv` |
| `mamba` | Mamba SSM block | Model type key |
| `ssm` | State Space Model | Model type key |
| `cnn` | Convolutional Neural Network | Encoder stage |
| `fb` | Feedback | Feedback vector / buffer |
| `fb_dim` | Feedback dimension | `dz` for projected GaWF; direct feedback uses output or hidden dimensions |
| `ih` | Input-to-Hidden | RNN weight matrix |
| `hh` | Hidden-to-Hidden | RNN recurrent weight |
| `hparam` | Hyperparameter | Used in log/result dir names |
| `wd` | Weight decay | CLI `--weight_decays` (`train_model.py`) |
| `cdo` | CNN dropout probability *p* | CLI `--cnn_dropout` (`train_model.py`); filename `_cdo{value}` |
| `rdo` | RNN/middle-path dropout *p* | CLI `--rnn_dropout` (`train_model.py`); filename `_rdo{value}` |
| `do` | (Legacy) unified dropout *p* | Old stems only: `_do{value}`; maps to both paths when parsing |
| `lr` | Learning rate | CLI and filename |
| `h` | Hidden size | Filename suffix, e.g. `h256` |
| `dmodel` | Mamba/SSM sequence width | Filename suffix for `mamba` / `ssm`, e.g. `dmodel170` |
| `state` | SSM latent state size | Filename suffix for `ssm`, e.g. `state189` |
| `dz` | Projected GaWF feedback context dimension | Optional GaWF filename suffix, e.g. `dz8` |
| `L` | Recurrent layer count | Multi-layer filename suffix, e.g. `L2` |
| `acc` | Acceleration / accuracy | Context-dependent (filename: acceleration) |
| `sector` | 3×3 spatial sector | Label mode name |
| `coord` | Coordinate regression | Label mode name |
| `allchars` | Predict all characters | Label mode name |
| `glob` | Global (sum correct / sum frames) | Training metrics vs batch-mean curves |
| `fg_switch` | Foreground switch flag / window | Label column; pre5/post5 eval windows |
| `pre5` / `post5` | Five frames before / around fg switch | Transition-window accuracy (sector eval) |
| `agg` | Aggregation | `space` or `feature` axis collapse |
| `trans` | Transformation matrix | `trans_ih`, `trans_hh` |
| `outer` | Outer product component | Rank-1 decomposition term |
| `comp` | Component | Decomposition component index |
| `npz` | NumPy compressed archive | File format `.npz` |
| `pth` | PyTorch state dict | Checkpoint file `.pth` |
| `pkl` | Pickle | Training results file |
| `mmap` | Memory-mapped array | NumPy `mmap_mode='r'` |
| `FDR` | False Discovery Rate | Statistical correction (Benjamini-Hochberg) |
| `BH` | Benjamini-Hochberg | FDR correction method |
| `cosine` | Cosine similarity | Channel/unit reorder criterion |
| `dPCA` | Demixed PCA | Dimensionality reduction (utils_anal) |
| `UMAP` | Uniform Manifold Approximation | Dimensionality reduction (utils_viz) |
| `dpca` | dPCA | Module/script name |
| `umap` | UMAP | Module/script name |

## 2. Directory Name Conventions

| Directory | Purpose | Naming Rule |
|-----------|---------|-------------|
| `results/train_data/<suffix>/` | Training outputs | suffix = CLI `--result_suffix` |
| `results/anal_data/<module>/` | Analysis arrays | module = script basename |
| `results/anal_figs/<module>/` | Figures | module = script basename |
| `logs_hparam/` | Hparam sweep logs | always at project root |
| `experiments/generalization/artifacts/gen_hparam_full_grid/` | Full-grid hparam summaries and status | `hparam_best.*`, `hparam_full_grid_status.*`, `failed_task_ids.txt` |
| `experiments/amarel/artifacts/<run>/` | Amarel Slurm logs | one stdout/stderr pair per array task; e.g. `hparam_full_grid`, `hparam_4h_5epoch_test` |

## 2.1 Experiment Script Conventions

| Location | Pattern | Example |
|----------|---------|---------|
| Full-grid hparam utility | `experiments/generalization/hparam_full_grid.py` | Subcommands: `emit-task`, `validate`, `status`, `summarize` |
| Amarel launchers | `experiments/amarel/*hparam_full_grid*.sh` | `submit_hparam_full_grid_batches.sh`, `rerun_hparam_full_grid_failed.sh` |
| Amarel smoke-test launchers | `experiments/amarel/*hparam_4h_5epoch_test*.sh` | 4h-only, 5-epoch, 4-model fixed-hparam submission/status check |
| Amarel probe | `experiments/amarel/probe_amarel_slurm_limits.sh` | Slurm `sbatch --test-only` probe |
| Local hparam launchers | `experiments/local/*hparam*.sh` | `run_hparam_full_grid_2gpu.sh --scale 10 20 40`, local 2-GPU smoke/full-grid runs |

## 3. File Suffix Conventions

### Checkpoint filenames
```
{model_type}_{label_mode}{acc_suffix}_h{hidden}_lr{lr}_wd{wd}_cdo{cdo}_rdo{rdo}{dz_suffix}{fb_suffix}_model.pth
```
Example: `gawf_sector_acc_h256_lr0.0005_wd0.0001_cdo0_rdo0.5_dz32_fb50_model.pth`

Single-layer GaWF `dz` suffix is optional:
- with explicit feedback dim: `_dz{value}`
- legacy behavior (no explicit feedback dim): omit `_dz`

Multi-layer GaWF stems always add layer count. Projected runs also include `dz`:
```
gawf_{label_mode}{acc_suffix}_h{hidden}_L{layers}_lr{lr}_wd{wd}_cdo{cdo}_rdo{rdo}{_dz{dz}}{fb_suffix}_model.pth
```
`gawf` defaults to `L=1`; multi-layer direct feedback uses `--num_layers > 1`, and
`--dz > 0` enables projected feedback.

Mamba and SSM stems use model-native width/state names instead of `h`:
```
mamba_{label_mode}{acc_suffix}_dmodel{d_model}_lr{lr}_wd{wd}_cdo{cdo}_rdo{rdo}_model.pth
ssm_{label_mode}{acc_suffix}_dmodel{d_model}_state{state_size}_lr{lr}_wd{wd}_cdo{cdo}_rdo{rdo}_model.pth
```

Legacy (pre-split): `..._do{value}_...` — still supported by `parse_hparams_from_filename` for old checkpoints.

### Analysis output tags
```
tag = f"{mode}{selected_idx}_{agg}"
# e.g. "sector3_space", "digit7_feature"
```

### Figure filenames
```
{mode}{idx}_{agg}_{descriptor}.png
# e.g. "sector3_space_avg_gate_allcomp.png"
```

## 4. Python Identifier Conventions

### Class names
`PascalCase`. Model classes end in `Conv` when they include a CNN encoder:
`RNNConv`, `GRUConv`, `LSTMConv`, `GaWFRNNConv`, `MultiLayerGaWFRNNConv`,
`DendriticANNConv`, `FeedForwardConv`.

### Function names
`snake_case` starting with a verb:
- `compute_*` — pure calculation, returns array(s)
- `build_*` — constructs an object (model, loader, dataset)
- `export_*` — script-level: runs inference and writes to disk
- `load_*` / `save_*` — I/O wrappers
- `parse_*` — parsing logic
- `plot_*` — generates and saves a figure
- `finalize_*` — reduces accumulated stats to final metrics
- `format_*` — returns a display string

### Private helper names
Single underscore prefix: `_agg_ih`, `_draw_boundaries`, `_save_digit_boundaries`.

### Loop variables
Consistent single-letter conventions:
```python
for sidx in range(n_total):      # sample index
    for t in t_indices:           # time index
        for b in range(batch_size): # batch index
            for d in range(num_digits): # digit/component
                for c in range(num_channels): # channel
```

## 5. Tensor Dimension Order

All tensors follow PyTorch conventions:
```
(B, T, C, H, W)  — batch, time, channel, height, width  [input frames]
(B, T, 2)        — batch, time, [digit_id, sector_id]   [labels, sector mode]
(B, H, input)    — batch, hidden, input_size             [trans_ih]
(n_comp, H, I)   — component, hidden, input              [outer_all_acc]
(C, D)           — channels, digits                      [mean_activation in CNN stats]
```
