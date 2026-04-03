# CONVENTIONS.md — Naming Conventions & Abbreviation Table

## 1. Abbreviation Dictionary

| Abbreviation | Full term | Context |
|---|---|---|
| `gawf` | Gated-with-Feedback | Model family name |
| `rnn` | Recurrent Neural Network | Model type key in CLI |
| `gru` | Gated Recurrent Unit | Model type key |
| `lstm` | Long Short-Term Memory | Model type key |
| `ann` | Artificial Neural Network (feedforward) | Model type key |
| `dann` | Dendritic ANN | `DendriticANNConv` |
| `ffn` | Feedforward Network | `FeedForwardConv` |
| `cnn` | Convolutional Neural Network | Encoder stage |
| `fb` | Feedback | Feedback vector / buffer |
| `fb_dim` | Feedback dimension | `num_classes + num_pos` |
| `ih` | Input-to-Hidden | RNN weight matrix |
| `hh` | Hidden-to-Hidden | RNN recurrent weight |
| `hparam` | Hyperparameter | Used in log/result dir names |
| `wd` | Weight decay | CLI `--weight_decays` (`train_model.py`) |
| `do` | Dropout probability *p* | CLI `--dropout` (`train_model.py`); filename `_do{value}` |
| `lr` | Learning rate | CLI and filename |
| `h` | Hidden size | Filename suffix, e.g. `h256` |
| `acc` | Acceleration / accuracy | Context-dependent (filename: acceleration) |
| `sector` | 3×3 spatial sector | Label mode name |
| `coord` | Coordinate regression | Label mode name |
| `allchars` | Predict all characters | Label mode name |
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

## 3. File Suffix Conventions

### Checkpoint filenames
```
{model_type}_{label_mode}{acc_suffix}_h{hidden}_lr{lr}_wd{wd}_do{do}{fb_suffix}_model.pth
```
Example: `gawf_sector_acc_h256_lr0.0005_wd0.0001_do0_fb50_model.pth`

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
`RNNConv`, `GRUConv`, `LSTMConv`, `GaWFRNNConv`, `DendriticANNConv`, `FeedForwardConv`.

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
