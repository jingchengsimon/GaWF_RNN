# Generalization Experiments

These launchers compare training-data scale against a fixed 40h validation set. Training always
uses `train_model.py`; aggregation uses `collect_results.py`; plotting uses
`utils_viz/plot_generalization.py`.

## Shared protocol

- Train scales: 4h, 10h, 20h, and 40h.
- Validation: `--eval_data_suffix 40h-float32` for every scale.
- Official selection metric: highest fair `val_acc_at_best` for character recognition; sector
  metrics come from the same selected epoch/run.
- `--patience 0` disables early stopping; otherwise best weights are restored before saving.
- A preset `CUDA_VISIBLE_DEVICES` is preserved for parallel launchers.

## Three-phase pipeline

Run `bash experiments/generalization/run_all_scales_2gpu.sh [short|full]`; the default is `short`.

### Short

1. `phase1_gawf_search.sh <scale> short` searches all four scales with the reduced grid.
2. `collect_results.py phase1` and `emit_hparams_shared` produce shared short hyperparameters.
3. Phase 2 is skipped.
4. `phase3_train_scale.sh <scale> short` trains all models into one
   `gen_phase3_short_<scale>_ep<N>` directory per scale.
5. Plot with CSV tag `${CSV_TAG}_ep${NUM_EPOCHS}` (default `_short_ep100`).

### Full

1. `phase1_gawf_search.sh <scale> full` runs the full Phase-1 grid.
2. `collect_results.py phase1` writes `phase1_best.json`.
3. `phase2_lr_check.sh` writes `phase2_final_hparams.json`.
4. `phase3_train_scale.sh <scale> full` writes one `gen_phase3_<scale>_ep<N>` directory per
   scale.
5. Plot with CSV tag `_ep${NUM_EPOCHS}`.

Legacy three-directory short aggregation remains available as `phase1_short`. Legacy Phase-3
per-model folders can be migrated with `merge_phase3_result_dirs.py --apply`. Archive-only
aggregate and Phase-3 tools live under `experiments/archive/` and are not part of the default
pipeline.

## Aggregation commands

| Command | Output |
|---|---|
| `phase1` | best Phase-1 row for four ordered scale directories |
| `phase1_short` | legacy three directories plus preset 40h |
| `emit_hparams_shared` | short shared hyperparameters |
| `phase2` | final full-pipeline hyperparameters |
| `phase3` | one scale CSV with character and sector metrics |
| `phase3_import` | one scale CSV from a pre-existing four-model result directory |

Legacy metrics without at-best fields fall back to the best character metrics. Old Phase-3
sector-at-best values can be restored from pickles with the archived backfill utility.

## Single-stage 1024-run grid

`hparam_full_grid.py` maps:

```text
4 scales
x 4 models (rnn, lstm, gru, gawf)
x 4 hidden sizes (64, 128, 256, 512)
x 4 learning rates (1e-4, 5e-4, 1e-3, 5e-3)
x 4 weight decays (0, 1e-5, 1e-4, 1e-3)
= 1024 tasks
```

All tasks use sector mode, acceleration, `cnn_dropout=0`, `rnn_dropout=0.5`, 100 epochs,
patience 15, seed 42, and fixed 40h validation. Each task writes to
`results/train_data/gen_hparam_full_grid/task_<id>/`.

The utility supports task emission, validation, status, and summarization. Summary outputs include
best JSON/CSV, all trials, a Markdown summary, and per-scale CSV files consumed by the standard
generalization plotter.

## Fixed-best six-model multi-seed confirmation

`clutter_best6_multiseed.py` freezes the selected 40h configurations for GaWF, RNN, LSTM, GRU,
Mamba, and S5. It maps seeds 1--10 to 60 tasks, trains every task for 150 epochs with
`--patience 0`, and validates the metrics, checkpoint, pickle, seed, and full-epoch completion.
Amarel submission uses ten independent jobs, each containing the six model tasks for one seed.
The 40h training and validation splits both use `40h-float32`; checkpoints retain the
best-validation state observed during the complete 150-epoch trajectory.

## Launch environments

- Amarel submission/status/rerun wrappers are documented in `experiments/amarel/README.md`.
- Local two-GPU execution uses `experiments/local/run_hparam_full_grid_2gpu.sh --scale ...` and
  schedules at most two processes while reusing the same task mapping.
- Generated metrics figures must use `utils_viz/plot_generalization.py`; training-curve figures
  use the repository `visualize_batch.sh` entry point.
