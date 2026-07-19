# Conventions

This document owns public argument names, identifiers, tensor layout, and saved-result naming.
Architecture and workflow rules live in `ARCHITECTURE.md` and `DEVELOPMENT_WORKFLOWS.md`.

## Vocabulary

| Name | Meaning |
|---|---|
| `gawf` | Gated-Weight-on-Feedback model; unified single/multi-layer public type |
| `fb`, `dz` | feedback vector and optional projected feedback dimension |
| `ih`, `hh` | input-to-hidden and hidden-to-hidden paths |
| `cdo`, `rdo` | CNN and recurrent/middle-path dropout filename fields |
| `h`, `dmodel`, `state` | recurrent hidden size, sequence-model width, S5 state size |
| `L` | recurrent/readout layer count in filenames |
| `fs`, `stack` | ALE frame skip and observation frame stack |
| `glob` | global correct frames divided by global frame count |
| `pre5`, `post5` | foreground-switch evaluation windows |
| `agg` | analysis aggregation axis: `space` or `feature` |
| `trans`, `outer` | feedback-conditioned transform and rank-one component |

## Public identifiers

- Python functions: verb-led `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: one leading underscore.
- Common loop indices: `sidx` sample, `t` time, `b` batch, `d` digit/component, `c` channel.
- Public model keys use lowercase: `ann`, `rnn`, `gru`, `lstm`, `gawf`, `mamba`, `s5`.

Do not introduce a second name for an existing public argument or model. Historical aliases may
remain parsable for compatibility but must not appear in new result names.

## Common CLI arguments

| Argument | Meaning |
|---|---|
| `--ckpt` | checkpoint path |
| `--save_dir` | analysis output directory |
| `--data_dir` | input dataset or analysis directory |
| `--device` | `cuda`, `mps`, or `cpu` as supported by the script |
| `--seed` | random seed |
| `--batch_size` | DataLoader batch size |
| `--use_mmap` | load large NumPy stimuli with mmap |
| `--use_sector_mode` | 3x3 sector classification |
| `--predict_all_chars` | predict foreground and background characters |
| `--sector`, `--digit` | selected sector 0–8 or digit 0–9 |
| `--agg` | `space` or `feature` aggregation |

Clutter training uses:

| Argument | Contract |
|---|---|
| `--cnn_dropout` | one or more CNN dropout values; default `[0]` |
| `--rnn_dropout` | one middle-path dropout value; default `0.5` |
| `--mamba_d_models` | one or more Mamba widths |
| `--ssm_d_models` | one or more S5 sequence widths |
| `--s5_state_sizes` | one or more S5 latent state sizes |
| `--feedback_dim`, `--dz` | GaWF projected feedback dimension; positive enables projectors |
| `--num_layers` | ANN/RNN/GRU/LSTM/GaWF depth; integer >= 1 |
| `--gawf_feedback_lr_scale` | U/V/projector LR multiplier; default `1.0` |
| `--data_suffix` | training and default validation data suffix; default `40h-uint8` |
| `--eval_data_suffix` | optional validation-only suffix |
| `--input_cast_mode` | `sample`, `batch_cpu`, or `device`; default `device` |
| `--frame_layout` | `stacked` or `compact`; default `compact` |
| `--shuffle_block_size` | `-1` uses effective batch size (default), `0` is global random |
| `--patience` | early stopping on fair val character accuracy; `0` disables |
| `--checkpoint_interval_epochs` | atomically save resumable state every N completed epochs |
| `--auto_resume` | load the deterministic per-experiment `*_train_state.pth` when present |
| `--resume_from` | explicit training-state checkpoint for a single experiment |

Clutter metrics JSON records `seed`, `patience`, `use_acceleration`, `use_mmap`,
`input_cast_mode`, `frame_layout`, and `shuffle_block_size` in addition
to the model/dataset hyperparameters and epoch summaries. Multi-seed result directories must
encode the seed even though checkpoint stems retain the standard model naming contract.

Atari DQN additionally uses `--frame_skip`, `--frame_stack`, `--task_schedule`,
`--replay_sampling`, `--amp_dtype`, `--allow_tf32`, `--compile_model`, and `--feedback_mode`.

MiniGrid PPO exposes the same CUDA acceleration names plus `--env_backend {sync,async}`,
`--cudnn_benchmark`, and `--fused_optimizer`. Saved metrics must record the active backend and
all acceleration settings. Amarel accelerated reruns append a distinct tag such as `_accel_v1`
to the result suffix so historical baselines are not overwritten.

The paper-aligned MiniGrid PPO entry point additionally uses
`--checkpoint_interval_updates` for atomic periodic checkpoints and `--resume_from` for
continuation. Resume restores model, optimizer, counters, and process RNG state. Because the
Gymnasium/MiniGrid environment is reset instead of serialized, saved metadata identifies the
continuation as `fresh_reset`; it is a statistically valid continuation, not bitwise replay of
the interrupted trajectory. A runner must never append to an existing history when no compatible
checkpoint is present.

Multi-task collection defaults to `transition_balanced`; historical `round_robin` remains
selectable. New Pong result suffixes must contain both `fs` and `stack`. GaWF DQN feedback is
named `qvalues`; A2C GaWF output feedback is named `output`.

## Tensor and label layout

```text
(B, T, C, H, W)       movie/observation sequences
(B, T, 2) int64       [digit_id, sector_id]
(B, T, 3) float32     [digit_id, x, y]
(B, T, max_chars)     ordered character IDs; -1 is padding
(B, H, input_size)    per-sample input-hidden transform
(n_comp, H, I)        rank-one transform components
```

Use PyTorch batch-first layouts at task boundaries unless an underlying core explicitly documents
another internal representation.

## Result directories

| Directory | Contents |
|---|---|
| `results/train_data/rl/atari/pong_6action/` | curated six-action Pong run bundles |
| `results/train_data/rl/atari/multitask_18action/` | curated full-18-action Atari controls and multi-task runs |
| `results/train_data/rl/minigrid/` | curated MiniGrid run bundles |
| `results/train_data/clutter/` | Clutter checkpoints and training metrics |
| `results/train_figs/rl/{atari,minigrid}/` | curated RL learning curves |
| `results/train_figs/clutter/` | Clutter training figures |
| `results/archive/` | historical, superseded, validation-only, or protocol-mismatched results |
| `results/anal_index/<CATEGORY>/<module>/data/` | analysis arrays and metadata |
| `results/anal_index/<CATEGORY>/<module>/figs/` | figures |
| `results/anal_index/<CATEGORY>/<module>/manifest.json` | run provenance and key numbers |
| `experiments/generalization/artifacts/` | aggregated experiment tables/configs |
| `experiments/amarel/artifacts/<run>/` | ignored Slurm logs/status artifacts |

The analysis/figure module directory matches the producing script basename. Analysis scripts
must obtain these directories from `utils_anal.anal_paths.output_dir`; legacy
`results/anal_data/` and `results/anal_figs/` are migration-only paths and must not be recreated.

Training jobs may first write a flat suffix directory as a staging artifact. Curated copies are
then placed in the task hierarchy above. Inside curated Atari paths, omit the redundant
`atari` filename prefix. Ordinary single-seed figures are files directly below their task
directory; a multi-seed campaign keeps one group directory with `seed<N>.png` files and writes
`mean_std.png` only after every declared seed is complete. Seed and step count are carried by
saved metadata and plot titles rather than repeated in the curated protocol filename. Raw
training data remains directory-based because a checkpoint, final metrics, and history form one
run bundle.

Active `pong_6action` results must report `action_space_mode=minimal`, `num_actions=6`, and a
strict matched frame protocol (`fs1_stack1` or `fs4_stack4`). Active
`multitask_18action` results must report `action_space_mode=full18` and `num_actions=18`.
Move mismatched or ambiguous historical results to `results/archive/` instead of relabelling
them.

## Checkpoint names

Standard recurrent Clutter form:

```text
{model}_{mode}{acc}_h{hidden}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}{suffixes}_model.pth
```

- Multi-layer recurrent runs add `_L{layers}`.
- Explicit/projected GaWF feedback adds `_dz{dimension}`.
- Legacy single-layer GaWF may omit `_dz` and infer task-output feedback.
- Historical `gawf_multi_` and unified `_do{dropout}` names remain readable but are not emitted.

Mamba/S5 use model-native width fields:

```text
mamba_{mode}{acc}_dmodel{width}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}_model.pth
s5_{mode}{acc}_dmodel{width}_state{size}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}_model.pth
```

Resumable Clutter training state uses the same stem with `_train_state.pth`. It is not an
inference checkpoint and must not replace the final `_model.pth` best-validation artifact.

Atari names must encode algorithm, model, feedback, optional layer count, environment, frame skip,
and stack. `pong_fs1_stack1` and `pong_fs4_stack1` are valid protocol tags; `pong1f` is not.

## Analysis output names

```text
tag = f"{mode}{selected_idx}_{agg}"
<descriptor>_<tag>.npy
<descriptor>_meta_<tag>.json
<mode><idx>_<agg>_<descriptor>.png
```

Save one array as `.npy`, related arrays as `.npz`, and metadata as JSON. Arrays written for
downstream use must be explicitly `np.float32` or `np.int64`.

The symmetric GaWF relevance/timing analysis writes its decomposition, relevance, timing, and
control artifacts under categories D, E, F, and H respectively. Part 2 must preserve both
`interaction_excluded` and
`interaction_included` results; Part 3 defines gate reconfiguration as a strict
`negative -> nonnegative` crossing after the switch.

The GaWF gate robustness audit writes compact JSON/CSV/NPZ results and figures below its
category-indexed script directories. Source/destination relevance, interaction policy, and
top-percent
selection must remain explicit columns. Final variance-fraction CIs state whether they are full
gate or sampled-synapse intervals; sampled intervals are recentered on the exact full-gate point.

## Compatibility and naming changes

When a public module, symbol, flag, metrics field, or filename changes:

1. Search all Python, shell, notebooks, analysis, and visualisation call sites.
2. Update producers and consumers together.
3. Preserve parsing/loading compatibility when historical results remain scientifically useful.
4. Document migrations in the owning reference; add to `EXPERIMENT_LOG.md` only when the change
   alters the research model, protocol, or interpretation.
