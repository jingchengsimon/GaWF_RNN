# Architecture

This document owns module responsibilities, dependency direction, model composition, and data
flow. Naming and file formats live in `CONVENTIONS.md`; implementation checklists live in
`DEVELOPMENT_WORKFLOWS.md`.

## Task families

| Task | Entry point | Model/data modules |
|---|---|---|
| Clutter vision | `train_model.py` | `utils/clutter_*`, `source/clutter/` |
| IMDB | `train_imdb.py` | `utils/text_task_models.py`, `utils/text_imdb_data.py` |
| SentiHood | `train_sentihood.py` | `utils/text_task_models.py`, `utils/text_sentihood_*` |
| Atari A2C | `train_atari.py` | `utils/atari_task_models.py`, `utils/atari_envs.py` |
| Atari DQN/DRQN | `train_atari_dqn.py` | `utils/atari_dqn_models.py`, `utils/atari_replay.py` |
| MiniGrid DQN/DRQN | `train_minigrid_dqn.py` | `utils/minigrid_models.py`, `utils/minigrid_envs.py` |
| MiniGrid PPO | `train_minigrid_ppo.py` | `utils/minigrid_ppo_models.py`, `utils/minigrid_envs.py` |

All task families compose the same recurrent implementations from `utils/recurrent_cores/`.
Task wrappers own encoders, heads, data shapes, and feedback selection; they do not reimplement
recurrent equations.

Task-specific experiment definitions live under `experiments/clutter/`, `experiments/atari/`,
`experiments/minigrid/`, and `experiments/text/`. Execution wrappers remain grouped by backend in
`experiments/amarel/` and `experiments/local/`; those directories do not imply separate repos.

## Dependency direction

```text
stdlib / third-party
        |
        v
utils/recurrent_cores/
        |
        v
utils/<task>_task_models.py + task data/train helpers
        |
        v
train_<task>.py
        |
        v
utils_anal/  ->  results/anal_data/
        |
        v
utils_viz/   ->  results/anal_figs/
```

The arrows are one-way:

- `utils/` must not import from `utils_anal/` or `utils_viz/`.
- `utils_anal/` must not import from `utils_viz/`.
- `utils_viz/` reads saved results rather than becoming a model dependency.
- `source/` contains data/environment preparation and must not become a training core.

## Shared recurrent cores

`utils/recurrent_cores/` provides:

- `RNNCore`, `GRUCore`, and `LSTMCore`, including unified `num_layers` handling.
- `GaWFCore`, with single- and multi-layer paths behind one public model type.
- `MambaCore` and `S5Core` sequence models.

GaWF uses feedback-conditioned input/hidden transforms. For feedback vector `fb`:

```text
U: (hidden_size, fb_dim)
V: (fb_dim, input_size + hidden_size)
gate = sigmoid(U @ (fb * V) / 0.5)
```

For one layer, omitted Clutter `--dz` retains output-sized legacy feedback; explicit `--dz > 0`
uses a projector. For multiple layers, direct feedback uses the detached adjacent upper hidden
state at non-final layers and the detached previous task output at the final layer. Projected
mode gives each layer its own U/V pair and projector dimension.

`prev_feedback` is runtime state and must be detached before storage. Checkpoint loading filters
that key and uses `strict=False`.

## Clutter architecture

### Encoder contract

`ClutterCNNEncoder` consumes two-channel movie frames. The fixed large configuration is:

```text
2x96x96
  -> Conv 2->32, same padding -> MaxPool 2x2 -> LayerNorm [32,48,48]
  -> Conv 32->64             -> MaxPool 4x4 -> LayerNorm [64,12,12]
  -> Conv 64->32, 1x1        -> AdaptiveAvgPool 6x6
  -> 32x6x6 = 1152 features
```

Changing the output requires updating model input sizes and every analysis that assumes 32
channels or 6x6 spatial structure.

### Model and training composition

`ClutterSequenceModel` composes the CNN, a middle recurrent/sequence model, and
`ClutterCharPosHead`. Public wrappers include `RNNConv`, `GRUConv`, `LSTMConv`, `GaWFRNNConv`,
`MambaConv`, and `S5Conv`. Historical multi-layer class/checkpoint names remain readable, while
new runs use `gawf --num_layers N`.

`clutter_train_helpers.py` owns CLI construction, paths, dataset creation, logging, model
registration, seeding, and saved summaries. `clutter_train_acceleration.py` owns loaders, AMP,
gradient accumulation, and `TrainStepper`. `clutter_train_engine.py` owns the epoch/batch loop.
Loss and metric implementations remain in `clutter_train_sector.py` and
`clutter_train_predict_all_chars.py`.

Official train and validation curves come from full evaluation passes, not online batch means.
Sector single-character data may include `fg_switch`; when present, evaluation also records
strict global and `pre5`/`post5` foreground-transition accuracies.

## Text architecture

`TextSequenceClassifier` combines embeddings, a shared recurrent core, and classification heads.
Dataset-specific modules own tokenization/data preparation and metrics. IMDB and SentiHood must
not introduce separate RNN/GaWF implementations.

## Atari architecture

### A2C

`AtariActorCritic` uses `AtariNatureEncoder`, a recurrent core, and separate policy/value heads.
The recurrent input contains encoded observation features plus previous action and reward.

- Model types: `lstm`, `gawf`.
- Feedback modes: `none`; GaWF may use `output`.
- `output` feedback is detached previous policy logits concatenated with previous value.

### DQN/DRQN

`AtariQNetwork` follows the DRQN family. All variants share the Nature-DQN convolutional feature
stack and a final linear Q head; only the readout slot changes.

- `ann`: dense Nature-DQN readout.
- `rnn`, `gru`, `lstm`, `gawf`: stepwise recurrent readout.
- `s5`, `mamba`: full-window sequence readout with rolling online context.
- GaWF `qvalues`: detached previous-step Q values gate recurrence.

The replay buffer and training loop preserve episode-reset metadata. RNN/GRU/LSTM use a
whole-sequence cuDNN fast path only when a sampled window has no internal reset, otherwise they
fall back to the reset-aware stepwise path. GaWF remains stepwise because feedback evolves at
each timestep.

Task-blind multi-task collection selects a task only at episode boundaries. The default
`transition_balanced` scheduler chooses the task with the fewest collected environment steps,
with cyclic tie-breaking; shorter tasks may therefore run more episodes. Replay remains a
separate concern: `task_balanced` sampling gives tasks equal update weight without exposing the
task identifier to the model.

CUDA autocast, TF32, gradient scaling, compilation configuration, cuDNN benchmarking, and fused
Adam live in `utils/atari_train_acceleration.py`. The current Amarel PyTorch build compiles ANN
only; recurrent-state dataclasses are not passed through Dynamo. Replay reset detection remains
on CPU, and logging-only scalar synchronization occurs at log intervals. These optimizations
must not alter sampled indices, losses, update cadence, environment steps, or network structure.

Pong result labels always encode both frame skip and stack. Historical five-seed results use
`pong_fs4_stack1`; strict one-decision/one-ALE-frame runs use `pong_fs1_stack1`.

## MiniGrid architecture

MiniGrid uses symbolic partial observations and a task-specific encoder while reusing the shared
recurrent cores. Recurrent PPO collects a fixed `(num_steps, num_envs)` rollout and replays each
complete per-environment sequence during every PPO epoch, preserving recurrent state and reset
masks.

MiniGrid acceleration is configured by `utils/minigrid_train_acceleration.py`. CUDA autocast,
TF32, gradient scaling, cuDNN benchmarking, fused Adam, and optional callable compilation must
not change environment samples, PPO losses, rollout length, update cadence, or model structure.
The `async` vector backend parallelizes environment stepping across subprocesses but preserves
the ordered vector-slot interface and per-environment seeds.

## Data and result flow

```text
source/<task>/ or external datasets/environments
        -> CPU arrays/datasets/replay
        -> DataLoader or replay sampling
        -> task wrapper + recurrent core
        -> results/train_data/<suffix>/
             *_model.pth
             *.pkl
             *_metrics.json
        -> utils_anal/ -> results/anal_data/<module>/
        -> utils_viz/  -> results/anal_figs/<module>/
```

Clutter data resolution order is CLI `--data_dir`, `AIM3_STIMULI_PATH`,
`FAW_RNN_DATA_PATH`, then `<repo>/stimuli`. Large arrays stay on CPU or mmap.

## Shared interfaces

- New clutter model types register through `get_model_classes()`.
- Clutter label modes implement the existing metrics-mode lifecycle used by the engine.
- Analysis model/dataset construction uses `build_model_from_ckpt` and `build_test_dataset` from
  `utils_anal.anal_helpers`.
- Model and accelerator changes must preserve public training arguments, metrics fields, and
  checkpoint compatibility unless a documented migration is part of the same change.
