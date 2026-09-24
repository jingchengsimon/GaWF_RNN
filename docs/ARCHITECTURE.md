# Architecture

This document owns module responsibilities, dependency direction, model composition, and data
flow. Naming and file formats live in `CONVENTIONS.md`; implementation checklists live in
`DEVELOPMENT_WORKFLOWS.md`.

## Task families

| Task | Entry point | Model/data modules |
|---|---|---|
| Clutter vision | `run_task.py clutter` | `utils/training/train_scripts/clutter.py`, `utils/training/clutter/`, `source/clutter/` |
| IMDB | `run_task.py imdb` | `utils/training/train_scripts/imdb.py`, `utils/training/text/` |
| SentiHood | `run_task.py sentihood` | `utils/training/train_scripts/sentihood.py`, `utils/training/text/` |
| Atari A2C | `run_task.py atari-a2c` | `utils/training/train_scripts/atari_a2c.py`, `utils/training/atari/` |
| Atari DQN/DRQN | `run_task.py atari-dqn` | `utils/training/train_scripts/atari_dqn.py`, `utils/training/atari/` |
| MiniGrid DQN/DRQN | `run_task.py minigrid-dqn` | `utils/training/train_scripts/minigrid_dqn.py`, `utils/training/minigrid/` |
| MiniGrid PPO | `run_task.py minigrid-ppo` | `utils/training/train_scripts/minigrid_ppo.py`, `utils/training/minigrid/` |

All task families compose the same recurrent implementations from `utils/training/recurrent_cores/`.
Task wrappers own encoders, heads, data shapes, and feedback selection; they do not reimplement
recurrent equations.

Task-specific experiment definitions live under `experiments/clutter/`, `experiments/rl/atari/`,
`experiments/rl/minigrid/`, and `experiments/text/`. Execution wrappers remain grouped by backend in
`experiments/<task>/amarel/` and `experiments/<task>/amarel/`; those directories do not imply separate repos.

## Dependency direction

```text
stdlib / third-party
        |
        v
utils/training/recurrent_cores/
        |
        v
utils/training/<task>/ + task data/train helpers
        |
        v
run_task.py <task> -> utils/training/train_scripts/<task>.py
        |
        v
utils/analysis/  ->  results/data/analysis/<CATEGORY>/<script>/
        |
        v
utils/analysis/  ->  results/figs/<CATEGORY>/
```

The arrows are one-way:

- `utils/training/` must not import from `utils/analysis/`.
- `utils/analysis/` owns numeric analysis and plotting from saved structured results; it is not a
  model dependency.
- `source/` contains data/environment preparation and must not become a training core.

## Shared recurrent cores

`utils/training/recurrent_cores/` provides:

- `GaWFCore`, the feedback-gated recurrence used by the reported model.
- `RNNCore`, the matched ungated recurrence.
- `GRUCore` and `LSTMCore`, direct wrappers around `nn.GRU` and `nn.LSTM`.
- `MambaCore` and `S5Core`, projected residual sequence stacks.

Historical ablations and reviewer-only recurrent definitions are frozen under
`utils/training/recurrent_cores/archive/pre_iclr_2026_09_24/`. Active training and analysis code
must not import from that directory.

GaWF uses feedback-conditioned input/hidden transforms. For feedback vector `fb`:

```text
U: (hidden_size, fb_dim)
V: (fb_dim, input_size + hidden_size)
gate = sigmoid(U @ (fb * V) / 0.5)
```

The gate multiplies every element of `W_ih` and `W_hh`. The current reported GaWF definition is:

```text
z_t = (gate_ih * W_ih) x_t + (gate_hh * W_hh) h_{t-1} + b_ih + b_hh
h_t = dropout(ReLU(LayerNorm(z_t)))  # layer output and next recurrent state
```

The matched RNN removes only the feedback-conditioned gates:

```text
z_t = W_ih x_t + W_hh h_{t-1} + b_ih + b_hh
h_t = dropout(ReLU(LayerNorm(z_t)))  # layer output and next recurrent state
```

GaWF and RNN compute these recurrences explicitly. Their ``rnn.weight_ih_l0``,
``rnn.weight_hh_l0``, ``rnn.bias_ih_l0``, and ``rnn.bias_hh_l0`` names are retained through an
activation-free affine parameter container for checkpoint and analysis compatibility; neither core
calls ``nn.RNN`` or inherits its default ``tanh`` activation.

For one layer, omitted Clutter `--dz` retains output-sized legacy feedback; explicit `--dz > 0`
uses a projector. For multiple layers, direct feedback uses the detached adjacent upper hidden
state at non-final layers and the detached previous task output at the final layer. Projected
mode gives each layer its own U/V pair and projector dimension. Each layer carries the wrapped
activity above as both its output and its next recurrent state.

`prev_feedback` is detached runtime state, registered as a non-persistent buffer in Clutter.
Resume and best-validation loading filter legacy copies, reset the runtime cache, and use
`strict=False` while still rejecting any missing or unexpected learned-state keys.

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
`ClutterCharPosHead`. The public wrappers are exactly `RNNConv`, `GRUConv`, `LSTMConv`,
`GaWFRNNConv`, `MambaConv`, and `S5Conv`. `GaWFRNNConv` owns both single- and multi-layer paths via
`--num_layers`; there is no separate multi-layer class. GaWF feedback is always active after the
zero-initialized first frame. GRU/LSTM return the native PyTorch layer output. Mamba/S5 return the
projected residual stack output. None of these four baselines receives an external
`LayerNorm -> ReLU -> dropout` wrap.

`clutter_train_helpers.py` owns CLI construction, paths, dataset creation, logging, model
registration, seeding, and saved summaries. `clutter_train_acceleration.py` owns loaders, AMP,
gradient accumulation, and `TrainStepper`. `clutter_train_engine.py` owns the epoch/batch loop.
Loss and metric implementations remain in `clutter_train_sector.py` and
`clutter_train_predict_all_chars.py`.

Official train and validation curves come from full evaluation passes, not online batch means.
Sector single-character data may include `fg_switch`; when present, evaluation also records
strict global and `pre5`/`post5` foreground-transition accuracies.

Long Clutter runs checkpoint only at completed epoch boundaries. The resumable state contains the
current model, optimizer, AMP scaler, completed epoch count, metric arrays, best-validation state,
early-stopping counters, process RNG, and DataLoader/sampler RNG. Final `_model.pth` files remain
best-validation inference artifacts and are distinct from `*_train_state.pth` continuation files.

## Text architecture

`TextSequenceClassifier` combines embeddings, a shared recurrent core, and classification heads.
Dataset-specific modules own tokenization/data preparation and metrics. IMDB and SentiHood must
not introduce separate RNN/GaWF implementations.

## Atari architecture

### A2C

`AtariActorCritic` uses `AtariNatureEncoder`, a recurrent core, and separate policy/value heads.
The recurrent input contains encoded observation features plus previous action and reward.

- Model types: `lstm`, `gawf`.
- LSTM has no task-output feedback; GaWF always uses `output` feedback.
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

DQN/DRQN TD targets clip rewards to `[-1, 1]` by default. `--no_reward_clip` retains raw
environment rewards for an explicitly named protocol; this setting is recorded in metrics and is
resume-validated, while checkpoints predating the flag retain the historical clipped behavior.

Task-blind multi-task collection selects a task only at episode boundaries. The default
`transition_balanced` scheduler chooses the task with the fewest collected environment steps,
with cyclic tie-breaking; shorter tasks may therefore run more episodes. Replay remains a
separate concern: `task_balanced` sampling gives tasks equal update weight without exposing the
task identifier to the model. Replay batch remainders rotate across tasks so non-divisible batch
sizes remain balanced over successive updates. Recoverable runs persist each vector slot's task
counts and tie-breaking cursor; ALE and recurrent state still resume from a fresh reset. Formal
multi-task runs may additionally require a per-task collection threshold before the first update.
The shared output remains 18-dimensional even when ALE exposes fewer legal actions for a game:
unsupported fire variants map to the corresponding legal non-fire movement, and unsupported
standalone FIRE maps to NOOP. No task-specific action mask is given to the model.

The historical `baseline` environment protocol retains its original wrapper order and
multi-task action mapping for exact reproduction. `skiing-stall-actionfix-v1` is a distinct MDP:
each task receives exactly one canonical full-18 wrapper, so the four games with all 18 ALE
actions are identity mappings while Skiing maps the 18 Q outputs once onto its nine legal
non-FIRE actions. Skiing course progress is the change in ALE RAM course-object y slots 86:94,
not score, clock, or lateral motion. After 450 agent steps without such progress, the wrapper
sets `truncated=True`, reports `end_reason=stalled`, and adjusts total raw return to at most
-30,000. Training resets recurrent/episode state at this artificial boundary but keeps the TD
bootstrap from its final observation; natural termination still stops bootstrap.

CUDA autocast, TF32, gradient scaling, compilation configuration, cuDNN benchmarking, and fused
Adam live in `utils/atari_train_acceleration.py`. The current Amarel PyTorch build compiles ANN
only; recurrent-state dataclasses are not passed through Dynamo. Replay reset detection remains
on CPU, and logging-only scalar synchronization occurs at log intervals. These optimizations
must not alter sampled indices, losses, update cadence, environment steps, or network structure.

Pong result labels always encode both frame skip and stack. Curated active six-action results use
matched strict protocols: `pong_fs1_stack1` for one-decision/one-ALE-frame runs and
`pong_fs4_stack4` for the standard four-frame protocol. Historical `pong_fs4_stack1` results
remain scientifically identifiable but belong under `results/archive/`, not the active curated
tree.

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
        -> results/data/<task>/runs/<suffix>/  (job-local staging)
             *_model.pth
             *.pkl
             *_metrics.json
        -> results/data/{rl,clutter,text}/...  (curated task hierarchy)
        -> utils/analysis/ -> results/data/analysis/<CATEGORY>/<module>/
        -> utils/analysis/ -> results/figs/<CATEGORY>/
```

Clutter data resolution order is CLI `--data_dir`, `AIM3_STIMULI_PATH`,
`FAW_RNN_DATA_PATH`, then `<repo>/source/clutter/stimuli`. Standard 40h arrays stay mmap-backed
uint8 on CPU.
The Dataset emits a compact `(T+C-1,H,W)` window; the training/evaluation boundary transfers
the uint8 batch, casts once to float32 on the target device, and expands it to
`(B,T,C,H,W)`. `BlockShuffleSampler` preserves exact epoch coverage while shuffling contiguous
blocks; the default block equals the effective batch size and can be disabled with
`--shuffle_block_size 0` for historical global-random reproduction.

## Shared interfaces

- New clutter model types register through `get_model_classes()`.
- Clutter label modes implement the existing metrics-mode lifecycle used by the engine.
- Analysis model/dataset construction uses `build_model_from_ckpt` and `build_test_dataset` from
  `utils.analysis.anal_helpers`.
- Model and accelerator changes must preserve public training arguments, metrics fields, and
  checkpoint compatibility unless a documented migration is part of the same change.
