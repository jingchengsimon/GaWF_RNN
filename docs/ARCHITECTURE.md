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

- `RNNCore`, `GRUCore`, and `LSTMCore`, including unified `num_layers` handling.
- `GaWFCore`, the later RNN-aligned experimental branch with single- and multi-layer paths. It is
  `nn.RNN` plus an element-wise gate on the input and hidden weight matrices, so the recurrence,
  both biases, the in-recurrence activation (`rnn_activation`: built-in `tanh` or `relu`) and the
  state convention are the built-in ones; `rnn_activation="identity"` is the single non-built-in
  branch and yields a linear recurrence. `gawf_legacy.GaWFCoreLegacy` freezes the original GaWF
  definition, in which the wrapped activity is the recurrent state, and is reachable as the
  `gawf_legacy` model type.
- `RNNCore` additionally supports `wrap_recurrent_state=True` for matched controls. The
  `rnn_inloop_notanh` control uses identity inner activation and feeds
  `dropout(ReLU(LayerNorm(preactivation)))` into the next time step.
- `AdditiveFeedbackRNNCore` and `ConcatenatedFeedbackCellCore`, used only by the
  non-multiplicative Clutter feedback controls.
- `MambaCore` and `S5Core` sequence models.
- `MLSTMCore`, `HyperLSTMCore`, and `BRIMsCore` for the open-loop
  dynamic/input-conditioned reviewer baselines. HyperLSTM uses the pinned minimal labml subset
  under `third_party/labml_nn/`; BRIMs is a clean-room implementation because the inspected
  upstream repository has no license grant.

GaWF uses feedback-conditioned input/hidden transforms. For feedback vector `fb`:

```text
U: (hidden_size, fb_dim)
V: (fb_dim, input_size + hidden_size)
gate = sigmoid(U @ (fb * V) / 0.5)
```

The gate multiplies every element of `W_ih` and `W_hh`. The original GaWF definition used by the
completed Clutter results is:

```text
pre_t = (gate_ih * W_ih) x_t + (gate_hh * W_hh) h_{t-1} + b_ih + b_hh
h_t   = dropout(ReLU(LayerNorm(activation(pre_t))))  # readout and next recurrent state
```

The later RNN-aligned branch instead carries only `activation(pre_t)` and applies the wrap to the
readout. It remains available for provenance as `GaWFCore`/`gawf_rnncore`, but its planned formal
rerun was cancelled after the original in-loop activity was reconfirmed as the intended GaWF
definition. The matched `rnn_inloop_notanh` behavioral control removes both the feedback gate and
the inner `tanh`, while retaining the in-loop wrap:

```text
pre_t = W_ih x_t + W_hh h_{t-1} + b_ih + b_hh
h_t   = dropout(ReLU(LayerNorm(pre_t)))             # readout and next recurrent state
```

For one layer, omitted Clutter `--dz` retains output-sized legacy feedback; explicit `--dz > 0`
uses a projector. For multiple layers, direct feedback uses the detached adjacent upper hidden
state at non-final layers and the detached previous task output at the final layer. Projected
mode gives each layer its own U/V pair and projector dimension. Multi-layer GaWF stacks built-in
`nn.RNN` layers, applies the external wrap between layers (each layer's readout is the next
layer's input), and carries each layer's raw `activation(preactivation)` as that layer's state.

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
`ClutterCharPosHead`. Public wrappers include `RNNConv`, `GRUConv`, `LSTMConv`, `GaWFRNNConv`,
`MambaConv`, `S5Conv`, `MLSTMConv`, `HyperLSTMConv`, and `BRIMsConv`. Historical multi-layer
class/checkpoint names remain readable, while new runs use `gawf --num_layers N`.

The three dynamic/input-conditioned baselines are open-loop: none receives task-head output.
`MLSTMConv` implements Krause et al. (2017) Equations (17)--(21), not the xLSTM matrix-memory
mLSTM. `HyperLSTMConv` preserves labml's four-tensor main/hyper state. `BRIMsConv` contains the
paper's two internal layers, bottom-up/current and top-down/previous-timestep attention, sparse
module updates, and within-layer communication; its default MNIST structure is blocks `(6, 3)`
and top-k `(4, 2)`. Its fixed attention dimensions follow the inspected MNIST core: input
attention uses 4 heads with `d_k=64`, and within-layer communication uses 4 heads with
`d_k=d_v=32`. All three return `(B,T,H)` and reset when called without an explicit state,
then use the same external `LayerNorm -> ReLU -> dropout` contract as existing recurrent cores.

The separate feedback-control model types are `gawf_additive`, `rnn_fb`, `gru_fb`, and
`lstm_fb`. They reuse `GaWFRNNConv._compute_feedback`: the previous frame's detached raw digit
and sector logits are concatenated into a 19-dimensional vector, with an all-zero vector at the
first frame of every independent rollout. `gawf_additive` adds `Linear(19, hidden_size)` to the
RNN preactivation and initializes its input/recurrent weights at `0.5W`, matching GaWF's
zero-feedback `sigmoid(0)=0.5` effective weight. The other three controls concatenate feedback
to every cell input and therefore use per-frame `RNNCell`, `GRUCell`, or `LSTMCell` execution.
These model types are single-layer controls and do not alter the original open-loop paths.
All four controls follow the same aligned readout contract as GaWF: the state that is fed back is
the raw activation and the wrap is applied to the readout only. Because the additive projection
and the concatenated input block are algebraically interchangeable, the two RNN controls differ
after this alignment only by that parameterization, by the additive bias and by the `0.5W`
initialization.

The corrected additive-feedback controls are `rnn_fb_add`, `gru_fb_add`, and `lstm_fb_add`.
They use the native no-wrap RNN-tanh, GRU, or LSTM output/state semantics and add one independent
affine source `W_fb f_(t-1) + b_fb` to the cell preactivation (all three GRU gates or all four
LSTM gates). `W_fb` and trainable `b_fb` use the same PyTorch uniform initialization bound as the
cell parameters and remain in the same optimizer parameter group. Disabling feedback omits the
entire affine source, including `b_fb`; the zero feedback vector on the first closed-loop step
still retains the learned source bias.

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
