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
- The six canonical Clutter model keys are `rnn`, `gru`, `lstm`, `gawf`, `mamba`, and `s5`.
  Additive-feedback controls use `rnn_fb_add`, `gru_fb_add`, and `lstm_fb_add`.
- `gawf` is the reported feedback-gated recurrence with no bounded inner activation; its
  `LayerNorm -> ReLU` activity is the clean next recurrent state, and dropout applies only to
  the readout-facing layer output.
- `rnn` is the matched ungated recurrence with the same in-loop activity definition.
- `gru`, `lstm`, `mamba`, and `s5` use their native no-wrap layer outputs.
- Historical ablation names remain only as checkpoint-filename aliases for the six manuscript
  checkpoints; they are not accepted by the training CLI and are never emitted for new runs.

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
| `--sequence_length` | analysis rollout and feedback-shuffle window length in frames; default `32` |
| `--use_mmap` | load large NumPy stimuli with mmap |
| `--use_sector_mode` | 3x3 sector classification |
| `--predict_all_chars` | predict foreground and background characters |
| `--sector`, `--digit` | selected sector 0–8 or digit 0–9 |
| `--agg` | `space` or `feature` aggregation |

Clutter training uses:

| Argument | Contract |
|---|---|
| `--cnn_dropout` | fixed at `0` for the output-only sequence dropout protocol |
| `--dropout` | shared output dropout for the six cores and three additive controls; default `0.5`, recorded as `rdo` |
| `--rnn_dropout` | legacy alias of `--dropout` |
| `--s5_dropout` | legacy option accepted only at `0`; S5 uses the shared dropout value |
| `--mamba_d_models` | one or more Mamba widths |
| `--ssm_d_models` | one or more S5 sequence widths |
| `--s5_state_sizes` | one or more S5 latent state sizes |
| `--feedback_dim`, `--dz` | GaWF projected feedback dimension; positive enables projectors |
| `--num_layers` | RNN/GRU/LSTM/GaWF depth |
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
The formal data-scale campaign uses
`data_scale/clutter_formal_4scale_ep150/<scale>/<model>-seedNN` as its `--result_suffix`; it keeps
each scale/model/seed unit in an independent resumable result leaf.

Atari DQN additionally uses `--frame_skip`, `--frame_stack`, `--task_schedule`,
`--replay_sampling`, `--replay_layout`, `--learning_starts_per_task`,
`--learning_rate_decay_per_task_steps`, `--amp_dtype`, `--allow_tf32`, `--compile_model`, and
`--feedback_mode`. `--atari_env_protocol` selects the versioned environment boundary/action
mapping contract; its default `baseline` retains historical behavior.
`--init_weights_from` accepts only a pure model `state_dict` and starts with a fresh
optimizer, replay, global step, RNG trajectory, epsilon schedule, and LR schedule. It is mutually
exclusive with `--resume_from` and `--auto_resume`. The SJC Skiing launcher requires a completed
20M final source by default; its diagnostic-only `--allow-incomplete-source` path may use a model
`state_dict` extracted from a stable copied resumable checkpoint, with the exact positive source
step encoded in metadata and the result leaf. Multi-task `--replay_layout per_task` creates one independent replay partition
per task; `--buffer_size` is the capacity of each partition, rather than a global shared capacity.
`--allow_total_timesteps_extension` applies only to resumable checkpoints and may only increase
the saved target while every other `RESUME_ARG_KEYS` field remains identical. The original target
must persist in later checkpoints and final metrics. A completed-run weights-only extension is a
new phase with fresh training state and must be named and documented separately from continuous
resume.
A finalized cumulative-2M Skiing model extends to cumulative 4M only through the fixed
`--extend-from-skiing-2m` weights-only phase. Its additional 2M uses a distinct
`extend2mto4m_2m_<model>_seed1` leaf and fresh optimizer/replay/global step; analysis adds a 2M
x-axis offset rather than describing the phase-local step as a strict continuation counter.
Atari DQN epsilon linearly decays over fixed `--exploration_steps` global steps (default
`500000`), independent of `--total_timesteps`. Historical `--exploration_fraction` remains
available only as an explicit compatibility option and cannot be combined with
`--exploration_steps`; new launchers and result protocols must record/use the fixed step count.
`--record_timing` is benchmark-only: it records host-wall environment/replay I/O and optimizer
timing in final metrics without changing the training protocol.

Atari DQN recovery uses `--checkpoint_interval_steps` (0 disables, and is the default),
`--resume_from` or `--auto_resume`, `--replay_backing {memory,mmap}`, and
`--keep_replay_on_success`. Because a DQN checkpoint is only meaningful together with its
replay buffer, resume requires `--replay_backing mmap`: the six replay arrays are then backed
by files under `<save_dir>/replay/` and the checkpoint carries only the position, task counts,
and sampler RNG state. Each checkpoint flushes the replay memmaps before the atomic save, so
the recorded position never outruns the durable data. A completed run deletes both the replay
directory and the scaffolding `checkpoint.pth`, leaving the usual single final `.pth`.
SIGTERM and SIGUSR1 are caught, converted into one final checkpoint, and reported as
`status=preempted`; pair them with `--requeue` and `--signal=B:USR1@120`.
As with paper-aligned MiniGrid PPO, the ALE environment and the recurrent state are reset
rather than serialized, so a resumed run is a valid continuation and not a bitwise replay;
metrics record `resume_count` and `resumed_at_steps` so interruptions stay visible in the
result. A runner must never append to an existing history when no compatible checkpoint is
present. The mmap backing costs roughly 27 GiB per fs4/stack4 shared-replay unit against a 1 TiB
`/scratch` soft quota. The five-task full-18 `per_task` protocol with five 0.5M partitions costs
roughly 65.8 GiB per unit. The formal five-task 1M-per-task protocol costs roughly 131.6 GiB per
unit, so its guard reserves at least 140 GiB for replay, checkpoint, and logs; its throttle is at
most `min(6, floor(available_writable_GiB / 140))`. When compute-node user quota cannot be parsed,
use a conservative throttle rather than reusing `%5`. Check quota before starting via
`experiments/<task>/amarel/scratch_quota_guard.py`. Measure headroom on a
`gpuk###` node, not the login node: Amarel serves one `/scratch` namespace from two GPFS
clusters whose fileset accounting disagrees for identical data (DSSP reports ~652 GiB free,
DSSK ~284 GiB), and a task may be enforced by either.

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
| `results/data/rl/atari/{runs,pong_6action,multitask_18action,5task_18action}/` | Atari run bundles; named protocol paths are curated |
| `results/data/rl/atari/breakout_4action/<protocol>/<model>/seed<N>/` | curated minimal-four-action Breakout bundles; use explicit `fs`, `stack`, layer, and plain/flicker protocol names |
| `results/data/rl/minigrid/{runs,...}/` | MiniGrid run bundles |
| `results/data/clutter/runs/` | Clutter checkpoints and training metrics |
| `results/data/text/runs/` | Text-task checkpoints and metrics |
| `results/data/rl/{atari,minigrid}/parameter_match/` | task-specific recurrent-core parameter-match tables |
| `results/figs/rl/{atari,minigrid}/` | curated RL learning curves |
| `results/figs/<CATEGORY>/` | analysis and development figures |
| `results/save/` | human-curated final figure files: `Fig*` as PDF only; `Supple*` as PNG only |
| `results/save/iclr_figs/` | ICLR-width alternate layouts; canonical saved figures stay unchanged |
| `results/save_data/<figure>/` | minimum numeric inputs for one saved figure; shared inputs have one owner |
| `../../6-Writing/Aim3/Figures/` | official publication PDFs |
| `experiments/clutter/artifacts/` | aggregated experiment tables/configs |
| `experiments/<task>/amarel/artifacts/<run>/` | ignored Slurm logs/status artifacts |

Analysis data remains grouped by producing script basename below
`results/data/analysis/<CATEGORY>/<module>/`, while figures are flat within
`results/figs/<CATEGORY>/`. Analysis scripts must obtain these directories from
`utils.analysis.anal_paths.output_dir` unless a task-specific convention below explicitly routes
one curated final figure to `results/save/`; its structured data still uses `output_dir`.

`results/save_data/` is a publication-curation boundary rather than an active analysis-output
directory. Fig1 consumes the single GaWF ablation copy owned by `save_data/fig2/`; Supple3
consumes the cache collection owned by `save_data/fig7/`. Do not duplicate either shared input.

Within `results/save/`, use the filename prefix to select exactly one output format: `Fig*` files
are PDFs and `Supple*` files are PNGs. Do not create a same-basename companion in the other format.

Development PNGs remain in their canonical result directories. Official publication PDFs are
written to `../../6-Writing/Aim3/Figures/`; set `AIM3_PUBLICATION_FIGURES_DIR` to override this
sibling-tree location on another host. A missing sibling writing tree does not authorize creating
one remotely: configure the environment variable explicitly or skip the publication PDF.
`core_objects_aggregate_2x2.png` and `best6_multiseed_summary_2x3.png` remain in their canonical
analysis/training result directories; their same-basename PDFs are official publication outputs
and therefore live only in the publication figure directory.

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
The distinct `5task_18action` namespace is reserved for the fixed
Pong/Breakout/Assault/Seaquest/Skiing L3 protocol and must never overwrite
`multitask_18action`. Its `smoke/`, `pilot/`, `parameter_match/`, and `figs/` leaves remain
separate.
Weights-only, single-Skiing adaptations from five-task seed snapshots use
`5task_18action/formal_20m_4mpertask_raw_seeds/`. Smoke and formal runs must use distinct leaves
inside this one parent; they must not create another sibling result parent or write back into the
source `formal_20m_4mpertask` leaves. An incomplete diagnostic source must include its exact source
step in both metadata and leaf naming and must not be described as a resume.
Do not relabel mismatched or ambiguous historical results. Retain them only when explicitly
curated into a task-specific `results/data/` path; otherwise remove them through the confirmed
cleanup workflow.

## Checkpoint names

Standard recurrent Clutter form:

```text
{model}_{mode}{acc}_h{hidden}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}{suffixes}_model.pth
```

- Multi-layer recurrent runs add `_L{layers}`.
- Explicit/projected GaWF feedback adds `_dz{dimension}`.
- Omitted single-layer GaWF `--dz` uses the concatenated task logits directly.
- New runs always emit one of the six public model keys. Historical stems are input-only aliases
  used to load the already-trained manuscript checkpoints.

Mamba/S5 use model-native width fields:

```text
mamba_{mode}{acc}_dmodel{width}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}_model.pth
s5_{mode}{acc}_dmodel{width}_state{size}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}_model.pth
```

HyperLSTM records both hidden widths and the feature size:

```text
hyperlstm_{mode}{acc}_h{hidden}_hh{hyper_hidden}_nz{n_z}_lr{lr}_wd{wd}_cdo{cnn}_rdo{rnn}_model.pth
```

`mlstm` and `brims` use the standard recurrent form. Their metrics JSON records the complete
architecture and `open_loop=true`.

The formal dynamic-baseline campaign uses
`dynamic_weight_baselines/clutter_dynamic_weight_baselines_ep150_v1/{model}-seed{seed:02d}`
below `data/clutter/runs/`. Its reset-excluded test outputs and ten-seed summary are written to
`data/analysis/dynamic_weight_baselines_reset_excluded_test_10seed_v1/` and
`data/analysis/dynamic_weight_baselines_formal_10seed_v1/`, respectively. Preflight runs use the
separate `preflight/dynamic_weight_baselines_2epoch_seed1_v1/` training subtree.

Resumable Clutter training state uses the same stem with `_train_state.pth`. It is not an
inference checkpoint and must not replace the final `_model.pth` best-validation artifact.

Atari names must encode algorithm, model, feedback, optional layer count, environment, frame skip,
and stack. `pong_fs1_stack1` and `pong_fs4_stack1` are valid protocol tags; `pong1f` is not.

## Analysis output names

The formal ten-seed Clutter data-scale comparison writes
``data_scale_seed_runs_10seed.csv``, ``data_scale_mean_sem_10seed.csv``, and
``data_scale_summary_10seed.json`` below
``results/data/analysis/G_behaviour/data_scale_comparison/``. Its curated 2-by-3 figure is
``results/save/data_scale_performance_2x3_10seed.pdf``. Historical files below
``results/figs/G_behaviour/`` remain retained but are not regenerated.

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
`negative -> nonnegative` crossing after the switch. Continuous-alignment significance uses a
two-sided permutation test of the absolute `diagonal mean - off-diagonal mean` contrast.
The relevance-distribution extensions use the same primary interaction-excluded top-10% masks as
Part 2 and write one normalized raw-gate density figure per context. Recurrent/sector outputs stay
below `E_relevance_alignment/gawf_recurrent_sector_relevance_distributions/`; input/sector,
input/digit, and recurrent/digit outputs are written below
`E_relevance_alignment/gawf_remaining_relevance_distributions/`.

The GaWF gate robustness audit writes compact JSON/CSV/NPZ results and figures below its
category-indexed script directories. Source/destination relevance, interaction policy, and
top-percent
selection must remain explicit columns. Final variance-fraction CIs state whether they are full
gate or sampled-synapse intervals; sampled intervals are recentered on the exact full-gate point.

The sequential sector input-gate mean analysis writes
`sector_gate_mean_sequential_equal_n.npz`,
`sector_gate_mean_sequential_equal_n_meta.json`, and paired
`fig2_sector_gate_mean_sequential_equal_n_point_{included,excluded}.{png,pdf}` artifacts below
`B_gate_by_context/sector_sigmoid_gate_sequential/`. It reconstructs the gates actually applied at
each timestep from aligned pre-step feedback and uses an equal-n frame sample across sectors. The
historical `fig2_sector_gate_mean.{png,pdf}` remains the explicitly labelled one-step/reset view;
the obsolete max-gate figure is not regenerated.

The GaWF/LSTM/GRU unit-gate context analysis writes `unit_gate_context_variance.{json,csv,npz}` to
`results/data/analysis/D_variance_decomposition/rnn_unit_gate_context_specificity/` and
three Figure-03-style per-model PNGs directly to
`results/figs/D_variance_decomposition/`. The poster summary writes
`03_unit_gate_marginalization_1x3.png` beside them and writes the official
`03_unit_gate_marginalization_1x3.pdf` to the publication figure directory. It contains only the
condition-mean marginalization panels; individual per-model PDFs are not generated. For GaWF,
`input_mean` and
`recurrent_mean` are destination-unit projections formed by arithmetically averaging raw sigmoid
gates across the corresponding incoming synapse axis on each frame. They are derived unit-level
views and do not replace the canonical connection-level GaWF results. LSTM reports sigmoid
input/forget/output gates and GRU reports sigmoid reset/update gates; candidate activations are
excluded. Plot titles and legends must distinguish native LSTM/GRU `unit-level gates` from the
derived GaWF `destination-unit projection` and from GaWF connection-level gate matrices in the
individual diagnostic figures. The compact poster summary uses the shorter panel titles
`GaWF afferent gates`, `LSTM gates`, and `GRU gates`; its caption or surrounding text carries the
unit-projection distinction.

The ICLR gate-specialization summary is generated by
`utils.analysis.clutter.fig4_combined_gate_specialization`. It combines the reset-excluded
ten-seed GaWF activation/synapse-gate decomposition, in that left-to-right order, in a centered
two-panel top row with the
GaWF/LSTM/GRU unit-gate comparison in a centered three-panel bottom row. The development PNG is
`results/figs/D_variance_decomposition/Fig4_gate_task_variable_specialization_2x3_10seed.png`;
the 5.5-inch-wide manuscript PDF is
`results/save/iclr_figs/Fig4_gate_task_variable_specialization_2x3_10seed.pdf`. The retained
standalone Figure 4 and Figure 5 outputs remain unchanged.
The same script's `--top_row_only --pdf_only --output_pdf
results/save/Fig4_gate_task_variable_specialization_1x2_10seed.pdf` exports only the
core results, with Encoder/Hidden activation on the left and Input/Recurrent gate on the right.

The combined ICLR recurrent-gating summary is generated by
`utils.analysis.clutter.fig7_combined_recurrent_gate_and_current`. Its centered three-block top
row contains the Digit and Sector recurrent delta-g bars plus the stacked Digit/Sector T-to-T
sign-magnitude curves; its centered two-panel bottom row contains the connection-normalized
Digit and Sector gate-dependent-current bars. The development PNG is
`results/figs/E_relevance_alignment/Fig7_recurrent_gate_disinhibition_and_current_2x3_10seed.png`;
the 5.5-inch-wide manuscript PDF is
`results/save/iclr_figs/Fig7_recurrent_gate_disinhibition_and_current_2x3_10seed.pdf`. The retained
standalone Figure 7 remains unchanged. The combined Figure 7 current panels and standalone Figure
8 share one visualization convention: every gray point is one training-seed mean across conditions,
matching the values used for the bar mean, SEM, and seed-level inference.

For the current manuscript Figure 6, pass `--pdf_only --output_pdf
results/save/iclr_figs/Fig6_recurrent_gate_disinhibition_and_current_2x3_10seed.pdf`
to overwrite only that PDF, without a PNG or metadata export. The combined figure's top
Digit/Sector panels retain the red/blue weight-sign bars and add one black Balanced bar
per TT/TR/RT/RR group. Red/blue bars retain the shared weight-magnitude overlap band;
the Balanced bar averages all nonzero connection/context delta-g rows within each seed,
without a magnitude restriction. Bars show ten-seed mean ± SEM, gray seed points, and
an uncorrected exact two-sided seed sign-flip test against zero. The sign-magnitude curves
and connection-normalized current panels retain their existing definitions.
Use `--top_row_only --pdf_only --output_pdf
results/save/Fig7_recurrent_gate_disinhibition_1x3_10seed.pdf` for the standalone
three-column gate figure, including both stacked TT curves and excluding current panels.

The GaWF gate-distribution summary writes `gawf_gate_histogram_summary_2x4.png` and
`01_pooled_all_gate_histogram.png` directly below `results/figs/A_raw_gate/`; only the
all-gate panel also has a local PDF by default. The all-gate figure combines input and recurrent
pooled histograms. The retained `2x4` filename is historical: the rendered layout is four rows
(pooled, weight-sign, sector, digit) by two columns (input, recurrent) so each panel is large
enough for its detail inset. Curves show per-bin probability percentages (`Probability (%)`), so
each curve's bin values sum to 100%. Mean/median reference lines appear in every 4-by-2 subplot; the
standalone all-gate panel shows only the pooled probability curve. All panels use a small x-axis
margin beyond 0 and 1 while retaining ticks over the observed gate range. To show the narrow
central peak without distorting the full distribution, the standalone all-gate figure and both
pooled/weight-sign columns of the 4-by-2 summary include 0.48--0.52 zoom insets (one pooled inset;
separate positive- and negative-weight insets for each weight-sign panel). Publication copies are
opt-in via `--publication_fig_dir` and are not created by the default command.

## Compatibility and naming changes

When a public module, symbol, flag, metrics field, or filename changes:

1. Search all Python, shell, notebooks, analysis, and visualisation call sites.
2. Update producers and consumers together.
3. Preserve parsing/loading compatibility when historical results remain scientifically useful.
4. Document migrations in the owning reference; add to `EXPERIMENT_LOG.md` only when the change
   alters the research model, protocol, or interpretation.
