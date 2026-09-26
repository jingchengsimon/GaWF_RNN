# Clutter protocol

**Protocol status (2026-09-25):** The earlier tanh GaWF and the later
`gawf_legacy_notanh` results are historical, non-formal results for the current
output-only dropout definition. The existing result trees and frozen figure-input manifest
remain provenance for those older protocols. No 40h ten-seed result has been completed under
the current six-model definition. The 60 planned units are six models × seeds 1–10; they require
a new result root and protocol-tagged checkpoints.

The pending 40h rerun has **60 logical model-seed units**, with no new scheduler IDs assigned yet:

| Model | Width | Seeds | Units |
|---|---:|---|---:|
| RNN-ReLU | 275 | 01–10 | 10 |
| LSTM | 80 | 01–10 | 10 |
| GRU | 105 | 01–10 | 10 |
| GaWF-ReLU | 256 | 01–10 | 10 |
| Mamba | 170 | 01–10 | 10 |
| S5 | 256, state 128 | 01–10 | 10 |

The widths and optimizer settings remain the existing parameter-matched choices; this rerun
changes dropout placement and uses one shared `--dropout 0.5` with `--cnn_dropout 0`.

## Output-only dropout 330-unit rolling campaign

`experiments/clutter/amarel/submit_clutter_330_rolling.sh` is the one-command Amarel launcher.
It submits one GPU smoke covering the nine model types, then a dependent CPU controller on the
`main` partition. The controller submits sparse GPU arrays into a rolling window of at most 90
campaign tasks and counts every user's expanded `PENDING`/`RUNNING` array element before filling
vacancies. It caps the user-wide submitted count at 99 against the observed 100-task limit, with
one spare slot. Each short controller job schedules a successor 15 minutes later; no process stays
on a login node. An ambiguous submission or missing final artifact blocks that unit for review
instead of risking a duplicate writer.

The fixed task map is `0–59` 40h parameter-matched six-model, `60–89` 40h additive-feedback
three-model, `90–149` 4h equal-width H=128 six-model, then `150–209`, `210–269`, and `270–329`
for parameter-matched six-model 4h, 10h, and 20h respectively. Every group has ten seeds per
model. All units train 150 epochs with `patience=0`, shared 40h validation, `--dropout 0.5`,
`--cnn_dropout 0`, standard uint8 mmap/device-cast pipeline, and five-epoch auto-resume.
RNN-FB uses H=271 for the updated parameter match. Old results from Amarel or SJC never satisfy
the new campaign's completion gate.

Training results use the isolated suffix
`results/data/clutter/runs/clutter_output_only_330_v1/<group>/<scale>/<model>-seedNN/`.
Controller state, source stamp, smoke report, and Slurm logs live below
`results/artifacts/clutter_output_only_330_v1/`. The controller requires a matching source
commit, `.done` receipt, final metrics with the output-only protocol tag, model checkpoint, and
history before counting a unit complete.

```bash
AIM3_RESULTS_PATH=/scratch/js3269/results \
AIM3_CLUTTER_DATA_DIR=/scratch/js3269/stimuli \
bash experiments/clutter/amarel/submit_clutter_330_rolling.sh --dry-run
```

Omit `--dry-run` only after the new source is committed and fast-forwarded into a clean Amarel
checkout. The submitter refuses a reused result or artifact root.

The earlier publication-oriented Clutter training is complete. Its retained figures and minimal
numeric inputs live in `results/save/` and `results/save_data/`; reproduction scripts remain in
`utils/analysis/`. Historical grids are not part of the active project tree.

The six manuscript-figure input paths are frozen in
`experiments/clutter/iclr_figure_data_manifest.json`. Use its exact Amarel source paths and local
visualization-cache mappings for layout-only redraws; do not search for substitute result trees.

The historical data-scale behavior campaign used
`experiments/clutter/amarel/submit_clutter_data_scale_formal.sh`. Each scale has 60 task IDs:
`task_id = model_index * 10 + seed - 1`, with model order
`rnn,lstm,gru,gawf,mamba,s5` and seeds 1-10. Runs use the fixed best6 hyperparameters, 150 epochs,
`patience=0`, a shared `40h-uint8` validation set, and the current standard uint8 pipeline.
Results are isolated below
`results/data/clutter/runs/data_scale/clutter_formal_4scale_ep150/<scale>/<model>-seedNN/`.

The active seed2-9 campaign uses a 48-unit rolling window rather than whole-seed barriers.
Seed2 and seed3 provide 48 dependency-free scale/model roots. For each later seed, every task
depends on the same scale/model task two seeds earlier: `seed4 <- seed2`,
`seed5 <- seed3`, through `seed9 <- seed7`. Each successful run releases exactly one
successor, so at most 48 seed2-9 campaign units are eligible or running. A failed run blocks only
its corresponding chain. Seed10 is submitted separately as 24 dependency-free units, outside the
rolling chains; while both batches are active, at most 72 data-scale units are eligible or running.

The launcher accepts Slurm `--dependency afterok:JOBID[_TASKID]` specifications. Dependency
rewiring must resolve completed predecessors as already satisfied because Slurm may reject
adding a new dependency on an already-completed job.
For an incomplete unit with a resumable checkpoint, use `--resume-existing` and a fresh
`--status-tag NAME`; this preserves prior failure markers and refuses completed result artifacts.
New runs still reject any existing result leaf. Check all active writers before resubmission.
For a long dependency chain, set `AIM3_ROOT` to a read-only source snapshot and
`AIM3_ARTIFACT_ROOT` to the writable formal artifact directory outside that snapshot.

Generate the required `4h-uint8`, `10h-uint8`, and `20h-uint8` training splits with
`experiments/clutter/amarel/submit_clutter_data_scale_generate_uint8.sh`. The generator uses the
same exclusive-switch stimulus settings as 40h, records seed 42, and publishes each NPY/TSV pair
plus its completion manifest directly beside the retained `40h-uint8` files. Scale training keeps
using the existing `40h-uint8` validation split, so shorter validation/test copies are not made.

## h128 comparison (4h)

The SJC `h128 comparison` uses the six best6 model-specific LR/WD/dropout settings from
`clutter-data-scale-formal-40h-ep150-rnn-lstm-gru-gawf-mamba-s5-seed1`, with every recurrent
hidden size (Mamba/S5 `d_model`) set to 128 and one layer. S5 retains `state_size=128`.
It trains seeds 1-10 for 150 epochs with `patience=0`, batch 256, two DataLoader workers,
5-epoch checkpoints and automatic resume, using generated `4h-uint8` train and existing
`40h-uint8` validation data. This is equal-width, not parameter-matched, and its hyperparameters
are transferred rather than tuned at 4h/h128.

Each GPU has a serial 30-unit queue; total concurrency is two. Odd seeds use GPU0 and even
seeds use GPU1, in model order `rnn,lstm,gru,gawf,mamba,s5`. A completed unit is validated before
the same lane starts its successor. A failed or interrupted unit stops that lane for recovery.
Results use `data/clutter/runs/data_scale/clutter_h128_comparison_4h_ep150/<model>-seedNN/`.
The exact source snapshot, launcher, commands, generation evidence and smoke records are
registered in `experiments/monitoring/jobs/clutter-4h-h128-comparison-l1-ep150-rnn-lstm-gru-gawf-mamba-s5-seeds1-10.json`.

After all 60 training units validate, evaluate every identity-selected checkpoint on the
joint-balanced movie with 512-frame rollouts while excluding each rollout's initial frame. Copy
those test-accuracy and target-switch-recovery artifacts beside the local metrics JSON and PKL
files, then render the equal-width multiseed summary:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n aim3_rnn python -B \
  -m utils.analysis.clutter.h128_4h_multiseed_summary
```

The script writes seed-level CSV/NPZ and a manifest below
`results/data/analysis/G_behaviour/h128_4h_multiseed_summary/`, a development PNG below
`results/figs/G_behaviour/`, and the requested PDF at
`results/save/Fig2_clutter_4h_h128_multiseed_2x4.pdf`. The retained filename is historical; the
figure is a 2-by-3 layout showing 512-frame joint-balanced test accuracy, validation-loss
trajectories, and 512-frame joint-balanced target-switch recovery for Location and Identity.
Every point and SEM sample is one independent training seed.

## Non-multiplicative feedback controls

The reviewer-control campaign adds four isolated single-layer model types without changing the
original six model configurations:

| Model | Width | Full trainable parameters | LR | Weight decay |
|---|---:|---:|---:|---:|
| `gawf_additive` | 271 | 585,401 | 0.005 | 0.001 |
| `rnn_fb` | 272 | 586,867 | 0.001 | 0.00001 |
| `gru_fb` | 103 | 584,562 | 0.005 | 0.001 |
| `lstm_fb` | 79 | 585,406 | 0.001 | 0.001 |

The parameter-matching target is the complete GaWF `H=256` Clutter model with 586,067 trainable
parameters. All controls use the same detached previous-step 19-D raw-logit feedback and the
same model-family hyperparameters as the corresponding formal baseline; there is no tuning.

The earlier seeds 1--10 additive-feedback campaign used native no-wrap recurrent semantics and
an independent feedback affine with its own trainable bias. These counts and results are historical
under the current output-only dropout and RNN-ReLU definition:

| Model | Width | Full trainable parameters | LR | Weight decay |
|---|---:|---:|---:|---:|
| `rnn_fb_add` | 272 | 586,595 | 0.001 | 0.00001 |
| `gru_fb_add` | 103 | 584,665 | 0.005 | 0.001 |
| `lstm_fb_add` | 79 | 585,564 | 0.001 | 0.001 |

For each model, `W_fb f_(t-1) + b_fb` is added to the native cell preactivation. The feedback
parameters share the cell optimizer settings and initialization bound. This campaign reuses the
family LR/weight decay without additional tuning.

The pending rerun keeps that feedback affine and uses clean-state output-only dropout for all
three controls. `rnn_fb_add` additionally switches to the canonical in-loop LayerNorm-ReLU
equation, without tanh. At the previously used widths, the new full-model parameter counts are
587,139 (RNN-FB, H=272), 584,665 (GRU-FB, H=103), and 585,564 (LSTM-FB, H=79). Against the
586,067-parameter GaWF target, RNN-FB H=271 gives the closer match at 585,401 parameters;
use H=271 for the new parameter-matched rerun. The other widths and family optimizer settings
stay as recorded above. Use a distinct result root and protocol label for all 30 new units.

On SJC, `experiments/remote/run_sjc_clutter_feedback_controls.sh` runs a mandatory seed-1,
200-step sanity gate and then distributes four models times ten seeds across GPUs 0 and 1. Each
unit trains for 150 epochs with `patience=0`, evaluates reset-excluded test accuracy, and runs the
existing sequence-512 reset-excluded feedback-shuffle protocol. Outputs are isolated below
`results/data/clutter/runs/feedback_controls/clutter_feedback_controls_ep150_v1/` and
`results/data/analysis/feedback_controls_*_v1/`. The final CSV, JSON, and Markdown tables report
mean and SEM across ten seeds and are not copied into any paper source.

The Amarel replica uses
`experiments/clutter/amarel/submit_clutter_feedback_controls_formal.sh`. The login-node submitter
performs only bounded path/hash checks and submits a GPU preflight, a dependent `0-39` training
array, and a dependent aggregate job. The aggregate consumes a small immutable prerequisite leaf
containing the previously validated original-model test CSV and ten GaWF shuffle JSON files;
`SHA256SUMS` must validate before submission. Amarel writes the same relative result hierarchy
under its independent `AIM3_RESULTS_PATH`, so it never shares a writer with the SJC campaign.

## Dynamic/input-conditioned open-loop baselines

The reviewer comparison adds `mlstm`, `hyperlstm`, and `brims` without changing the CNN encoder,
task heads, loss, or training loop. All three are open-loop. Parameter matching follows the
historical complete-model procedure against the 586,067-parameter GaWF `H=256` model:

| Model | Architecture | Core parameters | Full parameters | Absolute target difference |
|---|---|---:|---:|---:|
| `mlstm` | `H=65` | 395,915 | 585,265 | 802 |
| `hyperlstm` | `H=70`, `hyper_H=10`, `n_z=4` | 396,572 | 586,017 | 50 |
| `brims` | `H=84`, blocks `6 3`, top-k `4 2` | 386,532 | 576,243 | 9,824 |

HyperLSTM's ratio `hyper_H/H=0.143` remains close to the paper configuration `128/1000=0.128`.
For BRIMs, `H` must be divisible by both 6 and 3; `H=84` is the closest candidate when only
`nhid` is scaled and attention/module structure is frozen. No auxiliary projection or attention
width was changed to pad the parameter count. Its official-core attention dimensions are input
attention `(heads=4, d_k=64, d_v=4H/n_blocks)` and within-layer communication
`(heads=4, d_k=32, d_v=32)`, with attention dropout `0.1`. Reproduce the deterministic search with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n aim3_rnn python -B \
  -m experiments.clutter.dynamic_weight_param_match
```

The mLSTM recurrence follows Krause et al. (2017), arXiv:1609.07959, Equations (17)--(21), and is
not the 2024 xLSTM matrix-memory model. HyperLSTM vendors only the required MIT-licensed labml
files at commit `33ab02281c2b928e6b32792909cc79cbdcfe1d6a`; equation deviations are recorded in
`third_party/labml_nn/UPSTREAM.md`. BRIMs is a clean-room implementation from Mittal et al.
(2020), arXiv:2006.16981. The official repository was inspected at commit
`f8af67e863ea751b45b70cc7a7b91fb277beb329`, but its absent license prevents source vendoring;
see `third_party/brims/PROVENANCE.md`.

Formal Amarel execution is split at the required sanity boundary. First,
`submit_clutter_dynamic_baselines_preflight.sh` runs LSTM plus the three new models for two full
epochs on seed 1 and reports finite loss reduction and wall-clock seconds per epoch relative to
LSTM. Only after that summary passes may `submit_clutter_dynamic_baselines_formal.sh` submit the
thirty independent units (`mlstm`, `hyperlstm`, and `brims`, seeds 1--10). The formal protocol is
150 epochs with `patience=0`, the standard 40h uint8 train/validation/test data, reset-excluded
test evaluation, and an isolated mean ± SEM aggregation. To avoid an unrequested search, all
three new models inherit the existing LSTM baseline optimizer setting: AdamW, learning rate
`0.001`, and weight decay `0.001`.
