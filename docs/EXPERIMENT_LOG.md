# Experiment Log

This file records confirmed model changes and experiment extensions for GaWF.
Keep entries factual: what changed, why it was introduced, and the current
implementation choice.

## 2026-07-13 - Multi-host Atari acceleration baseline

- Canonical code is an immutable Git commit, deployed to per-run directories with a
  `.source_commit` marker; long-lived dirty checkouts on Mac or Amarel are not deployment sources.
- Canonical full-training runtime is Python 3.11. Python 3.14 can import PyTorch 2.9 but cannot
  execute `torch.compile`, so compilation requests now warn and fall back to equivalent eager
  execution.
- `environments/aim3_rnn-linux-cuda.yml` captures the validated Amarel-compatible CUDA stack;
  `environments/aim3_rnn-macos.yml` is the development/test profile without CUDA-only Mamba.
- Single-task and multi-task Atari share BF16, TF32, cuDNN benchmark, fused Adam, replay, and GaWF
  feedback acceleration code. Multi-task collection defaults to transition-balanced scheduling;
  this is independent from task-balanced replay sampling.

## 2026-06-14 - GaWF Feedback Generalization

### Step 1: Baseline GaWF Feedback

- Model: `gawf`.
- Task heads: two linear classification heads are used, one for position
classification and one for character classification.
- Feedback source: the feedback vector is generated from the output vector of
the combined linear head.
- Current limitation: the legacy feedback dimension is `dz=19`, which is tied
to the current task design. If the task design changes, or even if one digit
class is removed from the same task, a GaWF model trained with this task-bound
`dz=19` feedback is no longer directly reusable.



### Step 2: Projected Feedback Dimension

- Goal: improve GaWF generalization by making the feedback dimension less tied
to a specific task output structure.
- Design reference: CFL-style projector layer.
- Projected feedback runs use a generic feedback dimension such as `dz=8`.
- Projected ClutterMNIST implementation: the original `dz=19` output is first
linearly projected into a `dz=8` feedback vector, then used by the existing
`U * fb * V` gating pathway.
- Difference from the CFL paper: this implementation uses a projector layer
only; it does not adopt the LoRA design from the original paper.
- Current experimental direction: evaluate whether `dz=8` preserves GaWF
performance while making the feedback representation more reusable across
task designs.



### Step 3: Multi-layer Projected GaWF

- Model: `gawf_multi`.
- Scope: implemented as a separate model type from single-layer `gawf` so
existing legacy and projected single-layer behavior is unchanged.
- Direct feedback is the default; specifying `--dz > 0` enables projected feedback
such as `dz=8`.
- Default recurrent depth: `--gawf_layers 2`; the CLI supports deeper stacks.
- Parameter sharing choice: U and V are not shared across layers. The model uses
one U/V pair per recurrent layer because V shape depends on each layer's input
size.
- Feedback source: in direct mode, the final recurrent layer uses previous
classifier output and non-final layers use the detached previous timestep's
adjacent upper-layer hidden state. In projected mode, both sources are linearly
projected to `dz`.



## 2026-06-18 - Single-layer GaWF Gated Matmul Memory Optimization

- Model: `gawf`.
- Motivation: `gawf hidden_size=512` full-grid jobs failed with CUDA OOM during
the first training epoch. The issue was peak activation memory, not parameter
count.
- Previous implementation: explicitly materialized `gated_weight_ih` and
`gated_weight_hh` with shapes `(B, H, I)` and `(B, H, H)` before batched
matrix multiplication.
- Current implementation: keeps the same per-sample `gate_ih` and `gate_hh`
definitions, but computes the contractions directly with `torch.einsum`.
This is algebraically equivalent to `(gate * weight)` followed by the same
input/hidden reductions, while avoiding the extra `gated_weight_*` tensors.
- Validation: on `sjc-remote`, a synthetic 5-step training A/B with the same
seed gave maximum loss difference `4.77e-7`; repeating the optimized path gave
`0.0` loss difference. For `hidden_size=512`, batch size 256, AMP enabled,
peak allocated GPU memory dropped from `2603.7 MB` to `1709.5 MB`.
- Optimizer details for the `gawf hidden_size=512` full-grid rerun:
the submitted job uses the single-layer `gawf` model path, not `gawf_multi`.
Single-layer `gawf` splits optimizer parameters into base parameters and
GaWF gating parameters (`U`, `V`). Base parameters use the task learning rate
and searched weight decay. `U` and `V` use the same task learning rate but
always set `weight_decay=0.0`. No `0.1` learning-rate scale is applied to
`U` or `V` for single-layer `gawf`.
- Related multi-layer note: `gawf_multi` uses only
`--gawf_multi_feedback_lr_scale` (default `0.1`) for learning-rate scaling.
Base parameters use the searched learning rate directly. U/V feedback-gating
parameter groups use `searched_lr * gawf_multi_feedback_lr_scale` and still
set `weight_decay=0.0`.



## 2026-06-18 - Multi-layer GaWF Learning-rate Scale Simplification

- Model: `gawf_multi`.
- Change: removed the CLI argument `--gawf_multi_lr_scale`.
- Reason: the previous implementation had two multiplicative scale knobs for
multi-layer GaWF learning rates, which made the effective base and feedback
learning rates harder to interpret.
- Current implementation: base parameters use the searched or requested
learning rate directly. The only multi-layer feedback-specific scale is
`--gawf_multi_feedback_lr_scale`, default `0.1`; U/V use
`lr * gawf_multi_feedback_lr_scale`.
- Weight decay: U/V remain in a no-weight-decay optimizer group
(`weight_decay=0.0`). Base parameters use the searched/requested weight decay.



## 2026-06-25 - GAWF Full-grid Scale and Hidden-size Selection

- Scope: single-layer `gawf` full hyperparameter grid over train scales `4h`,
`10h`, `20h`, and `40h`; hidden sizes `{64, 128, 256, 512}`;
`lr ∈ {0.0001, 0.0005, 0.001, 0.005}`; and
`weight_decay ∈ {0.0, 1e-5, 1e-4, 0.001}`.
- Completion: all 256 expected metrics units were present with companion `.pkl`
and `_model.pth` files. The final Amarel array `56778872` covered only the
`hidden_size=512` slice (`global task 240–255`, `496–511`, `752–767`, and
`1008–1023`) and wrote those rerun outputs into the same canonical
`gen_hparam_full_grid/task_*` result locations as the earlier full-grid jobs,
so `56778872` must not be interpreted as the full hidden-size grid by itself.
- Validation comparison: all entries below use
`eval_dataset_suffix=40h-float32`.
- Selection criterion: highest `val_acc_at_best` / character validation accuracy
within each train scale, allowing all hidden sizes.


| Train scale   | Best task   | Hidden size | lr    | weight decay | Best val char acc | Best val sector acc | Best val loss char | Best val loss pos | Epochs |
| ------------- | ----------- | ----------- | ----- | ------------ | ----------------- | ------------------- | ------------------ | ----------------- | ------ |
| `4h-float32`  | `task_0251` | 512         | 0.001 | 0.001        | 72.340912         | 86.585976           | 1.095375           | 0.433275          | 69     |
| `10h-float32` | `task_0494` | 256         | 0.005 | 0.0001       | 80.505589         | 89.774432           | 0.726175           | 0.342137          | 48     |
| `20h-float32` | `task_0762` | 512         | 0.001 | 0.0001       | 86.255484         | 92.014076           | 0.539668           | 0.260618          | 64     |
| `40h-float32` | `task_1007` | 256         | 0.005 | 0.001        | 90.093559         | 93.642535           | 0.361951           | 0.194324          | 72     |


- Current conclusion: under the shared 40h validation set, the best GAWF model
still improves with train scale, but the optimal hidden size is scale
dependent. The overall best 40h model remains `hidden_size=256` from
`task_1007` (`gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5`), which is
slightly above the best 40h `hidden_size=512` run (`task_1019`, best val char
`89.962573`).



## 2026-06-27 - 40h Param-matched Six-model Comparison (GaWF h=256 reference)

- Provenance: the earlier same-day six-model table in this file was **newly
  aggregated in chat** from Amarel metrics; it was **not** a pre-existing log
  entry. That table selected each model's best run from `gen_hparam_full_grid`
  / `gen_hparam_mamba_s5_grid` without width matching, so RNN/LSTM/GRU used
  `hidden_size=512`. This section replaces it with **parameter-count-matched**
  widths aligned to GaWF `task_1007` (`hidden_size=256`).
- Reference checkpoint: `gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5`
  (`gen_hparam_full_grid` / `task_1007`).
- Matched middle-path widths (from `utils_anal/model_param_counts.py`,
  sector mode, `cnn_dropout=0.0`, `rnn_dropout=0.5`, legacy GaWF feedback
  `dz=19`):
  - `rnn` -> `hidden_size=275`
  - `lstm` -> `hidden_size=80`
  - `gru` -> `hidden_size=105`
  - `gawf` -> `hidden_size=256` (reference)
  - `mamba` -> `d_model=170` (fixed in `ssm_mamba_hparam_grid.py`)
  - `s5` -> `d_model=256`, `state_size=128`, `s5_ssm_lr_scale=0.1` (128 chosen via
    `model_param_counts.py` to match GaWF h=256; legacy `state=189` was copied from
    DiagLTI and is ~8% over-parameterized)
- Search grids (all: `train_data_suffix=40h-float32`,
  `eval_data_suffix=40h-float32`, `lr in {0.0001, 0.0005, 0.001, 0.005}`,
  `weight_decay in {0.0, 1e-5, 1e-4, 0.001}`, selection = highest
  `val_acc_at_best` within each model at matched width):
  - `gawf`: `gen_hparam_full_grid` (reference row = `task_1007`).
  - `rnn` / `lstm` / `gru`: `gen_hparam_40h_param_match` with
    `--gawf-ref-hidden 256` (`48/48` valid on Amarel).
  - `mamba` / `s5`: `gen_hparam_mamba_s5_grid` (`mamba` 40h slice `39/40` valid;
    `s5` 40h slice `20/20` valid at `state=128`, rerun complete `2026-06-28`,
    superseding legacy `state=189` outputs; results stored on `/scratch`).
- Test evaluation backfill: on 2026-07-06, the six copied best checkpoints in
  `results/train_data/clutter_best_6model_param_matched_40h` were evaluated on
  `stimulus_reg-test-40h-float32` from `/scratch/js3269/stimuli`. The output
  artifact is
  `results/train_data/clutter_best_6model_param_matched_40h/test_acc_40h_eval.json`.
  Evaluation reused the training eval protocol; Mamba was run on CPU with
  PyTorch reference kernels (`causal_conv1d_fn=None`,
  `selective_scan_fn=selective_scan_ref`) because the installed fast kernels
  require CUDA.

| Model | Params | Grid | Best task | Width | lr | weight decay | Train char acc | Val char acc | Test char acc | Train sector acc | Val sector acc | Test sector acc | Char gap | Val loss char | Val loss pos | Epochs |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gawf` | 586,067 | `gen_hparam_full_grid` | `task_1007` | `hidden_size=256` | 0.005 | 0.001 | 93.332563 | 90.093559 | 85.619700 | 95.230321 | 93.642535 | 92.699464 | 3.24 | 0.361951 | 0.194324 | 72 |
| `mamba` | 587,275 | `gen_hparam_mamba_s5_grid` | `task_0131` | `d_model=170` | 0.001 | 0.001 | 91.620527 | 86.565073 | 82.672282 | 95.321002 | 93.375942 | 92.224585 | 5.06 | 0.444332 | 0.188731 | 34 |
| `gru` | 586,905 | `gen_hparam_40h_param_match` | `task_0047` | `hidden_size=105` | 0.005 | 0.001 | 88.757107 | 84.831456 | 80.168615 | 92.553929 | 92.281887 | 90.595980 | 3.93 | 0.504102 | 0.226703 | 31 |
| `rnn` | 586,865 | `gen_hparam_40h_param_match` | `task_0009` | `hidden_size=275` | 0.001 | 1e-05 | 91.302709 | 84.149388 | 79.674000 | 93.384007 | 91.892570 | 89.924517 | 7.15 | 0.536043 | 0.241285 | 62 |
| `lstm` | 584,675 | `gen_hparam_40h_param_match` | `task_0027` | `hidden_size=80` | 0.001 | 0.001 | 89.948382 | 83.607920 | 79.806536 | 92.823792 | 91.607884 | 90.232043 | 6.34 | 0.557105 | 0.253663 | 42 |
| `s5` | 587,475 | `gen_hparam_mamba_s5_grid` | `task_0148` | `d_model=256`, `state=128` | 0.001 | 0.0 | 86.168562 | 79.998780 | 75.387371 | 91.545977 | 90.728542 | 88.939664 | 6.17 | 0.653471 | 0.263917 | 98 |

- Param counts: `python utils_anal/model_param_counts.py --hidden_rnn 275
  --hidden_lstm 80 --hidden_gru 105 --hidden_gawf 256 --mamba_d_model 170
  --s5_d_model 256 --s5_state_size 128` (Amarel `aim3_rnn`, 2026-06-27).
  RNN/LSTM/GRU/GAWF/Mamba/S5 are within ~0.3% of each other at these widths.
- Legacy note: the first S5 grid used `state=189` (DiagLTI default, 634,445
  params). That campaign was removed on Amarel; reruns use the standard launcher
  `experiments/amarel/submit_ssm_mamba_hparam_grid_batches.sh --model s5`.
- Checkpoint stems:

| Model | Checkpoint stem |
| --- | --- |
| `gawf` | `gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5` |
| `mamba` | `mamba_sector_acc_dmodel170_lr0.001_wd0.001_cdo0.0_rdo0.5` |
| `gru` | `gru_sector_acc_h105_lr0.005_wd0.001_cdo0.0_rdo0.5` |
| `rnn` | `rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5` |
| `lstm` | `lstm_sector_acc_h80_lr0.001_wd0.001_cdo0.0_rdo0.5` |
| `s5` | `s5_sector_acc_dmodel256_state128_lr0.001_wd0.0_cdo0.0_rdo0.5` |

- Current conclusion (param-matched, 40h train + 40h validation):
  1. **GAWF** (`task_1007`) still leads on val char (`90.09%`) and test char
     (`85.62%`) with the smallest char gap (`3.24` pt) among the top models.
  2. **Mamba** (`task_0131`) is second on char (`86.57%`) and closest on sector
     (`93.38%` vs GAWF `93.64%`).
  3. At matched width, **GRU/RNN/LSTM** all fall to `83.6-84.8%` val char,
     substantially below the earlier unmatched `h=512` full-grid peaks
     (`85.5-86.0%`), indicating the prior advantage was partly capacity-driven.
  4. **S5** at fair `state=128` (`task_0148`) is weakest of the six: `80.00%` val
     char and lowest val sector (`90.73%` vs GAWF `93.64%`), ran the full grid
     without early stop (98 epochs) at a small best `lr=0.001`—suggesting weak
     convergence/underfitting at this ~587K param budget rather than a capacity
     gap. (Supersedes the removed legacy `state=189` campaign.)
