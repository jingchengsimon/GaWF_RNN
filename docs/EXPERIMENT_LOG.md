# Experiment Log

This file records confirmed model changes and experiment extensions for GaWF.
Keep entries factual: what changed, why it was introduced, and the current
implementation choice.

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
