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
- Current choice: use a generic feedback dimension of `dz=8`.
- Current ClutterMNIST implementation: the original `dz=19` output is first
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
- Default projected feedback dimension: `dz=8`.
- Default recurrent depth: `--gawf_layers 2`; the CLI supports deeper stacks.
- Parameter sharing choice: U and V are not shared across layers. The model uses
  one U/V pair per recurrent layer because V shape depends on each layer's input
  size.
- Feedback source: the final recurrent layer uses projected classifier output;
  non-final layers use a projector from the previous timestep's upper-layer
  hidden state.
