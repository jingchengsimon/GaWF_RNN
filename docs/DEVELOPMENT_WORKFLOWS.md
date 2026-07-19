# Development Workflows

Read this document before adding or changing training, analysis, or visualisation scripts. Public
names and output formats are defined in `CONVENTIONS.md`; architecture boundaries are defined in
`ARCHITECTURE.md`.

## General Python rules

- Use Python 3.10+ type hints on public functions and `from __future__ import annotations`.
- Start every module with a docstring describing purpose, inputs, and outputs.
- Keep lines within 100 characters and avoid wildcard imports.
- Use the project logger for training progress; `print()` is acceptable for standalone analysis
  progress and explicit CLI diagnostics.
- Create output directories before writing and save NumPy arrays as `float32` or `int64`.
- Preserve unrelated worktree changes and update all imports/call sites after public renames.

## Training changes

### Clutter

- Keep `train_model.py` as orchestration; heavy loop logic belongs in
  `utils/clutter_train_engine.py`.
- Register model types through `utils/clutter_train_helpers.get_model_classes()`.
- Build losses through the factories in `clutter_train_sector.py` or
  `clutter_train_predict_all_chars.py`.
- Put recurrent computation in `utils/recurrent_cores/`, not task wrappers.
- Extend `AccelerationConfig` rather than adding acceleration branches to the training loop.
- Official curves use the train-eval and validation passes, not online batch averages.
- Standard long jobs pass `--checkpoint_interval_epochs 5 --auto_resume`. Checkpoints are atomic
  and occur only after a complete epoch, so interruption loses at most four completed epochs.
  Resume rejects model, optimizer, data-pipeline, seed, or hyperparameter mismatches. Dataset
  samples are deterministic, so restoring the loader and sampler generators is sufficient even
  though persistent worker process internals are recreated.
- A signal-triggered stop must retain the last periodic training checkpoint and exit without
  writing final `.pkl`, metrics JSON, or best-model artifacts from partial state.

### mmap and devices

- **Historical reproduction only:** the legacy `40h-float32` mmap pipeline used
  `num_workers=0` and `pin_memory=False`. Keep this configuration available when reproducing
  historical runs, but do not treat it as a general mmap requirement or the default for new
  experiments.
- **Current standard Clutter 40h configuration:** use `40h-uint8` with `--use_mmap`,
  `--input_cast_mode device`, `--frame_layout compact`, and `--shuffle_block_size -1` together
  with `AIM3_NUM_WORKERS=2` and `AIM3_PIN_MEMORY=1` on CUDA compute nodes. The loaders use
  persistent workers and `prefetch_factor=2`; the batch-sized block sampler preserves epoch
  coverage while reducing random shared-filesystem access.
- mmap and pinned memory are not inherently incompatible: mmap backs the CPU dataset, while
  pinning applies to the collated uint8 batches transferred asynchronously to CUDA. Do not raise
  the standard two-worker value without an endpoint-specific benchmark; more workers can increase
  page faults and shared-filesystem contention.
- Convert float64 inputs to float32 before MPS/CUDA transfer.
- Do not load the full dataset onto the accelerator.
- Scope `torch.no_grad()` to evaluation/inference blocks.
- For complex-parameter cores such as S5, AMP autocast may remain enabled while GradScaler and
  foreach clipping exclude unsupported complex gradients.

### Checkpoint loading

Use the canonical compatibility pattern and always report incompatibilities:

```python
state_dict = torch.load(ckpt_path, map_location=device)
state_dict = {key: value for key, value in state_dict.items() if key != "prev_feedback"}
incompatible = model.load_state_dict(state_dict, strict=False)
print("missing_keys:", incompatible.missing_keys)
print("unexpected_keys:", incompatible.unexpected_keys)
```

## Analysis scripts

New analysis belongs in `utils_anal/` and exports one logical result set per invocation.

Required structure:

```python
"""One-line summary.

Describe inputs, computation, and every output with shape and dtype.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Reuse the canonical helpers:

```python
from utils_anal.anal_helpers import build_model_from_ckpt, build_test_dataset
```

Analysis requirements:

- Resolve every analysis data/figure destination with
  `utils_anal.anal_paths.output_dir(category, script_name, kind)`. The only output tree is
  `results/anal_index/<CATEGORY>/<script_name>/{data,figs}/`; do not recreate the legacy split
  roots or a symlink index view.
- Each run writes a sibling `manifest.json` containing script path, commit, timestamp, category,
  files written, and a flat dictionary of key numerical results.
- Accumulate averages/statistics in float64 and cast to float32 before saving.
- Use `.npy` for one array and `.npz` for related arrays.
- Save companion metadata with mode, selected index, frame/sample counts, model/input sizes,
  aggregation mode, absolute checkpoint path, and spatial/feature shapes.
- Print qualifying-sample progress every 200 samples.
- Raise `RuntimeError` when no frames match; do not silently emit empty outputs.
- Do not import plotting code from `utils_viz/`.

### Unified GaWF variance decomposition

Use `utils_anal.variance_decomposition` for encoder activation, input/recurrent gate synapses,
effective input/recurrent weights, hidden state, and feedback/readout vectors. Every run balances
all 90 sector-digit cells to a common `n`, repeats the subsample for 20 fixed-seed draws, and
reports aggregate plus per-unit condition-mean and trial-level fractions. Gate/effective-weight
unit axes index synapses, not neurons. Trial-level gate analysis must stream second-order moments
under an explicit memory budget; a trial-by-synapse array is forbidden.

`utils_anal/run_unified_variance_decomposition.py` reads saved mmap `.npy` representations,
including the input and recurrent gate tensors. A saved GaWF trajectory may supply labels,
feedback, and static weights only; the runner never reconstructs gates from `U/V`, reruns the
model, or regenerates activations. Missing trial-level representations are a hard failure. When
those saved representations do not yet exist, run
`utils_anal/export_unified_variance_sources.py` once on a CUDA host with enough disk space. The
exporter loads the canonical checkpoint/test dataset, writes frame-major float32 mmap sources
without materializing a complete trial-by-synapse tensor, and emits the runner input manifest.
Use
`utils_anal/migrate_analysis_outputs.py` to plan or apply the one-time legacy output move;
ambiguous mixed artifacts remain in place and appear in its migration report.

## Visualisation scripts

New plotting belongs in `utils_viz/` and reads saved result files rather than loading models.

- Call `matplotlib.use("Agg")` before importing pyplot.
- Default to 150 DPI and save with `bbox_inches="tight", pad_inches=0.06`.
- Close every figure immediately after saving.
- Use `RdBu_r` with symmetric limits for diverging heatmaps and `viridis` for sequential data
  unless the CLI provides another colormap.
- Load feature order from `channel_order_by_cosine_similarity.npy` and hidden-unit order from
  `sorted_npz_order.npy`; gracefully fall back to natural order.
- Draw boundary/highlight lines in red with linewidth 0.7.
- For N components plus sum and full panels, use three columns and
  `ceil((N + 2) / 3)` rows; hide unused axes.
- Use `visualize_batch.sh` for saved training metrics unless a custom figure is explicitly
  requested. It dispatches clutter `.pkl` histories and Atari `metrics_history.jsonl` to their
  task-specific plotting modules. For a multi-task Atari run, the default figure contains one
  `episodic_return_100` curve per environment; use `--include_combined` only for diagnostic plots
  that intentionally pool episodes with different score scales.

## Shell launchers

- Use non-interactive commands where possible and fail early with `set -euo pipefail` (bash) or
  the equivalent zsh options.
- Preserve a preset `CUDA_VISIBLE_DEVICES`.
- Keep reusable launchers tracked, generated Slurm files ignored, and one-off recovery scripts
  clearly marked and removed before synchronization.
- Result suffixes must encode protocol-changing settings, including Pong frame skip and stack.

## Review checklist

- [ ] Read the owning architecture/convention/runbook document.
- [ ] Updated all imports and call sites after public changes.
- [ ] Preserved dependency direction and shared recurrent cores.
- [ ] Added/updated tests for changed behavior and compatibility.
- [ ] Kept dataset memory and device handling safe.
- [ ] Preserved result names, dtypes, metadata, and checkpoint loading behavior.
- [ ] Closed figures and handled empty analysis selections explicitly.
- [ ] Updated the owning documentation without duplicating it elsewhere.
- [ ] Added an `EXPERIMENT_LOG.md` entry only if the change is research-significant and confirmed.
