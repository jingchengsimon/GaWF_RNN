# AGENTS.md — FAW_RNN Project Constraints

始终用中文回答用户问题，公式和术语用英文。
This is the single entry point for coding agents. Keep it short. Read the routed document for
the task before editing; detailed architecture, naming, experiment, and remote procedures do not
belong here.

## Project map

The repository trains and analyses recurrent models, especially GaWF
(Gated-Weight-on-Feedback), across clutter vision, text, and control tasks.

| Task | Entry points | Required reference |
|---|---|---|
| Any model or training change | `train_*.py`, `utils/` | `docs/ARCHITECTURE.md` |
| CLI, filenames, saved results | all public scripts | `docs/CONVENTIONS.md` |
| Analysis or visualisation | `utils_anal/`, `utils_viz/` | `docs/DEVELOPMENT_WORKFLOWS.md` |
| Task experiment definitions | `experiments/{clutter,atari,minigrid,text}/` | `experiments/README.md` and the task README |
| Generalization experiments | `experiments/generalization/` | `experiments/generalization/README.md` |
| Amarel jobs | `experiments/amarel/` | `experiments/amarel/README.md` and `docs/operations/REMOTE_EXECUTION.md` |
| sjc-remote jobs | `remote/`, `experiments/local/` | `remote/README.md` and `docs/operations/REMOTE_EXECUTION.md` |
| Research history | confirmed model/protocol changes | `docs/EXPERIMENT_LOG.md` |

Local host aliases and absolute paths live in `.agents/local.md`, which is ignored by Git. If it
is missing, copy `.agents/local.example.md` and fill it in. Do not guess remote endpoints.

## Non-negotiable repository rules

- When renaming, moving, splitting, or deleting a module or public symbol, search the entire
  repository and update every affected import and call site in the same change.
- Task-specific entry points and data/model wrappers stay separate, while recurrent mathematics
  is shared through `utils/recurrent_cores/`. Do not reimplement GaWF/RNN logic in task wrappers.
- Dependency direction is `utils/` -> task entry points -> `utils_anal/` -> file outputs read by
  `utils_viz/`. Never import `utils_anal` or `utils_viz` from `utils`, and never import
  `utils_viz` from `utils_anal`.
- New analysis belongs in `utils_anal/`; new plotting belongs in `utils_viz/`.
- Register clutter model types in `utils/clutter_train_helpers.get_model_classes()`.
- Construct clutter losses through the factories in `utils/clutter_train_sector.py` or
  `utils/clutter_train_predict_all_chars.py`; keep the loop body in
  `utils/clutter_train_engine.py`.
- `AccelerationConfig` is the single source of truth for clutter AMP and gradient accumulation.
  Acceleration must not change sampling, losses, update cadence, UTD, or model structure.
  `--shuffle_block_size` is an explicit recorded data-pipeline protocol independent of AMP.

## Architecture contracts

- The clutter CNN output is fixed at `(32, 6, 6)`, flattened to 1152 features. Any shape change
  requires a migration note and simultaneous updates to all downstream analyses.
- Atari uses a separate Nature-DQN encoder for 84x84 observations. Do not reuse the clutter CNN.
- Atari A2C supports LSTM/GaWF with feedback modes `none` and GaWF `output`. Atari DQN/DRQN
  supports ANN/RNN/GRU/LSTM/GaWF/S5/Mamba; GaWF `qvalues` feedback is the detached previous
  Q-value vector.
- Pong result labels must state both protocol settings: `pong_fs1_stack1` or
  `pong_fs4_stack1`. Never introduce `1frame` or `pong1f` as a protocol name.
- GaWF has one public model type, `gawf`, and uses `--num_layers`. Direct multi-layer feedback
  uses adjacent upper hidden state for non-final layers and previous output for the final layer;
  `--dz > 0` enables per-layer projected feedback.
- GaWF U/V/projector parameters use no weight decay and
  `base_lr * --gawf_feedback_lr_scale` (default scale `1.0`).
- `prev_feedback` is runtime state, not a learned parameter. Filter it when loading checkpoints,
  use `strict=False`, and report missing and unexpected keys.

## Data and result safety

- Keep datasets on CPU or mmap; do not load a full dataset into GPU memory.
- Standard Clutter 40h training uses mmap uint8, device-side float32 cast, compact frame windows,
  block shuffle sized to the effective batch, `num_workers=2`, and CUDA pinned memory. The
  legacy `sample/stacked/global/0-workers` path remains an explicit reproduction fallback.
- Standard long Clutter runs atomically checkpoint every 5 completed epochs and enable automatic
  resume. A preempted run must not emit or overwrite final result artifacts from partial state.
- Explicitly cast saved NumPy arrays to `np.uint8`, `np.float32`, or `np.int64` as required
  by the documented storage/tensor contract.
- Preserve existing checkpoint and result naming contracts in `docs/CONVENTIONS.md`, including
  compatibility with historical `gawf_multi_` and `_do` filenames.
- Do not delete experiment results, checkpoints, or pending-cleanup records without explicit
  human confirmation. Completion, failure, timeout, or staleness is not deletion permission.

## Remote synchronization safety

- Never run `rsync --delete` against a repository root, `results/`, `stimuli/`, or another broad
  ancestor, and never combine `--delete` with multiple sources.
- Never flatten a trailing-slash source directory into a repository root. Sync one source to its
  exact homologous leaf destination.
- A deletion-enabled sync is allowed only for an explicitly requested generated-output leaf and
  requires the exact command to pass `--dry-run --itemize-changes` inspection first.
- Before `rm -rf`, `find -delete`, or equivalent cleanup, require non-empty variables, resolve the
  target, and assert it is the exact human-authorized leaf with a verified recovery copy.
- After synchronization, verify the destination, expected file count, and protected siblings.
  Missing or unexpected paths are a stop condition, not permission for follow-up cleanup.

## Python and script baseline

- Target Python 3.10+, PyTorch 2.0+, and the `aim3_rnn` Conda environment.
- Public functions require type hints; modules require purpose/input/output docstrings; wildcard
  imports are forbidden; line length is 100 characters.
- Training progress uses the logger, not `print()`.
- Scope `torch.no_grad()` to inference blocks; it must not become persistent model state.
- Analysis/model loading must reuse the canonical helpers from
  `utils_anal.anal_helpers` rather than rebuilding models independently.
- Visualisation is headless, closes every figure after saving, and follows the styles in
  `docs/DEVELOPMENT_WORKFLOWS.md`.

## Remote execution

- Before remote diagnostics, tests, training, or result inspection, read the remote runbook and
  local configuration. Use the `aim3_rnn` environment; never use the remote default Python.
- Treat every Amarel login node as control-plane only. A `submit_*.sh` launcher may perform
  bounded shell/stdlib validation and scheduler/file-status operations, but must never activate
  Conda or directly run training, inference, preprocessing, parameter matching, smoke tests,
  visualization, project-module imports, or any PyTorch/NumPy/JAX/TensorFlow workload. Put all
  such work in an `sbatch`-launched `run_*.sh`, including preflight jobs, and connect dependent
  arrays with `afterok`.
- Before synchronizing or executing any new or modified Amarel `submit_*.sh`, run
  `python -m pytest -q tests/test_amarel_submit_safety.py` on the local development host (or in a
  Slurm compute job), never on an Amarel login node. A failing safety test is a stop condition; do
  not bypass it or add an exception for a new launcher. Also use the launcher's `--dry-run` when
  available, and verify submitted work is assigned to a compute node.
- Consolidate related Amarel queries into one foreground SSH session. Do not open background SSH
  sessions; use the documented single-heredoc fallback only when direct SSH cannot proceed.
- Codex-submitted Amarel training requests use one Ada Lovelace GPU, 16 CPUs, 64G memory,
  and an explicit `AIM3_RESULTS_PATH`, unless a human explicitly specifies otherwise. General
  tasks use `AIM3_NUM_WORKERS=12`, while standard Clutter 40h mmap runs use the benchmarked
  `AIM3_NUM_WORKERS=2`; both use `AIM3_PIN_MEMORY=1` on CUDA compute nodes.
- After submission, report the job/run ID, remote root, result location, requested resources, and
  the status/check command, then register it in `experiments/monitoring/`.
- Before resubmitting an existing experiment unit, query all active scheduler jobs and process
  commands for the exact result suffix across historical job IDs/worktrees. Final-result absence
  alone is not evidence that no older writer is still active.
- Reusable launchers may be tracked. Generated Slurm scripts stay under
  `experiments/amarel/generated/`; clearly marked one-off scripts must be removed before branch
  synchronization.
- Maintain one long-lived repository per endpoint. Task-named worktrees are temporary local
  development aids only; formal runs use an explicit commit or read-only snapshot.

## Documentation maintenance

- Update the owning reference document whenever behavior, public CLI, defaults, paths, metrics,
  or result naming changes. Do not duplicate the same detailed rule in multiple documents.
- `docs/EXPERIMENT_LOG.md` is a concise human research history, not an engineering changelog.
  Write it in Chinese while preserving English technical terms, identifiers, metrics, and
  formulas. Add only confirmed method changes, protocol corrections, decisive evidence, or
  conclusions.
