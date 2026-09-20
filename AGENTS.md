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
| Any model or training change | `run_task.py`, `utils/training/` | `docs/ARCHITECTURE.md` |
| CLI, filenames, saved results | all public scripts | `docs/CONVENTIONS.md` |
| Analysis or visualisation | `utils/analysis/` | `docs/DEVELOPMENT_WORKFLOWS.md` |
| ICLR paper prose or captions | manuscript drafts and figure captions | `docs/ICLR_PAPER_WRITING.md` |
| Task experiment definitions | `experiments/{clutter,atari,minigrid,text}/` | `experiments/README.md` and the task README |
| Generalization experiments | `experiments/clutter/` | `experiments/clutter/README.md` |
| Amarel jobs | `experiments/<task>/amarel/` | `experiments/<task>/amarel/README.md` and `docs/operations/REMOTE_EXECUTION.md` |
| sjc-remote jobs | `experiments/remote/`, `experiments/<task>/amarel/` | `experiments/remote/README.md` and `docs/operations/REMOTE_EXECUTION.md` |
| Research history | confirmed model/protocol changes | `docs/EXPERIMENT_LOG.md` |

Local host aliases and absolute paths live in `.agents/local.md`, which is ignored by Git. If it
is missing, copy `.agents/local.example.md` and fill it in. Do not guess remote endpoints.

## Non-negotiable repository rules

- When renaming, moving, splitting, or deleting a module or public symbol, search the entire
  repository and update every affected import and call site in the same change.
- Task-specific entry points and data/model wrappers stay separate, while recurrent mathematics
  is shared through `utils/training/recurrent_cores/`. Do not reimplement GaWF/RNN logic in task wrappers.
- Dependency direction is `utils/training/` -> task entry points -> `utils/analysis/` -> file
  outputs. Never import `utils.analysis` from `utils.training`. Each retained analysis entry point owns both its
  numeric analysis and plotting from structured numeric outputs.
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
- Acceleration or optimization changes are accepted by a task-level variance protocol, not by
  requiring bitwise or fixed-tolerance eager equivalence.  Compare matched repeated baseline and
  accelerated runs across the available GPUs: the accelerated runs must not exceed the baseline
  run-to-run numerical/metric dispersion and must not show a systematic shift outside that
  baseline envelope.  Record the fixed-input numerical diagnostic separately; it identifies the
  source of differences but cannot replace the end-to-end RL variance check.

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
  An accepted smoke is the sole exception: its exact smoke result leaf and its exact smoke
  artifact/log leaf is temporary validation output handled under the global smoke-acceptance and
  exact-target cleanup rules. Failed, paused, or recovering smoke outputs remain protected.
- User-authorized data deletion must name exact target leaves. When the user explicitly authorizes
  irreversible deletion, a recovery copy is not required; still resolve every target, inspect an
  itemized pre-delete listing, and never delete a parent directory or unlisted sibling.

## Python and script baseline

- Target Python 3.10+, PyTorch 2.0+, and the `aim3_rnn` Conda environment.
- All project experiment, analysis, and remote Python invocations must set
  `PYTHONDONTWRITEBYTECODE=1` (or use `python -B`) so local and remote runs do not create
  `__pycache__/` directories. Tracked launchers and activation wrappers must export it by default.
- Public functions require type hints; modules require purpose/input/output docstrings; wildcard
  imports are forbidden; line length is 100 characters.
- Training progress uses the logger, not `print()`.
- Scope `torch.no_grad()` to inference blocks; it must not become persistent model state.
- Analysis/model loading must reuse the canonical helpers from
  `utils.analysis.anal_helpers` rather than rebuilding models independently.
- Visualisation is headless, closes every figure after saving, and follows the styles in
  `docs/DEVELOPMENT_WORKFLOWS.md`.
- Every visualisation result must be regenerated by a plotting script from its raw or structured
  numeric outputs (for example CSV, NPZ, NPY, or PKL), regardless of the model used. Never
  create a new result by simply cropping, compositing, or relabelling an existing raster figure.

## Remote execution

### 常规操作模板

对于已验证 launcher 的常规训练提交、状态查询和结果可视化，遵循
`docs/operations/REMOTE_EXECUTION.md` 中的 "Routine operation templates"。除非 launcher、
资源规格、恢复语义或结果协议发生改变，不要把首次开发的逐项排查流程重复用于常规操作。
一次操作应以模板规定的一次合并预检、一次提交/本地渲染和一次验证为界；不要为无变化的
状态重复同步、重复 dry-run、重复安全测试或逐轮提交绘图 job。

小规模分析、可视化和已有结构化结果的图更新默认在本地直接运行，不需要 smoke，也不得仅为
执行这类工作提交 Amarel job。只有源数据无法安全获取到本地或确实需要远端专用计算时，才按
远端 runbook 申请计算节点。

训练默认遵循所属 protocol 的 smoke gate；人类 prompt 明确要求跳过 smoke 时允许跳过，
并在 launcher 参数与实验 manifest 中记录该授权。跳过 smoke 不豁免 SSH 复用、提交脚本
安全检查、精确输出路径检查或 checkpoint/recovery 约束。

- Before remote diagnostics, tests, training, or result inspection, read the remote runbook and
  local configuration. Use the `aim3_rnn` environment; never use the remote default Python.
- Treat every Amarel login node as control-plane only. A `submit_*.sh` launcher may perform
  bounded shell/stdlib validation and scheduler/file-status operations, but must never activate
  Conda or directly run training, inference, preprocessing, parameter matching, smoke tests,
  visualization, project-module imports, or any PyTorch/NumPy/JAX/TensorFlow workload. Put all
  such work in an `sbatch`-launched `run_*.sh`, including preflight jobs, and connect dependent
  arrays with `afterok`.
- Amarel compute nodes are not guaranteed to provide `git` or login-node paths. Resolve source
  identity through `experiments/remote/amarel_source_guard.sh` plus the submit-time
  `source_commit.txt` stamp, never by calling git or hard-coding a login-node path inside a
  `run_*.sh`, and always write a fail marker before a guard exit.
- Before synchronizing or executing any new or modified Amarel `submit_*.sh`, run
  `python -m pytest -q experiments/tests/test_amarel_submit_safety.py` on the local development host (or in a
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
- Apply the global remote-experiment identity and SSH-socket workflow using
  `experiments/monitoring/jobs/<experiment-id>.json` as AIM3's only per-run fact source,
  `experiments/monitoring/JOBS.md` as the human/agent index, and
  `experiments/monitoring/active_jobs.json` as the active index. After every submission, run
  `PYTHONDONTWRITEBYTECODE=1 python -m experiments.monitoring.job_registry rebuild`.
- AIM3 experiment IDs are host-neutral. Prefixes encode the applicable hierarchy exactly:
  `clutter-...`, `text-...`, `rl-minigrid-...`, `rl-atari-multitask-...`,
  `rl-atari-breakout-...`, or `rl-atari-pong-...`. Follow the prefix with protocol, variant,
  budget, models, and seeds; never omit a required hierarchy level to create a shorter ambiguous
  ID. Store host and execution identity separately: logical host in `host`, Amarel Slurm IDs in
  `scheduler.job_ids`, and SJC tmux/run IDs in `scheduler.run_ids`.
- For AIM3 status checks, invoke
  `PYTHONDONTWRITEBYTECODE=1 python -m experiments.monitoring.progress` exactly once with
  `<experiment-id> --timeout 30 --no-update`. Report `completed/expected`, latest step or unit
  progress, ETA when supported, and any explicit failure reason. Do not use manual remote
  diagnostics unless the user expands the scope after an explicit checker error.
- Before resubmitting an existing experiment unit, query all active scheduler jobs and process
  commands for the exact result suffix across historical job IDs/worktrees. Final-result absence
  alone is not evidence that no older writer is still active.
- Reusable launchers may be tracked. Generated Slurm scripts stay under
  `experiments/<task>/amarel/generated/`; clearly marked one-off scripts must be removed before branch
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
