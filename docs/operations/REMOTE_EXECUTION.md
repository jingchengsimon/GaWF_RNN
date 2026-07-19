# Remote Execution Runbook

This runbook owns agent-facing remote access and execution rules. Machine-specific aliases,
project roots, data roots, and Conda initialization paths live in the ignored
`.agents/local.md`; wrapper-specific values live in the ignored `remote/config.sh`.

## Before connecting

1. Read `.agents/local.md`. If it is missing, copy `.agents/local.example.md` and ask the human
   for the missing values; never guess endpoints or paths.
2. Identify the target host, project root, data root, Conda initialization script, and status
   command before starting work.
3. Preserve uncommitted local and remote changes. Remote synchronization must not implicitly add,
   commit, reset, or discard files.

## Environment

- On every configured remote host, activate Conda environment `aim3_rnn` before tests, training,
  visualization, or Python diagnostics.
- Do not use the default shell Python, `module`, or an unrelated virtual environment.
- Amarel Slurm scripts source the configured Conda initialization script and then run
  `conda activate aim3_rnn`.
- Amarel training submissions explicitly export `AIM3_RESULTS_PATH`; do not depend on `.bashrc`
  or `--export=ALL` for the physical result root.
- Standard Clutter 40h runs on both endpoints use the shared benchmarked pipeline:
  `num_workers=2`, CUDA pinned memory, uint8 mmap, device cast, compact layout, and
block size equal to the effective batch. Per-run overrides remain available for reproduction.
Standard long Clutter runners save atomic training state every five completed epochs and enable
automatic resume. Requeueing or explicit resubmission may lose at most the epochs completed after
the last checkpoint; partial jobs must not write final result artifacts while stopping.

Result-root resolution is: explicit `--results_dir`, then `AIM3_RESULTS_PATH`, then
`FAW_RNN_RESULTS_PATH`, then `<repo>/results`. Emitters, validators, status tools, rerun logic,
and training must use the same resolved root.

## SSH diagnostics

- Prefer the SSH aliases recorded in `.agents/local.md`; do not embed usernames or addresses in
  tracked documentation.
- Combine related scheduler, log, metrics, and filesystem checks into one foreground SSH call.
- Never start background SSH sessions. Allow up to 120 seconds for a consolidated diagnostic.
- If a reused connection is stale, clean up the matching local SSH process before one retry.
- For complex checks, send one script and execute it in the same SSH session rather than opening
  several connections.

If direct SSH requires interactive authentication and the agent cannot continue, give the human
one pasteable block:

```bash
bash <<'EOF'
# all related commands
EOF
```

Do not split loops, variables, or related checks across multiple paste blocks.

## Amarel training submissions

Unless a human explicitly requests a different allocation, Codex-submitted training uses:

- partition `gpu-redhat`, account `general`;
- one GPU with `constraint=adalovelace`;
- `cpus-per-task=16`, `mem=64G`;
- `AIM3_PIN_MEMORY=1` exported at submission time;
- general tasks use `AIM3_NUM_WORKERS=12`; standard Clutter 40h mmap tasks use the benchmarked
  `AIM3_NUM_WORKERS=2`.

The Clutter input-pipeline defaults are shared across sjc-remote and Amarel. Other task families
retain their task-specific DataLoader values. Use a specific node only when reproducing a
documented node-level experiment.

After submission, verify and report:

- Slurm job ID or stable remote run/process ID;
- target host and remote project root;
- requested GPU/CPU/memory/constraint and active environment;
- result suffix or result directory;
- the single command used to inspect status and valid outputs.

In the same turn, register the job in the project-local monitoring registry described in
`experiments/monitoring/README.md`. Record every scheduler/run ID, the exact remote root, log and
result paths, expected units, and validity evidence. This internal registry replaces ad-hoc
cross-thread searching.

Scheduler state is not result validity. Completion checks must inspect expected metrics and any
required checkpoint/pickle companions.

Before a replacement or recovery submission, search active scheduler entries and process command
lines for the exact target result suffix across every known historical job ID and worktree. Do not
infer that a unit is idle merely because its final files are absent or its current manifest omits
an older job. If an older writer targets the same path, cancel and confirm it has exited before
starting the replacement.

### Login-node safety boundary

Amarel login nodes are control-plane hosts, not execution hosts. This boundary applies even to a
short smoke test or one-time setup command, and an SSH timeout is not process containment: a child
process may survive after the SSH client returns.

Allowed in `experiments/amarel/submit_*.sh`:

- bounded shell argument, path, and environment validation;
- `sbatch`, `squeue`, `sacct`, and narrowly scoped job-status queries;
- creation of small submission logs/task lists and exact-path metadata checks;
- `/usr/bin/python3` with Python standard library only for small, bounded metadata validation.

Forbidden on a login node or directly from a submitter:

- `conda activate`, environment initialization for a workload, `torchrun`, or `accelerate`;
- importing or executing PyTorch, NumPy, JAX, TensorFlow, CuPy, scikit-learn, Mamba, or S5;
- training, inference, preprocessing, parameter counting/matching, smoke tests, benchmarks,
  visualization, model construction/loading, dataset scans, or broad recursive filesystem scans;
- using a local/background process, `nohup`, `tmux`, or an SSH timeout in place of Slurm.

Every workload uses a two-part launcher:

1. `submit_<run>.sh` stays lightweight and only validates/submits.
2. `run_<run>.sh` contains `#SBATCH` resources, activates `aim3_rnn`, and executes on a compute
   node. Computational preflight work is a separate Slurm job; submit the training array with
   `--dependency=afterok:<preflight-job-id>`.

Before a new or modified submitter is synchronized or run, execute the following on the local
development host. If local execution is impossible, run the test in a Slurm compute job; never run
`pytest` on an Amarel login node.

```bash
bash -n experiments/amarel/submit_<run>.sh experiments/amarel/run_<run>.sh
python -m pytest -q tests/test_amarel_submit_safety.py
bash experiments/amarel/submit_<run>.sh --dry-run  # when supported
```

On the Amarel login node, limit verification to `bash -n`, `--dry-run`, and scheduler queries. The
safety test scans every tracked `submit_*.sh`. Its small allowlist covers only existing,
reviewed standard-library metadata commands. Do not extend that allowlist for a new launcher;
move the operation into a Slurm preflight runner instead. After submission, confirm with `squeue`
that each running task has a compute-node assignment. If a workload is ever found on a login node,
stop that process first, then repair the launcher before resubmitting.

## Long-running sjc jobs

- Use the wrappers described in `remote/README.md` or the task-specific two-GPU launcher.
- Prefer a stable tmux session/run ID and keep the result suffix unique.
- Fetch only results created for the run when possible; avoid copying the entire result history.
- Report the process/session ID, result root, and status/fetch command after launch.

## Result and script safety

- Never delete remote results or artifacts without explicit human confirmation at deletion time.
- Keep generated Slurm scripts in `experiments/amarel/generated/`; they are ignored by Git.
- Reusable launchers may be tracked. One-off rerun scripts must say that they are one-off and be
  removed before synchronizing branches or worktrees.
- Store Slurm stdout/stderr under the task's `experiments/amarel/artifacts/` directory.
- Use `visualize_batch.sh` for training metrics after activating `aim3_rnn`.

## Synchronization and deletion safety

On 2026-07-16, a multi-source `rsync --delete` flattened a small source into a repository root
and recursively removed unrelated code, stimuli, and results. The following rules are mandatory:

1. Never use `rsync --delete` on a repository root or broad ancestor, including `results/`,
   `stimuli/`, `experiments/`, shared code roots, or scratch roots.
2. Never combine `--delete` with multiple sources. A deletion-enabled sync has exactly one source
   directory and one exact homologous leaf destination.
3. Never send a trailing-slash source directory to a repository root. The destination must name
   the corresponding leaf explicitly.
4. Treat `.git/`, code, stimuli, checkpoints, `results/train_data/`, and `results/archive/` as
   protected unless the human names the exact path and a recovery copy is verified.
5. Before a deletion-enabled sync, run the exact command with `--dry-run --itemize-changes` and
   inspect every `*deleting` entry. Any parent, sibling, or unexpected file is a stop condition.
6. Before recursive cleanup, require non-empty variables, resolve the target, and assert it equals
   the exact authorized leaf.
7. Prefer non-deleting, one-path-at-a-time synchronization.
8. After synchronization, verify the destination file count and protected sibling directories.

Safe:

```bash
rsync -az local/results/anal_figs/run/ host:/exact/repo/results/anal_figs/run/
```

Forbidden:

```bash
rsync -az --delete file1 local/results/anal_figs/run/ file2 host:/exact/repo/
```
