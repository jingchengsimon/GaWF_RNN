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
- Local/sjc runs must not inherit Amarel-only DataLoader worker and pin-memory settings when that
  would exceed host memory.

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
- `AIM3_NUM_WORKERS=12`, `AIM3_PIN_MEMORY=1` exported at submission time.

Do not bake the Amarel DataLoader values into local training defaults. Use a specific node only
when reproducing a documented node-level experiment.

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
