# Amarel Experiment Launchers

This directory contains reusable Slurm submission, array-runner, status, rerun, and summary
scripts. Read `docs/operations/REMOTE_EXECUTION.md` before connecting or submitting work.

## Required execution contract

Codex-submitted training jobs use the project `aim3_rnn` Conda environment and, unless explicitly
overridden by a human:

```text
partition=gpu-redhat
account=general
gres=gpu:1
constraint=adalovelace
cpus-per-task=16
mem=64G
AIM3_NUM_WORKERS=12
AIM3_PIN_MEMORY=1
```

Export the DataLoader values with `sbatch`; do not add them to local launchers or training
defaults. Slurm scripts must source the configured Amarel Conda initialization script and run
`conda activate aim3_rnn`. Do not use `module`, the default Python, or an unrelated environment.

## File roles

- `submit_*.sh`: validate arguments/resources and call `sbatch`.
- `run_*_array.sh` / `run_*.sh`: execute one array task or named run on a compute node.
- `check_*`: inspect scheduler state, logs, and result validity.
- `rerun_*`: explicitly resubmit failed or missing tasks; there is no automatic retry loop.
- `summarize_*`: validate result contents and produce experiment summaries.
- `artifacts/<run>/`: ignored stdout, stderr, submission logs, and status artifacts.
- `generated/`: ignored generated Slurm scripts.

Reusable launchers and grid utilities may be tracked. One-off recovery scripts must be clearly
marked in their filename/header and deleted before branch or worktree synchronization.

## Full-grid workflow

The Clutter 1024-task mapping is owned by
`experiments/generalization/hparam_full_grid.py`. The main wrappers submit selected scales, run
array tasks, check validity, rerun explicit missing/failed IDs, and summarize results.

Use `submit_hparam_full_grid_batches.sh --scale <4|10|20|40|all ...>` to restrict the submitted
scale slices. Batch submission waits before sending the next batch to respect scheduler limits.

Before a full campaign, use the four-job 4h/5-epoch smoke test. It runs RNN/LSTM/GRU/GaWF at
hidden size 256, learning rate `5e-4`, and weight decay `1e-4`; validate it with the matching
status script.

## Other experiment families

The directory also contains reusable launchers for:

- parameter-matched Clutter baselines and Mamba/S5 grids;
- projected-feedback GaWF comparisons;
- IMDB and SentiHood runs;
- Atari Pong single-layer and depth comparisons;
- MiniGrid recurrent PPO comparisons with explicit environment/CUDA acceleration settings;
- optimizer and environment smoke checks.

The fixed-best Clutter confirmation run uses `submit_clutter_best6_10seed.sh` for six models by
ten seeds. It submits ten independent Slurm jobs, one six-task model array per seed;
`check_clutter_best6_10seed.sh <job-id>` performs strict result validation. Because the 40h
training array is larger than node memory, this launcher uses mmap with
`AIM3_NUM_WORKERS=0` and `AIM3_PIN_MEMORY=0`, overriding the usual Amarel DataLoader defaults.

Each family must define expected result files and use a status/check script that validates result
contents rather than relying only on Slurm state.

## Submission handoff

After a successful submission, report in the same conversation:

- job ID and experiment/task description;
- remote root and result suffix/path;
- requested partition, GPU, constraint, CPUs, memory, and Conda environment;
- exported DataLoader settings;
- the status/check command.

In the same turn, add the job to `experiments/monitoring/jobs/` with
`experiments.monitoring.job_registry`. No external registry or Dashboard step is required.

## Results and cleanup

- Do not delete results because a job completed, failed, timed out, or became stale.
- Pending cleanup documents record replacement conditions; obtain explicit human confirmation at
  deletion time.
- Use `visualize_batch.sh` after activating `aim3_rnn` for saved training metrics.
- Use a fixed node only when reproducing a specifically documented node-level execution path.
