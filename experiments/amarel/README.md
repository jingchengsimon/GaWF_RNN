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
AIM3_RESULTS_PATH=<configured-scratch-results-root>
```

General tasks export the DataLoader values above with `sbatch`. Standard Clutter 40h mmap runs
instead use the cross-endpoint benchmarked default `AIM3_NUM_WORKERS=2`,
`AIM3_PIN_MEMORY=1`, device-side cast, compact windows, and block size equal to the effective
batch. Slurm scripts must source the configured Amarel Conda initialization script and run
`conda activate aim3_rnn`. Do not use `module`, the default Python, or an unrelated environment.
Training outputs belong under `$AIM3_RESULTS_PATH/train_data/<result_suffix>/`; status and rerun
tools must resolve that same physical root rather than assuming `<repo>/results`.
Long Clutter runners also use `--checkpoint_interval_epochs 5 --auto_resume`; resubmitting the
same compatible unit continues from its latest complete five-epoch boundary.

## File roles

- `submit_*.sh`: login-node control plane only; validate arguments/resources and call `sbatch`.
  Never activate Conda or execute project Python/ML work here. Small reviewed metadata checks may
  use `/usr/bin/python3` standard library only.
- `run_*_array.sh` / `run_*.sh`: execute one array task or named run on a compute node.
- `check_*`: inspect scheduler state, logs, and result validity.
- `rerun_*`: explicitly resubmit failed or missing tasks; there is no automatic retry loop.
- `summarize_*`: validate result contents and produce experiment summaries.
- `artifacts/<run>/`: ignored stdout, stderr, submission logs, and status artifacts.
- `generated/`: ignored generated Slurm scripts.

Reusable launchers and grid utilities may be tracked. One-off recovery scripts must be clearly
marked in their filename/header and deleted before branch or worktree synchronization.

Any parameter matching, model construction, preprocessing, smoke testing, benchmarking, or
visualization is compute work even if expected to finish quickly. Submit it as a `run_*.sh`
preflight job and use `afterok` for dependent jobs. Before synchronizing or running a new or
modified submitter, `bash -n` it and run
`python -m pytest -q tests/test_amarel_submit_safety.py` locally (or in a Slurm compute job), never
on an Amarel login node; do not add new safety-test allowlist entries. The detailed policy is in
`docs/operations/REMOTE_EXECUTION.md`.

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
training array is larger than node memory, this launcher uses mmap uint8 with device-side cast,
compact windows, batch-sized block shuffle, `AIM3_NUM_WORKERS=2`, and
`AIM3_PIN_MEMORY=1`. The full array remains mmap-backed; workers do not materialize it in RAM.

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
`experiments.monitoring.job_registry`.

## Results and cleanup

- Do not delete results because a job completed, failed, timed out, or became stale.
- Pending cleanup documents record replacement conditions; obtain explicit human confirmation at
  deletion time.
- Use `visualize_batch.sh` after activating `aim3_rnn` for saved training metrics.
- Use a fixed node only when reproducing a specifically documented node-level execution path.
