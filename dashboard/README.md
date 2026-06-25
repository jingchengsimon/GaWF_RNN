# FAW_RNN progress dashboard

The dashboard is a standalone UI served by the Mac mini. Each card shows the human-readable task
description, execution host, remote job ID(s), and `valid / submitted` progress. Result paths,
validators, and retention metadata remain in `tasks.json`.

Open it from the Mac mini or another device on the same LAN/private remote network:

[FAW_RNN Dashboard](http://Jingchengs-Mac-mini.local:8765/)

`127.0.0.1` only works in a browser running on the Mac mini itself.

Progress is derived from task-specific result evidence, not Slurm exit state. A metrics-grid
task counts unique configurations whose metrics JSON matches its registered filters and whose
PKL/checkpoint companions exist. An explicit-units task counts units only when every registered
required file exists.

## Run

```bash
python3 -m dashboard.server --host private-lan
# http://Jingchengs-Mac-mini.local:8765/
```

One remote refresh without starting the server:

```bash
python3 -m dashboard.server --once
```

Install as a macOS service independent of Codex:

```bash
python3 dashboard/install_launch_agent.py install
```

The installer places a read-only runtime copy in `~/Library/Application Support/FAW_RNN
Dashboard` because macOS blocks background LaunchAgents from loading Python modules directly
from Desktop. `dashboard/tasks.json` remains the project source of truth; the registration CLI
atomically syncs changes to the installed runtime registry.

If SSH is unavailable, the service keeps the last confirmed counts, marks them stale, and
retries automatically. It never converts an SSH outage into failed experiment units.

## Mandatory registration for every remote submission

Every conversation that submits an Amarel Slurm job or launches work on the sjc-remote
dual-GPU machine must register the work in the same turn. A conversational training summary
does not replace Dashboard registration.

Register one JSON object containing:

- a human-readable `description`;
- `machine` (`amarel` or `sjc-remote`);
- non-empty `job_ids`: Slurm IDs on Amarel; launcher IDs, PIDs, tmux session names, or explicit
  run IDs on sjc-remote;
- `remote_root` and a task-specific result tracker.
- `retention_policy: "human_confirmation_required"` (added automatically by the CLI if omitted).

The tracker is the remote completion-progress metric. Use `metrics_grid` for a sweep whose
valid outputs are selected by metrics content, or `explicit_units` for named runs with explicit
required files. Do not use scheduler state alone as completion evidence.

```bash
python3 -m dashboard.manage_tasks register --spec '{
  "id": "example-grid",
  "description": "Clear experiment description",
  "machine": "amarel",
  "job_ids": ["12345678"],
  "retention_policy": "human_confirmation_required",
  "remote_root": "~/FAW_RNN",
  "tracker": {
    "type": "metrics_grid",
    "expected_total": 16,
    "result_glob": "results/train_data/example/task_*/*_metrics.json",
    "companion_files": ["{stem}.pkl", "{stem}_model.pth"],
    "match": {"model_type": {"equals": "rnn"}},
    "uniqueness_fields": ["model_type", "hidden_size", "lr", "weight_decay"]
  }
}'
```

After registering, request `/api/refresh` and verify that the new card displays the expected
machine, job ID, and initial progress. The service reloads `tasks.json` on every refresh, so
registration does not require a restart.

## Retention and removal

Keep records after completion, failure, timeout, or staleness. Report the training summary in
conversation when results are ready, but remove the Dashboard record only after a human
explicitly confirms it is no longer needed. The CLI enforces this acknowledgement:

```bash
python3 -m dashboard.manage_tasks remove example-grid --human-confirmed
```

Calling `remove` without `--human-confirmed` fails without changing `tasks.json`.
