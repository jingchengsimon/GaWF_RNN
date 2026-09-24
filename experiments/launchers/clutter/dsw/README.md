# Clutter / DSW

This directory owns cross-node Clutter DSW controls and immutable snapshots of one-off DSW
coordinators. Training mathematics remains in `run_task.py` and `utils/training/`; the canonical
single-unit runner for the aligned RNN is
`experiments/clutter/amarel/run_clutter_rnn_inloop_notanh.sh`.

`run_nowrap_unit.sh` is the prepared DSW adapter for migrating pending Amarel array elements
`61753234_0-29`. It preserves the Amarel task map with seed offset 5, reuses the canonical
Clutter runner, refuses duplicate writers or final artifacts, and writes into a host-local result
root. GPU selection and the Amarel handoff gate remain the responsibility of the migration
monitor: an Amarel element is cancelled only after the DSW log has completed epoch 1.

`run_fixed_gpu_nowrap_queue.sh` is the deterministic continuation layer. Each invocation owns one
fixed GPU, waits only for explicitly supplied incumbent PID/command-token pairs, and then runs an
explicit task list serially through `run_nowrap_unit.sh`. It does not discover free GPUs, move a
task between GPUs, or cancel Amarel jobs; a failed task stops that lane and preserves its logs.

`run_fixed_gpu_feedback_add_queue.sh` queues the corrected `rnn_fb_add`, `gru_fb_add`, and
`lstm_fb_add` on fixed GPUs after explicitly named incumbent processes exit. Task IDs 0--14 map
to five seeds for each model; `AIM3_FEEDBACK_ADD_SEED_OFFSET=0` selects seeds 1--5 and offset 5
selects seeds 6--10. The first released lane owns a shared 200-step optimization smoke; every lane
requires its pass record before starting formal 150-epoch units. Each unit preserves one model,
metrics, and history file plus a reset-excluded evaluation under unique campaign result roots.

`run_feedback_add_recovery_pairs.sh` is the bounded recovery path for the 2026-09-24 DSW
interruption: it resumes RNN seed5 and GRU seed1 together, then runs the two LSTM units stranded
behind the failed 8095 lane. Both pairs share one 96-GiB GPU; each unit still uses the canonical
single-unit runner, checkpoint continuation, duplicate-writer guard, and final-artifact checks.

## Coordinated pause trigger

Create the ignored local config from the example and fill in the current SSH commands:

```bash
cp experiments/launchers/clutter/dsw/pause_campaigns.example.json \
  experiments/launchers/clutter/dsw/pause_campaigns.local.json
```

Read-only plan:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B \
  experiments/launchers/clutter/dsw/pause_campaigns.py \
  --config experiments/launchers/clutter/dsw/pause_campaigns.local.json
```

Only after an explicit human pause request:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B \
  experiments/launchers/clutter/dsw/pause_campaigns.py \
  --config experiments/launchers/clutter/dsw/pause_campaigns.local.json \
  --execute --confirm PAUSE-CHECKPOINTED-CAMPAIGNS
```

The trigger performs a two-host read-only preflight, stops coordinator shells from releasing new
seeds, waits for each active writer's next 5-epoch checkpoint, sends `SIGINT` to the root training
process, verifies exit and checkpoint presence, and writes a non-destructive pause receipt. It
does not delete result files or final artifacts. `SIGSTOP` is used only on lightweight coordinator
shells as a queue barrier; training writers are checkpointed and terminated so GPU memory is
released.

The operation is coordinated but not transactionally atomic across hosts. A partial host failure
is reported explicitly and must be resolved before any resume or redistribution.
