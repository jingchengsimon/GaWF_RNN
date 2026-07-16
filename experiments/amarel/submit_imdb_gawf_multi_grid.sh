#!/usr/bin/env bash
# Submit Task C: IMDB 2-layer GaWF direct-feedback grid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

GRID_UTIL="experiments/generalization/imdb_gawf_multi_grid.py"
RUN_SCRIPT="$SCRIPT_DIR/run_imdb_gawf_multi_grid_array.sh"
LOG_DIR="$ROOT/experiments/amarel/artifacts/imdb_gawf_multi_grid"
SUBMIT_LOG="$LOG_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-32}"
TASKS_PER_ARRAY="${TASKS_PER_ARRAY:-4}"
mkdir -p "$LOG_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

total_tasks="$(python "$GRID_UTIL" list-task-ids | wc -l | tr -d ' ')"
if [[ "$total_tasks" -le 0 ]]; then
  echo "No tasks emitted by $GRID_UTIL" >&2
  exit 2
fi
array_tasks=$(((total_tasks + TASKS_PER_ARRAY - 1) / TASKS_PER_ARRAY))
last_task=$((array_tasks - 1))
throttle="$ARRAY_CONCURRENCY"
if [[ "$array_tasks" -lt "$throttle" ]]; then
  throttle="$array_tasks"
fi

{
  echo "AIM3 IMDB 2-layer GaWF grid submission"
  echo "timestamp=$(date -Is)"
  echo "root=$ROOT"
  echo "grid_util=$GRID_UTIL"
  echo "run_script=$RUN_SCRIPT"
  echo "hidden_grid=64 96 128 192"
  echo "lr_grid=1e-4 5e-4 1e-3 5e-3"
  echo "wd_grid=0 1e-5 1e-4 1e-3"
  echo "fixed=gawf_layers=2 feedback_dim=0 gawf_multi_feedback_lr_scale=0.1"
  echo "logical_total_tasks=$total_tasks"
  echo "tasks_per_array=$TASKS_PER_ARRAY"
  echo "slurm_array_tasks=$array_tasks"
  echo "array=0-${last_task}%${throttle}"
  echo "resources=partition=gpu-redhat account=general gres=gpu:1 constraint=adalovelace cpus=16 mem=64G time=24:00:00"
  echo "env=AIM3_NUM_WORKERS=12 AIM3_PIN_MEMORY=1 conda=aim3_rnn"
} | tee "$SUBMIT_LOG"

job_id="$(
  sbatch --parsable \
    --export=ALL,AIM3_ROOT="$ROOT",AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1,TASKS_PER_ARRAY="$TASKS_PER_ARRAY" \
    --array="0-${last_task}%${throttle}" \
    "$RUN_SCRIPT"
)"

{
  echo "Submitted job_id=$job_id"
  echo "Next:"
  echo "  squeue -j $job_id"
  echo "  python $GRID_UTIL status --root \"$ROOT\""
} | tee -a "$SUBMIT_LOG"
