#!/usr/bin/env bash
# Submit the revised Task A single-layer GaWF feedback-LR-scale search.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

GRID_UTIL="experiments/generalization/gawf_single_feedback_lr_grid.py"
RUN_SCRIPT="$SCRIPT_DIR/run_gawf_single_feedback_lr_grid_array.sh"
LOG_DIR="$ROOT/experiments/amarel/artifacts/gawf_single_feedback_lr_grid"
SUBMIT_LOG="$LOG_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-96}"
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
last_task=$((total_tasks - 1))
throttle="$ARRAY_CONCURRENCY"
if [[ "$total_tasks" -lt "$throttle" ]]; then
  throttle="$total_tasks"
fi

{
  echo "AIM3 single-layer GaWF feedback-LR-scale submission"
  echo "timestamp=$(date -Is)"
  echo "root=$ROOT"
  echo "grid_util=$GRID_UTIL"
  echo "run_script=$RUN_SCRIPT"
  echo "total_tasks=$total_tasks"
  echo "array=0-${last_task}%${throttle}"
  echo "resources=partition=gpu-redhat account=general gres=gpu:1 constraint=adalovelace cpus=16 mem=64G time=72:00:00"
  echo "env=AIM3_NUM_WORKERS=12 AIM3_PIN_MEMORY=1 conda=aim3_rnn"
} | tee "$SUBMIT_LOG"

job_id="$(
  sbatch --parsable \
    --export=ALL,AIM3_ROOT="$ROOT",AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1 \
    --array="0-${last_task}%${throttle}" \
    "$RUN_SCRIPT"
)"

{
  echo "Submitted job_id=$job_id"
  echo "Next:"
  echo "  squeue -j $job_id"
  echo "  python $GRID_UTIL status --root \"$ROOT\""
} | tee -a "$SUBMIT_LOG"
