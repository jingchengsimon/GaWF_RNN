#!/usr/bin/env bash
# Submit the 1024-task full-grid search in <=200-task batches.
#
# From ~/aim3_runner:
#   bash experiments/amarel/submit_hparam_full_grid_batches.sh
#
# The script waits for each batch to finish before submitting the next one so
# the user's QOSMaxSubmitJobPerUserLimit is not exceeded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

TOTAL_TASKS="${TOTAL_TASKS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-200}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-96}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="$SCRIPT_DIR/run_hparam_full_grid_array.sh"
SUBMIT_LOG_DIR="$ROOT/artifacts/amarel_logs/hparam_full_grid"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submissions_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$SUBMIT_LOG_DIR"

log() {
  printf '%s\n' "$*" | tee -a "$SUBMIT_LOG"
}

wait_for_job() {
  local job_id="$1"
  log "Waiting for batch job $job_id ..."
  while squeue -j "$job_id" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$job_id" -h)" ]]; do
    squeue -j "$job_id" | tee -a "$SUBMIT_LOG" || true
    sleep "$POLL_SECONDS"
  done
  log "Batch job $job_id is no longer in squeue."
}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

log "AIM3 full-grid hparam submission"
log "timestamp=$(date -Is)"
log "root=$ROOT"
log "total_tasks=$TOTAL_TASKS"
log "batch_size=$BATCH_SIZE"
log "array_concurrency=$ARRAY_CONCURRENCY"
log "run_script=$RUN_SCRIPT"
log "submit_log=$SUBMIT_LOG"

start=0
while [[ "$start" -lt "$TOTAL_TASKS" ]]; do
  remaining=$((TOTAL_TASKS - start))
  if [[ "$remaining" -lt "$BATCH_SIZE" ]]; then
    count="$remaining"
  else
    count="$BATCH_SIZE"
  fi
  end=$((start + count - 1))
  array_last=$((count - 1))
  throttle="$ARRAY_CONCURRENCY"
  if [[ "$count" -lt "$throttle" ]]; then
    throttle="$count"
  fi

  log ""
  log "Submitting task_id ${start}-${end} as array 0-${array_last}%${throttle}"
  job_id="$(
    sbatch --parsable \
      --export=ALL,TASK_OFFSET="$start" \
      --array="0-${array_last}%${throttle}" \
      "$RUN_SCRIPT"
  )"
  log "Submitted job_id=$job_id for task_id ${start}-${end}"
  wait_for_job "$job_id"
  start=$((end + 1))
done

log ""
log "All batches completed. Run:"
log "  bash experiments/amarel/check_hparam_full_grid_status.sh"
