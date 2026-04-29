#!/usr/bin/env bash
# Submit the 1024-task full-grid search in <=200-task batches.
#
# From ~/FAW_RNN/experiments/amarel:
#   bash submit_hparam_full_grid_batches.sh
#   bash submit_hparam_full_grid_batches.sh --scale 10
#   bash submit_hparam_full_grid_batches.sh -scale 20
#
# The script waits for each batch to finish before submitting the next one so
# the user's QOSMaxSubmitJobPerUserLimit is not exceeded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

BATCH_SIZE="${BATCH_SIZE:-200}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-96}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="$SCRIPT_DIR/run_hparam_full_grid_array.sh"
SUBMIT_LOG_DIR="$ROOT/experiments/amarel/artifacts/hparam_full_grid"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submissions_$(date +%Y%m%d_%H%M%S).log"
SCALE="all"
START_TASK=""
END_TASK=""

usage() {
  cat <<'EOF'
Usage:
  bash submit_hparam_full_grid_batches.sh [--scale 4|10|20|40|all]
  bash submit_hparam_full_grid_batches.sh [-scale 4|10|20|40|all]

Defaults:
  batch size = 200, array concurrency = 96, full range = 0-1023.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale|-scale)
      SCALE="$2"
      shift 2
      ;;
    --start-task)
      START_TASK="$2"
      shift 2
      ;;
    --end-task)
      END_TASK="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

scale_to_range() {
  case "$1" in
    4|4h) echo "0 255" ;;
    10|10h) echo "256 511" ;;
    20|20h) echo "512 767" ;;
    40|40h) echo "768 1023" ;;
    all|"") echo "0 1023" ;;
    *)
      echo "Invalid scale: $1 (expected 4, 10, 20, 40, or all)" >&2
      return 2
      ;;
  esac
}

read -r default_start default_end <<< "$(scale_to_range "$SCALE")"
START_TASK="${START_TASK:-$default_start}"
END_TASK="${END_TASK:-$default_end}"
TOTAL_TASKS=$((END_TASK - START_TASK + 1))

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
log "scale=$SCALE"
log "task_range=${START_TASK}-${END_TASK}"
log "total_tasks=$TOTAL_TASKS"
log "batch_size=$BATCH_SIZE"
log "array_concurrency=$ARRAY_CONCURRENCY"
log "run_script=$RUN_SCRIPT"
log "submit_log=$SUBMIT_LOG"

start="$START_TASK"
while [[ "$start" -le "$END_TASK" ]]; do
  remaining=$((END_TASK - start + 1))
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
      --export=ALL,AIM3_ROOT="$ROOT",TASK_OFFSET="$start" \
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
