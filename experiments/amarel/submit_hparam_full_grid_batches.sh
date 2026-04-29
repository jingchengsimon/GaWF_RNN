#!/usr/bin/env bash
# Submit the 1024-task full-grid search in <=200-task batches.
#
# From ~/FAW_RNN/experiments/amarel:
#   bash submit_hparam_full_grid_batches.sh
#   bash submit_hparam_full_grid_batches.sh --scale 10
#   bash submit_hparam_full_grid_batches.sh --scale 10 20 40
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
SCALES=(all)
START_TASK=""
END_TASK=""

usage() {
  cat <<'EOF'
Usage:
  bash submit_hparam_full_grid_batches.sh [--scale 4|10|20|40|all ...]
  bash submit_hparam_full_grid_batches.sh [-scale 10 20 40]

Defaults:
  batch size = 200, array concurrency = 96, full range = 0-1023.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale|-scale)
      SCALES=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        SCALES+=("$1")
        shift
      done
      if [[ "${#SCALES[@]}" -eq 0 ]]; then
        echo "--scale requires at least one value" >&2
        exit 2
      fi
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

TASK_IDS=()
if [[ -n "$START_TASK" || -n "$END_TASK" ]]; then
  if [[ -z "$START_TASK" || -z "$END_TASK" ]]; then
    echo "--start-task and --end-task must be provided together" >&2
    exit 2
  fi
  for ((task_id = START_TASK; task_id <= END_TASK; task_id++)); do
    TASK_IDS+=("$task_id")
  done
else
  for scale in "${SCALES[@]}"; do
    read -r range_start range_end <<< "$(scale_to_range "$scale")"
    for ((task_id = range_start; task_id <= range_end; task_id++)); do
      TASK_IDS+=("$task_id")
    done
  done
fi
TOTAL_TASKS="${#TASK_IDS[@]}"
TASK_LIST_DIR="$ROOT/experiments/amarel/artifacts/hparam_full_grid/task_lists"
TASK_LIST_FILE="$TASK_LIST_DIR/tasks_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$SUBMIT_LOG_DIR"
mkdir -p "$TASK_LIST_DIR"
printf '%s\n' "${TASK_IDS[@]}" > "$TASK_LIST_FILE"

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
log "scales=${SCALES[*]}"
log "task_list=$TASK_LIST_FILE"
if [[ "$TOTAL_TASKS" -gt 0 ]]; then
  log "task_range=${TASK_IDS[0]}-${TASK_IDS[$((TOTAL_TASKS - 1))]}"
else
  log "task_range=<empty>"
fi
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
  log "Submitting task-list rows ${start}-${end} as array 0-${array_last}%${throttle}"
  job_id="$(
    sbatch --parsable \
      --export=ALL,AIM3_ROOT="$ROOT",TASK_ID_FILE="$TASK_LIST_FILE",TASK_FILE_OFFSET="$start" \
      --array="0-${array_last}%${throttle}" \
      "$RUN_SCRIPT"
  )"
  log "Submitted job_id=$job_id for task-list rows ${start}-${end}"
  wait_for_job "$job_id"
  start=$((end + 1))
done

log ""
log "All batches completed. Run:"
log "  bash experiments/amarel/check_hparam_full_grid_status.sh"
