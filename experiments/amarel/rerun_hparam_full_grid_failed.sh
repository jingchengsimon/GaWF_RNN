#!/usr/bin/env bash
# Submit reruns for failed/missing full-grid hparam tasks.
#
# Run check_hparam_full_grid_status.sh first, or let this script run it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-experiments/generalization/artifacts/gen_hparam_full_grid}"
FAILED_IDS="${FAILED_IDS:-$OUT_DIR/failed_task_ids.txt}"
BATCH_SIZE="${BATCH_SIZE:-200}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-96}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="$SCRIPT_DIR/run_hparam_full_grid_array.sh"
RERUN_DIR="$OUT_DIR/rerun_lists"
RERUN_LOG_DIR="$ROOT/artifacts/amarel_logs/hparam_full_grid"
RERUN_LOG="$RERUN_LOG_DIR/rerun_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RERUN_DIR" "$RERUN_LOG_DIR"

log() {
  printf '%s\n' "$*" | tee -a "$RERUN_LOG"
}

wait_for_job() {
  local job_id="$1"
  log "Waiting for rerun job $job_id ..."
  while squeue -j "$job_id" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$job_id" -h)" ]]; do
    squeue -j "$job_id" | tee -a "$RERUN_LOG" || true
    sleep "$POLL_SECONDS"
  done
  log "Rerun job $job_id is no longer in squeue."
}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

bash "$SCRIPT_DIR/check_hparam_full_grid_status.sh"

if [[ ! -s "$FAILED_IDS" ]]; then
  log "No failed task ids found in $FAILED_IDS"
  exit 0
fi

mapfile -t ids < "$FAILED_IDS"
total="${#ids[@]}"
log "Rerunning $total failed task(s) from $FAILED_IDS"

chunk_idx=0
start=0
while [[ "$start" -lt "$total" ]]; do
  remaining=$((total - start))
  if [[ "$remaining" -lt "$BATCH_SIZE" ]]; then
    count="$remaining"
  else
    count="$BATCH_SIZE"
  fi
  chunk_file="$RERUN_DIR/rerun_chunk_${chunk_idx}.txt"
  : > "$chunk_file"
  for ((i = start; i < start + count; i++)); do
    printf '%s\n' "${ids[$i]}" >> "$chunk_file"
  done

  array_last=$((count - 1))
  throttle="$ARRAY_CONCURRENCY"
  if [[ "$count" -lt "$throttle" ]]; then
    throttle="$count"
  fi

  log ""
  log "Submitting rerun chunk $chunk_idx with $count task(s): $chunk_file"
  job_id="$(
    sbatch --parsable \
      --export=ALL,TASK_ID_FILE="$chunk_file" \
      --array="0-${array_last}%${throttle}" \
      "$RUN_SCRIPT"
  )"
  log "Submitted rerun job_id=$job_id"
  wait_for_job "$job_id"

  start=$((start + count))
  chunk_idx=$((chunk_idx + 1))
done

log ""
log "Reruns completed. Re-checking status..."
bash "$SCRIPT_DIR/check_hparam_full_grid_status.sh"
