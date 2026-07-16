#!/usr/bin/env bash
# Submit the IMDB LSTM hparam grid in bounded Slurm array batches.
#
# Usage:
#   bash experiments/amarel/submit_imdb_hparam_grid_batches.sh
#   bash experiments/amarel/submit_imdb_hparam_grid_batches.sh --start-task 0 --end-task 3   # dry-run slice
#
# Prerequisite: run source/text/prepare_imdb_data.py once on a login node so the
# pre-tokenized tensors exist under the data dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

BATCH_SIZE="${BATCH_SIZE:-48}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"
POLL_SECONDS="${POLL_SECONDS:-120}"
RUN_SCRIPT="$SCRIPT_DIR/run_imdb_hparam_grid_array.sh"
GRID_UTIL="$ROOT/experiments/text/imdb_hparam_grid.py"
SUBMIT_LOG_DIR="$ROOT/experiments/amarel/artifacts/imdb_lstm_hparam_grid"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submissions_$(date +%Y%m%d_%H%M%S).log"
TASK_LIST_DIR="$SUBMIT_LOG_DIR/task_lists"
START_TASK=""
END_TASK=""

usage() {
  cat <<'EOF'
Usage:
  bash submit_imdb_hparam_grid_batches.sh [--start-task N --end-task M]

Defaults:
  full grid (lstm x lr{1e-4,5e-4,1e-3,5e-3} x wd{0,1e-5,1e-4,1e-3} x hidden{128,256,512}),
  batch size = 48, array concurrency = 16. Use --start-task/--end-task for a dry-run slice.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-task) START_TASK="$2"; shift 2 ;;
    --end-task) END_TASK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$SUBMIT_LOG_DIR" "$TASK_LIST_DIR"
: "${AIM3_RESULTS_PATH:?Set AIM3_RESULTS_PATH to the configured Amarel result root}"

export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

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

TASK_LIST_FILE="$TASK_LIST_DIR/tasks_$(date +%Y%m%d_%H%M%S).txt"
if [[ -n "$START_TASK" || -n "$END_TASK" ]]; then
  if [[ -z "$START_TASK" || -z "$END_TASK" ]]; then
    echo "--start-task and --end-task must be provided together" >&2
    exit 2
  fi
  for ((task_id = START_TASK; task_id <= END_TASK; task_id++)); do
    printf '%s\n' "$task_id" >> "$TASK_LIST_FILE"
  done
else
  total_grid_tasks="$(
    python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('grid', '$GRID_UTIL'); mod=importlib.util.module_from_spec(spec); sys.modules['grid'] = mod; spec.loader.exec_module(mod); print(mod.TOTAL_TASKS)"
  )"
  for task_id in $(seq 0 "$((total_grid_tasks - 1))"); do
    printf '%s\n' "$task_id" >> "$TASK_LIST_FILE"
  done
fi

if [[ ! -s "$TASK_LIST_FILE" ]]; then
  echo "No task ids selected." >&2
  exit 2
fi

mapfile -t TASK_IDS < "$TASK_LIST_FILE"
TOTAL_TASKS="${#TASK_IDS[@]}"

log "AIM3 IMDB LSTM hparam submission"
log "timestamp=$(date -Is)"
log "root=$ROOT"
log "lr_grid=1e-4 5e-4 1e-3 5e-3"
log "wd_grid=0 1e-5 1e-4 1e-3"
log "hidden_grid=128 256 512"
log "task_list=$TASK_LIST_FILE"
log "total_tasks=$TOTAL_TASKS"
log "batch_size=$BATCH_SIZE array_concurrency=$ARRAY_CONCURRENCY"
log "cpus_per_task=16 mem=64G gres=gpu:1 constraint=adalovelace"
log "AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
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
      --export=ALL,AIM3_ROOT="$ROOT",AIM3_RESULTS_PATH="$AIM3_RESULTS_PATH",AIM3_NUM_WORKERS="$AIM3_NUM_WORKERS",AIM3_PIN_MEMORY="$AIM3_PIN_MEMORY",TASK_ID_FILE="$TASK_LIST_FILE",TASK_FILE_OFFSET="$start" \
      --constraint=adalovelace --cpus-per-task=16 --mem=64G --gres=gpu:1 \
      --array="0-${array_last}%${throttle}" \
      "$RUN_SCRIPT"
  )"
  log "Submitted job_id=$job_id for task-list rows ${start}-${end}"
  wait_for_job "$job_id"
  start=$((end + 1))
done

log ""
log "All selected batches completed. Summarize with:"
log "  python experiments/text/imdb_hparam_grid.py summarize --root \"$AIM3_RESULTS_PATH\""
