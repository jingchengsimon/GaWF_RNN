#!/usr/bin/env bash
# Submit the Mamba/S5 hparam grid in bounded Slurm array batches.
#
# Usage:
#   bash experiments/amarel/submit_ssm_mamba_hparam_grid_batches.sh
#   bash experiments/amarel/submit_ssm_mamba_hparam_grid_batches.sh --model mamba
#   bash experiments/amarel/submit_ssm_mamba_hparam_grid_batches.sh --model s5
#   bash experiments/amarel/submit_ssm_mamba_hparam_grid_batches.sh --scale 4 10 20 40
#
# Note: Mamba tasks require mamba-ssm and causal-conv1d; S5 tasks require
# s5-pytorch in AIM3_CONDA_ENV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

BATCH_SIZE="${BATCH_SIZE:-128}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-64}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="$SCRIPT_DIR/run_ssm_mamba_hparam_grid_array.sh"
GRID_UTIL="$ROOT/experiments/generalization/ssm_mamba_hparam_grid.py"
SUBMIT_LOG_DIR="$ROOT/experiments/amarel/artifacts/mamba_s5_hparam_grid"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submissions_$(date +%Y%m%d_%H%M%S).log"
TASK_LIST_DIR="$SUBMIT_LOG_DIR/task_lists"
SMOKE_MARKER="$ROOT/experiments/amarel/artifacts/mamba_s5_optimizer_smoke/optimizer_grouping_smoke.done"
SKIP_SMOKE_CHECK=0
MODELS=(mamba s5)
SCALES=(4h 10h 20h 40h)
START_TASK=""
END_TASK=""

usage() {
  cat <<'EOF'
Usage:
  bash submit_ssm_mamba_hparam_grid_batches.sh [--model mamba|s5|all ...] [--scale 4|10|20|40|all ...]
  bash submit_ssm_mamba_hparam_grid_batches.sh --start-task N --end-task M
  bash submit_ssm_mamba_hparam_grid_batches.sh --skip-smoke-check

Defaults:
  models = mamba s5, scales = 4h 10h 20h 40h, batch size = 128, array concurrency = 64.

Before full submission, run:
  sbatch experiments/amarel/run_mamba_s5_optimizer_smoke.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--models)
      MODELS=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        case "$1" in
          all) MODELS=(mamba s5) ;;
          mamba|s5) MODELS+=("$1") ;;
          *) echo "Invalid model: $1" >&2; exit 2 ;;
        esac
        shift
      done
      ;;
    --scale|--scales)
      SCALES=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        case "$1" in
          all) SCALES=(4h 10h 20h 40h) ;;
          4|4h) SCALES+=(4h) ;;
          10|10h) SCALES+=(10h) ;;
          20|20h) SCALES+=(20h) ;;
          40|40h) SCALES+=(40h) ;;
          *) echo "Invalid scale: $1" >&2; exit 2 ;;
        esac
        shift
      done
      ;;
    --start-task)
      START_TASK="$2"
      shift 2
      ;;
    --end-task)
      END_TASK="$2"
      shift 2
      ;;
    --skip-smoke-check)
      SKIP_SMOKE_CHECK=1
      shift
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

if [[ "${#MODELS[@]}" -eq 0 || "${#SCALES[@]}" -eq 0 ]]; then
  echo "At least one model and one scale are required." >&2
  exit 2
fi

if [[ "$SKIP_SMOKE_CHECK" -eq 0 && ! -f "$SMOKE_MARKER" ]]; then
  cat >&2 <<EOF
Optimizer grouping smoke marker is missing:
  $SMOKE_MARKER

Run this first and inspect its log for:
  Mamba no_decay
  S5 ssm_core
  S5 decay
  S5_AMP_SMOKE_OK

Submit smoke:
  sbatch experiments/amarel/run_mamba_s5_optimizer_smoke.sh

To override intentionally, pass --skip-smoke-check.
EOF
  exit 2
fi

mkdir -p "$SUBMIT_LOG_DIR" "$TASK_LIST_DIR"

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
  python "$GRID_UTIL" emit-task --task-id 0 --root "$ROOT" >/dev/null
  total_grid_tasks="$(
    python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('grid', '$GRID_UTIL'); mod=importlib.util.module_from_spec(spec); sys.modules['grid'] = mod; spec.loader.exec_module(mod); print(mod.TOTAL_TASKS)"
  )"
  for task_id in $(seq 0 "$((total_grid_tasks - 1))"); do
    cfg_json="$(python "$GRID_UTIL" emit-task --task-id "$task_id" --root "$ROOT" --format json)"
    model="$(python -c 'import json,sys; print(json.load(sys.stdin)["model"])' <<< "$cfg_json")"
    scale="$(python -c 'import json,sys; print(json.load(sys.stdin)["scale"])' <<< "$cfg_json")"
    model_match=0
    for wanted in "${MODELS[@]}"; do
      [[ "$model" == "$wanted" ]] && model_match=1
    done
    scale_match=0
    for wanted in "${SCALES[@]}"; do
      [[ "$scale" == "$wanted" ]] && scale_match=1
    done
    if [[ "$model_match" -eq 1 && "$scale_match" -eq 1 ]]; then
      printf '%s\n' "$task_id" >> "$TASK_LIST_FILE"
    fi
  done
fi

if [[ ! -s "$TASK_LIST_FILE" ]]; then
  echo "No task ids selected." >&2
  exit 2
fi

mapfile -t TASK_IDS < "$TASK_LIST_FILE"
TOTAL_TASKS="${#TASK_IDS[@]}"

log "AIM3 Mamba/S5 hparam submission"
log "timestamp=$(date -Is)"
log "root=$ROOT"
log "models=${MODELS[*]}"
log "scales=${SCALES[*]}"
log "lr_grid=1e-4 5e-4 1e-3 5e-3 1e-2"
log "wd_grid=0 1e-5 1e-4 1e-3"
log "s5_ssm_lr_scale=0.1"
log "task_list=$TASK_LIST_FILE"
log "total_tasks=$TOTAL_TASKS"
log "batch_size=$BATCH_SIZE"
log "array_concurrency=$ARRAY_CONCURRENCY"
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
      --constraint=adalovelace \
      --cpus-per-task=16 \
      --mem=64G \
      --export=ALL,AIM3_ROOT="$ROOT",AIM3_NUM_WORKERS="$AIM3_NUM_WORKERS",AIM3_PIN_MEMORY="$AIM3_PIN_MEMORY",TASK_ID_FILE="$TASK_LIST_FILE",TASK_FILE_OFFSET="$start" \
      --array="0-${array_last}%${throttle}" \
      "$RUN_SCRIPT"
  )"
  log "Submitted job_id=$job_id for task-list rows ${start}-${end}"
  wait_for_job "$job_id"
  start=$((end + 1))
done

log ""
log "All selected batches completed. Run:"
log "  bash experiments/amarel/check_ssm_mamba_hparam_grid_status.sh"
