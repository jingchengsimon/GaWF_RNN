#!/usr/bin/env bash
# Submit param-count matched RNN/GRU/LSTM hparam tasks without waiting.
#
# Examples:
#   bash experiments/amarel/submit_hparam_param_match.sh --scale 40
#   bash experiments/amarel/submit_hparam_param_match.sh --scale 40 --gawf-ref-hidden 256
#   bash experiments/amarel/submit_hparam_param_match.sh --scale all --gawf-ref-hidden 512

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_hparam_param_match_array.sh"
GRID_UTIL="$ROOT/experiments/generalization/hparam_param_match_grid.py"
SUBMIT_LOG_DIR="$ROOT/experiments/amarel/artifacts/hparam_param_match"
TASK_LIST_DIR="$SUBMIT_LOG_DIR/task_lists"
SUBMIT_LOG="$SUBMIT_LOG_DIR/submissions_$(date +%Y%m%d_%H%M%S).log"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-48}"
SCALES=(40h)
MODELS=(rnn lstm gru)
GAWF_REF_HIDDEN=512

usage() {
  cat <<'EOF'
Usage:
  bash submit_hparam_param_match.sh [--scale 4|10|20|40|all ...] [--model rnn|lstm|gru|all ...] [--gawf-ref-hidden 64|128|256|512]

Defaults:
  scale = 40h, models = rnn lstm gru, gawf-ref-hidden = 512, array concurrency = 48.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale|--scales)
      SCALES=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        case "$1" in
          all) SCALES=(all) ;;
          4|4h) SCALES+=(4h) ;;
          10|10h) SCALES+=(10h) ;;
          20|20h) SCALES+=(20h) ;;
          40|40h) SCALES+=(40h) ;;
          *) echo "Invalid scale: $1" >&2; exit 2 ;;
        esac
        shift
      done
      ;;
    --model|--models)
      MODELS=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        case "$1" in
          all) MODELS=(all) ;;
          rnn|lstm|gru) MODELS+=("$1") ;;
          *) echo "Invalid model: $1" >&2; exit 2 ;;
        esac
        shift
      done
      ;;
    --gawf-ref-hidden)
      GAWF_REF_HIDDEN="$2"
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

if [[ "${#SCALES[@]}" -eq 0 || "${#MODELS[@]}" -eq 0 ]]; then
  echo "At least one scale and one model are required." >&2
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

mkdir -p "$SUBMIT_LOG_DIR" "$TASK_LIST_DIR"
TASK_LIST_FILE="$TASK_LIST_DIR/param_match_gawf${GAWF_REF_HIDDEN}_$(date +%Y%m%d_%H%M%S).txt"

python "$GRID_UTIL" list-task-ids \
  --scales "${SCALES[@]}" \
  --models "${MODELS[@]}" \
  --gawf-ref-hidden "$GAWF_REF_HIDDEN" \
  > "$TASK_LIST_FILE"

if [[ ! -s "$TASK_LIST_FILE" ]]; then
  echo "No task ids selected." >&2
  exit 2
fi

mapfile -t TASK_IDS < "$TASK_LIST_FILE"
TOTAL_TASKS="${#TASK_IDS[@]}"
array_last=$((TOTAL_TASKS - 1))
throttle="$ARRAY_CONCURRENCY"
if [[ "$TOTAL_TASKS" -lt "$throttle" ]]; then
  throttle="$TOTAL_TASKS"
fi

{
  echo "AIM3 param-match hparam submission"
  echo "timestamp=$(date -Is)"
  echo "root=$ROOT"
  echo "scales=${SCALES[*]}"
  echo "models=${MODELS[*]}"
  echo "gawf_ref_hidden=$GAWF_REF_HIDDEN"
  echo "task_list=$TASK_LIST_FILE"
  echo "total_tasks=$TOTAL_TASKS"
  echo "array=0-${array_last}%${throttle}"
  echo "run_script=$RUN_SCRIPT"
} | tee "$SUBMIT_LOG"

job_id="$(
  sbatch --parsable \
    --constraint=adalovelace \
    --cpus-per-task=16 \
    --mem=64G \
    --export=ALL,AIM3_ROOT="$ROOT",TASK_ID_FILE="$TASK_LIST_FILE",GAWF_REF_HIDDEN="$GAWF_REF_HIDDEN" \
    --array="0-${array_last}%${throttle}" \
    "$RUN_SCRIPT"
)"

{
  echo "submitted_job_id=$job_id"
  echo "squeue_command=squeue -j $job_id"
} | tee -a "$SUBMIT_LOG"
