#!/usr/bin/env bash
# Run the full-grid hparam search locally with two GPUs.
#
# Examples (from repo root):
#   bash experiments/local/run_hparam_full_grid_2gpu.sh --scale 4
#   bash experiments/local/run_hparam_full_grid_2gpu.sh -scale 10 20 40
#   bash experiments/local/run_hparam_full_grid_2gpu.sh --scale all
#
# This reuses experiments/generalization/hparam_full_grid.py for the canonical
# task-id mapping and validation. It runs two tasks at a time: GPU 0 and GPU 1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SCALES=(all)
START_TASK=""
END_TASK=""
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
LOG_DIR="$ROOT/experiments/local/artifacts/hparam_full_grid"
STATUS_DIR="$ROOT/experiments/local/artifacts/hparam_full_grid/status"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/local/run_hparam_full_grid_2gpu.sh [--scale 4|10|20|40|all ...]
  bash experiments/local/run_hparam_full_grid_2gpu.sh [-scale 10 20 40]

Optional environment overrides:
  AIM3_DATA_DIR=/path/to/stimuli
  GPU0=0 GPU1=1
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

mkdir -p "$LOG_DIR" "$STATUS_DIR"

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
elif [[ -d "/G/MIMOlab/Codes/aim3_RNN/stimuli" ]]; then
  DATA_DIR="/G/MIMOlab/Codes/aim3_RNN/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create $ROOT/stimuli." >&2
  exit 2
fi

run_task() {
  local task_id="$1"
  local gpu="$2"
  local task_tag
  task_tag="$(printf 'task_%04d' "$task_id")"
  local log_prefix="$LOG_DIR/${task_tag}"
  local done_file="$STATUS_DIR/${task_tag}.done"
  local fail_file="$STATUS_DIR/${task_tag}.fail"

  eval "$(python experiments/generalization/hparam_full_grid.py emit-task --task-id "$task_id" --root "$ROOT")"

  if python experiments/generalization/hparam_full_grid.py validate --task-id "$task_id" --root "$ROOT" >/dev/null 2>&1; then
    echo "[$(date -Is)] skip completed task_id=$task_id model=$MODEL_TYPE scale=$SCALE"
    {
      echo "status=skipped_existing"
      echo "task_id=$task_id"
      echo "metrics_path=$METRICS_PATH"
      echo "timestamp=$(date -Is)"
    } > "$done_file"
    rm -f "$fail_file"
    return 0
  fi

  echo "[$(date -Is)] start task_id=$task_id gpu=$gpu scale=$SCALE model=$MODEL_TYPE h=$HIDDEN_SIZE lr=$LR wd=$WD"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 python train_model.py \
    --model_types "$MODEL_TYPE" \
    --hidden_sizes "$HIDDEN_SIZE" \
    --data_suffix "$DATA_SUFFIX" \
    --eval_data_suffix "$EVAL_DATA_SUFFIX" \
    --data_dir "$DATA_DIR" \
    --lrs "$LR" \
    --wds "$WD" \
    --cnn_dropout "$CNN_DROPOUT" \
    --rnn_dropout "$RNN_DROPOUT" \
    --num_epochs "$NUM_EPOCHS" \
    --patience "$PATIENCE" \
    --seed "$SEED" \
    --use_acceleration \
    --use_sector_mode \
    --result_suffix "$RESULT_SUFFIX" \
    > "${log_prefix}.out" 2> "${log_prefix}.err"
  local train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 ]]; then
    {
      echo "status=train_failed"
      echo "task_id=$task_id"
      echo "exit_code=$train_rc"
      echo "scale=$SCALE"
      echo "model=$MODEL_TYPE"
      echo "hidden_size=$HIDDEN_SIZE"
      echo "lr=$LR"
      echo "weight_decay=$WD"
      echo "metrics_path=$METRICS_PATH"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(date -Is)"
    } > "$fail_file"
    echo "[$(date -Is)] fail task_id=$task_id rc=$train_rc"
    return "$train_rc"
  fi

  if python experiments/generalization/hparam_full_grid.py validate --task-id "$task_id" --root "$ROOT" >/dev/null 2>&1; then
    {
      echo "status=done"
      echo "task_id=$task_id"
      echo "scale=$SCALE"
      echo "model=$MODEL_TYPE"
      echo "hidden_size=$HIDDEN_SIZE"
      echo "lr=$LR"
      echo "weight_decay=$WD"
      echo "metrics_path=$METRICS_PATH"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(date -Is)"
    } > "$done_file"
    rm -f "$fail_file"
    echo "[$(date -Is)] done task_id=$task_id"
  else
    {
      echo "status=validation_failed"
      echo "task_id=$task_id"
      echo "metrics_path=$METRICS_PATH"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(date -Is)"
    } > "$fail_file"
    echo "[$(date -Is)] validation failed task_id=$task_id"
    return 1
  fi
}

echo "AIM3 local 2-GPU full-grid runner"
echo "root=$ROOT"
echo "data_dir=$DATA_DIR"
echo "scales=${SCALES[*]}"
if [[ "$TOTAL_TASKS" -gt 0 ]]; then
  echo "task_range=${TASK_IDS[0]}-${TASK_IDS[$((TOTAL_TASKS - 1))]}"
else
  echo "task_range=<empty>"
fi
echo "total_tasks=$TOTAL_TASKS"
echo "gpu_ids=$GPU0,$GPU1"
echo "log_dir=$LOG_DIR"

idx=0
while [[ "$idx" -lt "$TOTAL_TASKS" ]]; do
  task="${TASK_IDS[$idx]}"
  run_task "$task" "$GPU0" &
  p0=$!

  next_idx=$((idx + 1))
  if [[ "$next_idx" -lt "$TOTAL_TASKS" ]]; then
    next="${TASK_IDS[$next_idx]}"
    run_task "$next" "$GPU1" &
    p1=$!
    set +e
    wait "$p0"
    rc0=$?
    wait "$p1"
    rc1=$?
    set -e
    if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
      echo "[$(date -Is)] one or more tasks in pair ${task},${next} failed; continuing"
    fi
  else
    set +e
    wait "$p0"
    rc0=$?
    set -e
    if [[ "$rc0" -ne 0 ]]; then
      echo "[$(date -Is)] task ${task} failed; continuing"
    fi
  fi

  idx=$((idx + 2))
done

echo "Completed local task list (${TOTAL_TASKS} task(s))."
echo "Logs: $LOG_DIR"
echo "Status: $STATUS_DIR"
