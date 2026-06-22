#!/usr/bin/env bash
# Run the parameter-count-matched RNN/LSTM/GRU grid on two local GPUs.
#
# Default experiment: 40h training/validation, model widths matched to GAWF-256.
# The launcher is resumable: outputs that pass the canonical validator are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

GRID_UTIL="experiments/generalization/hparam_param_match_grid.py"
SCALES=(40h)
MODELS=(rnn lstm gru)
GAWF_REF_HIDDEN=256
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_TAG="${RUN_TAG:-param_match_gawf256_40h_2gpu}"
LOG_DIR="$ROOT/experiments/local/artifacts/$RUN_TAG"
STATUS_DIR="$LOG_DIR/status"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/local/run_hparam_param_match_2gpu.sh
  bash experiments/local/run_hparam_param_match_2gpu.sh \
    --scale 40 --model rnn lstm gru --gawf-ref-hidden 256

Optional environment overrides:
  AIM3_DATA_DIR=/path/to/stimuli
  AIM3_CONDA_ENV=aim3_rnn
  AIM3_SETUP_CMD='source .../conda.sh && conda activate aim3_rnn'
  PYTHON_BIN=/absolute/path/to/python
  GPU0=0 GPU1=1
  AIM3_BATCH_SIZE=256
  USE_MMAP=1
  RUN_TAG=param_match_gawf256_40h_2gpu

The sjc-remote-safe DataLoader settings are enforced and cannot be overridden:
  AIM3_NUM_WORKERS=0
  AIM3_PIN_MEMORY=0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale|--scales)
      SCALES=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        SCALES+=("$1")
        shift
      done
      ;;
    --model|--models)
      MODELS=()
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        MODELS+=("$1")
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

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif [[ -f "/G/anaconda3/etc/profile.d/conda.sh" ]]; then
  source /G/anaconda3/etc/profile.d/conda.sh
  conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  echo "Python environment is not training-ready: $PYTHON_BIN cannot import torch." >&2
  exit 2
fi

# Never inherit Amarel's multi-worker/pinned-memory submission settings on sjc-remote.
export AIM3_NUM_WORKERS=0
export AIM3_PIN_MEMORY=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

mapfile -t TASK_IDS < <(
  "$PYTHON_BIN" "$GRID_UTIL" list-task-ids \
    --scales "${SCALES[@]}" \
    --models "${MODELS[@]}" \
    --gawf-ref-hidden "$GAWF_REF_HIDDEN"
)
TOTAL_TASKS="${#TASK_IDS[@]}"
if [[ "$TOTAL_TASKS" -eq 0 ]]; then
  echo "No tasks selected." >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$STATUS_DIR"

run_task() {
  local task_id="$1"
  local gpu="$2"
  local task_tag log_prefix done_file fail_file
  task_tag="matchgawf${GAWF_REF_HIDDEN}_task_$(printf '%04d' "$task_id")"
  log_prefix="$LOG_DIR/$task_tag"
  done_file="$STATUS_DIR/$task_tag.done"
  fail_file="$STATUS_DIR/$task_tag.fail"

  eval "$(
    "$PYTHON_BIN" "$GRID_UTIL" emit-task \
      --task-id "$task_id" \
      --root "$ROOT" \
      --gawf-ref-hidden "$GAWF_REF_HIDDEN"
  )"

  if "$PYTHON_BIN" "$GRID_UTIL" validate \
    --task-id "$task_id" \
    --root "$ROOT" \
    --gawf-ref-hidden "$GAWF_REF_HIDDEN" >/dev/null 2>&1; then
    echo "[$(date -Is)] skip task_id=$task_id model=$MODEL_TYPE h=$HIDDEN_SIZE"
    {
      echo "status=skipped_existing"
      echo "task_id=$task_id"
      echo "metrics_path=$METRICS_PATH"
      echo "timestamp=$(date -Is)"
    } > "$done_file"
    rm -f "$fail_file"
    return 0
  fi

  local mmap_args=()
  if [[ "${USE_MMAP:-1}" == "1" ]]; then
    mmap_args=(--use_mmap)
  fi

  echo "[$(date -Is)] start task_id=$task_id gpu=$gpu scale=$SCALE model=$MODEL_TYPE h=$HIDDEN_SIZE lr=$LR wd=$WD"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 "$PYTHON_BIN" train_model.py \
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
    "${mmap_args[@]}" \
    --result_suffix "$RESULT_SUFFIX" \
    > "${log_prefix}.out" 2> "${log_prefix}.err"
  local train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 ]]; then
    {
      echo "status=train_failed"
      echo "task_id=$task_id"
      echo "exit_code=$train_rc"
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

  if "$PYTHON_BIN" "$GRID_UTIL" validate \
    --task-id "$task_id" \
    --root "$ROOT" \
    --gawf-ref-hidden "$GAWF_REF_HIDDEN" >/dev/null 2>&1; then
    {
      echo "status=done"
      echo "task_id=$task_id"
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

echo "AIM3 local 2-GPU parameter-matched runner"
echo "root=$ROOT"
echo "python=$PYTHON_BIN"
echo "data_dir=$DATA_DIR"
echo "scales=${SCALES[*]}"
echo "models=${MODELS[*]}"
echo "gawf_ref_hidden=$GAWF_REF_HIDDEN"
echo "total_tasks=$TOTAL_TASKS"
echo "gpu_ids=$GPU0,$GPU1"
echo "AIM3_BATCH_SIZE=${AIM3_BATCH_SIZE:-256}"
echo "AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS"
echo "AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
echo "use_mmap=${USE_MMAP:-1}"
echo "log_dir=$LOG_DIR"

idx=0
while [[ "$idx" -lt "$TOTAL_TASKS" ]]; do
  task0="${TASK_IDS[$idx]}"
  run_task "$task0" "$GPU0" &
  pid0=$!

  next_idx=$((idx + 1))
  if [[ "$next_idx" -lt "$TOTAL_TASKS" ]]; then
    task1="${TASK_IDS[$next_idx]}"
    run_task "$task1" "$GPU1" &
    pid1=$!
    set +e
    wait "$pid0"; rc0=$?
    wait "$pid1"; rc1=$?
    set -e
    if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
      echo "[$(date -Is)] task pair $task0,$task1 had a failure; continuing"
    fi
  else
    set +e
    wait "$pid0"; rc0=$?
    set -e
    if [[ "$rc0" -ne 0 ]]; then
      echo "[$(date -Is)] task $task0 failed; continuing"
    fi
  fi
  idx=$((idx + 2))
done

echo "Completed local task list ($TOTAL_TASKS task(s))."
echo "Logs: $LOG_DIR"
echo "Status: $STATUS_DIR"
