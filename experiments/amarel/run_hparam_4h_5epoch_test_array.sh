#!/usr/bin/env bash
#SBATCH --job-name=aim3-hpa-test
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/amarel/artifacts/hparam_4h_5epoch_test/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/hparam_4h_5epoch_test/%A_%a.err

# Run one 4h/5-epoch hparam smoke-test task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MODELS=(rnn lstm gru gawf)
HIDDEN_SIZES=(64 128 256 512)
LRS=(0.0001 0.0005 0.001 0.005)
WDS=(0.0 1e-05 0.0001 0.001)
TOTAL_TASKS=256
NUM_EPOCHS=5
PATIENCE=15
SEED=42
CNN_DROPOUT=0.0
RNN_DROPOUT=0.5
RESULT_ROOT_SUFFIX="gen_hparam_4h_5epoch_test"

mkdir -p experiments/amarel/artifacts/hparam_4h_5epoch_test
mkdir -p experiments/generalization/artifacts/${RESULT_ROOT_SUFFIX}/status

TASK_OFFSET="${TASK_OFFSET:-0}"
TASK_ID="$((TASK_OFFSET + ${SLURM_ARRAY_TASK_ID:-0}))"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "$TOTAL_TASKS" ]]; then
  echo "TASK_ID must be in [0, $((TOTAL_TASKS - 1))], got $TASK_ID" >&2
  exit 2
fi

wd_idx=$((TASK_ID % 4))
lr_idx=$(((TASK_ID / 4) % 4))
hidden_idx=$(((TASK_ID / 16) % 4))
model_idx=$(((TASK_ID / 64) % 4))

MODEL_TYPE="${MODELS[$model_idx]}"
HIDDEN_SIZE="${HIDDEN_SIZES[$hidden_idx]}"
LR="${LRS[$lr_idx]}"
WD="${WDS[$wd_idx]}"
DATA_SUFFIX="4h-float32"
EVAL_DATA_SUFFIX="40h-float32"
RESULT_SUFFIX="${RESULT_ROOT_SUFFIX}/task_$(printf '%04d' "$TASK_ID")"
RESULT_STEM="${MODEL_TYPE}_sector_acc_h${HIDDEN_SIZE}_lr${LR}_wd${WD}_cdo${CNN_DROPOUT}_rdo${RNN_DROPOUT}"
METRICS_PATH="$ROOT/results/train_data/${RESULT_SUFFIX}/${RESULT_STEM}_metrics.json"
STATUS_DIR="$ROOT/experiments/generalization/artifacts/${RESULT_ROOT_SUFFIX}/status"
DONE_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").done"
FAIL_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").fail"

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create stimuli under scratch/cache/repo." | tee "$FAIL_FILE"
  exit 2
fi

echo "[$(date -Is)] test_task_id=$TASK_ID model=$MODEL_TYPE h=$HIDDEN_SIZE lr=$LR wd=$WD"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"

if [[ -f "$METRICS_PATH" ]]; then
  echo "Test task $TASK_ID already has metrics; skipping."
  {
    echo "status=skipped_existing"
    echo "task_id=$TASK_ID"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
  exit 0
fi

set +e
DISABLE_TQDM=1 python train_model.py \
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
  --result_suffix "$RESULT_SUFFIX"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed"
    echo "task_id=$TASK_ID"
    echo "exit_code=$train_rc"
    echo "model=$MODEL_TYPE"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if [[ -f "$METRICS_PATH" ]]; then
  {
    echo "status=done"
    echo "task_id=$TASK_ID"
    echo "model=$MODEL_TYPE"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
else
  {
    echo "status=missing_metrics"
    echo "task_id=$TASK_ID"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
