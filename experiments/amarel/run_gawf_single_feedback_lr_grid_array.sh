#!/usr/bin/env bash
#SBATCH --job-name=gawf1-fblr
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/amarel/artifacts/gawf_single_feedback_lr_grid/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/gawf_single_feedback_lr_grid/%A_%a.err

# Run one single-layer GaWF feedback-LR-scale search task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

GRID_UTIL="experiments/generalization/gawf_single_feedback_lr_grid.py"
GRID_NAME="gawf_single_feedback_lr_grid"
ART_ROOT="$ROOT/experiments/amarel/artifacts/$GRID_NAME"
STATUS_DIR="$ROOT/experiments/generalization/artifacts/gawf_single_fblr_finesearch_40h/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

TASKS_PER_ARRAY="${TASKS_PER_ARRAY:-1}"
if [[ "$TASKS_PER_ARRAY" -gt 1 && -z "${AIM3_INNER_TASK_ID:-}" ]]; then
  total_tasks="$(python "$GRID_UTIL" list-task-ids | wc -l | tr -d ' ')"
  start_task=$((${SLURM_ARRAY_TASK_ID:-0} * TASKS_PER_ARRAY))
  end_task=$((start_task + TASKS_PER_ARRAY - 1))
  if [[ "$end_task" -ge "$total_tasks" ]]; then
    end_task=$((total_tasks - 1))
  fi
  echo "Array index ${SLURM_ARRAY_TASK_ID:-0} running logical tasks ${start_task}-${end_task}"
  for logical_task_id in $(seq "$start_task" "$end_task"); do
    AIM3_INNER_TASK_ID="$logical_task_id" TASKS_PER_ARRAY=1 bash "$0"
  done
  exit 0
fi

TASK_ID="${AIM3_INNER_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DONE_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").done"
FAIL_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").fail"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn

export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create stimuli under scratch/cache/repo." \
    | tee "$FAIL_FILE"
  exit 2
fi

eval "$(python "$GRID_UTIL" emit-task --task-id "$TASK_ID" --root "$ROOT")"

echo "[$(date -Is)] task_id=$TASK_ID model=$MODEL_TYPE scale=$SCALE h=$HIDDEN_SIZE lr=$LR wd=$WD fblr=$GAWF_FEEDBACK_LR_SCALE"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"
echo "AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"

if python "$GRID_UTIL" validate --task-id "$TASK_ID" --root "$ROOT" >/dev/null 2>&1; then
  echo "Task $TASK_ID already complete; skipping."
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
  --results_dir "$ROOT/results" \
  --lrs "$LR" \
  --wds "$WD" \
  --cnn_dropout "$CNN_DROPOUT" \
  --rnn_dropout "$RNN_DROPOUT" \
  --gawf_feedback_lr_scale "$GAWF_FEEDBACK_LR_SCALE" \
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
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "gawf_feedback_lr_scale=$GAWF_FEEDBACK_LR_SCALE"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if python "$GRID_UTIL" validate --task-id "$TASK_ID" --root "$ROOT" --json; then
  {
    echo "status=done"
    echo "task_id=$TASK_ID"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "gawf_feedback_lr_scale=$GAWF_FEEDBACK_LR_SCALE"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
else
  {
    echo "status=validation_failed"
    echo "task_id=$TASK_ID"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "gawf_feedback_lr_scale=$GAWF_FEEDBACK_LR_SCALE"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
