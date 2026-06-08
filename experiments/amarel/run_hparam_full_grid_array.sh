#!/usr/bin/env bash
#SBATCH --job-name=aim3-hparam
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpu018,gpu043
#SBATCH --time=72:00:00
#SBATCH --output=experiments/amarel/artifacts/hparam_full_grid/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/hparam_full_grid/%A_%a.err

# Run one full-grid hparam task. Submit via submit_hparam_full_grid_batches.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

mkdir -p "$ROOT/experiments/amarel/artifacts/hparam_full_grid"
mkdir -p "$ROOT/experiments/generalization/artifacts/gen_hparam_full_grid/status"

if [[ -n "${TASK_ID_FILE:-}" ]]; then
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_ID_FILE requires SLURM_ARRAY_TASK_ID" >&2
    exit 2
  fi
  TASK_FILE_OFFSET="${TASK_FILE_OFFSET:-0}"
  TASK_ID="$(sed -n "$((TASK_FILE_OFFSET + SLURM_ARRAY_TASK_ID + 1))p" "$TASK_ID_FILE")"
else
  TASK_OFFSET="${TASK_OFFSET:-0}"
  TASK_ID="$((TASK_OFFSET + ${SLURM_ARRAY_TASK_ID:-0}))"
fi

if [[ -z "$TASK_ID" ]]; then
  echo "Empty TASK_ID resolved from TASK_ID_FILE=${TASK_ID_FILE:-<unset>}" >&2
  exit 2
fi

STATUS_DIR="$ROOT/experiments/generalization/artifacts/gen_hparam_full_grid/status"
DONE_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").done"
FAIL_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").fail"

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # Continue if the env is already active or if local Python is configured differently.
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi

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
  echo "Data directory not found. Set AIM3_DATA_DIR or create stimuli under scratch/cache/repo." | tee "$FAIL_FILE"
  exit 2
fi

eval "$(python experiments/generalization/hparam_full_grid.py emit-task --task-id "$TASK_ID" --root "$ROOT")"

echo "[$(date -Is)] task_id=$TASK_ID scale=$SCALE model=$MODEL_TYPE h=$HIDDEN_SIZE lr=$LR wd=$WD"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"

if python experiments/generalization/hparam_full_grid.py validate --task-id "$TASK_ID" --root "$ROOT" >/dev/null 2>&1; then
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
    echo "scale=$SCALE"
    echo "model=$MODEL_TYPE"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if python experiments/generalization/hparam_full_grid.py validate --task-id "$TASK_ID" --root "$ROOT" --json; then
  {
    echo "status=done"
    echo "task_id=$TASK_ID"
    echo "scale=$SCALE"
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
    echo "status=validation_failed"
    echo "task_id=$TASK_ID"
    echo "scale=$SCALE"
    echo "model=$MODEL_TYPE"
    echo "hidden_size=$HIDDEN_SIZE"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
