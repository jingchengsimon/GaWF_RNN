#!/usr/bin/env bash
#SBATCH --job-name=aim3-imdb-lstm
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/amarel/artifacts/imdb_lstm_hparam_grid/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/imdb_lstm_hparam_grid/%A_%a.err

# Run one IMDB LSTM hparam-grid task. Submit via
# experiments/amarel/submit_imdb_hparam_grid_batches.sh.
# IMDB jobs are light (no CNN, short sequences) -> smaller cpu/mem than the vision grid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_imdb.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

# Grid util + artifact name are overridable so the same runner drives both the
# LSTM anchor grid (default) and the GaWF param-match grid. Override at submit
# time via --export=...,AIM3_IMDB_GRID_UTIL=...,AIM3_IMDB_GRID_NAME=... (and set
# sbatch --job-name / --output accordingly).
GRID_UTIL="${AIM3_IMDB_GRID_UTIL:-experiments/generalization/imdb_hparam_grid.py}"
GRID_NAME="${AIM3_IMDB_GRID_NAME:-imdb_lstm_hparam_grid}"
ART_ROOT="$ROOT/experiments/amarel/artifacts/$GRID_NAME"
STATUS_DIR="$ROOT/experiments/generalization/artifacts/$GRID_NAME/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli/imdb" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli/imdb" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli/imdb" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "IMDB data not found. Run scripts/prepare_imdb_data.py or set AIM3_DATA_DIR." | tee "$FAIL_FILE"
  exit 2
fi

eval "$(python "$GRID_UTIL" emit-task --task-id "$TASK_ID" --root "$ROOT")"

echo "[$(date -Is)] task_id=$TASK_ID model=$MODEL_TYPE hidden=$HIDDEN lr=$LR wd=$WD"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"

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
DISABLE_TQDM=1 python train_imdb.py \
  --model_types "$MODEL_TYPE" \
  --data_dir "$DATA_DIR" \
  --result_dir "$ROOT" \
  --result_suffix "$RESULT_SUFFIX" \
  --embed_dim "$EMBED_DIM" \
  --hidden_sizes "$HIDDEN" \
  --lrs "$LR" \
  --wds "$WD" \
  --embed_dropout "$EMBED_DROPOUT" \
  --rnn_dropout "$RNN_DROPOUT" \
  --pooling "$POOLING" \
  --optim "$OPTIM" \
  --num_epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "${AIM3_NUM_WORKERS:-4}" \
  --device cuda \
  --use_acceleration
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed"
    echo "task_id=$TASK_ID"
    echo "exit_code=$train_rc"
    echo "model=$MODEL_TYPE"
    echo "hidden=$HIDDEN"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if python "$GRID_UTIL" validate --task-id "$TASK_ID" --root "$ROOT" --json; then
  {
    echo "status=done"
    echo "task_id=$TASK_ID"
    echo "model=$MODEL_TYPE"
    echo "hidden=$HIDDEN"
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
    echo "model=$MODEL_TYPE"
    echo "hidden=$HIDDEN"
    echo "lr=$LR"
    echo "weight_decay=$WD"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
