#!/usr/bin/env bash
#SBATCH --job-name=aim3-imdb-best10
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Run one fixed best-hyperparameter IMDB model/seed unit.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must be exported at submission}"
: "${AIM3_DATA_DIR:?AIM3_DATA_DIR must be exported at submission}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

GRID_UTIL="$ROOT/experiments/text/imdb_5model_best10seed.py"
RUN_TAG="imdb_5model_best10seed"
ART_ROOT="$ROOT/experiments/text/amarel/artifacts/$RUN_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

TASK_ID="$SLURM_ARRAY_TASK_ID"
DONE_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").done"
FAIL_FILE="$STATUS_DIR/task_$(printf '%04d' "$TASK_ID").fail"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

if [[ ! -s "$AIM3_DATA_DIR/imdb/imdb_meta.json" ]]; then
  echo "Processed IMDB data missing under $AIM3_DATA_DIR/imdb" > "$FAIL_FILE"
  exit 2
fi

eval "$(python -B "$GRID_UTIL" emit-task --task-id "$TASK_ID" \
  --root "$AIM3_RESULTS_PATH")"

echo "[$(date -Is)] task_id=$TASK_ID model=$MODEL_TYPE seed=$SEED"
echo "hidden=$HIDDEN lr=$LR wd=$WD num_layers=$NUM_LAYERS"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"

if python -B "$GRID_UTIL" validate --task-id "$TASK_ID" \
    --root "$AIM3_RESULTS_PATH" >/dev/null 2>&1; then
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
DISABLE_TQDM=1 python -B run_task.py imdb \
  --model_types "$MODEL_TYPE" \
  --data_dir "$AIM3_DATA_DIR" \
  --results_dir "$AIM3_RESULTS_PATH" \
  --result_suffix "$RESULT_SUFFIX" \
  --embed_dim "$EMBED_DIM" \
  --hidden_sizes "$HIDDEN" \
  --num_layers "$NUM_LAYERS" \
  --gawf_feedback_lr_scale "$GAWF_FEEDBACK_LR_SCALE" \
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
  --num_workers "$AIM3_NUM_WORKERS" \
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
    echo "seed=$SEED"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if python -B "$GRID_UTIL" validate --task-id "$TASK_ID" \
    --root "$AIM3_RESULTS_PATH" --json; then
  {
    echo "status=done"
    echo "task_id=$TASK_ID"
    echo "model=$MODEL_TYPE"
    echo "seed=$SEED"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
else
  {
    echo "status=validation_failed"
    echo "task_id=$TASK_ID"
    echo "model=$MODEL_TYPE"
    echo "seed=$SEED"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
