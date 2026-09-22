#!/usr/bin/env bash
# Train one GaWF-aligned no-tanh RNN unit on an Amarel compute node.

#SBATCH --job-name=aim3-rnn-inloop
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-2}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
SOURCE_COMMIT_STAMP="${AIM3_SOURCE_COMMIT_FILE:-$STATUS_DIR/source_commit.txt}"
RUN_MODE="${AIM3_RUN_MODE:-formal}"
MODEL="rnn_inloop_notanh"
WIDTH=275
LR=0.001
WD=0.00001

case "$RUN_MODE" in
  smoke)
    TASK_ID=smoke
    SEED=1
    EPOCHS=2
    LEAF="preflight/clutter_rnn_inloop_notanh_40h_2epoch_seed1_v1"
    ;;
  formal)
    TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
    (( TASK_ID >= 0 && TASK_ID < 10 )) || {
      echo "Task id must be in [0, 9]" >&2
      exit 2
    }
    SEED=$((TASK_ID + 1))
    EPOCHS=150
    LEAF="clutter_rnn_inloop_notanh_40h_ep150_v1"
    ;;
  *)
    echo "AIM3_RUN_MODE must be smoke or formal, got $RUN_MODE" >&2
    exit 2
    ;;
esac

printf -v SEED_TAG '%02d' "$SEED"
SUFFIX="$LEAF/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS/data/clutter/runs/$SUFFIX"
TEST_LEAF="${AIM3_TEST_LEAF:-clutter_ablation_reset_excluded_test_rnn_inloop_notanh_10seed_v1}"
TEST_DIR="$RESULTS/data/analysis/$TEST_LEAF/$MODEL-seed$SEED_TAG"
DONE_FILE="$STATUS_DIR/task_${TASK_ID}.done"
FAIL_FILE="$STATUS_DIR/task_${TASK_ID}.fail"
RUNNING_FILE="$STATUS_DIR/task_${TASK_ID}.running"

mkdir -p "$STATUS_DIR"
on_error() {
  status=$?
  trap - ERR
  printf 'status=failed mode=%s task=%s model=%s seed=%s exit=%s timestamp=%s\n' \
    "$RUN_MODE" "$TASK_ID" "$MODEL" "$SEED" "$status" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR
RUNNING_FORMAT='status=running mode=%s task=%s model=%s seed=%s width=%s lr=%s wd=%s '
RUNNING_FORMAT+='epochs=%s timestamp=%s\n'
printf "$RUNNING_FORMAT" \
  "$RUN_MODE" "$TASK_ID" "$MODEL" "$SEED" "$WIDTH" "$LR" "$WD" "$EPOCHS" \
  "$(date -Is)" > "$RUNNING_FILE"

cd "$ROOT"
AMAREL_SOURCE_GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
[[ -s "$AMAREL_SOURCE_GUARD" ]] || {
  echo "Missing source guard: $AMAREL_SOURCE_GUARD" >&2
  exit 1
}
# shellcheck source=../../remote/amarel_source_guard.sh
source "$AMAREL_SOURCE_GUARD"
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$SOURCE_COMMIT_STAMP"; then
  echo "Source commit check failed" >&2
  exit 1
fi

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

shopt -s nullglob
CHECKPOINTS=("$RESULT_DIR"/*_model.pth)
METRICS=("$RESULT_DIR"/*_metrics.json)
HISTORIES=("$RESULT_DIR"/*.pkl)
shopt -u nullglob
if (( ${#CHECKPOINTS[@]} || ${#METRICS[@]} || ${#HISTORIES[@]} )); then
  (( ${#CHECKPOINTS[@]} == 1 && ${#METRICS[@]} == 1 && ${#HISTORIES[@]} == 1 )) || {
    echo "Incomplete final training output contract in $RESULT_DIR" >&2
    exit 1
  }
else
  python -B run_task.py clutter \
    --model_types "$MODEL" --hidden_sizes "$WIDTH" \
    --num_layers 1 --num_epochs "$EPOCHS" --patience 0 \
    --lrs "$LR" --wds "$WD" --optim adamw \
    --cnn_dropout 0.0 --rnn_dropout 0.5 \
    --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
    --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
    --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
    --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
    --checkpoint_interval_epochs 5 --auto_resume \
    --result_suffix "$SUFFIX"
  shopt -s nullglob
  CHECKPOINTS=("$RESULT_DIR"/*_model.pth)
  METRICS=("$RESULT_DIR"/*_metrics.json)
  HISTORIES=("$RESULT_DIR"/*.pkl)
  shopt -u nullglob
  (( ${#CHECKPOINTS[@]} == 1 && ${#METRICS[@]} == 1 && ${#HISTORIES[@]} == 1 )) || {
    echo "Incomplete final training output contract in $RESULT_DIR" >&2
    exit 1
  }
fi

if [[ "$RUN_MODE" == formal && ! -f "$TEST_DIR/reset_excluded_test_accuracy.json" ]]; then
  [[ ! -e "$TEST_DIR" ]] || {
    echo "Incomplete reset-excluded test output: $TEST_DIR" >&2
    exit 1
  }
  python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
    --ckpt "${CHECKPOINTS[0]}" --model "$MODEL" --seed "$SEED" \
    --output_dir "$TEST_DIR" --data_dir "$DATA_DIR" --data_suffix 40h-uint8 \
    --sequence_length 32 --batch_size 256 --num_workers 2 --device cuda
fi

printf 'status=done mode=%s task=%s model=%s seed=%s timestamp=%s\n' \
  "$RUN_MODE" "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$DONE_FILE"
trap - ERR
