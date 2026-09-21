#!/usr/bin/env bash
#SBATCH --job-name=aim3-fbctrl-formal
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

# Train and evaluate one model/seed feedback-control unit on an Amarel compute node.

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
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 40 )) || { echo "Task id must be in [0, 39]" >&2; exit 2; }

MODELS=(gawf_additive rnn_fb gru_fb lstm_fb)
WIDTHS=(271 272 103 79)
LRS=(0.005 0.001 0.005 0.001)
WDS=(0.001 0.00001 0.001 0.001)
MODEL_INDEX=$((TASK_ID / 10))
SEED=$((TASK_ID % 10 + 1))
MODEL="${MODELS[MODEL_INDEX]}"
WIDTH="${WIDTHS[MODEL_INDEX]}"
LR="${LRS[MODEL_INDEX]}"
WD="${WDS[MODEL_INDEX]}"
printf -v SEED_TAG '%02d' "$SEED"

RUN_TAG="clutter_feedback_controls_ep150_v1"
SUFFIX="feedback_controls/$RUN_TAG/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS/data/clutter/runs/$SUFFIX"
TEST_DIR="$RESULTS/data/analysis/feedback_controls_reset_excluded_test_10seed_v1/$MODEL-seed$SEED_TAG"
ABLATION_DIR="$RESULTS/data/analysis/feedback_controls_shuffle_resetexcluded_10seed_v1/$MODEL-seed$SEED_TAG"
DONE_FILE="$STATUS_DIR/task_${TASK_ID}.done"
FAIL_FILE="$STATUS_DIR/task_${TASK_ID}.fail"
RUNNING_FILE="$STATUS_DIR/task_${TASK_ID}.running"
SOURCE_COMMIT_STAMP="${AIM3_SOURCE_COMMIT_FILE:-$STATUS_DIR/source_commit.txt}"

mkdir -p "$STATUS_DIR"
on_error() {
  status=$?
  trap - ERR
  printf 'status=failed task=%s model=%s seed=%s exit=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$status" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR
printf 'status=running task=%s model=%s seed=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$RUNNING_FILE"

cd "$ROOT"
AMAREL_SOURCE_GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
if [[ ! -s "$AMAREL_SOURCE_GUARD" ]]; then
  printf 'status=failed task=%s model=%s seed=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$FAIL_FILE"
  printf 'Missing compute-node source guard: %s\n' "$AMAREL_SOURCE_GUARD" >&2
  exit 1
fi
# shellcheck source=../../remote/amarel_source_guard.sh
source "$AMAREL_SOURCE_GUARD"
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$SOURCE_COMMIT_STAMP"; then
  printf 'status=failed task=%s model=%s seed=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$FAIL_FILE"
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
    echo "Incomplete final training output contract in $RESULT_DIR" >&2; exit 1;
  }
else
  python -B run_task.py clutter \
    --model_types "$MODEL" --hidden_sizes "$WIDTH" \
    --num_layers 1 --num_epochs 150 --patience 0 \
    --lrs "$LR" --wds "$WD" --optim adamw \
    --cnn_dropout 0.0 --rnn_dropout 0.5 \
    --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
    --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
    --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
    --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
    --checkpoint_interval_epochs 5 --auto_resume --result_suffix "$SUFFIX"
  shopt -s nullglob
  CHECKPOINTS=("$RESULT_DIR"/*_model.pth)
  METRICS=("$RESULT_DIR"/*_metrics.json)
  HISTORIES=("$RESULT_DIR"/*.pkl)
  shopt -u nullglob
  (( ${#CHECKPOINTS[@]} == 1 && ${#METRICS[@]} == 1 && ${#HISTORIES[@]} == 1 )) || {
    echo "Incomplete training output contract in $RESULT_DIR" >&2; exit 1;
  }
fi

if [[ ! -f "$TEST_DIR/reset_excluded_test_accuracy.json" ]]; then
  [[ ! -e "$TEST_DIR" ]] || { echo "Incomplete test output: $TEST_DIR" >&2; exit 1; }
  python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
    --ckpt "${CHECKPOINTS[0]}" --model "$MODEL" --seed "$SEED" --output_dir "$TEST_DIR" \
    --data_dir "$DATA_DIR" --data_suffix 40h-uint8 --sequence_length 32 \
    --batch_size 256 --num_workers 2 --device cuda
fi
if [[ ! -f "$ABLATION_DIR/ablation_metrics.json" ]]; then
  [[ ! -e "$ABLATION_DIR" ]] || {
    echo "Incomplete feedback-ablation output: $ABLATION_DIR" >&2; exit 1;
  }
  python -B -m utils.analysis.clutter.fig2_feedback_ablation \
    --ckpt "${CHECKPOINTS[0]}" --save_dir "$ABLATION_DIR" --data_dir "$DATA_DIR" \
    --data_suffix 40h-float32-jointswitch-balanced-10digit-unique \
    --conditions baseline shuffle_digit shuffle_sector shuffle_all \
    --K 10 --pre_K 10 --sequence_length 512 --batch_size 16 --device cuda \
    --seed "$SEED" --exclude_window_initial_frame
fi

printf 'status=done task=%s model=%s seed=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$DONE_FILE"
trap - ERR
