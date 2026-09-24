#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-cores
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue

# Train and evaluate one GaWF core-placement unit on an Amarel compute node.
#
# Task map (30 tasks, 10 seeds per model):
#   0-9   gawf_legacy_nowrap  legacy GaWF with the in-loop LayerNorm/ReLU/dropout wrap removed
#   10-19 gawf_legacy_notanh  legacy GaWF with the in-loop activation removed (wrap kept)
#   20-29 gawf_rnncore       new RNN-aligned GaWF, unmodified (wrap and tanh both enabled)
#
# The protocol smoke/preflight gate is waived by explicit human instruction recorded in
# $STATUS_DIR/smoke_waiver.txt; this launcher refuses to run when that file is absent.

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
WAIVER_FILE="$STATUS_DIR/smoke_waiver.txt"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 30 )) || { echo "Task id must be in [0, 29]" >&2; exit 2; }

MODELS=(gawf_legacy_nowrap gawf_legacy_notanh gawf_rnncore)
LEAVES=(
  clutter_ablate_legacy_nowrap_40h_ep150_v1
  clutter_ablate_legacy_notanh_40h_ep150_v1
  clutter_ablate_rnncore_40h_ep150_v1
)
SEEDS_PER_MODEL=10
WIDTH=256
LR=0.005
WD=0.001
TEST_LEAF="${AIM3_TEST_LEAF:-clutter_ablation_reset_excluded_test_legacy_rnncore_v1}"

INDEX=$((TASK_ID / SEEDS_PER_MODEL))
SEED=$((TASK_ID % SEEDS_PER_MODEL + ${AIM3_SEED_OFFSET:-0} + 1))
MODEL="${MODELS[INDEX]}"
LEAF="${LEAVES[INDEX]}"
printf -v SEED_TAG '%02d' "$SEED"

SUFFIX="$LEAF/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS/data/clutter/runs/$SUFFIX"
TEST_DIR="$RESULTS/data/analysis/$TEST_LEAF/$MODEL-seed$SEED_TAG"
DONE_FILE="$STATUS_DIR/task_${TASK_ID}.done"
FAIL_FILE="$STATUS_DIR/task_${TASK_ID}.fail"
RUNNING_FILE="$STATUS_DIR/task_${TASK_ID}.running"

mkdir -p "$STATUS_DIR"
on_error() {
  status=$?
  trap - ERR
  printf 'status=failed task=%s model=%s seed=%s exit=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$SEED" "$status" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR
printf 'status=running task=%s model=%s seed=%s width=%s lr=%s wd=%s leaf=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$WIDTH" "$LR" "$WD" "$LEAF" "$(date -Is)" > "$RUNNING_FILE"

[[ -s "$WAIVER_FILE" ]] || {
  printf 'status=failed task=%s reason=missing_smoke_waiver timestamp=%s\n' \
    "$TASK_ID" "$(date -Is)" > "$FAIL_FILE"
  exit 1
}

case "$MODEL" in
  *) WIDTH_ARGS=(--hidden_sizes "$WIDTH") ;;
esac

cd "$ROOT"
AMAREL_SOURCE_GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
[[ -s "$AMAREL_SOURCE_GUARD" ]] || {
  printf 'status=failed task=%s reason=missing_guard timestamp=%s\n' \
    "$TASK_ID" "$(date -Is)" > "$FAIL_FILE"
  exit 1
}
# shellcheck source=../../remote/amarel_source_guard.sh
source "$AMAREL_SOURCE_GUARD"
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$SOURCE_COMMIT_STAMP"; then
  printf 'status=failed task=%s reason=source_commit timestamp=%s\n' \
    "$TASK_ID" "$(date -Is)" > "$FAIL_FILE"
  exit 1
fi

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH" || {
  printf 'status=failed task=%s reason=conda_init timestamp=%s\n' \
    "$TASK_ID" "$(date -Is)" > "$FAIL_FILE"
  exit 1
}
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
    --model_types "$MODEL" "${WIDTH_ARGS[@]}" \
    --num_layers 1 --num_epochs 150 --patience 0 \
    --lrs "$LR" --wds "$WD" --optim adamw \
    --cnn_dropout 0.0 --rnn_dropout 0.5 \
    --gawf_feedback_lr_scale 1.0 \
    --s5_num_layers 1 --s5_dropout 0.0 --s5_ssm_lr_scale 0.1 \
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

if [[ ! -f "$TEST_DIR/reset_excluded_test_accuracy.json" ]]; then
  [[ ! -e "$TEST_DIR" ]] || {
    echo "Incomplete reset-excluded test output: $TEST_DIR" >&2
    exit 1
  }
  python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
    --ckpt "${CHECKPOINTS[0]}" --model "$MODEL" --seed "$SEED" --output_dir "$TEST_DIR" \
    --data_dir "$DATA_DIR" --data_suffix 40h-uint8 --sequence_length 32 \
    --batch_size 256 --num_workers 2 --device cuda
fi

printf 'status=done task=%s model=%s seed=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$(date -Is)" > "$DONE_FILE"
trap - ERR
