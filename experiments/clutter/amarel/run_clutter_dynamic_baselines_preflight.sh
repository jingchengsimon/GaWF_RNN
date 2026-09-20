#!/usr/bin/env bash
#SBATCH --job-name=aim3-dyn-preflight
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Run one exact two-epoch Clutter sanity unit on an Amarel compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-2}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
REPORTS_DIR="${AIM3_PREFLIGHT_REPORTS_DIR:?AIM3_PREFLIGHT_REPORTS_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
(( TASK_ID >= 0 && TASK_ID < 4 )) || { echo "Task id must be in [0, 3]" >&2; exit 2; }

MODELS=(lstm mlstm hyperlstm brims)
WIDTHS=(80 65 70 84)
MODEL="${MODELS[TASK_ID]}"
WIDTH="${WIDTHS[TASK_ID]}"
SEED=1
RUN_TAG="dynamic_weight_baselines_2epoch_seed1_v1"
SUFFIX="preflight/$RUN_TAG/$MODEL-seed01"
RESULT_DIR="$RESULTS/data/clutter/runs/$SUFFIX"
REPORT="$REPORTS_DIR/$MODEL-seed01.json"
DONE_FILE="$STATUS_DIR/task_${TASK_ID}.done"
FAIL_FILE="$STATUS_DIR/task_${TASK_ID}.fail"
SOURCE_COMMIT_STAMP="${AIM3_SOURCE_COMMIT_FILE:-$STATUS_DIR/source_commit.txt}"

mkdir -p "$STATUS_DIR" "$REPORTS_DIR"
on_error() {
  status=$?
  trap - ERR
  printf 'status=failed task=%s model=%s exit=%s timestamp=%s\n' \
    "$TASK_ID" "$MODEL" "$status" "$(date -Is)" > "$FAIL_FILE"
  exit "$status"
}
trap on_error ERR

cd "$ROOT"
# shellcheck source=../../remote/amarel_source_guard.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../remote" && pwd)/amarel_source_guard.sh"
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

MODEL_ARGS=()
if [[ "$MODEL" == hyperlstm ]]; then
  MODEL_ARGS+=(--hyper_hidden_sizes 10 --hyper_embedding_size 4)
elif [[ "$MODEL" == brims ]]; then
  MODEL_ARGS+=(
    --brims_num_blocks 6 3 --brims_topk 4 2
    --brims_input_attention_heads 4 --brims_input_attention_key_size 64
    --brims_communication_attention_heads 4
    --brims_communication_attention_key_size 32
    --brims_communication_attention_value_size 32 --brims_attention_dropout 0.1
  )
fi

START_SECONDS="$(date +%s)"
python -B run_task.py clutter \
  --model_types "$MODEL" --hidden_sizes "$WIDTH" "${MODEL_ARGS[@]}" \
  --num_layers 1 --num_epochs 2 --patience 0 \
  --lrs 0.001 --wds 0.001 --optim adamw \
  --cnn_dropout 0.0 --rnn_dropout 0.5 \
  --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
  --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
  --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
  --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
  --checkpoint_interval_epochs 0 --result_suffix "$SUFFIX"
ELAPSED_SECONDS=$(( $(date +%s) - START_SECONDS ))

shopt -s nullglob
HISTORIES=("$RESULT_DIR"/*.pkl)
METRICS=("$RESULT_DIR"/*_metrics.json)
MODELS_OUT=("$RESULT_DIR"/*_model.pth)
shopt -u nullglob
(( ${#HISTORIES[@]} == 1 && ${#METRICS[@]} == 1 && ${#MODELS_OUT[@]} == 1 )) || {
  echo "Incomplete two-epoch output contract in $RESULT_DIR" >&2; exit 1;
}
python -B -m experiments.clutter.dynamic_weight_sanity_report \
  --model "$MODEL" --seed "$SEED" --history "${HISTORIES[0]}" \
  --metrics "${METRICS[0]}" --elapsed-seconds "$ELAPSED_SECONDS" \
  --source-commit "$SOURCE_COMMIT" --output "$REPORT"

printf 'status=done task=%s model=%s seed=%s elapsed_seconds=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$ELAPSED_SECONDS" "$(date -Is)" > "$DONE_FILE"
trap - ERR
