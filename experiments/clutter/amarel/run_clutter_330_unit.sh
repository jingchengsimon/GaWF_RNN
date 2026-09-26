#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-330
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue

# Run one output-only-dropout Clutter model/seed unit on a GPU compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export AIM3_NUM_WORKERS=2
export AIM3_PIN_MEMORY=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
ART="${AIM3_ARTIFACT_ROOT:?AIM3_ARTIFACT_ROOT is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "$TASK_ID" =~ ^[0-9]+$ ]] && (( TASK_ID < 330 )) || {
  echo "Invalid 330-campaign task ID: $TASK_ID" >&2; exit 2;
}
STATUS="$ART/status"
mkdir -p "$STATUS"
FAIL="$STATUS/task_${TASK_ID}.fail"
DONE="$STATUS/task_${TASK_ID}.done"

on_error() {
  code=$?
  trap - ERR
  printf 'task=%s exit=%s time=%s\n' "$TASK_ID" "$code" "$(date -Is)" > "$FAIL"
  exit "$code"
}
trap on_error ERR

cd "$ROOT"
GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
if [[ ! -s "$GUARD" ]]; then
  echo "Missing source guard: $GUARD" >&2
  printf 'task=%s missing_guard\n' "$TASK_ID" > "$FAIL"
  exit 1
fi
# shellcheck source=../../remote/amarel_source_guard.sh
source "$GUARD" || {
  printf 'task=%s source_guard_load_failed\n' "$TASK_ID" > "$FAIL"
  exit 1
}
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$ART/status/source_commit.txt"; then
  printf 'task=%s source_mismatch\n' "$TASK_ID" > "$FAIL"
  exit 1
fi
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH" || {
  printf 'task=%s conda_init_failed\n' "$TASK_ID" > "$FAIL"
  exit 1
}
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || {
  printf 'task=%s conda_activate_failed\n' "$TASK_ID" > "$FAIL"
  exit 1
}
set -u

SPEC_TEXT="$(python -B -m experiments.clutter.amarel.rolling330 shell-spec --task-id "$TASK_ID")"
mapfile -t SPEC <<< "$SPEC_TEXT"
GROUP="${SPEC[0]}"
SCALE="${SPEC[1]}"
MODEL="${SPEC[2]}"
SEED="${SPEC[3]}"
WIDTH="${SPEC[4]}"
LR="${SPEC[5]}"
WD="${SPEC[6]}"
SUFFIX="${SPEC[7]}"
RESULT_DIR="$RESULTS/data/clutter/runs/$SUFFIX"

if [[ -e "$DONE" ]]; then
  python -B -m experiments.clutter.amarel.rolling330 validate --task-id "$TASK_ID"
  trap - ERR
  exit 0
fi
if [[ -d "$RESULT_DIR" ]]; then
  shopt -s nullglob
  FINALS=("$RESULT_DIR"/*_metrics.json "$RESULT_DIR"/*_model.pth "$RESULT_DIR"/*.pkl)
  shopt -u nullglob
  if (( ${#FINALS[@]} )); then
    echo "Incomplete or unregistered final artifacts in $RESULT_DIR" >&2
    printf 'task=%s incomplete_final_artifacts\n' "$TASK_ID" > "$FAIL"
    exit 1
  fi
fi

WIDTH_ARGS=(--hidden_sizes "$WIDTH")
if [[ "$MODEL" == mamba ]]; then
  WIDTH_ARGS=(--mamba_d_models "$WIDTH")
elif [[ "$MODEL" == s5 ]]; then
  WIDTH_ARGS=(--s5_d_models "$WIDTH" --s5_state_sizes 128)
fi

python -B run_task.py clutter \
  --model_types "$MODEL" "${WIDTH_ARGS[@]}" \
  --num_layers 1 --num_epochs 150 --patience 0 \
  --lrs "$LR" --wds "$WD" --optim adamw \
  --cnn_dropout 0.0 --dropout 0.5 \
  --gawf_feedback_lr_scale 1.0 \
  --s5_num_layers 1 --s5_dropout 0.0 --s5_ssm_lr_scale 0.1 \
  --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
  --data_dir "$DATA" --results_dir "$RESULTS" \
  --data_suffix "$SCALE-uint8" --eval_data_suffix 40h-uint8 \
  --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
  --checkpoint_interval_epochs 5 --auto_resume --result_suffix "$SUFFIX"

python -B -m experiments.clutter.amarel.rolling330 validate --task-id "$TASK_ID"
printf '{"task_id":%s,"source_commit":"%s"}\n' "$TASK_ID" "$SOURCE_COMMIT" > "$DONE"
rm -f "$FAIL"
printf 'done group=%s scale=%s model=%s seed=%s result=%s\n' \
  "$GROUP" "$SCALE" "$MODEL" "$SEED" "$RESULT_DIR"
trap - ERR
