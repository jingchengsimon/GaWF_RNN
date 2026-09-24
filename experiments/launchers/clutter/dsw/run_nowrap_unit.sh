#!/usr/bin/env bash
# Run one seeds6-10 Clutter no-wrap unit on a prepared DSW execution snapshot.

set -euo pipefail

if (( $# != 7 )); then
  echo "Usage: $0 ROOT ENV_ROOT DATA_ROOT RESULTS_ROOT RUN_ROOT TASK_ID GPU" >&2
  exit 2
fi

ROOT="$1"
ENV_ROOT="$2"
DATA_ROOT="$3"
RESULTS_ROOT="$4"
RUN_ROOT="$5"
TASK_ID="$6"
GPU="$7"
EXPECTED_COMMIT=988ee1b0f2727189479a56e494556651b589e1ca
RESULT_LEAF=clutter_ablate_outer_norm_40h_ep150_v1
TEST_LEAF=clutter_ablation_reset_excluded_test_10seed_v1
RUNNER="${AIM3_NOWRAP_RUNNER:-$ROOT/experiments/clutter/amarel/run_clutter_nonlinearity_ablation.sh}"

[[ "$TASK_ID" =~ ^[0-9]+$ ]] && (( TASK_ID >= 0 && TASK_ID < 30 )) || {
  echo "TASK_ID must be an integer in [0, 29]" >&2
  exit 2
}
[[ "$GPU" =~ ^[0-9]+$ ]] && (( GPU >= 0 && GPU < 8 )) || {
  echo "GPU must be an integer in [0, 7]" >&2
  exit 2
}
[[ -x "$ENV_ROOT/bin/python" && -f "$RUNNER" ]] || {
  echo "Prepared environment or canonical runner is missing" >&2
  exit 1
}
[[ "$(<"$ROOT/source_commit.txt")" == "$EXPECTED_COMMIT" ]] || {
  echo "Unexpected DSW source stamp" >&2
  exit 1
}

MODELS=(gawf_nowrap rnn_nowrap gru_nowrap lstm_nowrap mamba_nowrap s5_nowrap)
MODEL_INDEX=$((TASK_ID / 5))
MODEL="${MODELS[$MODEL_INDEX]}"
SEED=$((TASK_ID % 5 + 6))
printf -v SEED_TAG '%02d' "$SEED"
SUFFIX="$RESULT_LEAF/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS_ROOT/data/clutter/runs/$SUFFIX"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RECEIPT_DIR="$RUN_ROOT/receipts"
PROCESS_TOKEN="--result_suffix $SUFFIX"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RECEIPT_DIR"
[[ -s "$STATUS_DIR/smoke_waiver.txt" ]] || {
  echo "Prepared smoke waiver is missing: $STATUS_DIR/smoke_waiver.txt" >&2
  exit 1
}

exec 9>"$RUN_ROOT/launch.lock"
flock -x 9
if pgrep -af -- "$PROCESS_TOKEN" >/dev/null; then
  echo "An active writer already targets $SUFFIX" >&2
  exit 1
fi

shopt -s nullglob
models=("$RESULT_DIR"/*_model.pth)
metrics=("$RESULT_DIR"/*_metrics.json)
histories=("$RESULT_DIR"/*.pkl)
shopt -u nullglob
if (( ${#models[@]} || ${#metrics[@]} || ${#histories[@]} )); then
  echo "A final or incomplete final artifact already exists in $RESULT_DIR" >&2
  exit 1
fi

receipt="$RECEIPT_DIR/task_${TASK_ID}.launching"
printf 'task=%s model=%s seed=%s gpu=%s pid=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$GPU" "$$" "$(date -Is)" > "$receipt"
flock -u 9

export CUDA_VISIBLE_DEVICES="$GPU"
export AIM3_ROOT="$ROOT"
export AIM3_RESULTS_PATH="$RESULTS_ROOT"
export AIM3_CLUTTER_DATA_DIR="$DATA_ROOT"
export AIM3_STATUS_DIR="$STATUS_DIR"
export AIM3_SOURCE_COMMIT="$EXPECTED_COMMIT"
export AIM3_SOURCE_COMMIT_FILE="$ROOT/source_commit.txt"
export AIM3_SEED_OFFSET=5
export AIM3_BATCH1_LEAF="$RESULT_LEAF"
export AIM3_TEST_LEAF="$TEST_LEAF"
export AIM3_CONDA_SH=/mnt/workspace/sjc/miniconda3/etc/profile.d/conda.sh
export AIM3_CONDA_ENV="$ENV_ROOT"
export AIM3_NUM_WORKERS=2
export AIM3_PIN_MEMORY=1
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export SLURM_ARRAY_TASK_ID="$TASK_ID"

exec bash "$RUNNER"
