#!/usr/bin/env bash
# Run one corrected additive-feedback Clutter unit on a fixed DSW GPU.

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

[[ "$TASK_ID" =~ ^[0-9]+$ ]] && (( TASK_ID >= 0 && TASK_ID < 15 )) || {
  echo "TASK_ID must be an integer in [0, 14]" >&2
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be an integer in [0, 7]" >&2; exit 2; }
[[ -x "$ENV_ROOT/bin/python" && -f "$ROOT/run_task.py" ]] || {
  echo "Prepared environment or source checkout is missing" >&2
  exit 1
}
EXPECTED_COMMIT="$(<"$RUN_ROOT/expected_commit.txt")"
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "Unexpected DSW source commit: $ACTUAL_COMMIT" >&2
  exit 1
}
[[ -s "$RUN_ROOT/smoke/sanity.json" && -s "$RUN_ROOT/smoke/pass" ]] || {
  echo "Additive-feedback smoke gate has not passed" >&2
  exit 1
}

MODELS=(rnn_fb_add gru_fb_add lstm_fb_add)
WIDTHS=(272 103 79)
LRS=(0.001 0.005 0.001)
WDS=(0.00001 0.001 0.001)
MODEL_INDEX=$((TASK_ID / 5))
MODEL="${MODELS[$MODEL_INDEX]}"
WIDTH="${WIDTHS[$MODEL_INDEX]}"
LR="${LRS[$MODEL_INDEX]}"
WD="${WDS[$MODEL_INDEX]}"
SEED=$((TASK_ID % 5 + 1))
printf -v SEED_TAG '%02d' "$SEED"
LEAF=clutter_feedback_add_newsem_40h_ep150_v1
TEST_LEAF=clutter_feedback_add_newsem_reset_excluded_test_5seed_v1
SUFFIX="$LEAF/$MODEL-seed$SEED_TAG"
RESULT_DIR="$RESULTS_ROOT/data/clutter/runs/$SUFFIX"
TEST_DIR="$RESULTS_ROOT/data/analysis/$TEST_LEAF/$MODEL-seed$SEED_TAG"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
PROCESS_TOKEN="--result_suffix $SUFFIX"

mkdir -p "$STATUS_DIR" "$LOG_DIR"
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
flock -u 9

printf 'state=running task=%s model=%s seed=%s gpu=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$GPU" "$(date -Is)" > "$STATUS_DIR/task_${TASK_ID}.status"

export CUDA_VISIBLE_DEVICES="$GPU"
export AIM3_NUM_WORKERS=2
export AIM3_PIN_MEMORY=1
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
cd "$ROOT"
"$ENV_ROOT/bin/python" -B run_task.py clutter \
  --model_types "$MODEL" --hidden_sizes "$WIDTH" \
  --num_layers 1 --num_epochs 150 --patience 0 \
  --lrs "$LR" --wds "$WD" --optim adamw \
  --cnn_dropout 0.0 --rnn_dropout 0.5 \
  --seed "$SEED" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
  --data_dir "$DATA_ROOT" --results_dir "$RESULTS_ROOT" \
  --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
  --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
  --checkpoint_interval_epochs 5 --auto_resume --result_suffix "$SUFFIX"

shopt -s nullglob
models=("$RESULT_DIR"/*_model.pth)
metrics=("$RESULT_DIR"/*_metrics.json)
histories=("$RESULT_DIR"/*.pkl)
shopt -u nullglob
(( ${#models[@]} == 1 && ${#metrics[@]} == 1 && ${#histories[@]} == 1 )) || {
  echo "Training did not produce one model, metrics file, and history" >&2
  exit 1
}
"$ENV_ROOT/bin/python" -B - "${metrics[0]}" "$MODEL" "$SEED" <<'PY'
import json
import sys

path, model, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    metrics = json.load(handle)
assert metrics["model_type"] == model, metrics.get("model_type")
assert int(metrics["seed"]) == seed, metrics.get("seed")
assert int(metrics["actual_epochs"]) == 150, metrics.get("actual_epochs")
assert metrics["feedback_pathway"] == "additive_affine", metrics.get("feedback_pathway")
assert metrics["feedback_bias"] is True, metrics.get("feedback_bias")
assert metrics["core_output_wrap"] == "none", metrics.get("core_output_wrap")
PY

[[ ! -e "$TEST_DIR" ]] || {
  echo "Refusing to overwrite existing reset-excluded output: $TEST_DIR" >&2
  exit 1
}
"$ENV_ROOT/bin/python" -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
  --ckpt "${models[0]}" --model "$MODEL" --seed "$SEED" --output_dir "$TEST_DIR" \
  --data_dir "$DATA_ROOT" --data_suffix 40h-uint8 --sequence_length 32 \
  --batch_size 256 --num_workers 2 --device cuda

printf 'state=done task=%s model=%s seed=%s gpu=%s timestamp=%s\n' \
  "$TASK_ID" "$MODEL" "$SEED" "$GPU" "$(date -Is)" > "$STATUS_DIR/task_${TASK_ID}.status"
