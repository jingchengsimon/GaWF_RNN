#!/usr/bin/env bash
# Run the fixed five-model IMDB 10-seed campaign as two serial SJC GPU lanes.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_DATA_DIR:-$ROOT/source/text/data}"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
GRID_UTIL="$ROOT/experiments/text/imdb_5model_best10seed.py"
RUN_TAG="imdb_5model_best10seed"
RESULT_BASE="$RESULTS/data/text/runs/$RUN_TAG"
ARTIFACT_ROOT="$RESULTS/artifacts/imdb_5model_best10seed_sjc2gpu"
STATUS_DIR="$ARTIFACT_ROOT/status"
COMPLETE_FILE="$ARTIFACT_ROOT/.complete"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "root=$ROOT"
  echo "results=$RESULTS"
  echo "data_dir=$DATA_DIR"
  echo "result_base=$RESULT_BASE"
  echo "gpu0_tasks=0,2,...,48"
  echo "gpu1_tasks=1,3,...,49"
  for task_id in 0 1 9 10 20 30 40 48 49; do
    python -B "$GRID_UTIL" emit-task --task-id "$task_id" \
      --root "$RESULTS" --format json
  done
  exit 0
fi

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

required=(
  vocab.json imdb_meta.json
  imdb_train_ids.pt imdb_train_len.pt imdb_train_label.pt
  imdb_val_ids.pt imdb_val_len.pt imdb_val_label.pt
  imdb_test_ids.pt imdb_test_len.pt imdb_test_label.pt
)
for name in "${required[@]}"; do
  [[ -s "$DATA_DIR/imdb/$name" ]] || {
    echo "Missing processed IMDB input: $DATA_DIR/imdb/$name" >&2
    exit 1
  }
done

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
(( gpu_count >= 2 )) || { echo "Expected at least two visible GPUs, found $gpu_count" >&2; exit 1; }

mkdir -p "$ARTIFACT_ROOT" "$STATUS_DIR"
exec 9>"$ARTIFACT_ROOT/runner.lock"
flock -n 9 || { echo "Another SJC IMDB two-GPU runner is active" >&2; exit 1; }
[[ ! -f "$COMPLETE_FILE" ]] || { echo "Campaign already complete: $COMPLETE_FILE"; exit 0; }

run_unit() {
  local task_id="$1" gpu="$2" result_dir log done_file fail_file running_file train_rc
  eval "$(python -B "$GRID_UTIL" emit-task --task-id "$task_id" --root "$RESULTS")"
  result_dir="$RESULT_DIR"
  log="$ARTIFACT_ROOT/task_$(printf '%04d' "$task_id")_${MODEL_TYPE}_seed$(printf '%02d' "$SEED").log"
  done_file="$STATUS_DIR/task_$(printf '%04d' "$task_id").done"
  fail_file="$STATUS_DIR/task_$(printf '%04d' "$task_id").fail"
  running_file="$STATUS_DIR/task_$(printf '%04d' "$task_id").running"

  if python -B "$GRID_UTIL" validate --task-id "$task_id" --root "$RESULTS" >/dev/null 2>&1; then
    printf 'status=skipped_existing task=%s model=%s seed=%s timestamp=%s\n' \
      "$task_id" "$MODEL_TYPE" "$SEED" "$(date -Is)" > "$done_file"
    rm -f "$fail_file" "$running_file"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "Refusing incomplete or mismatched result leaf: $result_dir" >&2
    return 1
  }

  printf 'status=running task=%s gpu=%s model=%s seed=%s timestamp=%s\n' \
    "$task_id" "$gpu" "$MODEL_TYPE" "$SEED" "$(date -Is)" > "$running_file"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -B "$ROOT/run_task.py" imdb \
    --model_types "$MODEL_TYPE" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS" \
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
    --use_acceleration 2>&1 | tee "$log"
  train_rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$train_rc" -ne 0 ]] \
    || ! python -B "$GRID_UTIL" validate --task-id "$task_id" \
      --root "$RESULTS" --json; then
    printf 'status=failed task=%s gpu=%s model=%s seed=%s exit=%s timestamp=%s\n' \
      "$task_id" "$gpu" "$MODEL_TYPE" "$SEED" "$train_rc" "$(date -Is)" > "$fail_file"
    return 1
  fi
  printf 'status=done task=%s gpu=%s model=%s seed=%s timestamp=%s\n' \
    "$task_id" "$gpu" "$MODEL_TYPE" "$SEED" "$(date -Is)" > "$done_file"
  rm -f "$fail_file" "$running_file"
}

worker() {
  local gpu="$1" task_id
  for ((task_id=gpu; task_id<50; task_id+=2)); do
    run_unit "$task_id" "$gpu"
  done
}

cd "$ROOT"
worker 0 &
pid0=$!
worker 1 &
pid1=$!
set +e
wait "$pid0"
rc0=$?
wait "$pid1"
rc1=$?
set -e
if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
  echo "SJC IMDB lane failure: gpu0=$rc0 gpu1=$rc1" >&2
  exit 1
fi

for task_id in $(seq 0 49); do
  python -B "$GRID_UTIL" validate --task-id "$task_id" --root "$RESULTS" >/dev/null
done
touch "$COMPLETE_FILE"
echo "SJC IMDB two-GPU campaign complete: $RESULT_BASE"
