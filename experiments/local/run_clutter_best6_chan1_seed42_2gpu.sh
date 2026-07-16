#!/usr/bin/env bash
# Train the chan=1 frozen-best six-model protocol at seed42 on two local GPUs.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
GRID_UTIL="$ROOT/experiments/generalization/clutter_best6_chan1_seed42.py"
CONDA_INIT="${AIM3_CONDA_INIT:-/G/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${AIM3_CONDA_ENV:-aim3_rnn}"
DATA_DIR="${AIM3_DATA_DIR:-/G/MIMOlab/Codes/aim3_RNN/stimuli}"
RUN_ID="${AIM3_RUN_ID:-aim3-clutter-c1-s42-ep150}"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/clutter_best6_chan1_seed42_ep150"
LOG_PATH="$ARTIFACT_DIR/$RUN_ID.log"

mkdir -p "$ARTIFACT_DIR/status"
exec > >(tee -a "$LOG_PATH") 2>&1

if [[ ! -f "$CONDA_INIT" ]]; then
  echo "Missing Conda initialization script: $CONDA_INIT" >&2
  exit 2
fi
set +u
source "$CONDA_INIT"
conda activate "$CONDA_ENV"
set -u
cd "$ROOT"

for suffix in train-40h-float32 validation-40h-float32; do
  if [[ ! -f "$DATA_DIR/stimulus_reg-${suffix}.npy" ]]; then
    echo "Missing stimulus: $DATA_DIR/stimulus_reg-${suffix}.npy" >&2
    exit 2
  fi
done

run_one() {
  local task_id="$1"
  local gpu="$2"
  local rc

  eval "$(python "$GRID_UTIL" emit-task --task-id "$task_id" --root "$ROOT")"
  mkdir -p "$(dirname "$DONE_FILE")"
  echo "[$(date -Is)] gpu=$gpu task=$TASK_ID unit=$UNIT_ID phase=train"

  if ! python "$GRID_UTIL" validate --task-id "$task_id" --root "$ROOT" >/dev/null 2>&1; then
    MODEL_WIDTH_ARGS=(--hidden_sizes "$MODEL_WIDTH")
    if [[ "$WIDTH_KIND" == "mamba" ]]; then
      MODEL_WIDTH_ARGS=(--mamba_d_models "$MODEL_WIDTH")
    elif [[ "$WIDTH_KIND" == "s5" ]]; then
      MODEL_WIDTH_ARGS=(--s5_d_models "$MODEL_WIDTH" --s5_state_sizes "$S5_STATE_SIZE")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      AIM3_NUM_WORKERS=0 AIM3_PIN_MEMORY=0 python train_model.py \
      --model_types "$MODEL_TYPE" \
      "${MODEL_WIDTH_ARGS[@]}" \
      --num_layers 1 \
      --chan_num "$CHAN_NUM" \
      --data_suffix "$DATA_SUFFIX" \
      --eval_data_suffix "$EVAL_DATA_SUFFIX" \
      --data_dir "$DATA_DIR" \
      --lrs "$LR" \
      --wds "$WD" \
      --cnn_dropout "$CNN_DROPOUT" \
      --rnn_dropout "$RNN_DROPOUT" \
      --num_epochs "$NUM_EPOCHS" \
      --patience "$PATIENCE" \
      --seed "$SEED" \
      --s5_ssm_lr_scale "$S5_SSM_LR_SCALE" \
      --use_acceleration \
      --use_sector_mode \
      --use_mmap \
      --result_suffix "$RESULT_SUFFIX"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      printf 'status=failed\nphase=train\nexit_code=%s\ntask_id=%s\nunit_id=%s\ntimestamp=%s\n' \
        "$rc" "$task_id" "$UNIT_ID" "$(date -Is)" > "$FAIL_FILE"
      return "$rc"
    fi
  fi

  if ! python "$GRID_UTIL" validate --task-id "$task_id" --root "$ROOT" >/dev/null; then
    printf 'status=failed\nphase=validate\ntask_id=%s\nunit_id=%s\ntimestamp=%s\n' \
      "$task_id" "$UNIT_ID" "$(date -Is)" > "$FAIL_FILE"
    return 1
  fi

  {
    echo "status=done"
    echo "task_id=$task_id"
    echo "unit_id=$UNIT_ID"
    echo "model=$MODEL_TYPE"
    echo "seed=$SEED"
    echo "checkpoint=$CHECKPOINT_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
  echo "[$(date -Is)] gpu=$gpu task=$TASK_ID unit=$UNIT_ID phase=done"
}

run_worker() {
  local worker="$1"
  local gpu="$2"
  local task_id
  for ((task_id = worker; task_id < 6; task_id += 2)); do
    if ! run_one "$task_id" "$gpu"; then
      echo "[$(date -Is)] worker=$worker gpu=$gpu task=$task_id failed; continuing"
    fi
  done
}

run_worker 0 0 &
worker0_pid=$!
run_worker 1 1 &
worker1_pid=$!
wait "$worker0_pid"
worker0_rc=$?
wait "$worker1_pid"
worker1_rc=$?

python "$GRID_UTIL" status --root "$ROOT"
status_rc=$?
echo "[$(date -Is)] workers_done rc0=$worker0_rc rc1=$worker1_rc status_rc=$status_rc"
exit "$status_rc"
