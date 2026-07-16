#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-best6-s10
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/amarel/artifacts/clutter_best6_10seed_ep150/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/clutter_best6_10seed_ep150/%A_%a.err

# Run one fixed-best Clutter model/seed task for all 150 epochs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

GRID_UTIL="${AIM3_GRID_UTIL:-experiments/generalization/clutter_best6_multiseed.py}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TASK_OFFSET="${TASK_OFFSET:-0}"
if [[ ! "$TASK_OFFSET" =~ ^(0|6|12|18|24|30|36|42|48|54)$ ]]; then
  echo "TASK_OFFSET must select one seed block: 0,6,...,54." >&2
  exit 2
fi
TASK_ID="$((TASK_OFFSET + ARRAY_TASK_ID))"
if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
eval "$(python "$GRID_UTIL" emit-task --task-id "$TASK_ID" --root "$ROOT")"
mkdir -p "$(dirname "$DONE_FILE")"

write_failure() {
  local exit_code="$1"
  {
    echo "status=failed"
    echo "exit_code=$exit_code"
    echo "task_id=$TASK_ID"
    echo "unit_id=$UNIT_ID"
    echo "model=$MODEL_TYPE"
    echo "seed=$SEED"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
}
trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_failure "$rc"; fi' EXIT

DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
if [[ ! -f "$DATA_DIR/stimulus_reg-train-${DATA_SUFFIX}.npy" ]]; then
  echo "Missing Clutter training data under $DATA_DIR" >&2
  exit 2
fi
if [[ "${AIM3_NUM_WORKERS:-}" != "0" || "${AIM3_PIN_MEMORY:-}" != "0" ]]; then
  echo "mmap run requires AIM3_NUM_WORKERS=0 and AIM3_PIN_MEMORY=0." >&2
  exit 2
fi

if python "$GRID_UTIL" validate --task-id "$TASK_ID" --root "$ROOT" >/dev/null 2>&1; then
  echo "Task $TASK_ID ($UNIT_ID) already valid; skipping."
  printf 'status=skipped_existing\ntask_id=%s\nunit_id=%s\ntimestamp=%s\n' \
    "$TASK_ID" "$UNIT_ID" "$(date -Is)" > "$DONE_FILE"
  rm -f "$FAIL_FILE"
  trap - EXIT
  exit 0
fi

MODEL_WIDTH_ARGS=(--hidden_sizes "$MODEL_WIDTH")
if [[ "$WIDTH_KIND" == "mamba" ]]; then
  MODEL_WIDTH_ARGS=(--mamba_d_models "$MODEL_WIDTH")
elif [[ "$WIDTH_KIND" == "s5" ]]; then
  MODEL_WIDTH_ARGS=(--s5_d_models "$MODEL_WIDTH" --s5_state_sizes "$S5_STATE_SIZE")
fi

echo "[$(date -Is)] task=$TASK_ID unit=$UNIT_ID model=$MODEL_TYPE seed=$SEED"
echo "epochs=$NUM_EPOCHS patience=$PATIENCE data=$DATA_SUFFIX eval=$EVAL_DATA_SUFFIX"
echo "lr=$LR wd=$WD width=$MODEL_WIDTH chan_num=$CHAN_NUM mmap=true workers=0 pin_memory=false"
echo "result_suffix=$RESULT_SUFFIX"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DISABLE_TQDM=1 python train_model.py \
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

python "$GRID_UTIL" validate --task-id "$TASK_ID" --root "$ROOT" --json
{
  echo "status=done"
  echo "task_id=$TASK_ID"
  echo "unit_id=$UNIT_ID"
  echo "model=$MODEL_TYPE"
  echo "seed=$SEED"
  echo "metrics_path=$METRICS_PATH"
  echo "timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE"
trap - EXIT
