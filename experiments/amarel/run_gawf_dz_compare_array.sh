#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-dz
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpu018,gpu043
#SBATCH --time=72:00:00
#SBATCH --output=experiments/amarel/artifacts/gawf_dz_compare/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/gawf_dz_compare/%A_%a.err

# Run one GaWF dz-comparison condition on Amarel.
# Submit via experiments/amarel/submit_gawf_dz_compare.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

SCALE="${SCALE:-40h}"
DATA_SUFFIX="${DATA_SUFFIX:-${SCALE}-float32}"
EVAL_DATA_SUFFIX="${EVAL_DATA_SUFFIX:-40h-float32}"
RESULT_SUFFIX="${RESULT_SUFFIX:-gawf_dz_compare_${SCALE}_fullfb}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LR="${LR:-0.005}"
WD="${WD:-0.001}"
CNN_DROPOUT="${CNN_DROPOUT:-0.0}"
RNN_DROPOUT="${RNN_DROPOUT:-0.5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-42}"

CONDITION_LIST="${CONDITION_LIST:-legacy dz8 dz16 dz32 dz64}"
read -r -a CONDITIONS <<< "$CONDITION_LIST"
TASK_ID="$(( ${SLURM_ARRAY_TASK_ID:-0} + ${TASK_OFFSET:-0} ))"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#CONDITIONS[@]}" ]]; then
  echo "TASK_ID must be in [0, $((${#CONDITIONS[@]} - 1))], got $TASK_ID" >&2
  exit 2
fi
CONDITION="${CONDITIONS[$TASK_ID]}"

mkdir -p "$ROOT/experiments/amarel/artifacts/gawf_dz_compare"
mkdir -p "$ROOT/experiments/generalization/artifacts/gawf_dz_compare/status"

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create stimuli under scratch/cache/repo." >&2
  exit 2
fi

dz_suffix=""
dz_args=()
if [[ "$CONDITION" != "legacy" ]]; then
  DZ="${CONDITION#dz}"
  dz_suffix="_dz${DZ}"
  dz_args=(--dz "$DZ")
fi

RESULT_DIR_SUFFIX="${RESULT_SUFFIX}/${CONDITION}"
RESULT_STEM="gawf_sector_acc_h${HIDDEN_SIZE}_lr${LR}_wd${WD}_cdo${CNN_DROPOUT}_rdo${RNN_DROPOUT}${dz_suffix}"
METRICS_PATH="$ROOT/results/train_data/${RESULT_DIR_SUFFIX}/${RESULT_STEM}_metrics.json"
STATUS_DIR="$ROOT/experiments/generalization/artifacts/gawf_dz_compare/status"
DONE_FILE="$STATUS_DIR/${CONDITION}.done"
FAIL_FILE="$STATUS_DIR/${CONDITION}.fail"

echo "[$(date -Is)] condition=$CONDITION scale=$SCALE h=$HIDDEN_SIZE lr=$LR wd=$WD"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_DIR_SUFFIX"
echo "metrics_path=$METRICS_PATH"
echo "feedback=full"

if [[ -f "$METRICS_PATH" ]]; then
  echo "Condition $CONDITION already has metrics; skipping."
  {
    echo "status=skipped_existing"
    echo "condition=$CONDITION"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
  exit 0
fi

set +e
DISABLE_TQDM=1 python train_model.py \
  --model_types gawf \
  --hidden_sizes "$HIDDEN_SIZE" \
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
  --use_acceleration \
  --use_sector_mode \
  --result_suffix "$RESULT_DIR_SUFFIX" \
  "${dz_args[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed"
    echo "condition=$CONDITION"
    echo "exit_code=$train_rc"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if [[ -f "$METRICS_PATH" ]]; then
  {
    echo "status=done"
    echo "condition=$CONDITION"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
else
  {
    echo "status=missing_metrics"
    echo "condition=$CONDITION"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
