#!/usr/bin/env bash
#SBATCH --job-name=aim3-sentihood-lstm
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/amarel/artifacts/sentihood_lstm_final/%A.out
#SBATCH --error=experiments/amarel/artifacts/sentihood_lstm_final/%A.err

# Run the first SentiHood LSTM-Final reproducibility job. Submit via
# experiments/amarel/submit_sentihood_lstm_final.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_sentihood.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

RUN_TAG="${RUN_TAG:-sentihood_lstm_final}"
ART_ROOT="$ROOT/experiments/amarel/artifacts/$RUN_TAG"
STATUS_DIR="$ROOT/experiments/text/artifacts/$RUN_TAG/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

DONE_FILE="$STATUS_DIR/lstm_final.done"
FAIL_FILE="$STATUS_DIR/lstm_final.fail"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must be exported at submission}"

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "/scratch/${USER}/stimuli/sentihood" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
elif [[ -d "/cache/${USER}/stimuli/sentihood" ]]; then
  DATA_DIR="/cache/${USER}/stimuli"
elif [[ -d "$ROOT/stimuli/sentihood" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "SentiHood data not found. Run source/text/prepare_sentihood_data.py first." | tee "$FAIL_FILE"
  exit 2
fi

MODEL_TYPE="${MODEL_TYPE:-lstm}"
HIDDEN="${HIDDEN:-50}"
EMBED_DIM="${EMBED_DIM:-50}"
LR="${LR:-0.01}"
WD="${WD:-0.0}"
EMBED_DROPOUT="${EMBED_DROPOUT:-0.001}"
RNN_DROPOUT="${RNN_DROPOUT:-0.001}"
POOLING="${POOLING:-last}"
OPTIM="${OPTIM:-adam}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-150}"
RESULT_SUFFIX="${RESULT_SUFFIX:-$RUN_TAG}"
SELECTION_METRIC="${SELECTION_METRIC:-aspect_f1}"
BALANCE_TRAIN_LABELS="${BALANCE_TRAIN_LABELS:-1}"

RESULT_STEM="${MODEL_TYPE}_sentihood_h${HIDDEN}_emb${EMBED_DIM}"
RESULT_STEM+="_lr${LR}_wd${WD}_edo${EMBED_DROPOUT}_rdo${RNN_DROPOUT}"
METRICS_PATH="$AIM3_RESULTS_PATH/train_data/$RESULT_SUFFIX/${RESULT_STEM}_metrics.json"
PKL_PATH="$AIM3_RESULTS_PATH/train_data/$RESULT_SUFFIX/${RESULT_STEM}.pkl"
MODEL_PATH="$AIM3_RESULTS_PATH/train_data/$RESULT_SUFFIX/${RESULT_STEM}_model.pth"

echo "[$(date -Is)] SentiHood LSTM-Final"
echo "root=$ROOT"
echo "data_dir=$DATA_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "metrics_path=$METRICS_PATH"
echo "resources=partition=gpu-redhat gres=gpu:1 constraint=adalovelace cpus=16 mem=64G"
echo "env=AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
echo "balance_train_labels=$BALANCE_TRAIN_LABELS"

if [[ -f "$METRICS_PATH" && -f "$PKL_PATH" && -f "$MODEL_PATH" ]]; then
  echo "Existing complete output found; skipping."
  {
    echo "status=skipped_existing"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
  exit 0
fi

balance_args=()
case "$BALANCE_TRAIN_LABELS" in
  1|true|TRUE|True|yes|YES|Yes|y|Y)
    balance_args+=(--balance_train_labels)
    ;;
esac

set +e
DISABLE_TQDM=1 python train_sentihood.py \
  --model_types "$MODEL_TYPE" \
  --data_dir "$DATA_DIR" \
  --results_dir "$AIM3_RESULTS_PATH" \
  --result_suffix "$RESULT_SUFFIX" \
  --embed_dim "$EMBED_DIM" \
  --hidden_sizes "$HIDDEN" \
  --lrs "$LR" \
  --wds "$WD" \
  --embed_dropout "$EMBED_DROPOUT" \
  --rnn_dropout "$RNN_DROPOUT" \
  --pooling "$POOLING" \
  --optim "$OPTIM" \
  --num_epochs "$NUM_EPOCHS" \
  --patience "$PATIENCE" \
  --selection_metric "$SELECTION_METRIC" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$AIM3_NUM_WORKERS" \
  --device cuda \
  --use_acceleration \
  "${balance_args[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed"
    echo "exit_code=$train_rc"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

if [[ -f "$METRICS_PATH" && -f "$PKL_PATH" && -f "$MODEL_PATH" ]]; then
  python - "$METRICS_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    m = json.load(f)
print("test_aspect_f1_at_best=", m.get("test_aspect_f1_at_best"))
print("test_sentiment_acc_at_best=", m.get("test_sentiment_acc_at_best"))
print("test_aspect_auc_at_best=", m.get("test_aspect_auc_at_best"))
print("test_sentiment_auc_at_best=", m.get("test_sentiment_auc_at_best"))
PY
  {
    echo "status=done"
    echo "metrics_path=$METRICS_PATH"
    echo "timestamp=$(date -Is)"
  } > "$DONE_FILE"
  rm -f "$FAIL_FILE"
else
  {
    echo "status=validation_failed"
    echo "metrics_path=$METRICS_PATH"
    echo "pkl_path=$PKL_PATH"
    echo "model_path=$MODEL_PATH"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit 1
fi
