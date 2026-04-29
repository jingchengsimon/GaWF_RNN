#!/usr/bin/env bash
# Run a local 2-GPU 4h/5-epoch smoke test for the hparam pipeline.
#
# From repo root:
#   bash experiments/local/run_hparam_4h_5epoch_test_2gpu.sh
#
# Fixed hparams:
#   hidden_size=256, lr=5e-4, wd=1e-4, cnn_dropout=0.0, rnn_dropout=0.5
# Models:
#   rnn, lstm, gru, gawf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/experiments/local/artifacts/hparam_4h_5epoch_test"
RESULT_SUFFIX="gen_local_hparam_4h_5epoch_test"
DATA_SUFFIX="4h-float32"
EVAL_DATA_SUFFIX="40h-float32"
HIDDEN_SIZE=256
LR=0.0005
WD=0.0001
CNN_DROPOUT=0.0
RNN_DROPOUT=0.5
NUM_EPOCHS=5
PATIENCE=15
SEED=42

mkdir -p "$LOG_DIR"

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
elif [[ -d "/G/MIMOlab/Codes/aim3_RNN/stimuli" ]]; then
  DATA_DIR="/G/MIMOlab/Codes/aim3_RNN/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create $ROOT/stimuli." >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local model="$2"
  local log_prefix="$LOG_DIR/${model}_h${HIDDEN_SIZE}_lr${LR}_wd${WD}"

  echo "[$(date -Is)] start model=$model gpu=$gpu log=${log_prefix}.out"
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 python train_model.py \
    --model_types "$model" \
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
    --result_suffix "$RESULT_SUFFIX" \
    > "${log_prefix}.out" 2> "${log_prefix}.err"
  echo "[$(date -Is)] done model=$model gpu=$gpu"
}

echo "AIM3 local 2-GPU 4h/5-epoch smoke test"
echo "root=$ROOT"
echo "data_dir=$DATA_DIR"
echo "log_dir=$LOG_DIR"
echo "result_suffix=$RESULT_SUFFIX"
echo "fixed_hparams=hidden_size=$HIDDEN_SIZE,lr=$LR,wd=$WD"

run_one 0 rnn &
p1=$!
run_one 1 lstm &
p2=$!
wait "$p1" "$p2"

run_one 0 gru &
p3=$!
run_one 1 gawf &
p4=$!
wait "$p3" "$p4"

echo "All local smoke-test runs completed."
echo "Logs: $LOG_DIR"
echo "Results: $ROOT/results/train_data/$RESULT_SUFFIX"
