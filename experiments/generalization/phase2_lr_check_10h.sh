#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
HP="$SCRIPT_DIR/artifacts/phase1_best.json"
WD=$(python -c "import json; print(json.load(open('$HP'))['10h']['weight_decay'])")

SUFFIX=gen_phase2_10h
DS=10h-float32

python train_model.py \
  --model_types rnn --hidden_sizes 275 \
  --data_suffix "$DS" --eval_data_suffix 40h-float32 \
  --lrs 1e-4 3e-4 5e-4 1e-3 --wds "$WD" \
  --cnn_dropout 0.0 --rnn_dropout 0.5 --num_epochs 100 --patience 15 \
  --use_acceleration --use_sector_mode \
  --result_suffix "$SUFFIX"

python train_model.py \
  --model_types lstm --hidden_sizes 80 \
  --data_suffix "$DS" --eval_data_suffix 40h-float32 \
  --lrs 1e-4 3e-4 5e-4 1e-3 --wds "$WD" \
  --cnn_dropout 0.0 --rnn_dropout 0.5 --num_epochs 100 --patience 15 \
  --use_acceleration --use_sector_mode \
  --result_suffix "$SUFFIX"

python train_model.py \
  --model_types gru --hidden_sizes 105 \
  --data_suffix "$DS" --eval_data_suffix 40h-float32 \
  --lrs 1e-4 3e-4 5e-4 1e-3 --wds "$WD" \
  --cnn_dropout 0.0 --rnn_dropout 0.5 --num_epochs 100 --patience 15 \
  --use_acceleration --use_sector_mode \
  --result_suffix "$SUFFIX"

python experiments/generalization/collect_results.py phase2 \
  --scale 10h --metrics_dir "results/train_data/$SUFFIX"
