#!/usr/bin/env bash
# Short Phase 1: smaller LR/WD grid (4h train, 40h val).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

python train_model.py \
  --model_types gawf \
  --hidden_sizes 256 \
  --data_suffix 4h-float32 \
  --eval_data_suffix 40h-float32 \
  --lrs 1e-4 3e-4 5e-4 \
  --wds 1e-4 1e-3 \
  --cnn_dropout 0.0 \
  --rnn_dropout 0.5 \
  --num_epochs 50 \
  --patience 8 \
  --use_acceleration \
  --use_sector_mode \
  --result_suffix gen_phase1_short_gawf_4h
