#!/usr/bin/env bash
# Phase 1 (GAWF LR×WD grid) — train 4h, validate on 40h.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

python train_model.py \
  --model_types gawf \
  --hidden_sizes 256 \
  --data_suffix 4h-float32 \
  --eval_data_suffix 40h-float32 \
  --lrs 1e-4 3e-4 5e-4 1e-3 \
  --wds 1e-4 1e-3 \
  --cnn_dropout 0.0 \
  --rnn_dropout 0.5 \
  --num_epochs 100 \
  --patience 15 \
  --use_acceleration \
  --use_sector_mode \
  --result_suffix gen_phase1_gawf_4h
