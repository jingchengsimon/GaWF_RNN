#!/usr/bin/env bash
#
# Short pipeline: no Phase 2; Phase 1 only 4h/10h/20h (6-point GAWF grid, fb from epoch 1);
# 40h uses
# results/train_data/sector_40h_adamw for lr/wd (GAWF) + Phase3 metrics import (no retrain).
# Phase 1 & Phase 3 short training: num_epochs=50, patience=8 (see *_short.sh).
#
# Usage (repo root):
#   bash experiments/generalization/run_all_scales_2gpu_short.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "=== Phase 1 short: GAWF grid (4h/10h/20h), dynamic 20h on first free GPU ==="
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase1_gawf_search_4h_short.sh" &
pid_4h=$!f04
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase1_gawf_search_10h_short.sh" &
pid_10h=$!

wait -n "$pid_4h" "$pid_10h"
if ! kill -0 "$pid_4h" 2>/dev/null; then
  gpu_for_20h=0
  remaining_pid="$pid_10h"
else
  gpu_for_20h=1
  remaining_pid="$pid_4h"
fi

CUDA_VISIBLE_DEVICES="$gpu_for_20h" bash "$SCRIPT_DIR/phase1_gawf_search_20h_short.sh" &
pid_20h=$!
wait "$remaining_pid"
wait "$pid_20h"

bash "$SCRIPT_DIR/run_phase1_aggregate_short.sh"

echo "=== Phase 3 short: 4 models × (4h,10h,20h); 40h from preset folder ==="
bash "$SCRIPT_DIR/phase3_train_4h_short.sh"
bash "$SCRIPT_DIR/phase3_train_10h_short.sh"
bash "$SCRIPT_DIR/phase3_train_20h_short.sh"
bash "$SCRIPT_DIR/import_phase3_40h_short.sh"

echo "=== Plot (gap + train/val acc) ==="
python "$ROOT/plot_generalization.py" --csv_tag _short
echo "Done. Figures: results/anal_figs/generalization/*_short.* (gap, train_acc, val_acc)."
