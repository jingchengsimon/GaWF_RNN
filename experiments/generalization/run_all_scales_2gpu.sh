#!/usr/bin/env bash
#
# End-to-end generalization pipeline (all scales, all models): no manual LR/WD decisions.
# Phase 1 GAWF searches run with feedback enabled from epoch 1 (num_epochs=100).
# Requires 2× GPUs. Phase 1 and Phase 2 use both cards in parallel where safe; Phase 3
# runs one scale at a time (each scale already uses both GPUs for four models).
#
# Usage (from repo root):
#   bash experiments/generalization/run_all_scales_2gpu.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "=== Phase 1: GAWF grid (4h/10h/20h only), 2 jobs in parallel ==="
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase1_gawf_search_4h.sh" &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase1_gawf_search_10h.sh" &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase1_gawf_search_20h.sh"
wait

bash "$SCRIPT_DIR/run_phase1_aggregate.sh"

echo "=== Phase 2: RNN/LSTM/GRU LR check (4 scales), 2 scales in parallel ==="
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase2_lr_check_4h.sh" &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase2_lr_check_10h.sh" &
wait
CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase2_lr_check_20h.sh" &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase2_lr_check_40h.sh" &
wait

echo "=== Phase 3: final 4 models × scale (one scale at a time,2 GPUs inside) ==="
bash "$SCRIPT_DIR/phase3_train_4h.sh"
bash "$SCRIPT_DIR/phase3_train_10h.sh"
bash "$SCRIPT_DIR/phase3_train_20h.sh"
bash "$SCRIPT_DIR/phase3_train_40h.sh"

echo "=== Plot (overfit_gap, train_acc, val_acc vs scale) ==="
python "$ROOT/plot_generalization.py"
echo "Done. Figures: results/anal_figs/generalization/overfit_gap_vs_scale.*, train_acc_vs_scale.*, val_acc_vs_scale.*"
