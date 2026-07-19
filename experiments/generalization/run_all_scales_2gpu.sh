#!/usr/bin/env bash
#
# End-to-end generalization: Phase 1 → (Phase 2 only if full) → Phase 3 → plot.
# First argument selects pipeline: **short** (default) or **full**.
#
# - **short** — smaller Phase 1 grid + 4 scales (4/10/20/40h GAWF); no Phase 2; uses
#   phase2_final_hparams_short.json; short Phase3/CSV tag is ${CSV_TAG}_ep${NUM_EPOCHS} (default _short_ep${NUM_EPOCHS}).
# - **full** — larger Phase 1 + 4 scales; Phase 2 LR sanity; phase2_final_hparams.json;
#   plot --csv_tag _ep${NUM_EPOCHS}.
#
# Usage (from repo root):
#   bash experiments/generalization/run_all_scales_2gpu.sh
#   bash experiments/generalization/run_all_scales_2gpu.sh short
#   bash experiments/generalization/run_all_scales_2gpu.sh full
#   NUM_EPOCHS=50 bash ...   # override (Phase 3 epoch count / tags)
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PIPELINE="${1:-short}"
if [[ ! "$PIPELINE" =~ ^(short|full)$ ]]; then
  echo "Usage: $0 [short|full]   (default: short)" >&2
  exit 1
fi

P1_MODE="$PIPELINE"

phase1_two_by_two() {
  echo "=== Phase 1 ($P1_MODE): GAWF grid 4h/10h/20h/40h (2+2 on GPUs) ==="
  CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase1_gawf_search.sh" 4h "$P1_MODE" & _p1=$!
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase1_gawf_search.sh" 10h "$P1_MODE" & _p2=$!
  wait "$_p1" "$_p2"
  CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase1_gawf_search.sh" 20h "$P1_MODE" & _p3=$!
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase1_gawf_search.sh" 40h "$P1_MODE" & _p4=$!
  wait "$_p3" "$_p4"
}

if [[ "$PIPELINE" == full ]]; then
  export NUM_EPOCHS="${NUM_EPOCHS:-100}"

  phase1_two_by_two

  echo "=== Phase 1 aggregate -> artifacts/phase1_best.json ==="
  python "$ROOT/experiments/generalization/collect_results.py" phase1 \
    "results/train_data/gen_phase1_gawf_4h" \
    "results/train_data/gen_phase1_gawf_10h" \
    "results/train_data/gen_phase1_gawf_20h" \
    "results/train_data/gen_phase1_gawf_40h"

  echo "=== Phase 2: RNN/LSTM/GRU LR check (4 scales), 2 scales in parallel ==="
  CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase2_lr_check.sh" 4h &
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase2_lr_check.sh" 10h &
  wait
  CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/phase2_lr_check.sh" 20h &
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/phase2_lr_check.sh" 40h &
  wait

  echo "=== Phase 3 (full): 4 models × 4h,10h,20h,40h (phase3_train_scale.sh … full) ==="
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 4h full
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 10h full
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 20h full
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 40h full

  echo "=== Plot (overfit_gap, train_acc, val_acc vs scale) ==="
  python "$ROOT/utils_viz/plot_generalization.py" --csv_tag "_ep${NUM_EPOCHS}"
  echo "Done. Figures: results/anal_index/G_behaviour/plot_generalization/figs/overfit_gap_vs_scale_ep${NUM_EPOCHS}*.png, ... (add --save-pdf for PDF too); csv_tag=_ep${NUM_EPOCHS}"
else
  export NUM_EPOCHS="${NUM_EPOCHS:-100}"
  export CSV_TAG="${CSV_TAG:-_short}"

  phase1_two_by_two

  echo "=== Phase 1 aggregate -> phase1_best_short.json + phase2_final_hparams_short.json ==="
  python "$ROOT/experiments/generalization/collect_results.py" phase1 \
    "results/train_data/gen_phase1_short_gawf_4h" \
    "results/train_data/gen_phase1_short_gawf_10h" \
    "results/train_data/gen_phase1_short_gawf_20h" \
    "results/train_data/gen_phase1_short_gawf_40h" \
    --out "$ROOT/experiments/generalization/artifacts/phase1_best_short.json"
  python "$ROOT/experiments/generalization/collect_results.py" emit_hparams_shared \
    --phase1_best "$ROOT/experiments/generalization/artifacts/phase1_best_short.json" \
    --out "$ROOT/experiments/generalization/artifacts/phase2_final_hparams_short.json"

  echo "=== Phase 3 (short): 4 models × 4h,10h,20h,40h (phase3_train_scale.sh … short) ==="
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 4h short
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 10h short
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 20h short
  bash "$SCRIPT_DIR/phase3_train_scale.sh" 40h short

  echo "=== Plot (gap + train/val acc) ==="
  TAG="${CSV_TAG}_ep${NUM_EPOCHS}"
  python "$ROOT/utils_viz/plot_generalization.py" --csv_tag "$TAG"
  echo "Done. Figures: results/anal_index/G_behaviour/plot_generalization/figs/*${TAG#_}.* (gap, train_acc, val_acc)."
fi
