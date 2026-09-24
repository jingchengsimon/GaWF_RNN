#!/usr/bin/env bash
# Compute-node launcher for the CM-MNIST nonlinearity-placement ablation behaviour analysis.
#
# For one model/seed unit it (a) collects the reset-excluded test accuracy when its collect JSON is
# missing and (b) runs the target-switch recovery inference on the joint-switch balanced 10-digit
# test movie. Training is never re-run: a unit without a final checkpoint is skipped, so the same
# script is safe to re-submit while the training array is still finishing units.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODEL=""
SEED=""
while (( $# )); do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" && -n "$SEED" ]] || { echo "--model and --seed are required" >&2; exit 2; }

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

case "$MODEL" in
  *_notanh) BATCH_LEAF="clutter_ablate_inner_activation_40h_ep150_v1" ;;
  *) BATCH_LEAF="clutter_ablate_outer_norm_40h_ep150_v1" ;;
esac
printf -v SEED2 '%02d' "$SEED"
RUN_DIR="$AIM3_RESULTS_PATH/data/clutter/runs/$BATCH_LEAF/$MODEL-seed$SEED2"
TEST_BASE="$AIM3_RESULTS_PATH/data/analysis/clutter_ablation_reset_excluded_test_10seed_v1"
RECOVERY_BASE="$AIM3_RESULTS_PATH/data/analysis/ablation_behavior_recovery_jointswitch10digit_v1"

shopt -s nullglob
CHECKPOINTS=("$RUN_DIR"/*_model.pth)
shopt -u nullglob
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "unit-not-finished: $MODEL-seed$SEED2 has no final checkpoint yet; nothing to do" >&2
  exit 0
fi
(( ${#CHECKPOINTS[@]} == 1 )) || { echo "Expected one checkpoint in $RUN_DIR" >&2; exit 1; }
CHECKPOINT="${CHECKPOINTS[0]}"

source "${AIM3_CONDA_INIT:-/home/js3269/enter/etc/profile.d/conda.sh}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

if [[ "${AIM3_SKIP_COLLECT:-0}" != "1" && ! -f "$TEST_BASE/$MODEL-seed$SEED2/reset_excluded_test_accuracy.json" ]]; then
  python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
    --ckpt "$CHECKPOINT" --model "$MODEL" --seed "$SEED" \
    --output_dir "$TEST_BASE/$MODEL-seed$SEED2" \
    --data_dir "$AIM3_CLUTTER_DATA_DIR" --data_suffix 40h-uint8 --sequence_length 32
fi

if [[ -z "$(ls -A "$RECOVERY_BASE/$MODEL-seed$SEED2" 2>/dev/null || true)" ]]; then
  python -B -m utils.analysis.clutter.fig1_target_switch_recovery \
    --ckpts "$CHECKPOINT" --save_dir "$RECOVERY_BASE/$MODEL-seed$SEED2" \
    --data_dir "$AIM3_CLUTTER_DATA_DIR" \
    --data_suffix 40h-float32-jointswitch-balanced-10digit-unique \
    --window_radius 10 --batch_size 64 --device cuda --seed "$SEED" \
    --exclude_window_initial_frame
fi
echo "unit-complete: $MODEL-seed$SEED2"
