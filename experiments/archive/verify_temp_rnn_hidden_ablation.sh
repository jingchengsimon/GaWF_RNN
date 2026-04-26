#!/usr/bin/env bash
#
# TEMPORARY ABLATION — delete this file and verify_temp_rnn_hidden_ablation_plot.py
# when done. Not sourced by any other script.
#
# Trains plain RNN at hidden 256 and 275 on each train scale (4/10/20/40h) with
# fixed 40h val. LR/WD for RNN per scale come from
#   experiments/generalization/artifacts/phase2_final_hparams_short.json
# (same source as short Phase3 / phase2_final_hparams_short.json), NOT from per-run metrics files.
#
# Two GPUs: per scale, h256 and h275 in parallel; scales sequential.
# Then calls verify_temp_rnn_hidden_ablation_plot.py for gap/train/val figures
# (y-limits vs Phase3 summary CSVs).
#
# Usage (repo root):
#   bash experiments/generalization/verify_temp_rnn_hidden_ablation.sh
#
# Optional env:
#   NUM_EPOCHS (default 50)
#   HPARAMS_JSON  (default: experiments/generalization/artifacts/phase2_final_hparams_short.json)
#   REF_CSV_TAG   (default: _short_ep${NUM_EPOCHS})
#   VERIFY_SCALES  space-separated train scales to run (default: 4h 10h 20h 40h).
#                  Example: only redo 4h after overwrite — VERIFY_SCALES=4h bash ...
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

NUM_EPOCHS="${NUM_EPOCHS:-50}"
REF_CSV_TAG="${REF_CSV_TAG:-_short_ep${NUM_EPOCHS}}"
HPARAMS_JSON="${HPARAMS_JSON:-"$SCRIPT_DIR/artifacts/phase2_final_hparams_short.json"}"
if [[ ! -f "$HPARAMS_JSON" ]]; then
  echo "ERROR: missing hparams file: $HPARAMS_JSON" >&2
  exit 1
fi

SCRATCH_DATA="/scratch/${USER}/stimuli"
if [[ -d "$SCRATCH_DATA" ]]; then
  DATA_DIR="$SCRATCH_DATA"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Checked: $SCRATCH_DATA and $ROOT/stimuli" >&2
  exit 1
fi
VERIFY_SCALES="${VERIFY_SCALES:-4h 10h 20h 40h}"
echo "Using data_dir: $DATA_DIR  num_epochs: $NUM_EPOCHS  hparams: $HPARAMS_JSON  ref_csv_tag: $REF_CSV_TAG  verify_scales: $VERIFY_SCALES"

is_verify_scale() {
  local sk="$1"
  local s
  for s in $VERIFY_SCALES; do
    if [[ "$s" == "$sk" ]]; then
      return 0
    fi
  done
  return 1
}

read_rnn_lr_wd() {
  local SK="$1"
  python3 - "$HPARAMS_JSON" "$SK" <<'PY'
import json, sys
p, sk = sys.argv[1], sys.argv[2]
with open(p, "r", encoding="utf-8") as f:
    h = json.load(f)
r = h[sk]["rnn"]
print(float(r["lr"]))
print(float(r["weight_decay"]))
PY
}

run_one() {
  local SK="$1"
  local DS="$2"
  local HS="$3"
  local GPU="$4"
  local LR WD
  mapfile -t _lw < <(read_rnn_lr_wd "$SK")
  LR="${_lw[0]}"
  WD="${_lw[1]}"
  echo "  scale=$SK hidden=$HS lr=$LR wd=$WD (from phase2_final_hparams_short.json rnn)"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU python train_model.py \
    --data_suffix "$DS" \
    --eval_data_suffix 40h-float32 \
    --data_dir "$DATA_DIR" \
    --model_types rnn \
    --hidden_sizes "$HS" \
    --lrs "$LR" \
    --wds "$WD" \
    --cnn_dropout 0.0 \
    --rnn_dropout 0.5 \
    --num_epochs "$NUM_EPOCHS" \
    --patience 8 \
    --use_acceleration \
    --use_sector_mode \
    --result_suffix "verify_rnn_h${HS}_${SK}_ep${NUM_EPOCHS}"
}

for row in "4h 4h-float32" "10h 10h-float32" "20h 20h-float32" "40h 40h-float32"; do
  set -- $row
  SK="$1"
  DS="$2"
  if ! is_verify_scale "$SK"; then
    echo "=== Skip $SK (not in VERIFY_SCALES) ==="
    continue
  fi
  echo "=== Scale $SK ($DS) ==="
  run_one "$SK" "$DS" 256 0 &
  run_one "$SK" "$DS" 275 1 &
  wait
done

echo "=== Plotting (y-lims vs Phase3 ref tag $REF_CSV_TAG) ==="
python "$SCRIPT_DIR/verify_temp_rnn_hidden_ablation_plot.py" --epoch "$NUM_EPOCHS" \
  --ref_csv_tag "$REF_CSV_TAG" \
  --write_csv "experiments/generalization/artifacts/phase3_summary_verify_rnn_h256_h275_ep${NUM_EPOCHS}.csv"
echo "Done. Figures: results/anal_figs/generalization/overfit_gap_verify_rnn_h256_h275_ep${NUM_EPOCHS}.png (and train/val)."
