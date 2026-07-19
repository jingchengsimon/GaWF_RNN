#!/usr/bin/env bash
# Phase 2 (full pipeline only) — RNN/LSTM/GRU LR sweep at Phase-1 WD for one train scale
# (requires experiments/generalization/artifacts/phase1_best.json; full pipeline builds it after Phase1).
#
# Usage (repo root):
#   bash experiments/generalization/phase2_lr_check.sh 4h
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SK="${1:-}"
if [[ ! "$SK" =~ ^(4h|10h|20h|40h)$ ]]; then
  echo "Usage: $0 <4h|10h|20h|40h>" >&2
  exit 1
fi

case "$SK" in
  4h)  DS=4h-float32 ;;
  10h) DS=10h-float32 ;;
  20h) DS=20h-float32 ;;
  40h) DS=40h-uint8 ;;
esac

SCRATCH_DATA="/scratch/${USER}/stimuli"
if [[ -d "$SCRATCH_DATA" ]]; then
  DATA_DIR="$SCRATCH_DATA"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
else
  echo "Data directory not found. Checked: $SCRATCH_DATA and $ROOT/stimuli" >&2
  exit 1
fi

HP="$SCRIPT_DIR/artifacts/phase1_best.json"
WD=$(python -c "import json; print(json.load(open('$HP'))['$SK']['weight_decay'])")
SUFFIX="gen_phase2_${SK}"
echo "[Phase2 LR check $SK] data_dir=$DATA_DIR WD=$WD result_suffix=$SUFFIX"

BASE=(--data_suffix "$DS" --eval_data_suffix 40h-uint8 --data_dir "$DATA_DIR"
  --lrs 1e-4 3e-4 5e-4 1e-3 --wds "$WD"
  --cnn_dropout 0.0 --rnn_dropout 0.5 --num_epochs 100 --patience 15
  --use_acceleration --use_sector_mode --result_suffix "$SUFFIX")

python train_model.py --model_types rnn --hidden_sizes 275 "${BASE[@]}"
python train_model.py --model_types lstm --hidden_sizes 80 "${BASE[@]}"
python train_model.py --model_types gru --hidden_sizes 105 "${BASE[@]}"

python experiments/generalization/collect_results.py phase2 \
  --scale "$SK" --metrics_dir "results/train_data/$SUFFIX"
