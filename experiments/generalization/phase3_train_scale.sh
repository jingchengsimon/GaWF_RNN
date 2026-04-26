#!/usr/bin/env bash
# Unified Phase 3 for one train scale (4h/10h/20h/40h). Two profiles:
#   full  — phase2_final_hparams.json (after Phase 2 in full pipeline), patience 15,
#           all models share one results folder: gen_phase3_<scale>_ep<N>
#   short — phase2_final_hparams_short.json, patience 8,
#           all models share: gen_phase3_short_<scale>_ep<N>
# Both: --data_dir scratch/stimuli or repo stimuli; collect_results --out_tag TAG.
#
# Usage (repo root):
#   bash experiments/generalization/phase3_train_scale.sh 4h full
#   bash experiments/generalization/phase3_train_scale.sh 4h short
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SK="${1:-}"
PROFILE="${2:-}"
if [[ -z "$SK" || -z "$PROFILE" ]]; then
  echo "Usage: $0 <4h|10h|20h|40h> <full|short>" >&2
  exit 1
fi

case "$SK" in
  4h) DS=4h-float32 ;;
  10h) DS=10h-float32 ;;
  20h) DS=20h-float32 ;;
  40h) DS=40h-float32 ;;
  *) echo "Invalid scale: $SK" >&2; exit 1 ;;
esac

case "$PROFILE" in
  full) ;;
  short) ;;
  *) echo "Profile must be full or short, got: $PROFILE" >&2; exit 1 ;;
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

if [[ "$PROFILE" == short ]]; then
  : "${NUM_EPOCHS:=100}"
  : "${CSV_TAG:=_short}"
  TAG="${CSV_TAG}_ep${NUM_EPOCHS}"
  H="$SCRIPT_DIR/artifacts/phase2_final_hparams_short.json"
  PAT=8
  RS_PREFIX="gen_phase3_short_${SK}"
  echo "[Phase3 $SK profile=short] data_dir=$DATA_DIR NUM_EPOCHS=$NUM_EPOCHS TAG=$TAG"
else
  : "${NUM_EPOCHS:=100}"
  : "${CSV_TAG:=}"
  TAG="${CSV_TAG}_ep${NUM_EPOCHS}"
  H="$SCRIPT_DIR/artifacts/phase2_final_hparams.json"
  PAT=15
  RS_PREFIX="gen_phase3_${SK}"
  echo "[Phase3 $SK profile=full] data_dir=$DATA_DIR NUM_EPOCHS=$NUM_EPOCHS TAG=$TAG"
fi

RNN_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['rnn']['lr'])")
RNN_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['rnn']['weight_decay'])")
LSTM_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['lstm']['lr'])")
LSTM_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['lstm']['weight_decay'])")
GRU_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gru']['lr'])")
GRU_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gru']['weight_decay'])")
GAWF_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gawf']['lr'])")
GAWF_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gawf']['weight_decay'])")

EP_SUFFIX="_ep${NUM_EPOCHS}"
# One shared train_data (and train_figs, if any consumer uses result_suffix) dir per scale+epoch; model type is in file stems.
RS_COMBINED="${RS_PREFIX}${EP_SUFFIX}"

COMMON=(--data_suffix "$DS" --eval_data_suffix 40h-float32 --cnn_dropout 0.0 --rnn_dropout 0.5
  --num_epochs "$NUM_EPOCHS" --patience "$PAT" --use_acceleration --use_sector_mode --data_dir "$DATA_DIR"
  --result_suffix "$RS_COMBINED")

CUDA_VISIBLE_DEVICES=0 python train_model.py "${COMMON[@]}" \
  --model_types rnn --hidden_sizes 275 \
  --lrs "$RNN_LR" --wds "$RNN_WD" &
CUDA_VISIBLE_DEVICES=1 python train_model.py "${COMMON[@]}" \
  --model_types lstm --hidden_sizes 80 \
  --lrs "$LSTM_LR" --wds "$LSTM_WD" &
wait

CUDA_VISIBLE_DEVICES=0 python train_model.py "${COMMON[@]}" \
  --model_types gru --hidden_sizes 105 \
  --lrs "$GRU_LR" --wds "$GRU_WD" &
CUDA_VISIBLE_DEVICES=1 python train_model.py "${COMMON[@]}" \
  --model_types gawf --hidden_sizes 256 \
  --lrs "$GAWF_LR" --wds "$GAWF_WD" &
wait

python experiments/generalization/collect_results.py phase3 --scale "$SK" --out_tag "$TAG" \
  "results/train_data/${RS_COMBINED}"

echo "Wrote experiments/generalization/artifacts/phase3_summary_${SK}${TAG}.csv"
