#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
H="$SCRIPT_DIR/artifacts/phase2_final_hparams_short.json"
SK=20h
DS=20h-float32
TAG=_short

RNN_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['rnn']['lr'])")
RNN_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['rnn']['weight_decay'])")
LSTM_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['lstm']['lr'])")
LSTM_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['lstm']['weight_decay'])")
GRU_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gru']['lr'])")
GRU_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gru']['weight_decay'])")
GAWF_LR=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gawf']['lr'])")
GAWF_WD=$(python -c "import json; h=json.load(open('$H')); print(h['$SK']['gawf']['weight_decay'])")

COMMON=(--data_suffix "$DS" --eval_data_suffix 40h-float32 --cnn_dropout 0.0 --rnn_dropout 0.5
  --num_epochs 50 --patience 8 --use_acceleration --use_sector_mode)

CUDA_VISIBLE_DEVICES=0 python train_model.py "${COMMON[@]}" \
  --model_types rnn --hidden_sizes 275 \
  --lrs "$RNN_LR" --wds "$RNN_WD" \
  --result_suffix "gen_phase3_short_${SK}_rnn" &
CUDA_VISIBLE_DEVICES=1 python train_model.py "${COMMON[@]}" \
  --model_types lstm --hidden_sizes 80 \
  --lrs "$LSTM_LR" --wds "$LSTM_WD" \
  --result_suffix "gen_phase3_short_${SK}_lstm" &
wait

CUDA_VISIBLE_DEVICES=0 python train_model.py "${COMMON[@]}" \
  --model_types gru --hidden_sizes 105 \
  --lrs "$GRU_LR" --wds "$GRU_WD" \
  --result_suffix "gen_phase3_short_${SK}_gru" &
CUDA_VISIBLE_DEVICES=1 python train_model.py "${COMMON[@]}" \
  --model_types gawf --hidden_sizes 256 \
  --nofb --fb_start_epoch 50 \
  --lrs "$GAWF_LR" --wds "$GAWF_WD" \
  --result_suffix "gen_phase3_short_${SK}_gawf" &
wait

python experiments/generalization/collect_results.py phase3 --scale "$SK" --out_tag "$TAG" \
  "results/train_data/gen_phase3_short_${SK}_rnn" \
  "results/train_data/gen_phase3_short_${SK}_lstm" \
  "results/train_data/gen_phase3_short_${SK}_gru" \
  "results/train_data/gen_phase3_short_${SK}_gawf"
