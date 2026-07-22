#!/usr/bin/env bash
# Phase 1: GAWF LR×WD grid for one train scale. Val is always 40h except when train is 40h
# (then train and val both use 40h-uint8).
# Modes: "full" (larger grid, longer run) or "short" (smaller grid, fewer epochs).
#
# Usage (repo root; script cd's there):
#   bash experiments/generalization/phase1_gawf_search.sh 4h full
#   bash experiments/generalization/phase1_gawf_search.sh 40h short
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SCAL="${1:-}"
MODE="${2:-full}"

if [[ -z "$SCAL" ]] || [[ ! "$SCAL" =~ ^(4h|10h|20h|40h)$ ]]; then
  echo "Usage: $0 <4h|10h|20h|40h> <full|short>" >&2
  exit 1
fi
if [[ ! "$MODE" =~ ^(full|short)$ ]]; then
  echo "Mode must be full or short, got: $MODE" >&2
  exit 1
fi

if [[ "$SCAL" == 40h ]]; then
  DS="40h-uint8"
  EVAL="40h-uint8"
else
  DS="${SCAL}-float32"
  EVAL="40h-uint8"
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
echo "[Phase1 GAWF $SCAL mode=$MODE] data_dir=$DATA_DIR"

if [[ "$MODE" == short ]]; then
  LRS=(1e-4 3e-4 5e-4)
  NE=50
  PT=8
  RS="gen_phase1_short_gawf_${SCAL}"
else
  LRS=(1e-4 3e-4 5e-4 1e-3)
  NE=100
  PT=15
  RS="gen_phase1_gawf_${SCAL}"
fi

python train_model.py \
  --model_types gawf \
  --hidden_sizes 256 \
  --data_suffix "$DS" \
  --eval_data_suffix "$EVAL" \
  --data_dir "$DATA_DIR" \
  --lrs "${LRS[@]}" \
  --wds 1e-4 1e-3 \
  --cnn_dropout 0.0 \
  --rnn_dropout 0.5 \
  --num_epochs "$NE" \
  --patience "$PT" \
  --use_acceleration \
  --use_sector_mode \
  --result_suffix "$RS"
