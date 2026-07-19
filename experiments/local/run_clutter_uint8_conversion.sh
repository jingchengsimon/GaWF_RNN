#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CONDA_INIT="${AIM3_CONDA_INIT:-/G/anaconda3/etc/profile.d/conda.sh}"
DATA_DIR="${AIM3_DATA_DIR:-$ROOT/stimuli}"
ARTIFACT_DIR="$ROOT/experiments/local/artifacts/clutter_uint8_conversion"
LOG_PATH="$ARTIFACT_DIR/run.log"

if [[ ! -f "$CONDA_INIT" ]]; then
  echo "Missing Conda initialization script: $CONDA_INIT" >&2
  exit 2
fi

set +u
source "$CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
cd "$ROOT"
mkdir -p "$ARTIFACT_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -Is)] converting 40h float32 stimuli to uint8 under $DATA_DIR"

python source/clutter/convert_float32_to_uint8.py \
  --data-dir "$DATA_DIR" \
  --source-suffix 40h-float32 \
  --target-suffix 40h-uint8 \
  --splits train validation test

echo "[$(date -Is)] conversion complete"
