#!/usr/bin/env bash
# 40h short: no retrain; import metrics from a preset folder into phase3_summary_40h<tag>.csv
# Uses NUM_EPOCHS / CSV_TAG like phase3_train_*_short.sh (see phase3_short_env.inc.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
# shellcheck source=phase3_short_env.inc.sh
source "$SCRIPT_DIR/phase3_short_env.inc.sh"
METRICS_DIR="${METRICS_DIR:-results/train_data/sector_40h_adamw}"

python experiments/generalization/collect_results.py phase3_import \
  --scale 40h \
  --metrics_dir "$METRICS_DIR" \
  --out_tag "$TAG"
