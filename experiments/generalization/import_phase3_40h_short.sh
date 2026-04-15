#!/usr/bin/env bash
# 40h short: no retrain; import metrics from sector_40h_adamw into phase3_summary_40h_short.csv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

python experiments/generalization/collect_results.py phase3_import \
  --scale 40h \
  --metrics_dir results/train_data/sector_40h_adamw \
  --out_tag _short
