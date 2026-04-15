#!/usr/bin/env bash
# After 4h/10h/20h phase1 runs finish, aggregate best LR/WD; reuse preset 40h.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
python experiments/generalization/collect_results.py phase1_short \
  results/train_data/gen_phase1_gawf_4h \
  results/train_data/gen_phase1_gawf_10h \
  results/train_data/gen_phase1_gawf_20h \
  --preset_40h_dir results/train_data/sector_40h_adamw \
  --out experiments/generalization/artifacts/phase1_best.json
