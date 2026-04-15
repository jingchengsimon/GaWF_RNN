#!/usr/bin/env bash
# After 4h/10h/20h short Phase-1 runs: merge + 40h preset -> phase1_best_short.json + shared hparams.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

python experiments/generalization/collect_results.py phase1_short \
  results/train_data/gen_phase1_short_gawf_4h \
  results/train_data/gen_phase1_short_gawf_10h \
  results/train_data/gen_phase1_short_gawf_20h \
  --preset_40h_dir results/train_data/sector_40h_adamw

python experiments/generalization/collect_results.py emit_hparams_shared \
  --phase1_best experiments/generalization/artifacts/phase1_best_short.json \
  --out experiments/generalization/artifacts/phase2_final_hparams_short.json
