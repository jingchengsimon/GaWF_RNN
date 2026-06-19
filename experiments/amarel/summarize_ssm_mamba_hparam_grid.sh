#!/usr/bin/env bash
# Aggregate completed Mamba/S5 hparam-grid runs into best-hparam summaries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-experiments/generalization/artifacts/gen_hparam_mamba_s5_grid}"

python experiments/generalization/ssm_mamba_hparam_grid.py summarize \
  --root "$ROOT" \
  --out-dir "$OUT_DIR"

echo ""
echo "Summary artifacts:"
echo "  $OUT_DIR/mamba_s5_hparam_best.json"
echo "  $OUT_DIR/mamba_s5_hparam_best.csv"
echo "  $OUT_DIR/mamba_s5_hparam_best_summary.md"
echo "  $OUT_DIR/mamba_s5_hparam_all_trials.csv"
