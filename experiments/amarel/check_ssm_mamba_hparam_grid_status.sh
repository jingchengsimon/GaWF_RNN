#!/usr/bin/env bash
# Check completion of the SSM/Mamba hparam grid and write failed_task_ids.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-experiments/generalization/artifacts/gen_hparam_ssm_mamba_grid}"

python experiments/generalization/ssm_mamba_hparam_grid.py status \
  --root "$ROOT" \
  --out-dir "$OUT_DIR"

echo ""
echo "Status artifacts:"
echo "  $OUT_DIR/ssm_mamba_hparam_status.json"
echo "  $OUT_DIR/ssm_mamba_hparam_status.csv"
echo "  $OUT_DIR/failed_task_ids.txt"
