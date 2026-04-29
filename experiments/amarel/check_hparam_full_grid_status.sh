#!/usr/bin/env bash
# Check completion of the full-grid hparam search and write failed_task_ids.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-experiments/generalization/artifacts/gen_hparam_full_grid}"

python experiments/generalization/hparam_full_grid.py status \
  --root "$ROOT" \
  --out-dir "$OUT_DIR"

echo ""
echo "Status artifacts:"
echo "  $OUT_DIR/hparam_full_grid_status.json"
echo "  $OUT_DIR/hparam_full_grid_status.csv"
echo "  $OUT_DIR/failed_task_ids.txt"
