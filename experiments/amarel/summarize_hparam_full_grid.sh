#!/usr/bin/env bash
# Aggregate the completed full-grid search and generate generalization figures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

CSV_TAG="${CSV_TAG:-_hparam_full_grid}"
OUT_DIR="${OUT_DIR:-experiments/generalization/artifacts/gen_hparam_full_grid}"
SAVE_PDF="${SAVE_PDF:-0}"

python experiments/generalization/hparam_full_grid.py status \
  --root "$ROOT" \
  --out-dir "$OUT_DIR" \
  --fail-on-missing

python experiments/generalization/hparam_full_grid.py summarize \
  --root "$ROOT" \
  --out-dir "$OUT_DIR" \
  --csv-tag "$CSV_TAG"

plot_args=(--csv_tag "$CSV_TAG")
if [[ "$SAVE_PDF" == "1" ]]; then
  plot_args+=(--save-pdf)
fi
python utils_viz/plot_generalization.py "${plot_args[@]}"

echo "Summary written under $OUT_DIR"
echo "Figures written under results/anal_figs/generalization"
