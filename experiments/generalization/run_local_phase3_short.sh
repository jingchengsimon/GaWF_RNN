#!/usr/bin/env bash
#
# Local Phase 3 + 40h import + plots. Assumes Phase 1 is already done and
# experiments/generalization/artifacts/phase2_final_hparams_short.json exists
# (from collect_results emit_hparams_shared on phase1_best_short.json).
#
# Environment (optional):
#   NUM_EPOCHS  — max epochs for Phase 3 training (default 50); also in result dir names and CSV/plot tags.
#   CSV_TAG     — CSV/plot tag base (default _short); full tag is ${CSV_TAG}_ep${NUM_EPOCHS}.
#   METRICS_40H — dir for phase3_import 40h row (default results/train_data/sector_40h_adamw_0409).
#
# Specify epoch count:
#   NUM_EPOCHS=100 bash experiments/generalization/run_local_phase3_short.sh
#   bash experiments/generalization/run_local_phase3_short.sh 100
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_EPOCHS="$1"
fi

: "${NUM_EPOCHS:=50}"
: "${CSV_TAG:=_short}"
: "${METRICS_40H:=results/train_data/sector_40h_adamw_0409}"

HP_JSON="$SCRIPT_DIR/artifacts/phase2_final_hparams_short.json"
if [[ ! -f "$HP_JSON" ]]; then
  echo "Missing required file: $HP_JSON" >&2
  echo "Create it from Phase 1 best metrics, e.g.:" >&2
  echo "  python experiments/generalization/collect_results.py emit_hparams_shared \\" >&2
  echo "    --phase1_best experiments/generalization/artifacts/phase1_best_short.json \\" >&2
  echo "    --out $HP_JSON" >&2
  exit 1
fi

METRICS_ABS="$METRICS_40H"
if [[ "$METRICS_ABS" != /* ]]; then
  METRICS_ABS="$ROOT/$METRICS_ABS"
fi
if [[ ! -d "$METRICS_ABS" ]]; then
  echo "METRICS_40H directory not found: $METRICS_ABS" >&2
  exit 1
fi
if ! compgen -G "$METRICS_ABS"/*_metrics.json >/dev/null 2>&1; then
  echo "No *_metrics.json under: $METRICS_ABS" >&2
  exit 1
fi

# shellcheck source=phase3_short_env.inc.sh
source "$SCRIPT_DIR/phase3_short_env.inc.sh"

export NUM_EPOCHS CSV_TAG

bash "$SCRIPT_DIR/phase3_train_4h_short.sh"
bash "$SCRIPT_DIR/phase3_train_10h_short.sh"
bash "$SCRIPT_DIR/phase3_train_20h_short.sh"

python experiments/generalization/collect_results.py phase3_import \
  --scale 40h \
  --metrics_dir "$METRICS_40H" \
  --out_tag "$TAG"

python plot_generalization.py --csv_tag "$TAG"
echo "Done. CSV tag: $TAG  (artifacts: experiments/generalization/artifacts/phase3_summary_*${TAG}.csv)"
echo "Figures: results/anal_figs/generalization/*${TAG#_}.*"
