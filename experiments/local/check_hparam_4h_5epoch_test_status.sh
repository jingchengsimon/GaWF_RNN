#!/usr/bin/env bash
# Check local 2-GPU 4h/5-epoch smoke-test outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MODELS=(rnn lstm gru gawf)
RESULT_SUFFIX="gen_local_hparam_4h_5epoch_test"
OUT_DIR="$ROOT/experiments/local/artifacts/hparam_4h_5epoch_test"
STATUS_CSV="$OUT_DIR/status.csv"
HIDDEN_SIZE=256
LR=0.0005
WD=0.0001

mkdir -p "$OUT_DIR"
echo "model,valid,reason,metrics_path" > "$STATUS_CSV"

valid_count=0
for model in "${MODELS[@]}"; do
  stem="${model}_sector_acc_h${HIDDEN_SIZE}_lr${LR}_wd${WD}_cdo0.0_rdo0.5"
  metrics_path="$ROOT/results/train_data/${RESULT_SUFFIX}/${stem}_metrics.json"
  valid=0
  reason="missing_metrics"

  if [[ -f "$metrics_path" ]]; then
    if python - "$metrics_path" "$model" <<'PY'
import json
import math
import sys

path, model = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)
ok = (
    m.get("model_type") == model
    and int(m.get("hidden_size", -1)) == 256
    and math.isclose(float(m.get("lr", "nan")), 0.0005, rel_tol=0, abs_tol=1e-12)
    and math.isclose(float(m.get("weight_decay", "nan")), 0.0001, rel_tol=0, abs_tol=1e-12)
    and m.get("dataset_suffix") == "4h-float32"
    and m.get("eval_dataset_suffix") == "40h-float32"
    and int(m.get("num_epochs", -1)) == 5
)
raise SystemExit(0 if ok else 1)
PY
    then
      valid=1
      reason="ok"
      valid_count=$((valid_count + 1))
    else
      reason="metrics_mismatch"
    fi
  fi

  printf '%s,%s,%s,%s\n' "$model" "$valid" "$reason" "$metrics_path" >> "$STATUS_CSV"
done

failed_count=$((4 - valid_count))
cat > "$OUT_DIR/status.json" <<EOF
{
  "total": 4,
  "valid": $valid_count,
  "failed": $failed_count,
  "status_csv": "$STATUS_CSV"
}
EOF

cat "$OUT_DIR/status.json"
