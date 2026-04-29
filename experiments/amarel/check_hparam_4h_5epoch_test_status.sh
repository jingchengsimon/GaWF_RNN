#!/usr/bin/env bash
# Check completion for the 4h/5-epoch hparam smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MODELS=(rnn lstm gru gawf)
HIDDEN_SIZES=(64 128 256 512)
LRS=(0.0001 0.0005 0.001 0.005)
WDS=(0.0 1e-05 0.0001 0.001)
TOTAL_TASKS=256
RESULT_ROOT_SUFFIX="gen_hparam_4h_5epoch_test"
OUT_DIR="experiments/generalization/artifacts/${RESULT_ROOT_SUFFIX}"
STATUS_CSV="$OUT_DIR/hparam_4h_5epoch_test_status.csv"
FAILED_IDS="$OUT_DIR/failed_task_ids.txt"

mkdir -p "$OUT_DIR"
echo "task_id,model,hidden_size,lr,weight_decay,valid,reason,metrics_path" > "$STATUS_CSV"
: > "$FAILED_IDS"

valid_count=0
for ((task_id = 0; task_id < TOTAL_TASKS; task_id++)); do
  wd_idx=$((task_id % 4))
  lr_idx=$(((task_id / 4) % 4))
  hidden_idx=$(((task_id / 16) % 4))
  model_idx=$(((task_id / 64) % 4))

  model="${MODELS[$model_idx]}"
  hidden="${HIDDEN_SIZES[$hidden_idx]}"
  lr="${LRS[$lr_idx]}"
  wd="${WDS[$wd_idx]}"
  result_suffix="${RESULT_ROOT_SUFFIX}/task_$(printf '%04d' "$task_id")"
  stem="${model}_sector_acc_h${hidden}_lr${lr}_wd${wd}_cdo0.0_rdo0.5"
  metrics_path="$ROOT/results/train_data/${result_suffix}/${stem}_metrics.json"

  valid=0
  reason="missing_metrics"
  if [[ -f "$metrics_path" ]]; then
    if python - "$metrics_path" "$model" "$hidden" "$lr" "$wd" <<'PY'
import json
import math
import sys

path, model, hidden, lr, wd = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    m = json.load(f)
ok = (
    m.get("model_type") == model
    and int(m.get("hidden_size", -1)) == int(hidden)
    and math.isclose(float(m.get("lr", "nan")), float(lr), rel_tol=0, abs_tol=1e-12)
    and math.isclose(float(m.get("weight_decay", "nan")), float(wd), rel_tol=0, abs_tol=1e-12)
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

  if [[ "$valid" -ne 1 ]]; then
    printf '%s\n' "$task_id" >> "$FAILED_IDS"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$task_id" "$model" "$hidden" "$lr" "$wd" "$valid" "$reason" "$metrics_path" \
    >> "$STATUS_CSV"
done

failed_count=$((TOTAL_TASKS - valid_count))
cat > "$OUT_DIR/hparam_4h_5epoch_test_status.json" <<EOF
{
  "total": $TOTAL_TASKS,
  "valid": $valid_count,
  "failed": $failed_count,
  "status_csv": "$STATUS_CSV",
  "failed_task_ids_path": "$FAILED_IDS"
}
EOF

cat "$OUT_DIR/hparam_4h_5epoch_test_status.json"
