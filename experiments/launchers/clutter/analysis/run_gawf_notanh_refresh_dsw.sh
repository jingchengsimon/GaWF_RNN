#!/usr/bin/env bash
# Run the ten-seed GaWF legacy no-tanh refresh on seven fixed DSW GPUs.

set -euo pipefail

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
PYTHON="${AIM3_PYTHON:?AIM3_PYTHON is required}"
EXPECTED_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
CHECKPOINT_ROOT="${AIM3_CHECKPOINT_ROOT:?AIM3_CHECKPOINT_ROOT is required}"
DATA_DIR="${AIM3_DATA_DIR:?AIM3_DATA_DIR is required}"
OUTPUT_ROOT="${AIM3_OUTPUT_ROOT:?AIM3_OUTPUT_ROOT is required}"
ARTIFACT_ROOT="${AIM3_ARTIFACT_ROOT:?AIM3_ARTIFACT_ROOT is required}"
RUN_ROOT="${AIM3_RUN_ROOT:?AIM3_RUN_ROOT is required}"
FIGURE_ROOT="$ARTIFACT_ROOT/Figures_notanh"
RECORD_OUTPUT="$ARTIFACT_ROOT/GaWF_STATS_RECORD_notanh.md"
LAUNCHER="$ROOT/experiments/launchers/clutter/analysis/gawf_notanh_refresh.py"

test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$ROOT" status --porcelain)"
test ! -e "$OUTPUT_ROOT"
test ! -e "$ARTIFACT_ROOT"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status"
printf '%s\n' "$EXPECTED_COMMIT" > "$RUN_ROOT/source_commit.txt"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"

common=(
  --checkpoint-root "$CHECKPOINT_ROOT"
  --data-dir "$DATA_DIR"
  --output-root "$OUTPUT_ROOT"
  --figure-root "$FIGURE_ROOT"
  --record-output "$RECORD_OUTPUT"
  --python "$PYTHON"
  --device cuda
)

run_seed() {
  local seed="$1" gpu="$2" tag status
  printf -v tag '%02d' "$seed"
  printf 'state=running seed=%s gpu=%s timestamp=%s\n' \
    "$seed" "$gpu" "$(date -Is)" > "$RUN_ROOT/status/seed${tag}.status"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -B "$LAUNCHER" collect \
    "${common[@]}" --seed "$seed" --execute > "$RUN_ROOT/logs/seed${tag}.log" 2>&1; then
    printf 'state=done seed=%s gpu=%s timestamp=%s\n' \
      "$seed" "$gpu" "$(date -Is)" > "$RUN_ROOT/status/seed${tag}.status"
  else
    status=$?
    printf 'state=failed seed=%s gpu=%s exit=%s timestamp=%s\n' \
      "$seed" "$gpu" "$status" "$(date -Is)" > "$RUN_ROOT/status/seed${tag}.status"
    return "$status"
  fi
}

printf 'state=pilot seed=1 gpu=0 timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/campaign.status"
run_seed 1 0

printf 'state=collecting seeds=2-10 timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/campaign.status"
run_seed 2 0 & p2=$!
run_seed 3 1 & p3=$!
run_seed 4 2 & p4=$!
run_seed 5 3 & p5=$!
run_seed 6 4 & p6=$!
run_seed 7 5 & p7=$!
run_seed 8 7 & p8=$!
wait "$p2"
wait "$p3"
wait "$p4"
wait "$p5"
wait "$p6"
wait "$p7"
wait "$p8"
run_seed 9 0 & p9=$!
run_seed 10 1 & p10=$!
wait "$p9"
wait "$p10"

printf 'state=aggregating timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/campaign.status"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -B "$LAUNCHER" aggregate "${common[@]}" --execute \
  > "$RUN_ROOT/logs/aggregate.log" 2>&1
"$PYTHON" -B "$LAUNCHER" record "${common[@]}" --execute \
  > "$RUN_ROOT/logs/record.log" 2>&1
printf 'state=done timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/campaign.status"
