#!/usr/bin/env bash
# Recover two interrupted units, then run the two units stranded by a failed DSW lane.

set -euo pipefail

if (( $# != 7 )); then
  echo "Usage: $0 ROOT ENV DATA PRIMARY_RESULTS SECONDARY_RESULTS RUN_ROOT GPU" >&2
  exit 2
fi

ROOT="$1"
ENV_ROOT="$2"
DATA_ROOT="$3"
PRIMARY_RESULTS="$4"
SECONDARY_RESULTS="$5"
RUN_ROOT="$6"
GPU="$7"
UNIT="${AIM3_FEEDBACK_ADD_UNIT:-$ROOT/experiments/launchers/clutter/dsw/run_feedback_add_unit.sh}"

[[ "$GPU" =~ ^[0-7]$ && -x "$UNIT" ]] || exit 2
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status"

run_pair() {
  local results_root="$1" label="$2" task_a="$3" task_b="$4" pid_a pid_b status=0
  bash "$UNIT" "$ROOT" "$ENV_ROOT" "$DATA_ROOT" "$results_root" "$RUN_ROOT" \
    "$task_a" "$GPU" > "$RUN_ROOT/logs/${label}_task_${task_a}.out" \
    2> "$RUN_ROOT/logs/${label}_task_${task_a}.err" &
  pid_a=$!
  bash "$UNIT" "$ROOT" "$ENV_ROOT" "$DATA_ROOT" "$results_root" "$RUN_ROOT" \
    "$task_b" "$GPU" > "$RUN_ROOT/logs/${label}_task_${task_b}.out" \
    2> "$RUN_ROOT/logs/${label}_task_${task_b}.err" &
  pid_b=$!
  wait "$pid_a" || status=1
  wait "$pid_b" || status=1
  (( status == 0 )) || return 1
}

printf 'state=running phase=resume task_pair=4,5 gpu=%s timestamp=%s\n' \
  "$GPU" "$(date -Is)" > "$RUN_ROOT/status/supervisor.status"
run_pair "$PRIMARY_RESULTS" resume 4 5

printf 'state=running phase=stranded task_pair=11,14 gpu=%s timestamp=%s\n' \
  "$GPU" "$(date -Is)" > "$RUN_ROOT/status/supervisor.status"
run_pair "$SECONDARY_RESULTS" stranded 11 14

printf 'state=done gpu=%s timestamp=%s\n' "$GPU" "$(date -Is)" \
  > "$RUN_ROOT/status/supervisor.status"
