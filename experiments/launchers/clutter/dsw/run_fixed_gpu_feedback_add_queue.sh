#!/usr/bin/env bash
# Wait for exact incumbent processes, then run a fixed-GPU additive-feedback task queue.

set -euo pipefail

usage() {
  echo "Usage: $0 ROOT ENV DATA RESULTS RUN_ROOT GPU [--wait PID:TOKEN] -- TASK_ID..." >&2
  exit 2
}

(( $# >= 8 )) || usage
ROOT="$1"
ENV_ROOT="$2"
DATA_ROOT="$3"
RESULTS_ROOT="$4"
RUN_ROOT="$5"
GPU="$6"
shift 6

[[ "$GPU" =~ ^[0-7]$ ]] || usage
WAIT_SPECS=()
while [[ "${1:-}" == "--wait" ]]; do
  (( $# >= 2 )) || usage
  WAIT_SPECS+=("$2")
  shift 2
done
[[ "${1:-}" == "--" ]] || usage
shift
(( $# > 0 )) || usage
TASKS=("$@")

UNIT="$ROOT/experiments/launchers/clutter/dsw/run_feedback_add_unit.sh"
SMOKE="$ROOT/experiments/launchers/clutter/dsw/run_feedback_add_smoke.sh"
LANE_ROOT="$RUN_ROOT/lanes/gpu$GPU"
mkdir -p "$LANE_ROOT"
[[ -x "$UNIT" && -x "$SMOKE" ]] || { echo "Queue runners are missing" >&2; exit 1; }

declare -A SEEN=()
for task in "${TASKS[@]}"; do
  [[ "$task" =~ ^[0-9]+$ ]] && (( task >= 0 && task < 15 )) || usage
  [[ -z "${SEEN[$task]:-}" ]] || { echo "Duplicate task id: $task" >&2; exit 2; }
  SEEN[$task]=1
done

exec 9>"$LANE_ROOT/lane.lock"
flock -n 9 || { echo "Another coordinator owns GPU $GPU lane" >&2; exit 1; }
printf 'state=waiting gpu=%s tasks=%s timestamp=%s\n' \
  "$GPU" "${TASKS[*]}" "$(date -Is)" > "$LANE_ROOT/lane.status"

wait_spec_is_active() {
  local spec="$1" pid="${spec%%:*}" token="${spec#*:}" command
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command" == *"$token"* ]]
}

while true; do
  active=0
  for spec in "${WAIT_SPECS[@]}"; do
    if wait_spec_is_active "$spec"; then active=$((active + 1)); fi
  done
  (( active == 0 )) && break
  sleep 5
done

printf 'state=smoke_wait gpu=%s tasks=%s timestamp=%s\n' \
  "$GPU" "${TASKS[*]}" "$(date -Is)" > "$LANE_ROOT/lane.status"
bash "$SMOKE" "$ROOT" "$ENV_ROOT" "$RUN_ROOT" "$GPU"

for task in "${TASKS[@]}"; do
  printf 'state=running gpu=%s task=%s timestamp=%s\n' \
    "$GPU" "$task" "$(date -Is)" > "$LANE_ROOT/task_${task}.status"
  if bash "$UNIT" \
    "$ROOT" "$ENV_ROOT" "$DATA_ROOT" "$RESULTS_ROOT" "$RUN_ROOT" "$task" "$GPU" \
    > "$LANE_ROOT/task_${task}.out" 2> "$LANE_ROOT/task_${task}.err"; then
    printf 'state=done gpu=%s task=%s timestamp=%s\n' \
      "$GPU" "$task" "$(date -Is)" > "$LANE_ROOT/task_${task}.status"
  else
    status=$?
    printf 'state=failed gpu=%s task=%s exit=%s timestamp=%s\n' \
      "$GPU" "$task" "$status" "$(date -Is)" > "$LANE_ROOT/task_${task}.status"
    printf 'state=failed gpu=%s task=%s exit=%s timestamp=%s\n' \
      "$GPU" "$task" "$status" "$(date -Is)" > "$LANE_ROOT/lane.status"
    exit "$status"
  fi
done
printf 'state=done gpu=%s tasks=%s timestamp=%s\n' \
  "$GPU" "${TASKS[*]}" "$(date -Is)" > "$LANE_ROOT/lane.status"
