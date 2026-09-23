#!/usr/bin/env bash
# Run the one-time additive-feedback optimization smoke on the first released queue GPU.

set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: $0 ROOT ENV_ROOT RUN_ROOT GPU" >&2
  exit 2
fi
ROOT="$1"
ENV_ROOT="$2"
RUN_ROOT="$3"
GPU="$4"
SMOKE_ROOT="$RUN_ROOT/smoke"
mkdir -p "$SMOKE_ROOT"

exec 9>"$SMOKE_ROOT/lock"
flock -x 9
[[ ! -e "$SMOKE_ROOT/fail" ]] || { echo "Earlier smoke failed" >&2; exit 1; }
[[ ! -s "$SMOKE_ROOT/pass" ]] || exit 0

EXPECTED_COMMIT="$(<"$RUN_ROOT/expected_commit.txt")"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "Unexpected DSW source commit" >&2
  exit 1
}
[[ ! -e "$SMOKE_ROOT/sanity.json" ]] || {
  echo "Incomplete smoke output already exists" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
if "$ENV_ROOT/bin/python" -B -m experiments.clutter.feedback_control_sanity \
  --model-set additive --output "$SMOKE_ROOT/sanity.json" --device cuda --steps 200 --seed 1 \
  > "$SMOKE_ROOT/sanity.log" 2>&1; then
  printf 'commit=%s gpu=%s timestamp=%s\n' "$EXPECTED_COMMIT" "$GPU" "$(date -Is)" \
    > "$SMOKE_ROOT/pass"
else
  status=$?
  printf 'exit=%s gpu=%s timestamp=%s\n' "$status" "$GPU" "$(date -Is)" \
    > "$SMOKE_ROOT/fail"
  exit "$status"
fi
