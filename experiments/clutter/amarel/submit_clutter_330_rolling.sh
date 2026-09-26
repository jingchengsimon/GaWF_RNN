#!/usr/bin/env bash
# Submit the smoke gate and self-replenishing 330-unit output-only Clutter campaign.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( DRY_RUN )); then
  printf 'campaign=clutter_output_only_330_v1 units=330 window=90 user_submit_ceiling=99\n'
  printf 'groups=matched40:60,feedback40:30,equal4:60,scale4:60,scale10:60,scale20:60\n'
  printf 'submission=GPU preflight -> afterok CPU controller -> sparse GPU arrays + delayed controllers\n'
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
[[ -d "$ROOT/.git" || -f "$ROOT/.git" ]] || {
  echo "AIM3_ROOT is not a Git checkout: $ROOT" >&2; exit 1;
}
[[ -d "$AIM3_RESULTS_PATH" ]] || { echo "Missing results root" >&2; exit 1; }
[[ -d "$AIM3_CLUTTER_DATA_DIR" ]] || { echo "Missing data root" >&2; exit 1; }
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || {
  echo "Source checkout is dirty; use a clean immutable snapshot" >&2; exit 1;
}
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
RESULT_BASE="$AIM3_RESULTS_PATH/data/clutter/runs/clutter_output_only_330_v1"
ART="${AIM3_ARTIFACT_ROOT:-$AIM3_RESULTS_PATH/artifacts/clutter_output_only_330_v1}"
[[ ! -e "$RESULT_BASE" && ! -e "$ART" ]] || {
  echo "Campaign result or artifact root already exists; refusing duplicate submission" >&2
  exit 1
}
ACTIVE_COUNT="$(squeue -r -h -u "$USER" -o '%i' | wc -l | tr -d ' ')"
(( ACTIVE_COUNT <= 97 )) || {
  echo "Insufficient user-wide quota for preflight and controller: $ACTIVE_COUNT active" >&2
  exit 1
}
for file in \
  "$SCRIPT_DIR/run_clutter_330_preflight.sh" \
  "$SCRIPT_DIR/run_clutter_330_controller.sh" \
  "$SCRIPT_DIR/run_clutter_330_unit.sh" \
  "$SCRIPT_DIR/rolling330.py" \
  "$SCRIPT_DIR/preflight330.py"; do
  [[ -s "$file" ]] || { echo "Missing campaign source: $file" >&2; exit 1; }
done
mkdir -p "$ART/status"
printf '%s\n' "$SOURCE_COMMIT" > "$ART/status/source_commit.txt"
EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_ARTIFACT_ROOT=$ART"
EXPORTS+=",AIM3_SOURCE_COMMIT=$SOURCE_COMMIT"
SMOKE_RAW="$(sbatch --parsable --chdir="$ROOT" \
  --output="$ART/%j.preflight.out" --error="$ART/%j.preflight.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_330_preflight.sh")"
SMOKE_ID="${SMOKE_RAW%%;*}"
CONTROLLER_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$SMOKE_ID" \
  --output="$ART/%j.controller.out" --error="$ART/%j.controller.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_330_controller.sh")"
printf 'PREFLIGHT_JOB_ID=%s\n' "$SMOKE_ID"
printf 'CONTROLLER_JOB_ID=%s\n' "${CONTROLLER_RAW%%;*}"
printf 'SOURCE_COMMIT=%s\n' "$SOURCE_COMMIT"
printf 'RESULT_BASE=%s\n' "$RESULT_BASE"
printf 'ARTIFACT_ROOT=%s\n' "$ART"
