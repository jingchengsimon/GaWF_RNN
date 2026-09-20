#!/usr/bin/env bash
# Submit the four two-epoch dynamic-baseline sanity units and their aggregation.

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
  printf 'submit: two-epoch array=0-3%%4 -> preflight aggregate\n'
  printf 'models=lstm,mlstm,hyperlstm,brims seed=1 optimizer=adamw lr=0.001 wd=0.001\n'
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.tsv"; do
  [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

RUN_TAG="dynamic_weight_baselines_2epoch_seed1_v1"
RESULT_BASE="$AIM3_RESULTS_PATH/data/clutter/runs/preflight/$RUN_TAG"
ARTIFACT_ROOT="${AIM3_ARTIFACT_ROOT:-$AIM3_RESULTS_PATH/artifacts/$RUN_TAG}"
[[ ! -e "$RESULT_BASE" ]] || { echo "Result root already exists: $RESULT_BASE" >&2; exit 1; }
[[ ! -e "$ARTIFACT_ROOT" ]] || { echo "Artifact root already exists: $ARTIFACT_ROOT" >&2; exit 1; }

STATUS_DIR="$ARTIFACT_ROOT/status"
REPORTS_DIR="$ARTIFACT_ROOT/reports"
SUMMARY="$ARTIFACT_ROOT/summary.json"
mkdir -p "$STATUS_DIR" "$REPORTS_DIR"
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_STATUS_DIR=$STATUS_DIR"
EXPORTS+=",AIM3_PREFLIGHT_REPORTS_DIR=$REPORTS_DIR,AIM3_PREFLIGHT_SUMMARY=$SUMMARY"
EXPORTS+=",AIM3_SOURCE_COMMIT=$SOURCE_COMMIT,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"

ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --array="0-3%4" \
  --output="$ARTIFACT_ROOT/%A_%a.out" --error="$ARTIFACT_ROOT/%A_%a.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_dynamic_baselines_preflight.sh")"
ARRAY_ID="${ARRAY_RAW%%;*}"
AGGREGATE_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$ARRAY_ID" \
  --output="$ARTIFACT_ROOT/%j.aggregate.out" --error="$ARTIFACT_ROOT/%j.aggregate.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_dynamic_baselines_preflight_aggregate.sh")"

printf 'PREFLIGHT_ARRAY_JOB_ID=%s\n' "$ARRAY_ID"
printf 'PREFLIGHT_AGGREGATE_JOB_ID=%s\n' "${AGGREGATE_RAW%%;*}"
printf 'RESULT_BASE=%s\n' "$RESULT_BASE"
printf 'ARTIFACT_ROOT=%s\n' "$ARTIFACT_ROOT"
printf 'PREFLIGHT_SUMMARY=%s\n' "$SUMMARY"
