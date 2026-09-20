#!/usr/bin/env bash
# Submit thirty independent dynamic-baseline units and one summary job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAX_CONCURRENT=6
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent requires an integer}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--max-concurrent must be a positive integer" >&2; exit 2;
}

if (( DRY_RUN )); then
  printf 'submit: formal array=0-29%%%s -> aggregate\n' "$MAX_CONCURRENT"
  printf 'models=mlstm,hyperlstm,brims seeds=1-10 epochs=150 optimizer=adamw lr=0.001 wd=0.001\n'
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
: "${AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY:?Export AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.tsv" \
  "$AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY"; do
  [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
grep -Fq '"status": "passed"' "$AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY" || {
  echo "Preflight summary is not passed" >&2; exit 1;
}
grep -Fq "\"source_commit\": \"$SOURCE_COMMIT\"" \
  "$AIM3_DYNAMIC_BASELINE_PREFLIGHT_SUMMARY" || {
  echo "Preflight source commit does not match current source" >&2; exit 1;
}

RUN_TAG="clutter_dynamic_weight_baselines_ep150_v1"
RESULT_BASE="$AIM3_RESULTS_PATH/data/clutter/runs/dynamic_weight_baselines/$RUN_TAG"
TEST_BASE="$AIM3_RESULTS_PATH/data/analysis/dynamic_weight_baselines_reset_excluded_test_10seed_v1"
SUMMARY_ROOT="$AIM3_RESULTS_PATH/data/analysis/dynamic_weight_baselines_formal_10seed_v1"
ARTIFACT_ROOT="${AIM3_ARTIFACT_ROOT:-$AIM3_RESULTS_PATH/artifacts/$RUN_TAG}"
for target in "$RESULT_BASE" "$TEST_BASE" "$SUMMARY_ROOT" "$ARTIFACT_ROOT"; do
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing target: $target" >&2; exit 1; }
done

STATUS_DIR="$ARTIFACT_ROOT/status"
mkdir -p "$STATUS_DIR"
EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_STATUS_DIR=$STATUS_DIR"
EXPORTS+=",AIM3_SOURCE_COMMIT=$SOURCE_COMMIT,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"

ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --array="0-29%$MAX_CONCURRENT" \
  --output="$ARTIFACT_ROOT/%A_%a.out" --error="$ARTIFACT_ROOT/%A_%a.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_dynamic_baselines_formal.sh")"
ARRAY_ID="${ARRAY_RAW%%;*}"
AGGREGATE_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$ARRAY_ID" \
  --output="$ARTIFACT_ROOT/%j.aggregate.out" --error="$ARTIFACT_ROOT/%j.aggregate.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_dynamic_baselines_aggregate.sh")"

printf 'ARRAY_JOB_ID=%s\n' "$ARRAY_ID"
printf 'AGGREGATE_JOB_ID=%s\n' "${AGGREGATE_RAW%%;*}"
printf 'ARRAY_TASKS=0-29%%%s\n' "$MAX_CONCURRENT"
printf 'RESULT_BASE=%s\n' "$RESULT_BASE"
printf 'TEST_BASE=%s\n' "$TEST_BASE"
printf 'SUMMARY_ROOT=%s\n' "$SUMMARY_ROOT"
printf 'ARTIFACT_ROOT=%s\n' "$ARTIFACT_ROOT"
