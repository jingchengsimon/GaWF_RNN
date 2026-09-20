#!/usr/bin/env bash
# Submit preflight, forty training/evaluation units, and aggregation for feedback controls.

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
  printf 'submit: preflight -> array=0-39%%%s -> aggregate\n' "$MAX_CONCURRENT"
  printf 'models=gawf_additive,rnn_fb,gru_fb,lstm_fb seeds=1-10 epochs=150\n'
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
: "${AIM3_FEEDBACK_CONTROL_PREREQ:?Export AIM3_FEEDBACK_CONTROL_PREREQ}"

for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-float32-jointswitch-balanced-10digit-unique.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-float32-jointswitch-balanced-10digit-unique.tsv" \
  "$AIM3_FEEDBACK_CONTROL_PREREQ/original_test/reset_excluded_test_accuracy_10seed.csv" \
  "$AIM3_FEEDBACK_CONTROL_PREREQ/SHA256SUMS"; do
  [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
(( $(find "$AIM3_FEEDBACK_CONTROL_PREREQ/gawf_shuffle" \
  -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 10 )) || {
  echo "Expected ten GaWF shuffle prerequisite files" >&2; exit 1;
}
(cd "$AIM3_FEEDBACK_CONTROL_PREREQ" && sha256sum -c SHA256SUMS)

for target in \
  "$AIM3_RESULTS_PATH/data/clutter/runs/feedback_controls/clutter_feedback_controls_ep150_v1" \
  "$AIM3_RESULTS_PATH/data/analysis/feedback_controls_reset_excluded_test_10seed_v1" \
  "$AIM3_RESULTS_PATH/data/analysis/feedback_controls_shuffle_resetexcluded_10seed_v1" \
  "$AIM3_RESULTS_PATH/data/analysis/feedback_controls_formal_10seed_v1"; do
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing target: $target" >&2; exit 1; }
done

ARTIFACT_ROOT="${AIM3_ARTIFACT_ROOT:-$ROOT/experiments/clutter/amarel/artifacts/clutter_feedback_controls_ep150_v1}"
[[ ! -e "$ARTIFACT_ROOT" ]] || { echo "Artifact root already exists: $ARTIFACT_ROOT" >&2; exit 1; }
STATUS_DIR="$ARTIFACT_ROOT/status"
PREFLIGHT_DIR="$ARTIFACT_ROOT/preflight"
mkdir -p "$STATUS_DIR"
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_STATUS_DIR=$STATUS_DIR"
EXPORTS+=",AIM3_PREFLIGHT_DIR=$PREFLIGHT_DIR,AIM3_PREREQ_ROOT=$AIM3_FEEDBACK_CONTROL_PREREQ"
EXPORTS+=",AIM3_SOURCE_COMMIT=$SOURCE_COMMIT,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"

PREFLIGHT_RAW="$(sbatch --parsable --chdir="$ROOT" \
  --output="$ARTIFACT_ROOT/%j.preflight.out" --error="$ARTIFACT_ROOT/%j.preflight.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_feedback_controls_preflight.sh")"
PREFLIGHT_ID="${PREFLIGHT_RAW%%;*}"
ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$PREFLIGHT_ID" \
  --array="0-39%$MAX_CONCURRENT" --output="$ARTIFACT_ROOT/%A_%a.out" \
  --error="$ARTIFACT_ROOT/%A_%a.err" --export="$EXPORTS" \
  "$SCRIPT_DIR/run_clutter_feedback_controls_formal.sh")"
ARRAY_ID="${ARRAY_RAW%%;*}"
AGGREGATE_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$ARRAY_ID" \
  --output="$ARTIFACT_ROOT/%j.aggregate.out" --error="$ARTIFACT_ROOT/%j.aggregate.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_feedback_controls_aggregate.sh")"

printf 'PREFLIGHT_JOB_ID=%s\n' "$PREFLIGHT_ID"
printf 'ARRAY_JOB_ID=%s\n' "$ARRAY_ID"
printf 'AGGREGATE_JOB_ID=%s\n' "${AGGREGATE_RAW%%;*}"
printf 'ARRAY_TASKS=0-39%%%s\n' "$MAX_CONCURRENT"
printf 'RESULT_BASE=%s\n' \
  "$AIM3_RESULTS_PATH/data/clutter/runs/feedback_controls/clutter_feedback_controls_ep150_v1"
printf 'STATUS_DIR=%s\n' "$STATUS_DIR"
