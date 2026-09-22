#!/usr/bin/env bash
# Submit a two-epoch smoke followed by the ten-seed GaWF-aligned no-tanh RNN array.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
MAX_CONCURRENT=10
while (( $# )); do
  case "$1" in
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent requires an integer}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_CONCURRENT" =~ ^([1-9]|10)$ ]] || {
  echo "--max-concurrent must be an integer in [1, 10]" >&2
  exit 2
}

RUNNER="$SCRIPT_DIR/run_clutter_rnn_inloop_notanh.sh"
ARTIFACT_ROOT="$SCRIPT_DIR/artifacts/clutter_rnn_inloop_notanh_v1"
SMOKE_STATUS="$ARTIFACT_ROOT/status/smoke"
FORMAL_STATUS="$ARTIFACT_ROOT/status/formal"
RESULTS_PLACEHOLDER="${AIM3_RESULTS_PATH:-<AIM3_RESULTS_PATH>}"
RESULT_BASE="$RESULTS_PLACEHOLDER/data/clutter/runs/clutter_rnn_inloop_notanh_40h_ep150_v1"
SMOKE_RESULT="$RESULTS_PLACEHOLDER/data/clutter/runs/preflight/"
SMOKE_RESULT+="clutter_rnn_inloop_notanh_40h_2epoch_seed1_v1"
TEST_BASE="$RESULTS_PLACEHOLDER/data/analysis/"
TEST_BASE+="clutter_ablation_reset_excluded_test_rnn_inloop_notanh_10seed_v1"

if (( DRY_RUN )); then
  printf 'smoke: rnn_inloop_notanh seed=1 epochs=2 -> %s\n' "$SMOKE_RESULT"
  printf 'formal: afterok:<smoke_job_id> array=0-9%%%s seeds=1-10 -> %s\n' \
    "$MAX_CONCURRENT" "$RESULT_BASE"
  printf 'hyperparameters: hidden=275 lr=0.001 wd=0.00001 cdo=0.0 rdo=0.5 epochs=150 patience=0\n'
  printf 'recurrence: h_t=Dropout(ReLU(LayerNorm(W_ih*x_t+b_ih+W_hh*h_prev+b_hh)))\n'
  printf 'test export: %s\n' "$TEST_BASE"
  printf 'runner=%s artifacts=%s\n' "$RUNNER" "$ARTIFACT_ROOT"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.tsv" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.tsv"; do
  [[ -s "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

RESULT_BASE="$AIM3_RESULTS_PATH/data/clutter/runs/clutter_rnn_inloop_notanh_40h_ep150_v1"
SMOKE_RESULT="$AIM3_RESULTS_PATH/data/clutter/runs/preflight/"
SMOKE_RESULT+="clutter_rnn_inloop_notanh_40h_2epoch_seed1_v1"
TEST_BASE="$AIM3_RESULTS_PATH/data/analysis/"
TEST_BASE+="clutter_ablation_reset_excluded_test_rnn_inloop_notanh_10seed_v1"
[[ ! -e "$SMOKE_RESULT" ]] || {
  echo "Smoke result root already exists: $SMOKE_RESULT" >&2
  exit 1
}
[[ ! -e "$ARTIFACT_ROOT" ]] || { echo "Artifact root already exists: $ARTIFACT_ROOT" >&2; exit 1; }
for seed in {1..10}; do
  printf -v seed_tag '%02d' "$seed"
  [[ ! -e "$RESULT_BASE/rnn_inloop_notanh-seed$seed_tag" ]] || {
    echo "Formal result leaf already exists: $RESULT_BASE/rnn_inloop_notanh-seed$seed_tag" >&2
    exit 1
  }
  [[ ! -e "$TEST_BASE/rnn_inloop_notanh-seed$seed_tag" ]] || {
    echo "Test result leaf already exists: $TEST_BASE/rnn_inloop_notanh-seed$seed_tag" >&2
    exit 1
  }
done

mkdir -p "$SMOKE_STATUS" "$FORMAL_STATUS"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
printf '%s\n' "$SOURCE_COMMIT" > "$SMOKE_STATUS/source_commit.txt"
printf '%s\n' "$SOURCE_COMMIT" > "$FORMAL_STATUS/source_commit.txt"
COMMON_EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
COMMON_EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_SOURCE_COMMIT=$SOURCE_COMMIT"
COMMON_EXPORTS+=",AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"

SMOKE_RAW="$(sbatch --parsable --chdir="$ROOT" --time=02:00:00 \
  --output="$ARTIFACT_ROOT/%j.smoke.out" --error="$ARTIFACT_ROOT/%j.smoke.err" \
  --export="$COMMON_EXPORTS,AIM3_STATUS_DIR=$SMOKE_STATUS,AIM3_RUN_MODE=smoke" "$RUNNER")"
SMOKE_ID="${SMOKE_RAW%%;*}"
FORMAL_RAW="$(sbatch --parsable --chdir="$ROOT" --dependency="afterok:$SMOKE_ID" \
  --array="0-9%$MAX_CONCURRENT" \
  --output="$ARTIFACT_ROOT/%A_%a.out" --error="$ARTIFACT_ROOT/%A_%a.err" \
  --export="$COMMON_EXPORTS,AIM3_STATUS_DIR=$FORMAL_STATUS,AIM3_RUN_MODE=formal" "$RUNNER")"

printf 'SMOKE_JOB_ID=%s\n' "$SMOKE_ID"
printf 'FORMAL_ARRAY_JOB_ID=%s\n' "${FORMAL_RAW%%;*}"
printf 'RESULT_BASE=%s\n' "$RESULT_BASE"
printf 'TEST_BASE=%s\n' "$TEST_BASE"
printf 'ARTIFACT_ROOT=%s\n' "$ARTIFACT_ROOT"
