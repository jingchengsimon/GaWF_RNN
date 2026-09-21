#!/usr/bin/env bash
# Submit the forty-unit CM-MNIST nonlinearity-placement ablation array on Amarel.
#
# Batch 1 (tasks 0-29) removes the outer LayerNorm -> ReLU -> dropout wrap for six models;
# batch 2 (tasks 30-39) removes the nn.RNN activation for gawf and rnn. Seeds 1-5 per model,
# full concurrency (no array throttle), no preflight job and no aggregate job by explicit human
# request: this script requires --skip-smoke, records the waiver in the status directory, and the
# compute-node launcher refuses to run without that waiver file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
SKIP_SMOKE=0
MAX_CONCURRENT=0
while (( $# )); do
  case "$1" in
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent requires an integer}"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || {
  echo "--max-concurrent must be a non-negative integer (0 = full 40-way concurrency)" >&2; exit 2;
}

ARRAY_SPEC="0-39"
if (( MAX_CONCURRENT > 0 )); then
  ARRAY_SPEC="0-39%${MAX_CONCURRENT}"
fi
BATCH1_LEAF="clutter_ablate_outer_norm_40h_ep150_v1"
BATCH2_LEAF="clutter_ablate_inner_activation_40h_ep150_v1"

if (( DRY_RUN )); then
  printf 'submit: array=%s\n' "$ARRAY_SPEC"
  printf 'batch1 tasks 0-29: gawf_nowrap,rnn_nowrap,gru_nowrap,lstm_nowrap,mamba_nowrap,s5_nowrap x seeds 1-5 -> runs/%s\n' "$BATCH1_LEAF"
  printf 'batch2 tasks 30-39: gawf_notanh,rnn_notanh x seeds 1-5 -> runs/%s\n' "$BATCH2_LEAF"
  printf 'test export: data/analysis/clutter_ablation_reset_excluded_test_10seed_v1\n'
  printf 'preflight=skipped aggregate=skipped skip_smoke=%s max_concurrent=%s\n' \
    "$SKIP_SMOKE" "$MAX_CONCURRENT"
  exit 0
fi

(( SKIP_SMOKE )) || {
  echo "Refusing to submit: this campaign intentionally has no preflight gate, so --skip-smoke" >&2
  echo "is required and is recorded as the human-authorized waiver." >&2
  exit 2
}

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

ARTIFACT_ROOT="$ROOT/experiments/clutter/amarel/artifacts/clutter_nonlinearity_ablation_v1"
[[ ! -e "$ARTIFACT_ROOT" ]] || { echo "Artifact root already exists: $ARTIFACT_ROOT" >&2; exit 1; }
STATUS_DIR="$ARTIFACT_ROOT/status"
mkdir -p "$STATUS_DIR"

for leaf in "$BATCH1_LEAF" "$BATCH2_LEAF"; do
  target="$AIM3_RESULTS_PATH/data/clutter/runs/$leaf"
  [[ ! -e "$target" ]] || { echo "Refusing to overwrite existing result leaf: $target" >&2; exit 1; }
done
test_leaf="$AIM3_RESULTS_PATH/data/analysis/clutter_ablation_reset_excluded_test_10seed_v1"
[[ ! -e "$test_leaf" ]] || { echo "Refusing to overwrite existing test leaf: $test_leaf" >&2; exit 1; }

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
printf '%s\n' "$SOURCE_COMMIT" > "$STATUS_DIR/source_commit.txt"
printf 'waiver=smoke_gate_skipped\nauthorized_by=human_prompt\nauthorized_on=%s\nscope=batch1=30 units (6 models x seeds 1-5, outer wrap removed), batch2=10 units (gawf/rnn, nn.RNN activation removed)\npreflight=none\naggregate=none\narray=0-39 (full 40-way concurrency)\nnote=The human explicitly requested skipping the preflight and aggregate jobs on 2026-09-21.\n' \
  "$(date -Is)" > "$STATUS_DIR/smoke_waiver.txt"

EXPORTS="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",AIM3_CLUTTER_DATA_DIR=$AIM3_CLUTTER_DATA_DIR,AIM3_STATUS_DIR=$STATUS_DIR"
EXPORTS+=",AIM3_SOURCE_COMMIT=$SOURCE_COMMIT,AIM3_NUM_WORKERS=2,AIM3_PIN_MEMORY=1"

ARRAY_RAW="$(sbatch --parsable --chdir="$ROOT" --array="$ARRAY_SPEC" \
  --output="$ARTIFACT_ROOT/%A_%a.out" --error="$ARTIFACT_ROOT/%A_%a.err" \
  --export="$EXPORTS" "$SCRIPT_DIR/run_clutter_nonlinearity_ablation.sh")"
ARRAY_ID="${ARRAY_RAW%%;*}"

printf 'ARRAY_JOB_ID=%s\n' "$ARRAY_ID"
printf 'ARRAY_TASKS=%s\n' "$ARRAY_SPEC"
printf 'RESULT_LEAVES=%s %s\n' \
  "$AIM3_RESULTS_PATH/data/clutter/runs/$BATCH1_LEAF" \
  "$AIM3_RESULTS_PATH/data/clutter/runs/$BATCH2_LEAF"
printf 'TEST_LEAF=%s\n' "$test_leaf"
printf 'STATUS_DIR=%s\n' "$STATUS_DIR"
