#!/usr/bin/env bash
# Submit the strict Pong fs1/stack1 sweep without running PyTorch on a login node.
#
# The parameter match is a separate Slurm compute job. The 70-task training
# array starts only after that job succeeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_atari_pong_fs1_stack1_array.sh"
MATCH_SCRIPT="$SCRIPT_DIR/run_atari_param_match.sh"
ART_ROOT="$ROOT/experiments/amarel/artifacts/atari_pong_fs1_stack1"
MATCH_ART="$ROOT/experiments/amarel/artifacts/atari_param_match_fs1_stack1"
ARRAY_SPEC="0-69"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-20}"
DRY_RUN=0
while (( $# )); do
  case "$1" in
    --array) ARRAY_SPEC="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( DRY_RUN )); then
  echo "param_match=sbatch layers=1 models=rnn,gru,lstm,gawf,s5,mamba"
  echo "training=sbatch array=${ARRAY_SPEC}%${ARRAY_CONCURRENCY} dependency=afterok:param_match"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || {
  echo "AIM3_RESULTS_PATH must be absolute: $AIM3_RESULTS_PATH" >&2
  exit 2
}
mkdir -p "$ART_ROOT" "$MATCH_ART"

MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match"
MATCH_JSON="$MATCH_DIR/atari_param_match.json"
DEPENDENCY_ARGS=()
MATCH_JOB_ID=""
if [[ -z "${SKIP_PARAM_MATCH:-}" ]]; then
  MATCH_RAW="$(sbatch --parsable \
    --chdir="$ROOT" \
    --output="$MATCH_ART/%j.out" \
    --error="$MATCH_ART/%j.err" \
    --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,PARAM_MATCH_NUM_LAYERS=1,PARAM_MATCH_MODELS=rnn:gru:lstm:gawf:s5:mamba,PARAM_MATCH_REQUIRED=ann:rnn:gru:lstm:gawf:s5:mamba,PARAM_MATCH_OUT_DIR=$MATCH_DIR,ARTIFACT_TAG=atari_param_match_fs1_stack1" \
    "$MATCH_SCRIPT")"
  MATCH_JOB_ID="${MATCH_RAW%%;*}"
  DEPENDENCY_ARGS=(--dependency="afterok:$MATCH_JOB_ID")
else
  [[ -f "$MATCH_JSON" ]] || { echo "Missing $MATCH_JSON" >&2; exit 2; }
  /usr/bin/python3 - "$MATCH_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
required = {"ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba"}
assert required <= set(data["matched"])
assert data.get("candidate_num_layers") == 1
PY
fi

ARRAY_RAW="$(sbatch --parsable \
  --array="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}" \
  --chdir="$ROOT" \
  "${DEPENDENCY_ARGS[@]}" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}" \
  "$RUN_SCRIPT")"
ARRAY_JOB_ID="${ARRAY_RAW%%;*}"

echo "PARAM_MATCH_JOB_ID=$MATCH_JOB_ID"
echo "TRAINING_ARRAY_JOB_ID=$ARRAY_JOB_ID array=${ARRAY_SPEC}%${ARRAY_CONCURRENCY}"
