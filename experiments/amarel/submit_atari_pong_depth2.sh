#!/usr/bin/env bash
# Submit the parameter-matched Pong depth-2 sweep safely through Slurm.
# No PyTorch model construction runs in this login-node submit process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

SEEDS_CSV=42
DRY_RUN=0
CONCURRENCY=10
FRAME_SKIP=1
AMP_DTYPE=bfloat16
while (( $# )); do
  case "$1" in
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --frame-skip) FRAME_SKIP="$2"; shift 2 ;;
    --amp-dtype) AMP_DTYPE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
MODELS=(ann rnn gru lstm gawf)
N_TASKS=$((${#MODELS[@]} * ${#SEEDS[@]} * 2))

if (( DRY_RUN )); then
  echo "param_match=sbatch layers=2 models=rnn,gru,lstm,gawf"
  for ((task=0; task<N_TASKS; task++)); do
    model="${MODELS[$((task % ${#MODELS[@]}))]}"
    rest=$((task / ${#MODELS[@]}))
    setting=$((rest / ${#SEEDS[@]}))
    seed="${SEEDS[$((rest % ${#SEEDS[@]}))]}"
    compile=0
    [[ "$model" == "ann" ]] && compile=1
    printf 'task=%d model=%s setting=%d seed=%s layers=2 frame_skip=%s amp=%s compile=%s\n' \
      "$task" "$model" "$setting" "$seed" "$FRAME_SKIP" "$AMP_DTYPE" "$compile"
  done
  echo "training_tasks=$N_TASKS concurrency=$CONCURRENCY dependency=afterok:param_match"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || {
  echo "AIM3_RESULTS_PATH must be absolute: $AIM3_RESULTS_PATH" >&2
  exit 2
}

MATCH_SCRIPT="$SCRIPT_DIR/run_atari_param_match.sh"
RUN_SCRIPT="$SCRIPT_DIR/run_atari_pong_depth2_array.sh"
MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match_depth2"
MATCH_JSON="$MATCH_DIR/atari_param_match.json"
MATCH_ART="$ROOT/experiments/amarel/artifacts/atari_param_match_depth2"
TRAIN_ART="$ROOT/experiments/amarel/artifacts/atari_pong_depth2"
mkdir -p "$MATCH_ART" "$TRAIN_ART"

DEPENDENCY_ARGS=()
MATCH_JOB_ID=""
if [[ -z "${SKIP_PARAM_MATCH:-}" ]]; then
  MATCH_RAW="$(sbatch --parsable \
    --chdir="$ROOT" \
    --output="$MATCH_ART/%j.out" \
    --error="$MATCH_ART/%j.err" \
    --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,PARAM_MATCH_NUM_LAYERS=2,PARAM_MATCH_MODELS=rnn:gru:lstm:gawf,PARAM_MATCH_REQUIRED=ann:rnn:gru:lstm:gawf,PARAM_MATCH_OUT_DIR=$MATCH_DIR,ARTIFACT_TAG=atari_param_match_depth2" \
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
required = {"ann", "rnn", "gru", "lstm", "gawf"}
assert required <= set(data["matched"])
assert data.get("candidate_num_layers") == 2
assert all(data["matched"][model].get("num_layers") == 2 for model in required)
PY
fi

ARRAY_RAW="$(sbatch --parsable \
  --array="0-$((N_TASKS - 1))%${CONCURRENCY}" \
  --chdir="$ROOT" \
  "${DEPENDENCY_ARGS[@]}" \
  --export="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,SEEDS_CSV=$SEEDS_CSV,FRAME_SKIP=$FRAME_SKIP,AMP_DTYPE=$AMP_DTYPE,ALLOW_TF32=1,COMPILE_MODEL=1,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1" \
  "$RUN_SCRIPT")"
ARRAY_JOB_ID="${ARRAY_RAW%%;*}"

echo "PARAM_MATCH_JOB_ID=$MATCH_JOB_ID"
echo "TRAINING_ARRAY_JOB_ID=$ARRAY_JOB_ID tasks=$N_TASKS concurrency=$CONCURRENCY"
