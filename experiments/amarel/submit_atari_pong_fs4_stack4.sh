#!/usr/bin/env bash
# Submit strict frame-skip-4/frame-stack-4 Pong and flickering-Pong sweeps.
# Formal arrays: L1 70 tasks + parameter-matched L2 50 tasks = 120 tasks.
# A compute-node match job fills the missing L1 Mamba size when needed. By
# default, one GaWF task per depth then runs as a smoke gate for both arrays.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
SKIP_SMOKE=0
SMOKE_STEPS=25000
L1_CONCURRENCY=20
L2_CONCURRENCY=10
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --l1-concurrency) L1_CONCURRENCY="$2"; shift 2 ;;
    --l2-concurrency) L2_CONCURRENCY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

L1_MODELS=(ann rnn gru lstm gawf s5 mamba)
L2_MODELS=(ann rnn gru lstm gawf)
SEEDS=(42 1 2 3 4)
L1_TASKS=$((${#L1_MODELS[@]} * ${#SEEDS[@]} * 2))
L2_TASKS=$((${#L2_MODELS[@]} * ${#SEEDS[@]} * 2))

if (( DRY_RUN )); then
  echo "protocol: ALE/Pong-v5, minimal 6-action, frame_skip=4, frame_stack=4"
  echo "settings: Pong flicker=0.0; Flickering Pong flicker=0.5"
  echo "seeds: ${SEEDS[*]}"
  echo "L1: models=${L1_MODELS[*]} tasks=$L1_TASKS array=0-$((L1_TASKS - 1))%$L1_CONCURRENCY"
  echo "L2: models=${L2_MODELS[*]} tasks=$L2_TASKS array=0-$((L2_TASKS - 1))%$L2_CONCURRENCY"
  echo "formal total: $((L1_TASKS + L2_TASKS)); total_timesteps per task: 1000000"
  if (( SKIP_SMOKE )); then
    echo "smoke: skipped"
  else
    echo "smoke: L1 GaWF task 4 and L2 GaWF task 4, $SMOKE_STEPS steps; formal arrays depend on both"
  fi
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
if [[ "$AIM3_RESULTS_PATH" != /* ]]; then
  echo "AIM3_RESULTS_PATH must be an absolute path: $AIM3_RESULTS_PATH" >&2
  exit 2
fi

L1_MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match_fs4_stack4_l1"
L1_MATCH_JSON="$L1_MATCH_DIR/atari_param_match.json"
L2_MATCH_JSON="$AIM3_RESULTS_PATH/atari_param_match_depth2/atari_param_match.json"
BASE_L1_MATCH_JSON="$AIM3_RESULTS_PATH/atari_param_match/atari_param_match.json"
[[ -f "$BASE_L1_MATCH_JSON" ]] || { echo "Missing base L1 match JSON: $BASE_L1_MATCH_JSON" >&2; exit 2; }
[[ -f "$L2_MATCH_JSON" ]] || { echo "Missing L2 match JSON: $L2_MATCH_JSON" >&2; exit 2; }

L1_RUNNER="$ROOT/experiments/amarel/run_atari_pong_fs4_stack4_l1_array.sh"
L2_RUNNER="$ROOT/experiments/amarel/run_atari_pong_fs4_stack4_l2_array.sh"
MATCH_RUNNER="$ROOT/experiments/amarel/prepare_atari_pong_fs4_stack4_match.sh"
L1_ART="$ROOT/experiments/amarel/artifacts/atari_pong_fs4_stack4_l1"
L2_ART="$ROOT/experiments/amarel/artifacts/atari_pong_fs4_stack4_l2"
SMOKE_ART="$ROOT/experiments/amarel/artifacts/atari_pong_fs4_stack4_smoke"
MATCH_ART="$ROOT/experiments/amarel/artifacts/atari_pong_fs4_stack4_match"
mkdir -p "$L1_ART" "$L2_ART" "$SMOKE_ART" "$MATCH_ART"

COMMON_EXPORT="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
MATCH_RAW="$(sbatch --parsable \
  --chdir="$ROOT" \
  --output="$MATCH_ART/%j.out" \
  --error="$MATCH_ART/%j.err" \
  --export="ALL,$COMMON_EXPORT,BASE_MATCH_JSON=$BASE_L1_MATCH_JSON,MATCH_DIR=$L1_MATCH_DIR,L2_MATCH_JSON=$L2_MATCH_JSON" \
  "$MATCH_RUNNER")"
MATCH_JOB_ID="${MATCH_RAW%%;*}"

SMOKE_DEPENDENCY_ARGS=(--dependency="afterok:$MATCH_JOB_ID")
FORMAL_DEPENDENCY_ARGS=()
SMOKE_L1_JOB_ID=""
SMOKE_L2_JOB_ID=""
if (( ! SKIP_SMOKE )); then
  SMOKE_L1_RAW="$(sbatch --parsable \
    --job-name=aim3-pong-fs4s4-l1-smoke \
    --array=4 \
    --time=01:00:00 \
    --chdir="$ROOT" \
    --output="$SMOKE_ART/l1_%A_%a.out" \
    --error="$SMOKE_ART/l1_%A_%a.err" \
    "${SMOKE_DEPENDENCY_ARGS[@]}" \
    --export="ALL,$COMMON_EXPORT,MATCH_JSON=$L1_MATCH_JSON,TOTAL_TIMESTEPS=$SMOKE_STEPS,RUN_TAG=pong_fs4_stack4_l1_smoke,ARTIFACT_TAG=atari_pong_fs4_stack4_smoke_l1" \
    "$L1_RUNNER")"
  SMOKE_L1_JOB_ID="${SMOKE_L1_RAW%%;*}"
  SMOKE_L2_RAW="$(sbatch --parsable \
    --job-name=aim3-pong-fs4s4-l2-smoke \
    --array=4 \
    --time=01:00:00 \
    --chdir="$ROOT" \
    --output="$SMOKE_ART/l2_%A_%a.out" \
    --error="$SMOKE_ART/l2_%A_%a.err" \
    "${SMOKE_DEPENDENCY_ARGS[@]}" \
    --export="ALL,$COMMON_EXPORT,MATCH_JSON=$L2_MATCH_JSON,TOTAL_TIMESTEPS=$SMOKE_STEPS,RUN_TAG=pong_fs4_stack4_l2match_smoke,ARTIFACT_TAG=atari_pong_fs4_stack4_smoke_l2" \
    "$L2_RUNNER")"
  SMOKE_L2_JOB_ID="${SMOKE_L2_RAW%%;*}"
  FORMAL_DEPENDENCY_ARGS=(--dependency="afterok:$SMOKE_L1_JOB_ID:$SMOKE_L2_JOB_ID")
else
  FORMAL_DEPENDENCY_ARGS=(--dependency="afterok:$MATCH_JOB_ID")
fi

L1_RAW="$(sbatch --parsable \
  --job-name=aim3-pong-fs4s4-l1 \
  --array="0-$((L1_TASKS - 1))%$L1_CONCURRENCY" \
  --chdir="$ROOT" \
  --output="$L1_ART/%A_%a.out" \
  --error="$L1_ART/%A_%a.err" \
  "${FORMAL_DEPENDENCY_ARGS[@]}" \
  --export="ALL,$COMMON_EXPORT,MATCH_JSON=$L1_MATCH_JSON,TOTAL_TIMESTEPS=1000000,RUN_TAG=pong_fs4_stack4_l1,ARTIFACT_TAG=atari_pong_fs4_stack4_l1" \
  "$L1_RUNNER")"
L1_JOB_ID="${L1_RAW%%;*}"

L2_RAW="$(sbatch --parsable \
  --job-name=aim3-pong-fs4s4-l2 \
  --array="0-$((L2_TASKS - 1))%$L2_CONCURRENCY" \
  --chdir="$ROOT" \
  --output="$L2_ART/%A_%a.out" \
  --error="$L2_ART/%A_%a.err" \
  "${FORMAL_DEPENDENCY_ARGS[@]}" \
  --export="ALL,$COMMON_EXPORT,MATCH_JSON=$L2_MATCH_JSON,TOTAL_TIMESTEPS=1000000,RUN_TAG=pong_fs4_stack4_l2match,ARTIFACT_TAG=atari_pong_fs4_stack4_l2" \
  "$L2_RUNNER")"
L2_JOB_ID="${L2_RAW%%;*}"

echo "MATCH_JOB_ID=$MATCH_JOB_ID"
echo "SMOKE_L1_JOB_ID=$SMOKE_L1_JOB_ID"
echo "SMOKE_L2_JOB_ID=$SMOKE_L2_JOB_ID"
echo "FORMAL_L1_JOB_ID=$L1_JOB_ID tasks=$L1_TASKS"
echo "FORMAL_L2_JOB_ID=$L2_JOB_ID tasks=$L2_TASKS"
if (( ! SKIP_SMOKE )); then
  echo "FORMAL_DEPENDENCY=afterok:$SMOKE_L1_JOB_ID:$SMOKE_L2_JOB_ID"
elif [[ -n "$MATCH_JOB_ID" ]]; then
  echo "FORMAL_DEPENDENCY=afterok:$MATCH_JOB_ID"
fi
