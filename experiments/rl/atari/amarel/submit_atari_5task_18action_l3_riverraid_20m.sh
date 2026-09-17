#!/usr/bin/env bash
# Submit the 20M Riverraid five-task full18 L3 run for five models and seeds 1-5.

set -euo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DRY_RUN=0
ARRAY_CONCURRENCY=8

while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --array-concurrency)
      (( $# >= 2 )) || { echo "--array-concurrency requires a value" >&2; exit 2; }
      ARRAY_CONCURRENCY="$2"
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "--array-concurrency must be a positive integer" >&2
  exit 2
}
: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
[[ "$AIM3_RESULTS_PATH" == /* ]] || { echo "AIM3_RESULTS_PATH must be absolute" >&2; exit 2; }

MATCH_JSON="${AIM3_MATCH_JSON:-$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/parameter_match/l3_full18/atari_param_match.json}"
RESULT_PARENT="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/formal_20m_4mpertask_riverraid_c_eps005_20260917"
RUNNER="$ROOT/experiments/rl/atari/amarel/run_atari_5task_18action_l3_riverraid_20m_array.sh"
ARTIFACT_TAG="atari_5task_18action_l3_formal_20m_riverraid_c_eps005_20260917_amarel72h"
ARTIFACT_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/$ARTIFACT_TAG"
RESULT_TAG="atari_dqn_5task_riverraid_fs4_stack4_l3_buf0p5m_eps5m_end0p05_lrpertask2m_20m"
ARRAY_TASKS="0-24"

if (( DRY_RUN )); then
  echo "protocol: five-task full18 L3; 20M global steps; 500K mmap replay per task"
  echo "units: gawf,ann,rnn,gru,lstm × seeds 1-5; array tasks ${ARRAY_TASKS}%${ARRAY_CONCURRENCY}"
  echo "LR: per-task 2M-step decay; epsilon: 1.0 to 0.05 over 5M global steps"
  echo "resources: 1 Ada Lovelace GPU, 16 CPUs, 64G, 72 hours, requeue enabled"
  echo "results: $RESULT_PARENT"
  exit 0
fi

[[ -x "$RUNNER" ]] || { echo "Missing executable runner: $RUNNER" >&2; exit 2; }
[[ -f "$MATCH_JSON" ]] || { echo "Missing parameter match: $MATCH_JSON" >&2; exit 2; }
mkdir -p "$ARTIFACT_ROOT/formal"

EXPORTS="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,MATCH_JSON=$MATCH_JSON"
EXPORTS+=",AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1,RESULT_PARENT=$RESULT_PARENT"
EXPORTS+=",SEED_COUNT=5,SEED_OFFSET=0,RUN_PHASE=formal,TOTAL_TIMESTEPS=20000000"
EXPORTS+=",CHECKPOINT_INTERVAL_STEPS=50000,LR_DECAY_STEP=0,LR_DECAY_PER_TASK_STEPS=2000000"
EXPORTS+=",LEARNING_STARTS_PER_TASK=20000,EXPLORATION_STEPS=5000000,END_EPSILON=0.05"
EXPORTS+=",REQUIRED_GIB=65.8"
EXPORTS+=",ARTIFACT_TAG=$ARTIFACT_TAG,RESULT_TAG=$RESULT_TAG"

ARRAY_RAW="$(sbatch --parsable --job-name=aim3-atari5-rnn20m \
  --array="${ARRAY_TASKS}%${ARRAY_CONCURRENCY}" --time=72:00:00 --constraint=adalovelace \
  --chdir="$ROOT" --output="$ARTIFACT_ROOT/formal/%A_%a.out" \
  --error="$ARTIFACT_ROOT/formal/%A_%a.err" --export="ALL,$EXPORTS" "$RUNNER")"
echo "FORMAL_JOB_ID=${ARRAY_RAW%%;*}"
echo "RESULT_ROOT=$RESULT_PARENT"
