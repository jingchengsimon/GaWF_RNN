#!/usr/bin/env bash
# Submit the strict 4-action Breakout frame-skip-4/frame-stack-4 L1 seed search:
# 7 models x {Breakout, Flickering Breakout} x 5 seeds = 70 tasks.
#
# Every task is recoverable: it checkpoints periodically, keeps its replay buffer
# on disk, and resumes after a preemption or timeout instead of restarting.
#
# Concurrency defaults to 12 because each running task holds a ~28 GB mmap replay
# and /scratch enforces a 1 TiB per-user soft quota. 12 x 28 GB leaves roughly
# 300 GiB of headroom over the ~372 GB already stored. Raising this risks SIGBUS
# crashes once the quota is hit, so change it only after re-checking
# `mmlsquota -u "$USER" scratch`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$ROOT"

DRY_RUN=0
SKIP_SMOKE=0
SMOKE_STEPS=25000
CONCURRENCY=12
CHECKPOINT_INTERVAL_STEPS=50000
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-steps) SMOKE_STEPS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --checkpoint-interval-steps) CHECKPOINT_INTERVAL_STEPS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

MODELS=(ann rnn gru lstm gawf s5 mamba)
SEEDS=(42 1 2 3 4)
TASKS=$((${#MODELS[@]} * ${#SEEDS[@]} * 2))
REPLAY_GB_PER_TASK=28

if (( CONCURRENCY > 15 )); then
  echo "Refusing concurrency=$CONCURRENCY: ${REPLAY_GB_PER_TASK} GB of replay per task" >&2
  echo "would exceed the 1 TiB scratch soft quota. Check mmlsquota first." >&2
  exit 2
fi

if (( DRY_RUN )); then
  echo "protocol: ALE/Breakout-v5, minimal 4-action, frame_skip=4, frame_stack=4"
  echo "settings: Breakout flicker=0.0; Flickering Breakout flicker=0.5"
  echo "seeds: ${SEEDS[*]}"
  echo "L1: models=${MODELS[*]} tasks=$TASKS array=0-$((TASKS - 1))%$CONCURRENCY"
  echo "recovery: checkpoint_interval_steps=$CHECKPOINT_INTERVAL_STEPS replay_backing=mmap --requeue"
  echo "peak replay footprint: $((CONCURRENCY * REPLAY_GB_PER_TASK)) GB across $CONCURRENCY concurrent tasks"
  echo "total_timesteps per task: 1000000"
  if (( SKIP_SMOKE )); then
    echo "smoke: skipped"
  else
    echo "smoke: GaWF task 4, $SMOKE_STEPS steps; the formal array depends on it"
  fi
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH, normally /scratch/js3269/results}"
if [[ "$AIM3_RESULTS_PATH" != /* ]]; then
  echo "AIM3_RESULTS_PATH must be an absolute path: $AIM3_RESULTS_PATH" >&2
  exit 2
fi

MATCH_DIR="$AIM3_RESULTS_PATH/atari_param_match_breakout_fs4_stack4_l1"
MATCH_JSON="$MATCH_DIR/atari_param_match.json"

RUNNER="$ROOT/experiments/amarel/run_atari_breakout_fs4_stack4_l1_array.sh"
MATCH_RUNNER="$ROOT/experiments/amarel/prepare_atari_breakout_fs4_stack4_match.sh"
ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_l1"
SMOKE_ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_smoke"
MATCH_ART="$ROOT/experiments/amarel/artifacts/atari_breakout_fs4_stack4_match"
mkdir -p "$ART" "$SMOKE_ART" "$MATCH_ART"

COMMON_EXPORT="AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"

MATCH_RAW="$(sbatch --parsable \
  --chdir="$ROOT" \
  --output="$MATCH_ART/%j.out" \
  --error="$MATCH_ART/%j.err" \
  --export="ALL,$COMMON_EXPORT,MATCH_DIR=$MATCH_DIR,NUM_ACTIONS=4" \
  "$MATCH_RUNNER")"
MATCH_JOB_ID="${MATCH_RAW%%;*}"

FORMAL_DEPENDENCY_ARGS=(--dependency="afterok:$MATCH_JOB_ID")
SMOKE_JOB_ID=""
if (( ! SKIP_SMOKE )); then
  SMOKE_RAW="$(sbatch --parsable \
    --job-name=aim3-breakout-fs4s4-l1-smoke \
    --array=4 \
    --time=01:00:00 \
    --chdir="$ROOT" \
    --output="$SMOKE_ART/%A_%a.out" \
    --error="$SMOKE_ART/%A_%a.err" \
    --dependency="afterok:$MATCH_JOB_ID" \
    --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,TOTAL_TIMESTEPS=$SMOKE_STEPS,CHECKPOINT_INTERVAL_STEPS=5000,REQUIRED_GB=1,RUN_TAG=breakout_fs4_stack4_l1_smoke,ARTIFACT_TAG=atari_breakout_fs4_stack4_smoke" \
    "$RUNNER")"
  SMOKE_JOB_ID="${SMOKE_RAW%%;*}"
  FORMAL_DEPENDENCY_ARGS=(--dependency="afterok:$SMOKE_JOB_ID")
fi

FORMAL_RAW="$(sbatch --parsable \
  --job-name=aim3-breakout-fs4s4-l1 \
  --array="0-$((TASKS - 1))%$CONCURRENCY" \
  --chdir="$ROOT" \
  --output="$ART/%A_%a.out" \
  --error="$ART/%A_%a.err" \
  "${FORMAL_DEPENDENCY_ARGS[@]}" \
  --export="ALL,$COMMON_EXPORT,MATCH_JSON=$MATCH_JSON,TOTAL_TIMESTEPS=1000000,CHECKPOINT_INTERVAL_STEPS=$CHECKPOINT_INTERVAL_STEPS,RUN_TAG=breakout_fs4_stack4_l1,ARTIFACT_TAG=atari_breakout_fs4_stack4_l1" \
  "$RUNNER")"
FORMAL_JOB_ID="${FORMAL_RAW%%;*}"

echo "MATCH_JOB_ID=$MATCH_JOB_ID"
echo "SMOKE_JOB_ID=$SMOKE_JOB_ID"
echo "FORMAL_JOB_ID=$FORMAL_JOB_ID tasks=$TASKS concurrency=$CONCURRENCY"
echo "peak_replay_gb=$((CONCURRENCY * REPLAY_GB_PER_TASK))"
