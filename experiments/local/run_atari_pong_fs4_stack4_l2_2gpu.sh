#!/usr/bin/env bash
# Run the strict 6-action fs4/stack4 depth-2 Pong sweep on two local GPUs.
# Five models x Pong/Flickering Pong x five seeds = 50 formal units.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$ROOT/experiments/amarel/run_atari_pong_fs4_stack4_l2_array.sh"
RUN_ID="${AIM3_RUN_ID:-sjc-pong-fs4s4-l2-5seed}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-$ROOT/results}"
MATCH_JSON="${MATCH_JSON:-$RESULTS_ROOT/atari_param_match_depth2/atari_param_match.json}"
RESULT_TAG="${RESULT_TAG:-pong_fs4_stack4_l2match_sjc5seed}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_pong_fs4_stack4_l2_sjc5seed}"
SMOKE_TAG="${SMOKE_TAG:-pong_fs4_stack4_l2match_sjc5seed_smoke}"
SMOKE_ARTIFACT_TAG="${SMOKE_ARTIFACT_TAG:-atari_pong_fs4_stack4_l2_sjc5seed_smoke}"
LOG_DIR="$ROOT/experiments/remote/artifacts/$RUN_ID"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
SMOKE_STEPS="${SMOKE_STEPS:-25000}"
RUN_SMOKE="${RUN_SMOKE:-1}"
MODELS=(ann rnn gru lstm gawf)
SEEDS=(42 1 2 3 4)
# Longest estimated units first; the two workers atomically claim from this
# shared order so neither GPU is pinned to a particular model or setting.
MODEL_ORDER=(4 2 1 3 0)
TASK_ORDER=()

for model_idx in "${MODEL_ORDER[@]}"; do
  for setting in 0 1; do
    for seed_idx in 0 1 2 3 4; do
      TASK_ORDER+=($((model_idx + 5 * (seed_idx + 5 * setting))))
    done
  done
done

unit_meta() {
  local task="$1"
  local model_idx=$((task % 5))
  local rest=$((task / 5))
  local setting=$((rest / 5))
  local seed_idx=$((rest % 5))
  UNIT_MODEL="${MODELS[$model_idx]}"
  UNIT_SETTING="$setting"
  UNIT_SEED="${SEEDS[$seed_idx]}"
  if (( setting == 0 )); then
    UNIT_SUFFIX="atari_dqn_${RESULT_TAG}_${UNIT_MODEL}_seed${UNIT_SEED}"
  else
    UNIT_SUFFIX="atari_dqn_${RESULT_TAG}_flicker_${UNIT_MODEL}_seed${UNIT_SEED}"
  fi
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for task in "${TASK_ORDER[@]}"; do
    unit_meta "$task"
    echo "unit task=$task model=$UNIT_MODEL setting=$UNIT_SETTING seed=$UNIT_SEED suffix=$UNIT_SUFFIX"
  done
  echo "total=${#TASK_ORDER[@]} gpus=2 scheduling=dynamic-longest-first"
  exit 0
fi

if [[ ! -f "$ROOT/train_atari_dqn.py" || ! -f "$RUNNER" ]]; then
  echo "Invalid AIM3_ROOT=$ROOT or missing L2 runner" >&2
  exit 2
fi
if [[ ! -f "$MATCH_JSON" ]]; then
  echo "Missing L2 parameter-match JSON: $MATCH_JSON" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
set +u
source "${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export AIM3_CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
export AIM3_RESULTS_PATH="$RESULTS_ROOT"
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

STATUS_DIR="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG/status"
SMOKE_STATUS_DIR="$ROOT/experiments/amarel/artifacts/$SMOKE_ARTIFACT_TAG/status"
CLAIM_DIR="$LOG_DIR/claims"
mkdir -p "$STATUS_DIR" "$SMOKE_STATUS_DIR" "$CLAIM_DIR"

{
  echo "[$(date -Is)] run_id=$RUN_ID root=$ROOT results=$RESULTS_ROOT"
  echo "protocol=minimal6 fs4_stack4 layers=2 timesteps=$TOTAL_TIMESTEPS"
  echo "tasks=${#TASK_ORDER[@]} gpus=0,1 scheduling=dynamic-longest-first"
} > "$LOG_DIR/orchestrator.log"

if [[ "$RUN_SMOKE" == "1" ]]; then
  SMOKE_SUFFIX="atari_dqn_${SMOKE_TAG}_gawf_seed42"
  if [[ -f "$SMOKE_STATUS_DIR/${SMOKE_SUFFIX}.done" ]]; then
    echo "[$(date -Is)] smoke already valid; skipping $SMOKE_SUFFIX" >> "$LOG_DIR/orchestrator.log"
  else
    echo "[$(date -Is)] smoke start gpu=0 task=4 model=gawf steps=$SMOKE_STEPS" \
      >> "$LOG_DIR/orchestrator.log"
    CUDA_VISIBLE_DEVICES=0 \
    SLURM_ARRAY_TASK_ID=4 \
    AIM3_ROOT="$ROOT" \
    MATCH_JSON="$MATCH_JSON" \
    TOTAL_TIMESTEPS="$SMOKE_STEPS" \
    RUN_TAG="$SMOKE_TAG" \
    ARTIFACT_TAG="$SMOKE_ARTIFACT_TAG" \
    bash "$RUNNER" > "$LOG_DIR/smoke.out" 2> "$LOG_DIR/smoke.err"
    echo "[$(date -Is)] smoke passed" >> "$LOG_DIR/orchestrator.log"
  fi
fi

run_task() {
  local gpu="$1"
  local task="$2"
  unit_meta "$task"
  local log_stem="$LOG_DIR/task${task}_${UNIT_MODEL}_setting${UNIT_SETTING}_seed${UNIT_SEED}"
  {
    echo "[$(date -Is)] start gpu=$gpu task=$task model=$UNIT_MODEL setting=$UNIT_SETTING seed=$UNIT_SEED"
  } >> "$LOG_DIR/orchestrator.log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" \
  SLURM_ARRAY_TASK_ID="$task" \
  AIM3_ROOT="$ROOT" \
  MATCH_JSON="$MATCH_JSON" \
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  RUN_TAG="$RESULT_TAG" \
  ARTIFACT_TAG="$ARTIFACT_TAG" \
  bash "$RUNNER" > "${log_stem}.out" 2> "${log_stem}.err"
  local rc=$?
  set -e
  echo "[$(date -Is)] finish gpu=$gpu task=$task model=$UNIT_MODEL rc=$rc" \
    >> "$LOG_DIR/orchestrator.log"
  return "$rc"
}

run_worker() {
  local gpu="$1"
  local failed=0
  local task
  for task in "${TASK_ORDER[@]}"; do
    unit_meta "$task"
    if [[ -f "$STATUS_DIR/${UNIT_SUFFIX}.done" ]]; then
      echo "[$(date -Is)] skip-valid gpu=$gpu task=$task suffix=$UNIT_SUFFIX" \
        >> "$LOG_DIR/orchestrator.log"
      continue
    fi
    if mkdir "$CLAIM_DIR/task$task" 2>/dev/null; then
      echo "gpu=$gpu pid=$BASHPID claimed_at=$(date -Is)" > "$CLAIM_DIR/task$task/owner"
      run_task "$gpu" "$task" || failed=1
    fi
  done
  return "$failed"
}

run_worker 0 &
WORKER0=$!
run_worker 1 &
WORKER1=$!
echo "worker0_pid=$WORKER0 worker1_pid=$WORKER1" >> "$LOG_DIR/orchestrator.log"

OVERALL=0
wait "$WORKER0" || OVERALL=1
wait "$WORKER1" || OVERALL=1

DONE_COUNT="$(find "$STATUS_DIR" -maxdepth 1 -type f -name "atari_dqn_${RESULT_TAG}_*.done" | wc -l | tr -d ' ')"
FAIL_COUNT="$(find "$STATUS_DIR" -maxdepth 1 -type f -name "atari_dqn_${RESULT_TAG}_*.fail" | wc -l | tr -d ' ')"
echo "[$(date -Is)] overall=$OVERALL done=$DONE_COUNT/50 fail=$FAIL_COUNT" \
  >> "$LOG_DIR/orchestrator.log"
if [[ "$OVERALL" -eq 0 && "$DONE_COUNT" -eq 50 ]]; then
  touch "$LOG_DIR/all.done"
else
  touch "$LOG_DIR/incomplete.fail"
  exit 1
fi
