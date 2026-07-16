#!/usr/bin/env bash
# Run seven models for strict fs1/stack1 and fs4/stack4 plain Pong on two GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$ROOT/experiments/amarel/run_atari_pong_fs1_stack1_array.sh"
RUN_ID="${AIM3_RUN_ID:-sjc-pong-fscompare1m}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_pong_fs1_stack1_vs_fs4_stack4_1m_sjc}"
LOG_DIR="$ROOT/experiments/remote/artifacts/$RUN_ID"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
MODELS=(ann rnn gru lstm gawf s5 mamba)
TASK_ORDER=(4 5 3 2 1 6 0)

if [[ ! -f "$ROOT/train_atari_dqn.py" || ! -f "$RUNNER" ]]; then
  echo "Invalid AIM3_ROOT=$ROOT" >&2
  exit 2
fi

print_units() {
  local gpu="$1"
  local frame_skip="$2"
  local frame_stack="$3"
  local task model
  for task in "${TASK_ORDER[@]}"; do
    model="${MODELS[$task]}"
    echo "unit gpu=$gpu protocol=fs${frame_skip}_stack${frame_stack} task=$task model=$model"
  done
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  print_units 0 1 1
  print_units 1 4 4
  exit 0
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"
set +u
source "${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export AIM3_CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

run_task() {
  local gpu="$1"
  local frame_skip="$2"
  local frame_stack="$3"
  local task="$4"
  local model="${MODELS[$task]}"
  local protocol="fs${frame_skip}_stack${frame_stack}"
  local log_stem="$LOG_DIR/${protocol}_task${task}_${model}"
  {
    echo "[$(date -Is)] start gpu=$gpu protocol=$protocol task=$task model=$model"
    echo "AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
  } >> "$LOG_DIR/orchestrator.log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" \
  SLURM_ARRAY_TASK_ID="$task" \
  AIM3_ROOT="$ROOT" \
  FRAME_SKIP="$frame_skip" \
  FRAME_STACK="$frame_stack" \
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  SEEDS_OVERRIDE=42 \
  RESULT_TAG="pong_${protocol}_1m" \
  ARTIFACT_TAG="$ARTIFACT_TAG" \
  bash "$RUNNER" > "$log_stem.out" 2> "$log_stem.err"
  local rc=$?
  set -e
  echo "[$(date -Is)] finish gpu=$gpu protocol=$protocol task=$task model=$model rc=$rc" \
    >> "$LOG_DIR/orchestrator.log"
  return "$rc"
}

run_protocol() {
  local gpu="$1"
  local frame_skip="$2"
  local frame_stack="$3"
  local failed=0
  local task
  for task in "${TASK_ORDER[@]}"; do
    run_task "$gpu" "$frame_skip" "$frame_stack" "$task" || failed=1
  done
  return "$failed"
}

{
  echo "[$(date -Is)] run_id=$RUN_ID root=$ROOT total_timesteps=$TOTAL_TIMESTEPS"
  echo "gpu0=fs1_stack1 gpu1=fs4_stack4 task_order=${TASK_ORDER[*]}"
} > "$LOG_DIR/orchestrator.log"

run_protocol 0 1 1 &
worker0=$!
run_protocol 1 4 4 &
worker1=$!
echo "worker0_pid=$worker0 worker1_pid=$worker1" >> "$LOG_DIR/orchestrator.log"

overall=0
wait "$worker0" || overall=1
wait "$worker1" || overall=1

STATUS_DIR="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG/status"
done_count="$(find "$STATUS_DIR" -maxdepth 1 -type f -name '*.done' 2>/dev/null | wc -l)"
echo "[$(date -Is)] overall=$overall done_count=$done_count/14" \
  >> "$LOG_DIR/orchestrator.log"
if [[ "$overall" -eq 0 && "$done_count" -eq 14 ]]; then
  touch "$LOG_DIR/all.done"
else
  touch "$LOG_DIR/incomplete.fail"
  exit 1
fi
