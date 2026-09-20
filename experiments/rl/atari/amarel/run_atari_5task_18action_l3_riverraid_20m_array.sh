#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-5task-l3
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@600

# Run one recoverable Riverraid five-task/full-18 Atari L3 model and seed.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
cd "$ROOT"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${MATCH_JSON:?MATCH_JSON is required}"
: "${RESULT_PARENT:?RESULT_PARENT is required}"
: "${SEED_COUNT:?SEED_COUNT is required}"
: "${RUN_PHASE:?RUN_PHASE is required}"
[[ -f "$MATCH_JSON" ]] || { echo "Missing L3/full18 match table: $MATCH_JSON" >&2; exit 2; }
(( SEED_COUNT >= 1 && SEED_COUNT <= 5 )) || { echo "SEED_COUNT must be 1..5" >&2; exit 2; }

SEED_OFFSET="${SEED_OFFSET:-0}"
[[ "$SEED_OFFSET" =~ ^(0|[1-9][0-9]*)$ ]] || {
  echo "SEED_OFFSET must be a non-negative integer" >&2
  exit 2
}
SEED_OFFSET_DECIMAL=$((10#$SEED_OFFSET))
SEEDS=()
for (( seed=1; seed<=SEED_COUNT; seed++ )); do
  SEEDS+=("$((seed + SEED_OFFSET_DECIMAL))")
done

MODELS=(gawf ann rnn gru lstm)
ENV_IDS=(ALE/Pong-v5 ALE/Breakout-v5 ALE/Assault-v5 ALE/Seaquest-v5 ALE/Riverraid-v5)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
N_SEEDS=${#SEEDS[@]}
N_TASKS=$((${#MODELS[@]} * N_SEEDS))
if (( TASK_ID < 0 || TASK_ID >= N_TASKS )); then
  echo "task $TASK_ID outside valid range 0..$((N_TASKS - 1))" >&2
  exit 2
fi
CANONICAL_TASK_ID="$TASK_ID"
if [[ "${GAWF_FIRST_SCHEDULING:-0}" == "1" && "$N_TASKS" -eq 15 ]]; then
  if (( SEED_COUNT != 3 || N_TASKS != 15 )); then
    echo "GAWF_FIRST_SCHEDULING requires the 5-model x 3-seed pilot" >&2
    exit 2
  fi
  # Slurm normalizes an array specification to ascending numeric task IDs.
  # Remap the first three array ordinals to the three slow GaWF seeds, then
  # run the twelve non-GaWF units in their ordinary canonical order.
  GAWF_FIRST_ORDER=(12 13 14 0 1 2 3 4 5 6 7 8 9 10 11)
  CANONICAL_TASK_ID="${GAWF_FIRST_ORDER[$TASK_ID]}"
fi
MODEL="${MODELS[$((CANONICAL_TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((CANONICAL_TASK_ID % N_SEEDS))]}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:?TOTAL_TIMESTEPS is required}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-50000}"
LR_DECAY_STEP="${LR_DECAY_STEP:-1000000}"
LR_DECAY_PER_TASK_STEPS="${LR_DECAY_PER_TASK_STEPS:-0}"
LEARNING_STARTS_PER_TASK="${LEARNING_STARTS_PER_TASK:-20000}"
EXPLORATION_STEPS="${EXPLORATION_STEPS:-500000}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_5task_18action_l3_per_task_buf500k_${RUN_PHASE}}"
RESULT_TAG="${RESULT_TAG:-atari_dqn_5task_fs4_stack4_l3_buf0p5m_lrdecay1m_${RUN_PHASE}}"
[[ "$RESULT_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "RESULT_TAG contains unsafe characters: $RESULT_TAG" >&2
  exit 2
}
[[ "$EXPLORATION_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPLORATION_STEPS must be a positive integer: $EXPLORATION_STEPS" >&2
  exit 2
}

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

ART_ROOT="$ROOT/experiments/rl/atari/amarel/artifacts/$ARTIFACT_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$STATUS_DIR" "$RESULT_PARENT"
SUFFIX="${RESULT_TAG}_${MODEL}_seed${SEED}"
RESULT_DIR="$RESULT_PARENT/$SUFFIX"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"
DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"

python -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" --filesystem scratch \
  --required_gib "${REQUIRED_GIB:-27}" --headroom_factor "${QUOTA_HEADROOM_FACTOR:-2}" \
  --marker_path "$STATUS_DIR/${SUFFIX}.quota"

HIDDEN="$(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    match = json.load(handle)
model = sys.argv[2]
expected = {
    "anchor": "lstm",
    "anchor_num_layers": 1,
    "hidden_size": 512,
    "candidate_num_layers": 3,
    "num_actions": 18,
}
actual = {key: match.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid full18 L3 match table: expected={expected}, actual={actual}")
entry = match["matched"][model]
if entry.get("num_layers") != 3:
    raise RuntimeError(f"Unexpected layer count for {model}: {entry}")
print(entry["hidden_size"])
PY
)"

RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
  echo "[$(date -Is)] resuming from $CHECKPOINT"
elif [[ ! -f "$DONE_FILE" && \
  ( -f "$RESULT_DIR/metrics_history.jsonl" || -f "$RESULT_DIR/metrics.json" ) ]]; then
  echo "Refusing partial results without a resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer)
if [[ "$MODEL" == "ann" ]]; then
  ACCEL_ARGS+=(--compile_model)
fi

COMMAND=(python run_task.py atari-dqn \
  --env_ids "${ENV_IDS[@]}" --action_space_mode full18 --model_type "$MODEL" \
  --num_layers 3 --hidden_size "$HIDDEN" --gawf_feedback_lr_scale 1.0 \
  --frame_skip 4 --frame_stack 4 --flicker_prob 0.0 --total_timesteps "$TOTAL_TIMESTEPS" \
  --task_schedule transition_balanced --replay_sampling task_balanced \
  --learning_starts 20000 --learning_starts_per_task "$LEARNING_STARTS_PER_TASK" \
  --batch_size 32 --seq_len 16 --sequences_per_batch 8 \
  --exploration_steps "$EXPLORATION_STEPS" \
  --seed "$SEED" --device cuda --result_suffix "$SUFFIX" --save_dir "$RESULT_DIR" \
  --replay_backing mmap --replay_layout per_task --buffer_size 500000 \
  --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
  --learning_rate_decay_step "$LR_DECAY_STEP" \
  --learning_rate_decay_per_task_steps "$LR_DECAY_PER_TASK_STEPS" \
  --learning_rate_decay_scale 0.1)
if (( ${#RESUME_ARGS[@]} > 0 )); then
  COMMAND+=("${RESUME_ARGS[@]}")
fi
COMMAND+=("${ACCEL_ARGS[@]}")

set +e
DISABLE_TQDM=1 "${COMMAND[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  echo "status=train_failed task_id=$TASK_ID model=$MODEL seed=$SEED exit_code=$TRAIN_RC" \
    > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

if [[ ! -f "$RESULT_DIR/metrics.json" ]]; then
  echo "status=paused task_id=$TASK_ID checkpoint=$CHECKPOINT" \
    > "$STATUS_DIR/${SUFFIX}.paused"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
  exit 0
fi

python - "$RESULT_DIR" "$TOTAL_TIMESTEPS" <<'PY'
import glob
import json
import math
import os
import sys

result_dir, total_steps = sys.argv[1:]
with open(os.path.join(result_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
if int(metrics.get("global_step", -1)) < int(total_steps):
    raise RuntimeError(
        f"Training did not reach the requested total steps: {metrics.get('global_step')} < {total_steps}"
    )
if not math.isfinite(float(metrics.get("loss", float("nan")))):
    raise RuntimeError("Final loss is not finite")
if not os.path.isfile(os.path.join(result_dir, "metrics_history.jsonl")):
    raise RuntimeError("Missing metrics history")
if not glob.glob(os.path.join(result_dir, "*.pth")):
    raise RuntimeError("Missing final or resumable checkpoint")
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL seed=$SEED phase=$RUN_PHASE"
  echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE" "$STATUS_DIR/${SUFFIX}.quota" "$STATUS_DIR/${SUFFIX}.paused"
