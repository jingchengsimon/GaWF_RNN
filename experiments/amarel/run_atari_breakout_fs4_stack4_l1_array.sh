#!/usr/bin/env bash
#SBATCH --job-name=aim3-breakout-fs4s4-l1
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=experiments/amarel/artifacts/atari_breakout_fs4_stack4_l1/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_breakout_fs4_stack4_l1/%A_%a.err

# Strict 4-action Breakout sweep:
# 7 models x 2 observation settings x 5 seeds = 70 tasks.
# The protocol is fixed to frame_skip=4 and frame_stack=4, matching Pong L1.
#
# Unlike the Pong launchers this one is recoverable: each unit checkpoints every
# CHECKPOINT_INTERVAL_STEPS and keeps its replay buffer in <result>/replay, so a
# preempted or timed-out unit resumes instead of restarting from step 0.
#
# Submit with a concurrency cap; see the quota guard below for why:
#   sbatch --array=0-69%8 experiments/amarel/run_atari_breakout_fs4_stack4_l1_array.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
: "${MATCH_JSON:?MATCH_JSON must point to the Breakout L1 parameter-match JSON}"
[[ -f "$MATCH_JSON" ]] || { echo "Missing parameter match JSON: $MATCH_JSON" >&2; exit 2; }

FRAME_SKIP=4
FRAME_STACK=4
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
SEQ_LEN="${SEQ_LEN:-16}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-50000}"
RUN_TAG="${RUN_TAG:-breakout_fs4_stack4_l1}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_breakout_fs4_stack4_l1}"
if [[ "$RUN_TAG" != *"fs4_stack4"* ]]; then
  echo "RUN_TAG must include fs4_stack4: $RUN_TAG" >&2
  exit 2
fi

ART_ROOT="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

MODELS=(ann rnn gru lstm gawf s5 mamba)
SEEDS=(42 1 2 3 4)
N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
N_TASKS=$((N_MODELS * N_SEEDS * 2))
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= N_TASKS )); then
  echo "task $TASK_ID outside valid range 0..$((N_TASKS - 1))" >&2
  exit 2
fi

MODEL="${MODELS[$((TASK_ID % N_MODELS))]}"
REST=$((TASK_ID / N_MODELS))
SETTING=$((REST / N_SEEDS))
SEED="${SEEDS[$((REST % N_SEEDS))]}"
if (( SETTING == 0 )); then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_${RUN_TAG}_${MODEL}_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_${RUN_TAG}_flicker_${MODEL}_seed${SEED}"
fi

SIZE_ARGS=()
if [[ "$MODEL" != "ann" ]]; then
  read -r KIND V1 V2 < <(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    entry = json.load(handle)["matched"].get(sys.argv[2], {})
if "hidden_size" in entry:
    print("hidden", entry["hidden_size"], "")
elif "d_model" in entry:
    print("ssm", entry["d_model"], entry.get("state_size", 128))
else:
    print("none", "", "")
PY
)
  if [[ "$KIND" == "hidden" ]]; then
    SIZE_ARGS=(--hidden_size "$V1")
  elif [[ "$KIND" == "ssm" ]]; then
    SIZE_ARGS=(--ssm_d_model "$V1" --ssm_state_size "$V2")
  else
    echo "No matched sizing for model=$MODEL in $MATCH_JSON" >&2
    exit 2
  fi
fi

ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer)
FUSED_EXPECTED=true
if [[ "$MODEL" == "s5" ]]; then
  FUSED_EXPECTED=false
fi
COMPILE_EXPECTED=false
if [[ "$MODEL" == "ann" || "$MODEL" == "gawf" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_EXPECTED=true
fi

RESULT_DIR="$AIM3_RESULTS_PATH/train_data/$SUFFIX"
DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
CHECKPOINT="$RESULT_DIR/checkpoint.pth"

# Quota guard; see experiments/amarel/scratch_quota_guard.py for why running out
# of quota under an mmap replay is a SIGBUS, not a clean ENOSPC, and why the two
# GPFS clusters serving /scratch report different headroom for the same data.
# Headroom factor 2, not 3: the %8 array cap already bounds how many units hold a
# replay at once, so demanding three units of slack would block legitimate starts
# once the array reaches steady state (~68 GiB free with 8 buffers mapped).
REQUIRED_GIB="${REQUIRED_GIB:-27}"
set +e
python -m experiments.amarel.scratch_quota_guard \
  --user "${QUOTA_USER:-js3269}" \
  --filesystem scratch \
  --required_gib "$REQUIRED_GIB" \
  --headroom_factor "${QUOTA_HEADROOM_FACTOR:-2}" \
  --marker_path "$STATUS_DIR/${SUFFIX}.quota"
QUOTA_RC=$?
set -e
if (( QUOTA_RC != 0 )); then
  exit "$QUOTA_RC"
fi

# Resume guard, mirroring the paper-aligned MiniGrid runner: continue from a
# checkpoint when one exists, and refuse to append a second trajectory onto
# partial results that cannot be resumed.
RESUME_ARGS=()
if [[ -f "$CHECKPOINT" ]]; then
  RESUME_ARGS=(--resume_from "$CHECKPOINT")
  echo "[$(date -Is)] resuming from $CHECKPOINT"
elif [[ ! -f "$DONE_FILE" && ( -f "$RESULT_DIR/metrics_history.jsonl" || -f "$RESULT_DIR/metrics.json" ) ]]; then
  {
    echo "status=blocked_no_checkpoint task_id=$TASK_ID model=$MODEL seed=$SEED"
    echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
  } > "$STATUS_DIR/${SUFFIX}.blocked"
  echo "Refusing to append to partial results without a resumable checkpoint: $RESULT_DIR" >&2
  exit 3
fi

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
echo "protocol=4-action-minimal frame_skip=4 frame_stack=4 layers=1 flicker=$FLICKER_PROB"
echo "result_dir=$RESULT_DIR total_timesteps=$TOTAL_TIMESTEPS"
echo "checkpoint_interval_steps=$CHECKPOINT_INTERVAL_STEPS replay_backing=mmap"

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id ALE/Breakout-v5 \
  --action_space_mode minimal \
  --model_type "$MODEL" \
  --num_layers 1 \
  --frame_skip "$FRAME_SKIP" \
  --frame_stack "$FRAME_STACK" \
  --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len "$SEQ_LEN" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
  --save_dir "$RESULT_DIR" \
  --replay_backing mmap \
  --checkpoint_interval_steps "$CHECKPOINT_INTERVAL_STEPS" \
  "${RESUME_ARGS[@]}" \
  "${ACCEL_ARGS[@]}" \
  "${SIZE_ARGS[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  {
    echo "status=train_failed task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
    echo "result_dir=$RESULT_DIR exit_code=$TRAIN_RC timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

# A preemption checkpoint is a successful pause, not a completed unit: leave the
# checkpoint in place and write no done marker. Exiting 0 here would make Slurm
# mark the task COMPLETED and never restart it, so the task requeues itself;
# --requeue alone only covers preemption and node failure, not the wall-clock
# signal we handle deliberately.
if [[ ! -f "$RESULT_DIR/metrics.json" ]]; then
  echo "[$(date -Is)] paused before completion; checkpoint retained at $CHECKPOINT"
  {
    echo "status=paused task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
    echo "checkpoint=$CHECKPOINT timestamp=$(date -Is)"
  } > "$STATUS_DIR/${SUFFIX}.paused"
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    echo "[$(date -Is)] requeueing $SLURM_JOB_ID to continue from the checkpoint"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60  # give Slurm time to act before the script would otherwise exit
  fi
  exit 0
fi

python - "$RESULT_DIR" "$MODEL" "$TOTAL_TIMESTEPS" "$FUSED_EXPECTED" "$COMPILE_EXPECTED" <<'PY'
import glob
import json
import os
import sys

result_dir, model_type = sys.argv[1], sys.argv[2]
total_steps = int(sys.argv[3])
fused_expected = sys.argv[4].lower() == "true"
compile_expected = sys.argv[5].lower() == "true"
metrics_path = os.path.join(result_dir, "metrics.json")
history_path = os.path.join(result_dir, "metrics_history.jsonl")
with open(metrics_path, encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": total_steps,
    "frame_skip": 4,
    "frame_stack": 4,
    "num_layers": 1,
    "model_type": model_type,
    "action_space_mode": "minimal",
    "num_actions": 4,
    "optimizer": "adam",
    "fused_optimizer": fused_expected,
    "compile_model": compile_expected,
    "replay_backing": "mmap",
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics: expected={expected}, actual={actual}")
if not os.path.isfile(history_path):
    raise RuntimeError(f"Missing history: {history_path}")
checkpoints = glob.glob(os.path.join(result_dir, "*.pth"))
if len(checkpoints) != 1:
    raise RuntimeError(f"Expected one checkpoint in {result_dir}, found {checkpoints}")
if os.path.isdir(os.path.join(result_dir, "replay")):
    raise RuntimeError(f"Replay storage was not reclaimed: {result_dir}/replay")
steps = []
with open(history_path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            steps.append(int(json.loads(line)["global_step"]))
if steps != sorted(steps) or len(steps) != len(set(steps)):
    raise RuntimeError(f"History is not monotone after resume: {history_path}")
print(
    f"validated resume_count={metrics.get('resume_count')} "
    f"resumed_at_steps={metrics.get('resumed_at_steps')}"
)
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
  echo "frame_skip=4 frame_stack=4 layers=1 action_space=minimal num_actions=4"
  echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE" "$STATUS_DIR/${SUFFIX}.blocked" "$STATUS_DIR/${SUFFIX}.quota" \
  "$STATUS_DIR/${SUFFIX}.paused"
echo "[$(date -Is)] done -> $RESULT_DIR"
