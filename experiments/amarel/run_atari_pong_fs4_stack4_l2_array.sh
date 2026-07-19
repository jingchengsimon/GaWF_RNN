#!/usr/bin/env bash
#SBATCH --job-name=aim3-pong-fs4s4-l2
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Strict 6-action, parameter-matched depth-2 Pong sweep:
# 5 models x 2 observation settings x 5 seeds = 50 tasks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
: "${MATCH_JSON:?MATCH_JSON must point to the L2 parameter-match JSON}"
[[ -f "$MATCH_JSON" ]] || { echo "Missing parameter match JSON: $MATCH_JSON" >&2; exit 2; }

FRAME_SKIP=4
FRAME_STACK=4
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
SEQ_LEN="${SEQ_LEN:-16}"
RUN_TAG="${RUN_TAG:-pong_fs4_stack4_l2match}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_pong_fs4_stack4_l2}"
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

MODELS=(ann rnn gru lstm gawf)
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

HIDDEN="$(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    entry = json.load(handle)["matched"][sys.argv[2]]
print(entry["hidden_size"])
PY
)"

ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32)
COMPILE_EXPECTED=false
if [[ "$MODEL" == "ann" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_EXPECTED=true
fi

RESULT_DIR="$AIM3_RESULTS_PATH/train_data/$SUFFIX"
DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
if [[ ! -f "$DONE_FILE" && -f "$RESULT_DIR/metrics_history.jsonl" ]]; then
  PREEMPT_TAG="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-$TASK_ID}"
  mv "$RESULT_DIR/metrics_history.jsonl" \
    "$RESULT_DIR/metrics_history.pre_rerun_${PREEMPT_TAG}.jsonl"
fi

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED hidden=$HIDDEN"
echo "protocol=6-action-minimal frame_skip=4 frame_stack=4 layers=2 flicker=$FLICKER_PROB"
echo "result_dir=$RESULT_DIR total_timesteps=$TOTAL_TIMESTEPS"

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id ALE/Pong-v5 \
  --action_space_mode minimal \
  --model_type "$MODEL" \
  --hidden_size "$HIDDEN" \
  --num_layers 2 \
  --gawf_feedback_lr_scale 1.0 \
  --frame_skip "$FRAME_SKIP" \
  --frame_stack "$FRAME_STACK" \
  --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len "$SEQ_LEN" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
  --save_dir "$RESULT_DIR" \
  "${ACCEL_ARGS[@]}"
TRAIN_RC=$?
set -e
if (( TRAIN_RC != 0 )); then
  {
    echo "status=train_failed task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
    echo "result_dir=$RESULT_DIR exit_code=$TRAIN_RC timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$TRAIN_RC"
fi

python - "$RESULT_DIR" "$MODEL" "$TOTAL_TIMESTEPS" "$COMPILE_EXPECTED" <<'PY'
import glob
import json
import os
import sys

result_dir, model_type = sys.argv[1], sys.argv[2]
total_steps = int(sys.argv[3])
compile_expected = sys.argv[4].lower() == "true"
metrics_path = os.path.join(result_dir, "metrics.json")
history_path = os.path.join(result_dir, "metrics_history.jsonl")
with open(metrics_path, encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": total_steps,
    "frame_skip": 4,
    "frame_stack": 4,
    "num_layers": 2,
    "model_type": model_type,
    "action_space_mode": "minimal",
    "num_actions": 6,
    "optimizer": "adam",
    "fused_optimizer": False,
    "compile_model": compile_expected,
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics: expected={expected}, actual={actual}")
if not os.path.isfile(history_path):
    raise RuntimeError(f"Missing history: {history_path}")
checkpoints = glob.glob(os.path.join(result_dir, "*.pth"))
if len(checkpoints) != 1:
    raise RuntimeError(f"Expected one checkpoint in {result_dir}, found {checkpoints}")
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
  echo "frame_skip=4 frame_stack=4 layers=2 action_space=minimal num_actions=6"
  echo "result_dir=$RESULT_DIR timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done -> $RESULT_DIR"
