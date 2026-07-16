#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-pong-fs1s1
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/amarel/artifacts/atari_pong_fs1_stack1/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_pong_fs1_stack1/%A_%a.err

# Train one (model x setting x seed) cell of the Pong frame-skip-1/stack-1 sweep.
#
# 7 models x 2 settings x 5 seeds = 70 array tasks (SLURM_ARRAY_TASK_ID 0..69):
#   setting 0 = plain Pong              (flicker_prob=0.0)   -> MDP-ish control
#   setting 1 = flickering Pong         (flicker_prob=0.5)   -> POMDP, needs memory
# Both use frame_skip=1 and frame_stack=1: one ALE frame is advanced and one
# observed frame is supplied per environment step.
# The historical frame-skip-4/stack-1 sweep is stored separately under the
# explicit `pong_fs4_stack1` protocol label.
#
# task -> (model, setting, seed) with model varying fastest:
#   model   = MODELS[task % 7]
#   rest    = task / 7            (0..9)
#   setting = rest / 5           (0..1)
#   seed    = SEEDS[rest % 5]
#
# All recurrent cores are param-matched to the LSTM anchor (hidden=512) via
# experiments/atari/atari_ssm_param_match.py -> results/atari_param_match/
# atari_param_match.json. RNN/GRU/LSTM/GaWF get --hidden_size; S5/Mamba get
# --ssm_d_model/--ssm_state_size; CNN is the unmatched feedforward control.
# FRAME_SKIP/FRAME_STACK overrides are used only by explicit protocol-comparison
# launchers; their RESULT_TAG must encode the overridden values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

FRAME_SKIP="${FRAME_SKIP:-1}"
FRAME_STACK="${FRAME_STACK:-1}"
PROTOCOL_TAG="pong_fs${FRAME_SKIP}_stack${FRAME_STACK}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_${PROTOCOL_TAG}}"
RESULT_TAG="${RESULT_TAG:-$PROTOCOL_TAG}"
if [[ "$FRAME_SKIP" -lt 1 || "$FRAME_STACK" -lt 1 ]]; then
  echo "FRAME_SKIP and FRAME_STACK must both be >= 1" >&2
  exit 2
fi
if [[ "$RESULT_TAG" != *"$PROTOCOL_TAG"* ]]; then
  echo "RESULT_TAG must include the explicit protocol tag $PROTOCOL_TAG" >&2
  exit 2
fi
ART_ROOT="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

# ---- environment -----------------------------------------------------------
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS=12
export AIM3_PIN_MEMORY=1

# ---- task -> (model, setting, seed) ---------------------------------------
MODELS=(ann rnn gru lstm gawf s5 mamba)
SEEDS=(${SEEDS_OVERRIDE:-42 1 2 3 4})
N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

MODEL="${MODELS[$((TASK_ID % N_MODELS))]}"
REST=$((TASK_ID / N_MODELS))
SETTING=$((REST / N_SEEDS))
SEED="${SEEDS[$((REST % N_SEEDS))]}"
OPTIMIZER_NAME=adam
FUSED_OPTIMIZER=1
if [[ "$MODEL" == "s5" ]]; then
  FUSED_OPTIMIZER=0
fi
ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer)
COMPILE_ACTIVE=0
if [[ "$MODEL" == "ann" || "$MODEL" == "gawf" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_ACTIVE=1
fi

if [[ "$SETTING" -eq 0 ]]; then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_${RESULT_TAG}_${MODEL}_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_${RESULT_TAG}_flicker_${MODEL}_seed${SEED}"
fi

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
SEQ_LEN="${SEQ_LEN:-16}"

# ---- param-matched sizing from JSON ---------------------------------------
MATCH_JSON="$ROOT/results/atari_param_match/atari_param_match.json"
SIZE_ARGS=()
if [[ "$MODEL" != "ann" ]]; then
  if [[ ! -f "$MATCH_JSON" ]]; then
    echo "Missing $MATCH_JSON. Run atari_ssm_param_match.py first." >&2
    exit 2
  fi
  read -r KIND V1 V2 < <(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["matched"].get(sys.argv[2], {})
if "hidden_size" in m:
    print("hidden", m["hidden_size"], "")
elif "d_model" in m:
    print("ssm", m["d_model"], m.get("state_size", d.get("ssm_state_size", 128)))
else:
    print("none", "", "")
PY
)
  if [[ "$KIND" == "hidden" ]]; then
    SIZE_ARGS=(--hidden_size "$V1")
  elif [[ "$KIND" == "ssm" ]]; then
    SIZE_ARGS=(--ssm_d_model "$V1" --ssm_state_size "$V2")
  else
    echo "No matched sizing for model=$MODEL in $MATCH_JSON." >&2
    exit 2
  fi
fi

DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
RESULT_DIR="$ROOT/results/train_data/$SUFFIX"

# A preempted run may leave a partial history. Preserve it under a rerun-specific
# name so the new 0..TOTAL_TIMESTEPS curve is not appended to stale steps.
if [[ ! -f "$DONE_FILE" && -f "$RESULT_DIR/metrics_history.jsonl" ]]; then
  RERUN_JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
  RERUN_TASK_ID="${SLURM_ARRAY_TASK_ID:-$TASK_ID}"
  mv "$RESULT_DIR/metrics_history.jsonl" \
    "$RESULT_DIR/metrics_history.pre_rerun_${RERUN_JOB_ID}_${RERUN_TASK_ID}.jsonl"
fi

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED flicker=$FLICKER_PROB"
echo "result_suffix=$SUFFIX total_timesteps=$TOTAL_TIMESTEPS sizing=${SIZE_ARGS[*]:-none(ann)}"
echo "frame_skip=$FRAME_SKIP frame_stack=$FRAME_STACK amp=bfloat16 tf32=1 " \
  "cudnn_benchmark=1 optimizer=$OPTIMIZER_NAME fused_optimizer=$FUSED_OPTIMIZER " \
  "compile=$COMPILE_ACTIVE"

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id "ALE/Pong-v5" \
  --model_type "$MODEL" \
  --frame_stack "$FRAME_STACK" \
  --frame_skip "$FRAME_SKIP" \
  --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len "$SEQ_LEN" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
  "${ACCEL_ARGS[@]}" \
  "${SIZE_ARGS[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
    echo "flicker_prob=$FLICKER_PROB result_suffix=$SUFFIX exit_code=$train_rc"
    echo "timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

python - "$ROOT/results/train_data/$SUFFIX" "$MODEL" "$TOTAL_TIMESTEPS" \
  "$FRAME_SKIP" "$FRAME_STACK" <<'PY'
import glob
import json
import os
import sys

result_dir, model_type = sys.argv[1], sys.argv[2]
total_timesteps, frame_skip, frame_stack = map(int, sys.argv[3:6])
metrics_path = os.path.join(result_dir, "metrics.json")
with open(metrics_path, encoding="utf-8") as handle:
    metrics = json.load(handle)
expected = {
    "global_step": total_timesteps,
    "frame_skip": frame_skip,
    "frame_stack": frame_stack,
    "num_layers": 1,
    "model_type": model_type,
    "optimizer": "adam",
    "fused_optimizer": model_type != "s5",
}
actual = {key: metrics.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Invalid metrics in {metrics_path}: expected={expected}, actual={actual}")
checkpoints = glob.glob(os.path.join(result_dir, "*.pth"))
if len(checkpoints) != 1:
    raise RuntimeError(f"Expected one checkpoint in {result_dir}, found {checkpoints}")
PY

{
  echo "status=done task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
  echo "frame_skip=$FRAME_SKIP frame_stack=$FRAME_STACK result_suffix=$SUFFIX"
  echo "metrics_path=results/train_data/$SUFFIX/metrics.json timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done model=$MODEL setting=$SETTING seed=$SEED -> results/train_data/$SUFFIX"
