#!/usr/bin/env bash
#SBATCH --job-name=aim3-pong-d2
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/amarel/artifacts/atari_pong_depth2/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_pong_depth2/%A_%a.err

# One cell of the parameter-matched depth-2 Pong pilot.
# Pong defaults to one ALE frame per environment step and one observed frame.

set -euo pipefail

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/projects/FAW_RNN}}"
cd "$ROOT"
ART="$ROOT/experiments/amarel/artifacts/atari_pong_depth2"
STATUS_DIR="$ART/status"
mkdir -p "$ART" "$STATUS_DIR"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE
export AIM3_NUM_WORKERS=12
export AIM3_PIN_MEMORY=1

FRAME_SKIP="${FRAME_SKIP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
ALLOW_TF32="${ALLOW_TF32:-1}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"

MODELS=(ann rnn gru lstm gawf)
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV:-42}"
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
ACCEL_ARGS=(--amp_dtype "$AMP_DTYPE")
COMPILE_ACTIVE=0
if [[ "$ALLOW_TF32" == "1" ]]; then
  ACCEL_ARGS+=(--allow_tf32)
fi
# Amarel PyTorch 2.3 cannot compile the recurrent-state dataclass reliably.
# ANN has no recurrent state; recurrent models retain BF16/TF32 acceleration,
# and RNN/GRU/LSTM use the reset-safe fused cuDNN scan in AtariQNetwork.
if [[ "$COMPILE_MODEL" == "1" && "$MODEL" == "ann" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_ACTIVE=1
fi
if (( SETTING == 0 )); then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_pong_fs${FRAME_SKIP}_stack1_depth2match_${MODEL}_L2_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_pong_fs${FRAME_SKIP}_stack1_flicker_depth2match_${MODEL}_L2_seed${SEED}"
fi

MATCH_JSON="$ROOT/results/atari_param_match_depth2/atari_param_match.json"
[[ -f "$MATCH_JSON" ]] || { echo "Missing $MATCH_JSON" >&2; exit 2; }
HIDDEN="$(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json, sys
entry = json.load(open(sys.argv[1]))["matched"][sys.argv[2]]
print(entry["hidden_size"])
PY
)"

DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED hidden=$HIDDEN layers=2"
echo "frame_skip=$FRAME_SKIP frame_stack=1 amp=$AMP_DTYPE tf32=$ALLOW_TF32 compile=$COMPILE_ACTIVE"

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id ALE/Pong-v5 \
  --model_type "$MODEL" \
  --hidden_size "$HIDDEN" \
  --num_layers 2 \
  --gawf_feedback_lr_scale 1.0 \
  --frame_stack 1 \
  --frame_skip "$FRAME_SKIP" \
  --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "${TOTAL_TIMESTEPS:-1000000}" \
  --seq_len "${SEQ_LEN:-16}" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
  "${ACCEL_ARGS[@]}"
rc=$?
set -e
if (( rc != 0 )); then
  echo "status=fail task=$TASK_ID model=$MODEL seed=$SEED rc=$rc $(date -Is)" > "$FAIL_FILE"
  exit "$rc"
fi
METRICS_FILE="results/train_data/$SUFFIX/metrics.json"
TARGET_STEPS="${TOTAL_TIMESTEPS:-1000000}"
set +e
python - "$METRICS_FILE" "$TARGET_STEPS" "$FRAME_SKIP" <<'PY'
import json
import os
import sys

metrics_path = sys.argv[1]
target_steps = int(sys.argv[2])
target_frame_skip = int(sys.argv[3])
with open(metrics_path, encoding="utf-8") as handle:
    metrics = json.load(handle)
if int(metrics.get("global_step", -1)) != target_steps:
    raise RuntimeError(
        f"global_step={metrics.get('global_step')} does not match target={target_steps}"
    )
if int(metrics.get("frame_skip", -1)) != target_frame_skip:
    raise RuntimeError(
        f"frame_skip={metrics.get('frame_skip')} does not match target={target_frame_skip}"
    )
if int(metrics.get("frame_stack", -1)) != 1:
    raise RuntimeError(f"frame_stack={metrics.get('frame_stack')} does not match target=1")
if int(metrics.get("num_layers", -1)) != 2:
    raise RuntimeError(f"num_layers={metrics.get('num_layers')} does not match target=2")
checkpoint = metrics.get("checkpoint_path") or metrics.get("checkpoint")
if not checkpoint or not os.path.isfile(checkpoint):
    raise RuntimeError(f"checkpoint is missing: {checkpoint!r}")
PY
validate_rc=$?
set -e
if (( validate_rc != 0 )); then
  echo "status=fail task=$TASK_ID model=$MODEL seed=$SEED rc=$validate_rc validation=artifacts $(date -Is)" > "$FAIL_FILE"
  exit "$validate_rc"
fi
echo "status=done task=$TASK_ID model=$MODEL seed=$SEED metrics=$METRICS_FILE global_step=$TARGET_STEPS frame_skip=$FRAME_SKIP $(date -Is)" > "$DONE_FILE"
rm -f "$FAIL_FILE"
