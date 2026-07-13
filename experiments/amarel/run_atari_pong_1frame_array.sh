#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-pong1f
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/amarel/artifacts/atari_pong_1frame/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_pong_1frame/%A_%a.err

# Train one (model x setting x seed) cell of the Pong frame-skip-1/stack-1 sweep.
#
# 7 models x 2 settings x 5 seeds = 70 array tasks (SLURM_ARRAY_TASK_ID 0..69):
#   setting 0 = plain Pong              (flicker_prob=0.0)   -> MDP-ish control
#   setting 1 = flickering Pong         (flicker_prob=0.5)   -> POMDP, needs memory
# Both use frame_skip=1 and frame_stack=1: one ALE frame is advanced and one
# observed frame is supplied per environment step.
# The pre-existing 4-frame baseline sweep is separate and left untouched.
#
# task -> (model, setting, seed) with model varying fastest:
#   model   = MODELS[task % 7]
#   rest    = task / 7            (0..9)
#   setting = rest / 5           (0..1)
#   seed    = SEEDS[rest % 5]
#
# All recurrent cores are param-matched to the LSTM anchor (hidden=512) via
# experiments/generalization/atari_ssm_param_match.py -> results/atari_param_match/
# atari_param_match.json. RNN/GRU/LSTM/GaWF get --hidden_size; S5/Mamba get
# --ssm_d_model/--ssm_state_size; CNN is the unmatched feedforward control.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

ART_ROOT="$ROOT/experiments/amarel/artifacts/atari_pong_1frame"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

# ---- environment -----------------------------------------------------------
source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
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
ACCEL_ARGS=(--amp_dtype bfloat16 --allow_tf32)
COMPILE_ACTIVE=0
if [[ "$MODEL" == "ann" ]]; then
  ACCEL_ARGS+=(--compile_model)
  COMPILE_ACTIVE=1
fi

if [[ "$SETTING" -eq 0 ]]; then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_pong_fs1_stack1_${MODEL}_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_pong_fs1_stack1_flicker_${MODEL}_seed${SEED}"
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

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED flicker=$FLICKER_PROB"
echo "result_suffix=$SUFFIX total_timesteps=$TOTAL_TIMESTEPS sizing=${SIZE_ARGS[*]:-none(ann)}"
echo "frame_skip=1 frame_stack=1 amp=bfloat16 tf32=1 compile=$COMPILE_ACTIVE"

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id "ALE/Pong-v5" \
  --model_type "$MODEL" \
  --frame_stack 1 \
  --frame_skip 1 \
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
    echo "flicker_prob=$FLICKER_PROB result_suffix=$SUFFIX exit_code=$train_rc timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

{
  echo "status=done task_id=$TASK_ID model=$MODEL setting=$SETTING seed=$SEED"
  echo "flicker_prob=$FLICKER_PROB result_suffix=$SUFFIX"
  echo "metrics_path=results/train_data/$SUFFIX/metrics.json timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done model=$MODEL setting=$SETTING seed=$SEED -> results/train_data/$SUFFIX"
