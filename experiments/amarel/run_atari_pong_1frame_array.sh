#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-pong1f
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --exclude=gpu018,gpu043
#SBATCH --time=48:00:00
#SBATCH --output=experiments/amarel/artifacts/atari_pong_1frame/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_pong_1frame/%A_%a.err

# Train one (model x setting x seed) cell of the 1-frame Pong sweep.
#
# 7 models x 2 settings x 5 seeds = 70 array tasks (SLURM_ARRAY_TASK_ID 0..69):
#   setting 0 = plain 1-frame Pong        (flicker_prob=0.0)   -> MDP-ish control
#   setting 1 = 1-frame flickering Pong   (flicker_prob=0.5)   -> POMDP, needs memory
# Both use --frame_stack 1 so the recurrent core is the ONLY source of memory.
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
if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

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

if [[ "$SETTING" -eq 0 ]]; then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_pong1f_${MODEL}_seed${SEED}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_pong1f_flicker_${MODEL}_seed${SEED}"
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

set +e
DISABLE_TQDM=1 python train_atari_dqn.py \
  --env_id "ALE/Pong-v5" \
  --model_type "$MODEL" \
  --frame_stack 1 \
  --flicker_prob "$FLICKER_PROB" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len "$SEQ_LEN" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
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
