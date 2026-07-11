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

# Train one (model x setting) cell of the 1-frame Pong sweep.
#
# 7 models x 2 settings = 14 array tasks (SLURM_ARRAY_TASK_ID 0..13):
#   setting 0 = plain 1-frame Pong        (flicker_prob=0.0)   -> MDP-ish control
#   setting 1 = 1-frame flickering Pong   (flicker_prob=0.5)   -> POMDP, needs memory
# Both use --frame_stack 1 so the recurrent core is the ONLY source of memory.
# The pre-existing 4-frame baseline sweep is separate and left untouched.
#
# task -> (model, setting):
#   model   = MODELS[task % 7]
#   setting = task / 7
#
# S5/Mamba d_model come from the LSTM-anchored param match
# (experiments/generalization/atari_ssm_param_match.py); override via env if the
# JSON is absent: S5_D_MODEL, MAMBA_D_MODEL, SSM_STATE_SIZE.

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

# ---- task -> (model, setting) ---------------------------------------------
MODELS=(cnn rnn gru lstm gawf s5 mamba)
N_MODELS=${#MODELS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID % N_MODELS))]}"
SETTING=$((TASK_ID / N_MODELS))

if [[ "$SETTING" -eq 0 ]]; then
  FLICKER_PROB=0.0
  SUFFIX="atari_dqn_pong1f_${MODEL}"
else
  FLICKER_PROB=0.5
  SUFFIX="atari_dqn_pong1f_flicker_${MODEL}"
fi

SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
SEQ_LEN="${SEQ_LEN:-16}"

# ---- param-matched S5/Mamba d_model ---------------------------------------
MATCH_JSON="$ROOT/results/atari_param_match/atari_ssm_param_match.json"
SSM_STATE_SIZE="${SSM_STATE_SIZE:-128}"
S5_D_MODEL="${S5_D_MODEL:-}"
MAMBA_D_MODEL="${MAMBA_D_MODEL:-}"
if [[ -f "$MATCH_JSON" ]]; then
  read -r JS5 JMAMBA JSTATE < <(python - "$MATCH_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d.get("matched", {})
print(m.get("s5", {}).get("d_model", ""),
      m.get("mamba", {}).get("d_model", ""),
      d.get("ssm_state_size", ""))
PY
)
  [[ -z "$S5_D_MODEL"   && -n "$JS5"    ]] && S5_D_MODEL="$JS5"
  [[ -z "$MAMBA_D_MODEL" && -n "$JMAMBA" ]] && MAMBA_D_MODEL="$JMAMBA"
  [[ -n "$JSTATE" ]] && SSM_STATE_SIZE="$JSTATE"
fi

SSM_ARGS=()
if [[ "$MODEL" == "s5" ]]; then
  if [[ -z "$S5_D_MODEL" ]]; then
    echo "S5 d_model unknown: run atari_ssm_param_match.py or set S5_D_MODEL." >&2
    exit 2
  fi
  SSM_ARGS=(--ssm_d_model "$S5_D_MODEL" --ssm_state_size "$SSM_STATE_SIZE")
elif [[ "$MODEL" == "mamba" ]]; then
  if [[ -z "$MAMBA_D_MODEL" ]]; then
    echo "Mamba d_model unknown: run atari_ssm_param_match.py or set MAMBA_D_MODEL." >&2
    exit 2
  fi
  SSM_ARGS=(--ssm_d_model "$MAMBA_D_MODEL" --ssm_state_size "$SSM_STATE_SIZE")
fi

DONE_FILE="$STATUS_DIR/${SUFFIX}_seed${SEED}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}_seed${SEED}.fail"

echo "[$(date -Is)] task=$TASK_ID model=$MODEL setting=$SETTING flicker=$FLICKER_PROB"
echo "result_suffix=$SUFFIX seed=$SEED total_timesteps=$TOTAL_TIMESTEPS"
[[ ${#SSM_ARGS[@]} -gt 0 ]] && echo "ssm_args=${SSM_ARGS[*]}"

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
  "${SSM_ARGS[@]}"
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  {
    echo "status=train_failed"
    echo "task_id=$TASK_ID model=$MODEL setting=$SETTING"
    echo "flicker_prob=$FLICKER_PROB result_suffix=$SUFFIX seed=$SEED"
    echo "exit_code=$train_rc timestamp=$(date -Is)"
  } > "$FAIL_FILE"
  exit "$train_rc"
fi

{
  echo "status=done"
  echo "task_id=$TASK_ID model=$MODEL setting=$SETTING"
  echo "flicker_prob=$FLICKER_PROB result_suffix=$SUFFIX seed=$SEED"
  echo "metrics_path=results/train_data/$SUFFIX/metrics.json"
  echo "timestamp=$(date -Is)"
} > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done model=$MODEL setting=$SETTING -> results/train_data/$SUFFIX"
