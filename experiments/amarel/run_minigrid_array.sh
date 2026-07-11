#!/usr/bin/env bash
#SBATCH --job-name=aim3-minigrid
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpu018,gpu043
#SBATCH --time=24:00:00
#SBATCH --output=experiments/amarel/artifacts/minigrid/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/minigrid/%A_%a.err

# Train one (model x seed) cell of a MiniGrid sweep on a single env.
#
# task -> (model, seed) with model varying fastest:
#   model = MODELS[task % n_models];  seed = SEEDS[task / n_models]
# Env, seeds, and budget come from env vars (set by the submit wrapper):
#   ENV_ID (default MiniGrid-MemoryS7-v0), SEEDS_OVERRIDE, TOTAL_TIMESTEPS.
#
# "ann" is the feedforward baseline (no recurrent sizing). RNN/GRU/LSTM/GaWF get
# --hidden_size, S5/Mamba get --ssm_d_model/--ssm_state_size, from the MiniGrid
# param-match JSON (LSTM@128 anchor). Encoder defaults to mlp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_minigrid_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

ART_ROOT="$ROOT/experiments/amarel/artifacts/minigrid"
STATUS_DIR="$ART_ROOT/status"
mkdir -p "$ART_ROOT" "$STATUS_DIR"

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

MODELS=(ann rnn gru lstm gawf s5 mamba)
SEEDS=(${SEEDS_OVERRIDE:-42})
N_MODELS=${#MODELS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID % N_MODELS))]}"
SEED="${SEEDS[$((TASK_ID / N_MODELS))]}"

ENV_ID="${ENV_ID:-MiniGrid-MemoryS7-v0}"
ENV_TAG="$(echo "$ENV_ID" | sed 's#MiniGrid-##; s#-v0##')"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
ENCODER="${ENCODER:-mlp}"
SEQ_LEN="${SEQ_LEN:-32}"
SUFFIX="mg_${ENV_TAG}_${MODEL}_seed${SEED}"

# ---- param-matched sizing from the MiniGrid JSON ---------------------------
MATCH_JSON="$ROOT/results/minigrid_param_match/atari_param_match.json"
SIZE_ARGS=()
if [[ "$MODEL" != "ann" && "$MODEL" != "cnn" ]]; then
  if [[ ! -f "$MATCH_JSON" ]]; then
    echo "Missing $MATCH_JSON. Run atari_ssm_param_match --out_dir results/minigrid_param_match first." >&2
    exit 2
  fi
  read -r KIND V1 V2 < <(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["matched"].get(sys.argv[2], {})
if "hidden_size" in m: print("hidden", m["hidden_size"], "")
elif "d_model" in m: print("ssm", m["d_model"], m.get("state_size", 64))
else: print("none", "", "")
PY
)
  if [[ "$KIND" == "hidden" ]]; then SIZE_ARGS=(--hidden_size "$V1")
  elif [[ "$KIND" == "ssm" ]]; then SIZE_ARGS=(--ssm_d_model "$V1" --ssm_state_size "$V2")
  else echo "No matched sizing for model=$MODEL" >&2; exit 2; fi
fi

DONE_FILE="$STATUS_DIR/${SUFFIX}.done"
FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"

echo "[$(date -Is)] task=$TASK_ID model=$MODEL seed=$SEED env=$ENV_ID steps=$TOTAL_TIMESTEPS"
echo "result_suffix=$SUFFIX encoder=$ENCODER sizing=${SIZE_ARGS[*]:-none(ann)}"

set +e
DISABLE_TQDM=1 python train_minigrid_dqn.py \
  --env_id "$ENV_ID" \
  --model_type "$MODEL" \
  --encoder "$ENCODER" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --seq_len "$SEQ_LEN" \
  --seed "$SEED" \
  --device cuda \
  --result_suffix "$SUFFIX" \
  "${SIZE_ARGS[@]}"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "status=train_failed model=$MODEL seed=$SEED env=$ENV_ID rc=$rc $(date -Is)" > "$FAIL_FILE"
  exit "$rc"
fi
echo "status=done model=$MODEL seed=$SEED env=$ENV_ID metrics=results/train_data/$SUFFIX/metrics.json $(date -Is)" > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done $SUFFIX"
