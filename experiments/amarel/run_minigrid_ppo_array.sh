#!/usr/bin/env bash
#SBATCH --job-name=aim3-mg-ppo
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=gpu018,gpu043
#SBATCH --time=12:00:00
#SBATCH --output=experiments/amarel/artifacts/minigrid_ppo/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/minigrid_ppo/%A_%a.err

# Train one (model x seed) cell of a MiniGrid recurrent-PPO sweep on one env.
#
# task -> (model, seed): model = MODELS[task % 6];  seed = SEEDS[task / 6]
# 6 recurrent cores (no feedforward baseline, BabyAI-style). Env/seeds/budget from
# env vars: ENV_ID (default MemoryS7), SEEDS_OVERRIDE, TOTAL_TIMESTEPS.
# Cores param-matched to LSTM@128 via results/minigrid_param_match/atari_param_match.json.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_minigrid_ppo.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"
ART="$ROOT/experiments/amarel/artifacts/minigrid_ppo"; STATUS_DIR="$ART/status"
mkdir -p "$ART" "$STATUS_DIR"

if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  [[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True KMP_DUPLICATE_LIB_OK=TRUE

MODELS=(rnn gru lstm gawf s5 mamba)
SEEDS=(${SEEDS_OVERRIDE:-42 1 2})
N_MODELS=${#MODELS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$((TASK_ID % N_MODELS))]}"
SEED="${SEEDS[$((TASK_ID / N_MODELS))]}"
ENV_ID="${ENV_ID:-MiniGrid-MemoryS7-v0}"
ENV_TAG="$(echo "$ENV_ID" | sed 's#MiniGrid-##; s#-v0##')"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
NUM_ENVS="${NUM_ENVS:-16}"; NUM_STEPS="${NUM_STEPS:-40}"; ENCODER="${ENCODER:-mlp}"
VIEW_ARGS=(); VIEW_TAG=""
if [[ -n "${AGENT_VIEW_SIZE:-}" ]]; then VIEW_ARGS=(--agent_view_size "$AGENT_VIEW_SIZE"); VIEW_TAG="_fov${AGENT_VIEW_SIZE}"; fi
SUFFIX="mg_ppo_${ENV_TAG}${VIEW_TAG}_${MODEL}_seed${SEED}"

MATCH_JSON="$ROOT/results/minigrid_param_match/atari_param_match.json"
read -r KIND V1 V2 < <(python - "$MATCH_JSON" "$MODEL" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); m=d["matched"].get(sys.argv[2], {})
if "hidden_size" in m: print("hidden", m["hidden_size"], "")
elif "d_model" in m: print("ssm", m["d_model"], m.get("state_size", 64))
else: print("none", "", "")
PY
)
if [[ "$KIND" == "hidden" ]]; then SIZE_ARGS=(--hidden_size "$V1")
elif [[ "$KIND" == "ssm" ]]; then SIZE_ARGS=(--ssm_d_model "$V1" --ssm_state_size "$V2")
else echo "No sizing for $MODEL" >&2; exit 2; fi

LR_ARGS=()
[[ -n "${LEARNING_RATE:-}" ]] && LR_ARGS=(--learning_rate "$LEARNING_RATE")

DONE_FILE="$STATUS_DIR/${SUFFIX}.done"; FAIL_FILE="$STATUS_DIR/${SUFFIX}.fail"
echo "[$(date -Is)] task=$TASK_ID model=$MODEL seed=$SEED env=$ENV_ID steps=$TOTAL_TIMESTEPS sizing=${SIZE_ARGS[*]} lr=${LEARNING_RATE:-default}"

set +e
DISABLE_TQDM=1 python train_minigrid_ppo.py --env_id "$ENV_ID" --model_type "$MODEL" \
  --encoder "$ENCODER" --total_timesteps "$TOTAL_TIMESTEPS" --num_envs "$NUM_ENVS" \
  --num_steps "$NUM_STEPS" --update_epochs 4 --seed "$SEED" --device cuda \
  --result_suffix "$SUFFIX" "${SIZE_ARGS[@]}" "${VIEW_ARGS[@]}" "${LR_ARGS[@]}"
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then echo "status=fail model=$MODEL seed=$SEED rc=$rc $(date -Is)" > "$FAIL_FILE"; exit "$rc"; fi
echo "status=done model=$MODEL seed=$SEED metrics=results/train_data/$SUFFIX/metrics.json $(date -Is)" > "$DONE_FILE"
rm -f "$FAIL_FILE"
echo "[$(date -Is)] done $SUFFIX"
