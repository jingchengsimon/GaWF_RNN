#!/usr/bin/env bash
# Run one-update smoke tests for every paper MiniGrid PPO model on two GPUs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV_ID="${ENV_ID:-MiniGrid-RedBlueDoors-8x8-v0}"
ENV_TAG="${ENV_ID#MiniGrid-}"
ENV_TAG="${ENV_TAG%-v0}"
SEED="${SEED:-42}"
MODELS=(paper_lstm lstm_core rnn gru gawf s5 mamba)
ART="$ROOT/experiments/local/artifacts/minigrid_ppo_paper_smoke/$ENV_TAG"
mkdir -p "$ART"

size_args() {
  case "$1" in
    paper_lstm|lstm_core) echo "--hidden_size 128" ;;
    rnn) echo "--hidden_size 304" ;;
    gru) echo "--hidden_size 155" ;;
    gawf) echo "--hidden_size 297" ;;
    s5) echo "--ssm_d_model 408 --ssm_state_size 64" ;;
    mamba) echo "--ssm_d_model 110 --ssm_state_size 64" ;;
    *) return 2 ;;
  esac
}

run_group() {
  local gpu="$1"
  shift
  local model suffix
  local -a sizing
  for model in "$@"; do
    suffix="smoke_mg_ppo_paper_${ENV_TAG}_${model}_seed${SEED}"
    read -r -a sizing <<< "$(size_args "$model")"
    CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 python train_minigrid_ppo_paper.py \
      --env_id "$ENV_ID" --model_type "$model" --seed "$SEED" \
      --total_timesteps 1024 --num_envs 8 --num_steps 128 \
      --num_minibatches 8 --update_epochs 4 --device cuda \
      --result_suffix "$suffix" --log_interval_updates 1 --overwrite \
      "${sizing[@]}" >"$ART/${model}.log" 2>&1
  done
}

run_group 0 paper_lstm rnn gawf mamba &
pid0="$!"
run_group 1 lstm_core gru s5 &
pid1="$!"
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "One or more MiniGrid paper PPO smoke units failed; inspect $ART" >&2
  exit 1
fi
echo "All seven MiniGrid paper PPO smoke units passed."
