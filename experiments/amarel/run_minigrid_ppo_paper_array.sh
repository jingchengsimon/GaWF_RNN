#!/usr/bin/env bash
#SBATCH --job-name=aim3-mg-paper
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/amarel/artifacts/minigrid_ppo_paper/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/minigrid_ppo_paper/%A_%a.err

# One seed-42 paper-aligned PPO array for one MiniGrid environment. The strict
# paper LSTM is task 0; tasks 1-6 replace only its recurrent core.

set -euo pipefail

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_minigrid_ppo_paper.py" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
export AIM3_NUM_WORKERS=12
export AIM3_PIN_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must be exported at submission}"

MODELS=(paper_lstm lstm_core rnn gru gawf s5 mamba)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"
ENV_ID="${ENV_ID:?ENV_ID must be exported at submission}"
SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
ENV_TAG="${ENV_ID#MiniGrid-}"
ENV_TAG="${ENV_TAG%-v0}"
SUFFIX="mg_ppo_paper_${ENV_TAG}_fov3_${MODEL}_seed${SEED}_100m"
ART="$ROOT/experiments/amarel/artifacts/minigrid_ppo_paper"
STATUS_DIR="$ART/status"
SAVE_DIR="$AIM3_RESULTS_PATH/train_data/$SUFFIX"
mkdir -p "$ART" "$STATUS_DIR"

case "$MODEL" in
  paper_lstm|lstm_core) SIZE_ARGS=(--hidden_size 128) ;;
  rnn) SIZE_ARGS=(--hidden_size 304) ;;
  gru) SIZE_ARGS=(--hidden_size 155) ;;
  gawf) SIZE_ARGS=(--hidden_size 297) ;;
  s5) SIZE_ARGS=(--ssm_d_model 408 --ssm_state_size 64) ;;
  mamba) SIZE_ARGS=(--ssm_d_model 110 --ssm_state_size 64) ;;
  *) echo "Unknown model $MODEL" >&2; exit 2 ;;
esac

STATUS_FILE="$STATUS_DIR/${SUFFIX}.status"
echo "status=running model=$MODEL seed=$SEED env=$ENV_ID $(date -Is)" > "$STATUS_FILE"
set +e
DISABLE_TQDM=1 python train_minigrid_ppo_paper.py \
  --env_id "$ENV_ID" --model_type "$MODEL" --seed "$SEED" \
  --total_timesteps "$TOTAL_TIMESTEPS" --num_envs 8 --num_steps 128 \
  --num_minibatches 8 --update_epochs 4 --device cuda \
  --result_suffix "$SUFFIX" --save_dir "$SAVE_DIR" "${SIZE_ARGS[@]}"
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  echo "status=fail model=$MODEL seed=$SEED rc=$rc $(date -Is)" > "$STATUS_FILE"
  exit "$rc"
fi
echo \
  "status=done model=$MODEL seed=$SEED metrics=$SAVE_DIR/metrics.json" \
  > "$STATUS_FILE"
