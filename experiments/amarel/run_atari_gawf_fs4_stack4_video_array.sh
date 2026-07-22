#!/usr/bin/env bash
#SBATCH --job-name=aim3-pong-gawf-video
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/amarel/artifacts/atari_gawf_fs4_stack4_video/%A_%a.out
#SBATCH --error=experiments/amarel/artifacts/atari_gawf_fs4_stack4_video/%A_%a.err

# Render the best completed training seed for L1 and L2 strict fs4/stack4 GaWF Pong.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT}" || ! -f "${ROOT}/train_atari_dqn.py" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${ROOT}"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH must point to persistent Amarel storage}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ "${TASK_ID}" == "0" ]]; then
  LAYERS=1
  TRAINING_SEED="${L1_TRAINING_SEED:-1}"
  RUN_SUFFIX="atari_dqn_pong_fs4_stack4_l1_gawf_seed${TRAINING_SEED}"
elif [[ "${TASK_ID}" == "1" ]]; then
  LAYERS=2
  TRAINING_SEED="${L2_TRAINING_SEED:-1}"
  RUN_SUFFIX="atari_dqn_pong_fs4_stack4_l2match_gawf_seed${TRAINING_SEED}"
else
  echo "Expected array task 0 or 1, got ${TASK_ID}" >&2
  exit 2
fi

SOURCE_DIR="${AIM3_RESULTS_PATH}/train_data/${RUN_SUFFIX}"
METRICS_PATH="${SOURCE_DIR}/metrics.json"
OUTPUT_DIR="${AIM3_RESULTS_PATH}/train_figs/rl/atari/pong_6action/videos/"
OUTPUT_DIR+="fs4_stack4_gawf_l${LAYERS}_seed${TRAINING_SEED}"
OUTPUT_VIDEO="${OUTPUT_DIR}/fs4_stack4_gawf_l${LAYERS}_seed${TRAINING_SEED}_best_episode.mp4"
OUTPUT_META="${OUTPUT_DIR}/metadata.json"

[[ -f "${METRICS_PATH}" ]] || { echo "Missing metrics: ${METRICS_PATH}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "${CONDA_SH}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python utils_anal/evaluate_atari_dqn_video.py \
  --metrics_path "${METRICS_PATH}" \
  --output_path "${OUTPUT_VIDEO}" \
  --metadata_path "${OUTPUT_META}" \
  --num_episodes "${NUM_EVAL_EPISODES:-3}" \
  --eval_seed "${EVAL_SEED:-20260718}" \
  --fps "${VIDEO_FPS:-15}" \
  --device cuda \
  --amp_dtype bfloat16

[[ -s "${OUTPUT_VIDEO}" ]] || { echo "Missing final video: ${OUTPUT_VIDEO}" >&2; exit 3; }
[[ -s "${OUTPUT_META}" ]] || { echo "Missing video metadata: ${OUTPUT_META}" >&2; exit 3; }
echo "layers=${LAYERS} training_seed=${TRAINING_SEED} video=${OUTPUT_VIDEO}"
