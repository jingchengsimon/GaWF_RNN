#!/usr/bin/env bash
# Submit strict fs4/stack4 GaWF L1/L2 greedy-policy video evaluation jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_atari_gawf_fs4_stack4_video_array.sh"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-/scratch/js3269/results}"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

[[ -f "${RUNNER}" ]] || { echo "Missing runner: ${RUNNER}" >&2; exit 2; }
mkdir -p "${ROOT}/experiments/amarel/artifacts/atari_gawf_fs4_stack4_video"

SBATCH_ARGS=(
  --array=0-1
  --export="ALL,AIM3_ROOT=${ROOT},AIM3_RESULTS_PATH=${RESULTS_ROOT},AIM3_CONDA_ENV=aim3_rnn,AIM3_CONDA_SH=/home/js3269/enter/etc/profile.d/conda.sh,L1_TRAINING_SEED=1,L2_TRAINING_SEED=1,NUM_EVAL_EPISODES=3,EVAL_SEED=20260718,VIDEO_FPS=15"
  "${RUNNER}"
)

if [[ "${DRY_RUN}" == true ]]; then
  echo "sbatch ${SBATCH_ARGS[*]}"
  exit 0
fi

job_id="$(sbatch --parsable "${SBATCH_ARGS[@]}")"
echo "job_id=${job_id}"
echo "tasks=0:L1-seed1,1:L2-seed1"
echo "results=${RESULTS_ROOT}/train_figs/rl/atari/pong_6action/videos"
