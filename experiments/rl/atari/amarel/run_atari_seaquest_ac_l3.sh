#!/usr/bin/env bash
#SBATCH --job-name=aim3-sea-ac
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
set -euo pipefail
: "${AIM3_ROOT:?}" "${AIM3_RESULTS_PATH:?}" "${AIM3_CONDA_SH:?}"
: "${SWEEP_RESULTS:?}" "${SWEEP_ARTIFACTS:?}" "${SLURM_ARRAY_TASK_ID:?}"
cd "$AIM3_ROOT"
set +u
source "$AIM3_CONDA_SH"
conda activate aim3_rnn
set -u
export PYTHONDONTWRITEBYTECODE=1 AIM3_NUM_WORKERS=12 AIM3_PIN_MEMORY=1
export DISABLE_TQDM=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -B -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "$USER" --filesystem scratch --required_gib 27 --headroom_factor 1.2 \
  --marker_path "$SWEEP_ARTIFACTS/quota_${SLURM_ARRAY_TASK_ID}.txt"
python -B -m experiments.rl.atari.amarel.seaquest_ac_diagnostic \
  --cell "$SLURM_ARRAY_TASK_ID" --results "$SWEEP_RESULTS" --artifacts "$SWEEP_ARTIFACTS" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
wait "$child"
rc=$?
while (( rc >= 128 )) && kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
exit "$rc"
