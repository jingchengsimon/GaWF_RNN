#!/usr/bin/env bash
#SBATCH --job-name=aim3-skiing-unclip
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
# Compute-only wrapper for the existing Skiing weights-only training protocol.
set -euo pipefail
: "${AIM3_ROOT:?}" "${AIM3_RESULTS_PATH:?}" "${SKIING_SOURCE_ROOT:?}"
: "${AIM3_CONDA_SH:?}" "${RESULT_PARENT:?}" "${RUN_PHASE:?}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Requires a Slurm compute allocation' >&2; exit 2; }
MODELS=(lstm gru gawf)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
[[ "$TASK_ID" =~ ^[012]$ ]] || exit 2
MODEL="${MODELS[$TASK_ID]}"
cd "$AIM3_ROOT"
set +u
source "$AIM3_CONDA_SH"
conda activate aim3_rnn
set -u
export PYTHONDONTWRITEBYTECODE=1
export AIM3_NUM_WORKERS=12 AIM3_PIN_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULT_PARENT"
python -m experiments.rl.atari.amarel.scratch_quota_guard \
  --user "$USER" --filesystem scratch --required_gib 12 --headroom_factor 3 \
  --marker_path "$RESULT_PARENT/quota_${RUN_PHASE}_${MODEL}.json"
(cd "$SKIING_SOURCE_ROOT" && sha256sum -c SHA256SUMS)
EXTRA=()
[[ "$MODEL" != gawf ]] || EXTRA+=(--allow-incomplete-source)
case "$RUN_PHASE" in
  smoke) EXTRA+=(--smoke --skip-smoke-video); BUDGET=smoke25k ;;
  formal) BUDGET=4m ;;
  *) exit 2 ;;
esac
COMMAND=(bash experiments/remote/run_sjc_atari_skiing_warmstart_l3.sh \
  --model "$MODEL" --cuda-device 0 \
  --source-checkpoint "$SKIING_SOURCE_ROOT/$MODEL/model.pth" \
  --source-metrics "$SKIING_SOURCE_ROOT/$MODEL/metrics.json" \
  --results-root "$AIM3_RESULTS_PATH" --result-parent "$RESULT_PARENT/$RUN_PHASE" \
  --run-tag "atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1_unclipped_g0999_${BUDGET}_${MODEL}_seed1" \
  --total-timesteps 4000000 --gamma 0.999 --no-reward-clip \
  --keep-replay-on-success --requeue-on-pause)
if (( ${#EXTRA[@]} > 0 )); then
  COMMAND+=("${EXTRA[@]}")
fi
exec "${COMMAND[@]}"
