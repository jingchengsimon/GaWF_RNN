#!/usr/bin/env bash
# Submit the 1-frame Pong sweep: 7 models x 2 settings (plain + flickering).
#
# Steps:
#   1. (once) param-match S5/Mamba cores to the LSTM anchor, writing
#      results/atari_param_match/atari_ssm_param_match.json.
#   2. sbatch the 14-task array (SLURM_ARRAY_TASK_ID 0..13).
#
# Usage:
#   bash experiments/amarel/submit_atari_pong_1frame.sh                 # all 14
#   ARRAY_CONCURRENCY=7 bash experiments/amarel/submit_atari_pong_1frame.sh
#   SKIP_PARAM_MATCH=1 bash experiments/amarel/submit_atari_pong_1frame.sh
#   SEED=1 bash experiments/amarel/submit_atari_pong_1frame.sh          # different seed
#
# To run only a subset, pass --array explicitly, e.g. only the flickering half:
#   bash experiments/amarel/submit_atari_pong_1frame.sh --array 7-13

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_atari_pong_1frame_array.sh"
ART_ROOT="$ROOT/experiments/amarel/artifacts/atari_pong_1frame"
mkdir -p "$ART_ROOT"

ARRAY_SPEC="0-13"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-14}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --array) ARRAY_SPEC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- environment (for the param-match step) --------------------------------
if [[ -n "${AIM3_SETUP_CMD:-}" ]]; then
  eval "$AIM3_SETUP_CMD"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || true
  fi
fi
export KMP_DUPLICATE_LIB_OK=TRUE

# ---- 1. param match --------------------------------------------------------
if [[ -z "${SKIP_PARAM_MATCH:-}" ]]; then
  echo "[submit] running S5/Mamba param match (LSTM anchor)..."
  python -m experiments.generalization.atari_ssm_param_match \
    --conv_out 3136 --hidden_size 512 --ssm_state_size 128 --num_layers 1
fi

# ---- 2. submit array -------------------------------------------------------
echo "[submit] sbatch --array=${ARRAY_SPEC}%${ARRAY_CONCURRENCY} $RUN_SCRIPT"
sbatch --array="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}" \
  --export=ALL,AIM3_ROOT="$ROOT",SEED="${SEED:-42}",TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}" \
  "$RUN_SCRIPT"
