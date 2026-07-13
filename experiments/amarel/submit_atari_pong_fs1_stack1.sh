#!/usr/bin/env bash
# Submit the Pong frame-skip-1/stack-1 sweep: 7 models x 2 settings x 5 seeds.
#
# Steps:
#   1. (once) param-match every recurrent core to the LSTM anchor, writing
#      results/atari_param_match/atari_param_match.json.
#   2. sbatch the 70-task array (SLURM_ARRAY_TASK_ID 0..69).
#
# Usage:
#   bash experiments/amarel/submit_atari_pong_fs1_stack1.sh                 # all 70
#   ARRAY_CONCURRENCY=20 bash experiments/amarel/submit_atari_pong_fs1_stack1.sh
#   SKIP_PARAM_MATCH=1 bash experiments/amarel/submit_atari_pong_fs1_stack1.sh
#
# Subsets via --array, e.g. only the flickering half (settings map to tasks
# 35-69 for the default 5 seeds):
#   bash experiments/amarel/submit_atari_pong_fs1_stack1.sh --array 35-69

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_atari_pong_fs1_stack1_array.sh"
ART_ROOT="$ROOT/experiments/amarel/artifacts/atari_pong_fs1_stack1"
mkdir -p "$ART_ROOT"

ARRAY_SPEC="0-69"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-20}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --array) ARRAY_SPEC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- environment (for the param-match step) --------------------------------
source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
export KMP_DUPLICATE_LIB_OK=TRUE

# ---- 1. param match (all recurrent cores incl. mamba) ----------------------
if [[ -z "${SKIP_PARAM_MATCH:-}" ]]; then
  echo "[submit] param-matching cores to the LSTM anchor..."
  python -m experiments.generalization.atari_ssm_param_match \
    --conv_out 3136 --hidden_size 512 --ssm_state_size 128 --num_actions 6 --num_layers 1
fi

# ---- 2. submit array -------------------------------------------------------
echo "[submit] sbatch --array=${ARRAY_SPEC}%${ARRAY_CONCURRENCY} $RUN_SCRIPT"
sbatch --array="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}" \
  --export=ALL,AIM3_ROOT="$ROOT",TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}" \
  "$RUN_SCRIPT"
