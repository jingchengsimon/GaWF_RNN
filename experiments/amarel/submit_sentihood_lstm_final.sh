#!/usr/bin/env bash
# Submit the first SentiHood LSTM-Final reproducibility job.
#
# Usage:
#   bash experiments/amarel/submit_sentihood_lstm_final.sh
#
# Prerequisite on an Amarel login node:
#   source /home/js3269/enter/etc/profile.d/conda.sh
#   conda activate aim3_rnn
#   python source/text/prepare_sentihood_data.py --data_dir /scratch/$USER/stimuli

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-sentihood_lstm_final}"
RUN_SCRIPT="$SCRIPT_DIR/run_sentihood_lstm_final.sh"
ART_DIR="$ROOT/experiments/amarel/artifacts/$RUN_TAG"
mkdir -p "$ART_DIR"

export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-12}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"
AIM3_SETUP_CMD="source /home/js3269/enter/etc/profile.d/conda.sh && conda activate aim3_rnn"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

export_arg="ALL"
export_arg+=",AIM3_ROOT=$ROOT"
export_arg+=",AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS"
export_arg+=",AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
export_arg+=",AIM3_SETUP_CMD=$AIM3_SETUP_CMD"
export_arg+=",RUN_TAG=$RUN_TAG"

job_id="$(
  sbatch --parsable \
    --job-name=aim3-sentihood-lstm \
    --partition=gpu-redhat \
    --account=general \
    --gres=gpu:1 \
    --constraint=adalovelace \
    --cpus-per-task=16 \
    --mem=64G \
    --time=12:00:00 \
    --output="$ART_DIR/%A.out" \
    --error="$ART_DIR/%A.err" \
    --export="$export_arg" \
    "$RUN_SCRIPT"
)"

echo "job_id=$job_id"
echo "run_tag=$RUN_TAG"
echo "task=sentihood_lstm_final"
echo "result_suffix=$RUN_TAG"
echo "resources=partition=gpu-redhat gres=gpu:1 constraint=adalovelace cpus=16 mem=64G"
echo "env=AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
echo "setup=$AIM3_SETUP_CMD"
echo "next=python experiments/amarel/check_sentihood_lstm_final_status.py"
echo "dashboard=python experiments/amarel/register_sentihood_lstm_final_dashboard.py --job_id $job_id"
