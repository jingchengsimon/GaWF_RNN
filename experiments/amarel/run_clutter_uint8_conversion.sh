#!/usr/bin/env bash
#SBATCH --job-name=clutter-u8-convert
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/amarel/artifacts/clutter_uint8_conversion/%j.out
#SBATCH --error=experiments/amarel/artifacts/clutter_uint8_conversion/%j.err

set -euo pipefail

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  echo "AIM3_ROOT or SLURM_SUBMIT_DIR must identify the project root." >&2
  exit 2
fi
if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi

cd "$ROOT"
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

python source/clutter/convert_float32_to_uint8.py \
  --data-dir "${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}" \
  --source-suffix 40h-float32 \
  --target-suffix 40h-uint8 \
  --splits train validation test
