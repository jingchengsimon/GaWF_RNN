#!/usr/bin/env bash
#SBATCH --job-name=clutter-dl-smoke
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/amarel/artifacts/clutter_dataloader_smoke/%j.out
#SBATCH --error=experiments/amarel/artifacts/clutter_dataloader_smoke/%j.err

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

DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
OUTPUT_DIR="$RESULTS_ROOT/benchmarks/clutter_dataloader/amarel_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

for split in train validation; do
  if [[ ! -f "$DATA_DIR/stimulus_reg-${split}-40h-uint8.npy" ]]; then
    echo "Missing uint8 ${split} data under $DATA_DIR" >&2
    exit 2
  fi
done

python experiments/clutter/benchmark_dataloader_pipeline.py \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_DIR/loader.json" \
  --device cuda \
  --batch-size 256 \
  --num-workers 2 \
  --pin-memory \
  --warmup-batches 2 \
  --num-batches 8 \
  --mode loader \
  --variants uint8_sample_stacked_global uint8_device_compact_block256

python experiments/clutter/benchmark_dataloader_pipeline.py \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_DIR/e2e_amp.json" \
  --device cuda \
  --batch-size 8 \
  --num-workers 2 \
  --pin-memory \
  --warmup-batches 1 \
  --num-batches 2 \
  --mode e2e \
  --variants uint8_sample_stacked_global uint8_device_compact_block256

echo "status=complete"
echo "output_dir=$OUTPUT_DIR"
