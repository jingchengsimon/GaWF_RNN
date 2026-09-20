#!/usr/bin/env bash
#SBATCH --job-name=clutter-continuous-pca
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# One independent, frozen-checkpoint inference stream per model and seed.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$1"
CKPT_ROOT="$2"
DATA_DIR="$3"
OUT="$4"
RADIUS="$5"
CONDA_SH="$6"
TASK="${SLURM_ARRAY_TASK_ID:?}"
MODEL=rnn
TAG=rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5_model.pth
if (( TASK >= 10 )); then
  MODEL=gawf
  TAG=gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth
fi
printf -v SEED '%02d' "$(( TASK % 10 + 1 ))"
export AIM3_RESULTS_PATH="${OUT%/data/analysis/*}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
cd "$ROOT"
set +u
source "$CONDA_SH"
conda activate aim3_rnn
set -u
test -n "${SLURM_JOB_ID:-}"
python -B -m utils.analysis.clutter.continuous_switch_pca collect \
  --ckpt "$CKPT_ROOT/$MODEL-seed$SEED/$TAG" \
  --data_dir "$DATA_DIR" --data_suffix 40h-float32-jointswitch-balanced-10digit-unique \
  --device cuda --radius "$RADIUS" --chunk_size 32 \
  --model "$MODEL" --seed "$(( TASK % 10 + 1 ))" \
  --output_dir "$OUT/$MODEL-seed$SEED"
