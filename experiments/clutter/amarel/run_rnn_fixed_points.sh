#!/usr/bin/env bash
#SBATCH --job-name=clutter-rnn-fixedpoints
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$1"; CKPT_ROOT="$2"; EVENTS="$3"; MOVIE="$4"; OUT="$5"; CONDA_SH="$6"
export AIM3_RESULTS_PATH="${OUT%/data/analysis/*}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
cd "$ROOT"
set +u
source "$CONDA_SH"
conda activate aim3_rnn
set -u
TASK="${SLURM_ARRAY_TASK_ID:?}"
printf -v SEED '%02d' "$TASK"
python -B -m pytest -q experiments/tests/test_rnn_fixed_points.py
python -B -m utils.analysis.clutter.rnn_fixed_points \
 --ckpt "$CKPT_ROOT/rnn-seed$SEED/rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5_model.pth" \
 --event_dir "$EVENTS/rnn-seed$SEED" --movie "$MOVIE" --save_dir "$OUT/rnn-seed$SEED" \
 --seed "$TASK" --device cuda --starts 64 --iterations 2000
