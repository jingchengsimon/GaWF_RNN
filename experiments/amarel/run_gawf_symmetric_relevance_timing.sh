#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-relevance
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/amarel/artifacts/gawf_symmetric_relevance_timing/slurm-%j.out
#SBATCH --error=experiments/amarel/artifacts/gawf_symmetric_relevance_timing/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/cache/home/js3269/projects/aim3_gawf_rnn"
CONDA_INIT="/home/js3269/enter/etc/profile.d/conda.sh"
STIMULI_ROOT="/scratch/js3269/stimuli"
CHECKPOINT="/scratch/js3269/results/train_data/gen_hparam_full_grid/task_1007/"
CHECKPOINT+="gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model.pth"
: "${AIM3_RESULTS_PATH:=/scratch/js3269/results}"
: "${AIM3_NUM_WORKERS:=12}"
: "${AIM3_PIN_MEMORY:=1}"

source "$CONDA_INIT"
conda activate aim3_rnn
cd "$PROJECT_ROOT"

DATA_OUT="$AIM3_RESULTS_PATH/anal_data/gawf_symmetric_relevance_timing"
FIG_OUT="$AIM3_RESULTS_PATH/anal_figs/gawf_symmetric_relevance_timing"
mkdir -p "$DATA_OUT" "$FIG_OUT"

if [[ "${AIM3_PLOT_ONLY:-0}" != "1" ]]; then
  python utils_anal/gawf_symmetric_relevance_timing.py \
    --ckpt "$CHECKPOINT" \
    --data_dir "$STIMULI_ROOT" \
    --data_suffix 40h-uint8 \
    --save_dir "$DATA_OUT" \
    --device cuda \
    --batch_size 16 \
    --num_workers "$AIM3_NUM_WORKERS" \
    --gate_chunk_size 16 \
    --permutation_batch_size 10 \
    --resamples 1000 \
    --top_percent 10 20 30 \
    --post_frames 10 \
    --seed 260718
fi

MPL_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/gawf-symmetric-relevance-matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPL_CACHE_ROOT"
MPLCONFIGDIR="$MPL_CACHE_ROOT" \
python utils_viz/gawf_symmetric_relevance_timing.py \
  --data_dir "$DATA_OUT" \
  --save_dir "$FIG_OUT"
