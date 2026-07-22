#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-test-eval
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/amarel/artifacts/clutter_multiseed_test/%j.out
#SBATCH --error=experiments/amarel/artifacts/clutter_multiseed_test/%j.err

# Evaluate all completed fixed-best Clutter checkpoints on the 40h uint8 test split.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_model.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

if [[ -z "${AIM3_CONDA_INIT:-}" || ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "AIM3_CONDA_INIT must identify the Amarel Conda initialization script." >&2
  exit 2
fi
source "$AIM3_CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"

DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-/scratch/${USER}/results}"
CAMPAIGN_ROOT="${AIM3_CAMPAIGN_ROOT:-$RESULTS_ROOT/train_data/clutter_best6_multiseed_40h_ep150}"
ANALYSIS_DIR="$RESULTS_ROOT/anal_data/clutter_multiseed_test"
FIGURE_DIR="$RESULTS_ROOT/anal_figs/clutter_multiseed_test"
mkdir -p "$ANALYSIS_DIR" "$FIGURE_DIR"
COHORT_CSV="$ANALYSIS_DIR/per_seed_test_accuracy.csv"
if [[ ! -s "$COHORT_CSV" ]]; then
  echo "Existing test-accuracy cohort CSV is required: $COHORT_CSV" >&2
  exit 2
fi

python utils_anal/evaluate_clutter_multiseed_test.py \
  --checkpoint_root "$CAMPAIGN_ROOT" \
  --data_dir "$DATA_DIR" \
  --data_suffix 40h-uint8 \
  --seed_filter_csv "$COHORT_CSV" \
  --save_csv "$ANALYSIS_DIR/per_seed_test_metrics.csv" \
  --save_meta "$ANALYSIS_DIR/per_seed_test_metrics_meta.json" \
  --device cuda \
  --batch_size 256 \
  --num_workers 2 \
  --chan_num 2 \
  --use_mmap

python -m utils_viz.clutter_multiseed_test_bars \
  --data_csv "$ANALYSIS_DIR/per_seed_test_metrics.csv" \
  --save_png "$FIGURE_DIR/test_accuracy_mean_sd.png" \
  --save_summary_csv "$ANALYSIS_DIR/test_accuracy_mean_sd.csv" \
  --save_loss_png "$FIGURE_DIR/test_loss_mean_sd.png" \
  --save_loss_summary_csv "$ANALYSIS_DIR/test_loss_mean_sd.csv"
