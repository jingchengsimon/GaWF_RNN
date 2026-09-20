#!/usr/bin/env bash
#SBATCH --job-name=aim3-fbctrl-aggregate
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Aggregate the completed forty-unit Amarel feedback-control campaign.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/execution_snapshot_identity.sh"
ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
PREREQ="${AIM3_PREREQ_ROOT:?AIM3_PREREQ_ROOT is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
RUN_BASE="$RESULTS/data/clutter/runs/feedback_controls/clutter_feedback_controls_ep150_v1"
TEST_BASE="$RESULTS/data/analysis/feedback_controls_reset_excluded_test_10seed_v1"
ABLATION_BASE="$RESULTS/data/analysis/feedback_controls_shuffle_resetexcluded_10seed_v1"
SUMMARY_ROOT="$RESULTS/data/analysis/feedback_controls_formal_10seed_v1"

on_error() {
  status=$?
  trap - ERR
  printf 'status=failed exit=%s timestamp=%s\n' "$status" "$(date -Is)" \
    > "$STATUS_DIR/aggregate.fail"
  exit "$status"
}
trap on_error ERR

(( $(find "$RUN_BASE" -mindepth 2 -maxdepth 2 -name '*_model.pth' | wc -l) == 40 ))
(( $(find "$TEST_BASE" -mindepth 2 -maxdepth 2 -name reset_excluded_test_accuracy.json | wc -l) == 40 ))
(( $(find "$ABLATION_BASE" -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 40 ))
[[ -f "$PREREQ/original_test/reset_excluded_test_accuracy_10seed.csv" ]]
(( $(find "$PREREQ/gawf_shuffle" -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 10 ))
[[ ! -e "$SUMMARY_ROOT/final" ]] || { echo "Refusing to overwrite final summary" >&2; exit 1; }

cd "$ROOT"
assert_execution_snapshot_commit "$ROOT" "$SOURCE_COMMIT"
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

mkdir -p "$SUMMARY_ROOT"
python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy aggregate \
  --data_root "$TEST_BASE" \
  --output_csv "$SUMMARY_ROOT/feedback_controls_reset_excluded_test_accuracy_10seed.csv" \
  --models gawf_additive rnn_fb gru_fb lstm_fb
python -B -m utils.analysis.clutter.feedback_control_summary \
  --original_test_csv "$PREREQ/original_test/reset_excluded_test_accuracy_10seed.csv" \
  --feedback_test_csv "$SUMMARY_ROOT/feedback_controls_reset_excluded_test_accuracy_10seed.csv" \
  --gawf_ablation_dir "$PREREQ/gawf_shuffle" \
  --feedback_ablation_dir "$ABLATION_BASE" --output_dir "$SUMMARY_ROOT/final"
touch "$SUMMARY_ROOT/final/.complete"
printf 'status=done timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.done"
trap - ERR
