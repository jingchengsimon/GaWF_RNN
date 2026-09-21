#!/usr/bin/env bash
#SBATCH --job-name=aim3-dyn-aggregate
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

# Aggregate the completed thirty-unit dynamic-baseline campaign.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
SOURCE_COMMIT_STAMP="${AIM3_SOURCE_COMMIT_FILE:-$STATUS_DIR/source_commit.txt}"
RUN_BASE="$RESULTS/data/clutter/runs/dynamic_weight_baselines/clutter_dynamic_weight_baselines_ep150_v1"
TEST_BASE="$RESULTS/data/analysis/dynamic_weight_baselines_reset_excluded_test_10seed_v1"
SUMMARY_ROOT="$RESULTS/data/analysis/dynamic_weight_baselines_formal_10seed_v1"

on_error() {
  status=$?
  trap - ERR
  printf 'status=failed exit=%s timestamp=%s\n' "$status" "$(date -Is)" \
    > "$STATUS_DIR/aggregate.fail"
  exit "$status"
}
trap on_error ERR

(( $(find "$RUN_BASE" -mindepth 2 -maxdepth 2 -name '*_model.pth' | wc -l) == 30 ))
(( $(find "$TEST_BASE" -mindepth 2 -maxdepth 2 -name reset_excluded_test_accuracy.json | wc -l) == 30 ))
[[ ! -e "$SUMMARY_ROOT" ]] || { echo "Refusing to overwrite summary root" >&2; exit 1; }

cd "$ROOT"
AMAREL_SOURCE_GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
if [[ ! -s "$AMAREL_SOURCE_GUARD" ]]; then
  printf 'status=failed timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.fail"
  printf 'Missing compute-node source guard: %s\n' "$AMAREL_SOURCE_GUARD" >&2
  exit 1
fi
# shellcheck source=../../remote/amarel_source_guard.sh
source "$AMAREL_SOURCE_GUARD"
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$SOURCE_COMMIT_STAMP"; then
  printf 'status=failed timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.fail"
  exit 1
fi
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python -B -m experiments.clutter.dynamic_weight_aggregate \
  --run-base "$RUN_BASE" --test-base "$TEST_BASE" --output-dir "$SUMMARY_ROOT"
printf 'status=done timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.done"
trap - ERR
