#!/usr/bin/env bash
#SBATCH --job-name=aim3-dyn-preagg
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Aggregate the four completed two-epoch sanity reports on a compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
REPORTS_DIR="${AIM3_PREFLIGHT_REPORTS_DIR:?AIM3_PREFLIGHT_REPORTS_DIR is required}"
SUMMARY="${AIM3_PREFLIGHT_SUMMARY:?AIM3_PREFLIGHT_SUMMARY is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
SOURCE_COMMIT_STAMP="${AIM3_SOURCE_COMMIT_FILE:-$STATUS_DIR/source_commit.txt}"

on_error() {
  status=$?
  trap - ERR
  printf 'status=failed exit=%s timestamp=%s\n' "$status" "$(date -Is)" \
    > "$STATUS_DIR/aggregate.fail"
  exit "$status"
}
trap on_error ERR

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
source "$CONDA_SH" || {
  printf 'status=failed timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.fail"
  printf 'Cannot initialize Conda from %s\n' "$CONDA_SH" >&2
  exit 1
}
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python -B -m experiments.clutter.dynamic_weight_preflight_aggregate \
  --reports-root "$REPORTS_DIR" --source-commit "$SOURCE_COMMIT" --output "$SUMMARY"
printf 'status=done timestamp=%s\n' "$(date -Is)" > "$STATUS_DIR/aggregate.done"
trap - ERR
