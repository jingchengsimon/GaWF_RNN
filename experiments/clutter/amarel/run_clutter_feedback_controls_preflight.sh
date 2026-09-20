#!/usr/bin/env bash
#SBATCH --job-name=aim3-fbctrl-preflight
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Run protocol and short optimization gates on an Amarel compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
source "$ROOT/experiments/clutter/amarel/execution_snapshot_identity.sh"
STATUS_DIR="${AIM3_STATUS_DIR:?AIM3_STATUS_DIR is required}"
PREFLIGHT_DIR="${AIM3_PREFLIGHT_DIR:?AIM3_PREFLIGHT_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
mkdir -p "$STATUS_DIR" "$PREFLIGHT_DIR"

on_error() {
  status=$?
  trap - ERR
  printf 'status=failed exit=%s timestamp=%s\n' "$status" "$(date -Is)" \
    > "$STATUS_DIR/preflight.fail"
  exit "$status"
}
trap on_error ERR

cd "$ROOT"
assert_execution_snapshot_commit "$ROOT" "$SOURCE_COMMIT"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python -B -m experiments.clutter.feedback_control_protocol_check \
  --output "$PREFLIGHT_DIR/protocol_checks.json"
CUDA_VISIBLE_DEVICES=0 python -B -m experiments.clutter.feedback_control_sanity \
  --output "$PREFLIGHT_DIR/sanity_seed1_200steps.json" --device cuda --steps 200 --seed 1
printf 'status=done source_commit=%s timestamp=%s\n' "$SOURCE_COMMIT" "$(date -Is)" \
  > "$STATUS_DIR/preflight.done"
trap - ERR
