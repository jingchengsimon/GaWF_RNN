#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-330-smoke
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00

# Gate formal 330-unit submission on one finite GPU optimization step per model.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
ART="${AIM3_ARTIFACT_ROOT:?AIM3_ARTIFACT_ROOT is required}"
DATA="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
mkdir -p "$ART/status"
on_error() {
  code=$?
  trap - ERR
  printf 'smoke_failed exit=%s time=%s\n' "$code" "$(date -Is)" > "$ART/status/preflight.fail"
  exit "$code"
}
trap on_error ERR
for scale in 4h 10h 20h 40h; do
  [[ -s "$DATA/stimulus_reg-train-$scale-uint8.npy" ]] || {
    echo "Missing training data for $scale" >&2
    printf 'missing_training_data scale=%s\n' "$scale" > "$ART/status/preflight.fail"
    exit 1
  }
  [[ -s "$DATA/stimulus_reg-train-$scale-uint8.tsv" ]] || {
    echo "Missing training labels for $scale" >&2
    printf 'missing_training_labels scale=%s\n' "$scale" > "$ART/status/preflight.fail"
    exit 1
  }
done
for path in "$DATA/stimulus_reg-validation-40h-uint8.npy" \
  "$DATA/stimulus_reg-validation-40h-uint8.tsv"; do
  [[ -s "$path" ]] || {
    printf 'missing_validation_data path=%s\n' "$path" > "$ART/status/preflight.fail"
    exit 1
  }
done
cd "$ROOT"
GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
if [[ ! -s "$GUARD" ]]; then
  printf 'missing_source_guard\n' > "$ART/status/preflight.fail"
  exit 1
fi
# shellcheck source=../../remote/amarel_source_guard.sh
source "$GUARD" || {
  printf 'source_guard_load_failed\n' > "$ART/status/preflight.fail"
  exit 1
}
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$ART/status/source_commit.txt"; then
  printf 'source_mismatch\n' > "$ART/status/preflight.fail"
  exit 1
fi
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH" || {
  printf 'conda_init_failed\n' > "$ART/status/preflight.fail"
  exit 1
}
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || {
  printf 'conda_activate_failed\n' > "$ART/status/preflight.fail"
  exit 1
}
set -u
python -B -m experiments.clutter.amarel.preflight330 --output "$ART/preflight.json"
printf 'smoke_passed source_commit=%s time=%s\n' "$SOURCE_COMMIT" "$(date -Is)" \
  > "$ART/status/preflight.done"
trap - ERR
