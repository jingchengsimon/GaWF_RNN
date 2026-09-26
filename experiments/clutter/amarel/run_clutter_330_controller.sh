#!/usr/bin/env bash
#SBATCH --job-name=aim3-clutter-330-control
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --requeue

# One bounded reconciliation cycle on a CPU compute node; successor is another Slurm job.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
ART="${AIM3_ARTIFACT_ROOT:?AIM3_ARTIFACT_ROOT is required}"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:?AIM3_SOURCE_COMMIT is required}"
mkdir -p "$ART/status"
cd "$ROOT"
GUARD="$ROOT/experiments/remote/amarel_source_guard.sh"
if [[ ! -s "$GUARD" ]]; then
  printf 'missing_source_guard time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
fi
# shellcheck source=../../remote/amarel_source_guard.sh
source "$GUARD" || {
  printf 'source_guard_load_failed time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
}
if ! amarel_require_source_commit "$ROOT" "$SOURCE_COMMIT" "$ART/status/source_commit.txt"; then
  printf 'source_mismatch time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
fi
if [[ ! -s "$ART/preflight.json" ]] || ! grep -Fq '"status": "passed"' "$ART/preflight.json" \
  || [[ ! -s "$ART/status/preflight.done" ]]; then
  printf 'preflight_gate_missing time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
fi
command -v sbatch >/dev/null && command -v squeue >/dev/null || {
  printf 'slurm_cli_missing time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
}
CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH" || {
  printf 'conda_init_failed time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
}
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}" || {
  printf 'conda_activate_failed time=%s\n' "$(date -Is)" > "$ART/status/controller.fail"
  exit 1
}
set -u
python -B -m experiments.clutter.amarel.rolling330 reconcile
