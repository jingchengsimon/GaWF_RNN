#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_clutter_dataloader_smoke.sh"
ARTIFACT_DIR="$SCRIPT_DIR/artifacts/clutter_dataloader_smoke"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"
AIM3_DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
AIM3_RESULTS_PATH="${AIM3_RESULTS_PATH:-/scratch/${USER}/results}"
AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
AIM3_CONDA_ENV="${AIM3_CONDA_ENV:-aim3_rnn}"

SBATCH_CMD=(
  sbatch
  --parsable
  --export="AIM3_ROOT=$ROOT,AIM3_DATA_DIR=$AIM3_DATA_DIR,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_CONDA_INIT=$AIM3_CONDA_INIT,AIM3_CONDA_ENV=$AIM3_CONDA_ENV"
  "$RUNNER"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  exit 0
fi

"${SBATCH_CMD[@]}"
