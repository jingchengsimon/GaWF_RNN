#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$ROOT/experiments/amarel/run_clutter_uint8_conversion.sh"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/clutter_uint8_conversion"

if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR"

export AIM3_ROOT="$ROOT"
export AIM3_DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
export AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
export AIM3_CONDA_ENV="${AIM3_CONDA_ENV:-aim3_rnn}"

sbatch --parsable \
  --export="AIM3_ROOT,AIM3_DATA_DIR,AIM3_CONDA_INIT,AIM3_CONDA_ENV" \
  "$RUNNER"
