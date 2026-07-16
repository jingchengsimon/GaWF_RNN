#!/usr/bin/env bash
# Check Slurm state and strict output validity for the chan=1 fixed-best 60-task run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_ID="${1:-}"

if [[ -n "$JOB_ID" ]] && command -v squeue >/dev/null 2>&1; then
  squeue -j "$JOB_ID" -o '%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R'
fi
python "$ROOT/experiments/generalization/clutter_best6_chan1_multiseed.py" \
  status --root "$ROOT"
