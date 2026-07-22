#!/usr/bin/env bash
# Submit the GaWF symmetric relevance/timing analysis to an Amarel compute node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PROJECT_ROOT/experiments/amarel/run_gawf_symmetric_relevance_timing.sh"
RESULTS_ROOT="/scratch/js3269/results"
DRY_RUN=0
PLOT_ONLY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" == "--plots-only" ]]; then
  PLOT_ONLY=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run|--plots-only]" >&2
  exit 2
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "Missing runner: $RUNNER" >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/experiments/amarel/artifacts/gawf_symmetric_relevance_timing"

EXPORTS="ALL,AIM3_RESULTS_PATH=$RESULTS_ROOT,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
EXPORTS+=",AIM3_PLOT_ONLY=$PLOT_ONLY"
COMMAND=(
  sbatch
  --export="$EXPORTS"
  "$RUNNER"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
