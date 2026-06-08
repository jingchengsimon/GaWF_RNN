#!/usr/bin/env bash
# Submit GaWF legacy + dz comparison on Amarel as a small Slurm array.
#
# From repo root:
#   bash experiments/amarel/submit_gawf_dz_compare.sh
#
# Optional environment overrides:
#   ARRAY_CONCURRENCY=5
#   CONDITION_LIST="legacy dz8 dz16 dz32 dz64"
#   SCALE=40h HIDDEN_SIZE=256 LR=0.005 WD=0.001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_gawf_dz_compare_array.sh"
CONDITION_LIST="${CONDITION_LIST:-legacy dz8 dz16 dz32 dz64}"
read -r -a CONDITIONS <<< "$CONDITION_LIST"
TOTAL_TASKS="${#CONDITIONS[@]}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-$TOTAL_TASKS}"
if [[ "$ARRAY_CONCURRENCY" -gt "$TOTAL_TASKS" ]]; then
  ARRAY_CONCURRENCY="$TOTAL_TASKS"
fi

LOG_DIR="$ROOT/experiments/amarel/artifacts/gawf_dz_compare"
SUBMIT_LOG="$LOG_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

log() {
  printf '%s\n' "$*" | tee -a "$SUBMIT_LOG"
}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

log "AIM3 GaWF dz comparison submission"
log "timestamp=$(date -Is)"
log "root=$ROOT"
log "conditions=${CONDITIONS[*]}"
log "total_tasks=$TOTAL_TASKS"
log "array_concurrency=$ARRAY_CONCURRENCY"
log "run_script=$RUN_SCRIPT"
log "submit_log=$SUBMIT_LOG"
log "scale=${SCALE:-40h}"
log "h=${HIDDEN_SIZE:-256} lr=${LR:-0.005} wd=${WD:-0.001}"
log "result_suffix=${RESULT_SUFFIX:-gawf_dz_compare_${SCALE:-40h}_fullfb}"

job_id="$(
  sbatch --parsable \
    --export=ALL,AIM3_ROOT="$ROOT",CONDITION_LIST="$CONDITION_LIST" \
    --array="0-$((TOTAL_TASKS - 1))%${ARRAY_CONCURRENCY}" \
    "$RUN_SCRIPT"
)"
log "Submitted job_id=$job_id"
log "Check with:"
log "  squeue -j $job_id"
log "  bash experiments/amarel/summarize_gawf_dz_compare.sh"
