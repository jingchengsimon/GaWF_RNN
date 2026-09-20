#!/usr/bin/env bash
# Submit three 25k smoke units and a dependent three-model 4M Skiing array.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${AIM3_ROOT:?}" "${AIM3_RESULTS_PATH:?}" "${SKIING_SOURCE_ROOT:?}" "${AIM3_CONDA_SH:?}"
DRY_RUN=false
SKIP_SMOKE=false
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --skip-smoke) SKIP_SMOKE=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
RESULT_PARENT="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/single_skiing/unclipped_gamma0p999_4m_seed1"
ARTIFACT_ROOT="$AIM3_RESULTS_PATH/artifacts/rl/atari/skiing_unclipped_gamma0p999_4m_seed1"
RUNNER="$AIM3_ROOT/experiments/rl/atari/amarel/run_atari_skiing_unclipped_l3.sh"
for p in "$AIM3_ROOT" "$AIM3_RESULTS_PATH" "$SKIING_SOURCE_ROOT" "$AIM3_CONDA_SH"; do
  [[ "$p" == /* && "$p" != *','* ]] || { echo 'Requires absolute paths without commas' >&2; exit 2; }
done
for model in lstm gru gawf; do
  [[ -s "$SKIING_SOURCE_ROOT/$model/model.pth" && -s "$SKIING_SOURCE_ROOT/$model/metrics.json" ]] || exit 2
done
[[ -f "$RUNNER" && -s "$SKIING_SOURCE_ROOT/SHA256SUMS" ]] || exit 2
[[ ! -e "$RESULT_PARENT" && ! -e "$ARTIFACT_ROOT" ]] || {
  echo 'Refusing an existing experiment destination; use registered recovery workflow' >&2; exit 3;
}
EXPORTS="ALL,AIM3_ROOT=$AIM3_ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
EXPORTS+=",SKIING_SOURCE_ROOT=$SKIING_SOURCE_ROOT,AIM3_CONDA_SH=$AIM3_CONDA_SH"
EXPORTS+=",RESULT_PARENT=$RESULT_PARENT,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"
if [[ "$DRY_RUN" == true ]]; then
  echo "skip_smoke=$SKIP_SMOKE; formal: lstm/gru/gawf seed1, 4M each"
  echo "protocol: BF16, gamma=0.999, no reward clip, original weights-only initialization"
  echo "resources/unit: 1 Ada GPU, 16 CPUs, 64G; smoke 2h, formal 72h; concurrency 3"
  echo "results: $RESULT_PARENT"
  echo "artifacts: $ARTIFACT_ROOT"
  exit 0
fi
mkdir -p "$ARTIFACT_ROOT/formal"
DEPENDENCY=()
printf 'SKIP_SMOKE=%s\n' "$SKIP_SMOKE" > "$ARTIFACT_ROOT/submission.txt"
if [[ "$SKIP_SMOKE" == false ]]; then
  mkdir -p "$ARTIFACT_ROOT/smoke"
  SMOKE_RAW="$(sbatch --parsable --array=0-2%3 --time=02:00:00 \
    --chdir="$AIM3_ROOT" --output="$ARTIFACT_ROOT/smoke/%A_%a.out" \
    --error="$ARTIFACT_ROOT/smoke/%A_%a.err" --export="$EXPORTS,RUN_PHASE=smoke" "$RUNNER")"
  SMOKE_ID="${SMOKE_RAW%%;*}"
  printf 'SMOKE_JOB_ID=%s\n' "$SMOKE_ID" | tee -a "$ARTIFACT_ROOT/submission.txt"
  DEPENDENCY=(--dependency="afterok:$SMOKE_ID")
fi
FORMAL_SBATCH=(sbatch --parsable --array=0-2%3)
if (( ${#DEPENDENCY[@]} > 0 )); then
  FORMAL_SBATCH+=("${DEPENDENCY[@]}")
fi
FORMAL_RAW="$("${FORMAL_SBATCH[@]}" \
  --chdir="$AIM3_ROOT" --output="$ARTIFACT_ROOT/formal/%A_%a.out" \
  --error="$ARTIFACT_ROOT/formal/%A_%a.err" --export="$EXPORTS,RUN_PHASE=formal" "$RUNNER")"
printf 'FORMAL_JOB_ID=%s\n' "${FORMAL_RAW%%;*}" | tee -a "$ARTIFACT_ROOT/submission.txt"
