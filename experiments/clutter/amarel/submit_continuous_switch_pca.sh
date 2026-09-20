#!/usr/bin/env bash
# Submit only; all dataset/model/PCA computation belongs on compute nodes.
set -euo pipefail
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
if (( $# != 7 )); then
  echo 'usage: [--dry-run] ROOT CKPT_ROOT DATA_DIR OUTPUT_DIR LOG_DIR RADIUS CONDA_SH' >&2
  exit 2
fi
ROOT="$1"; CKPT_ROOT="$2"; DATA_DIR="$3"; OUT="$4"; LOGS="$5"; RADIUS="$6"; CONDA_SH="$7"
[[ "$RADIUS" =~ ^[1-9][0-9]*$ ]] || exit 2
RUNNER="$ROOT/experiments/clutter/amarel/run_continuous_switch_pca.sh"
DEPENDENCY=()
if [[ -n "${AIM3_PREFLIGHT_JOB:-}" ]]; then
  [[ "$AIM3_PREFLIGHT_JOB" =~ ^[0-9]+$ ]] || exit 2
  DEPENDENCY=(--dependency="afterok:$AIM3_PREFLIGHT_JOB")
fi
COMMAND=(sbatch --parsable)
if (( ${#DEPENDENCY[@]} > 0 )); then
  COMMAND+=("${DEPENDENCY[@]}")
fi
COMMAND+=(--array=0-19%4 --output="$LOGS/%A_%a.out" --error="$LOGS/%A_%a.err"
  "$RUNNER" "$ROOT" "$CKPT_ROOT" "$DATA_DIR" "$OUT" "$RADIUS" "$CONDA_SH")
if (( DRY )); then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
test -f "$RUNNER"
test -f "$CONDA_SH"
test -d "$DATA_DIR"
test -d "$CKPT_ROOT"
test ! -e "$OUT"
mkdir -p "$OUT" "$LOGS"
"${COMMAND[@]}"
