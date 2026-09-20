#!/usr/bin/env bash
# Submit one independent RNN fixed-point analysis per seed, without login-node workloads.
set -euo pipefail
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# == 7 )) || { echo 'usage: [--dry-run] ROOT CKPT_ROOT EVENTS MOVIE OUT LOGS CONDA_SH' >&2; exit 2; }
ROOT="$1"; CKPT_ROOT="$2"; EVENTS="$3"; MOVIE="$4"; OUT="$5"; LOGS="$6"; CONDA_SH="$7"
COMMAND=(sbatch --parsable --array=1-10%2 --output="$LOGS/%A_%a.out" --error="$LOGS/%A_%a.err"
 "$ROOT/experiments/clutter/amarel/run_rnn_fixed_points.sh" "$ROOT" "$CKPT_ROOT" "$EVENTS" "$MOVIE" "$OUT" "$CONDA_SH")
if (( DRY )); then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
test -f "$MOVIE"
test -d "$EVENTS"
test -d "$CKPT_ROOT"
test ! -e "$OUT"
mkdir -p "$OUT" "$LOGS"
"${COMMAND[@]}"
