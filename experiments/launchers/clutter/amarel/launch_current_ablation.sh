#!/usr/bin/env bash
# Classified entry point for the current Clutter Amarel ablation launchers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: launch_current_ablation.sh TARGET [launcher arguments]

TARGET:
  nonlinearity  submit_clutter_nonlinearity_ablation.sh
  gawf-core     submit_clutter_gawf_core_variants.sh
  rnn-inloop    submit_clutter_rnn_inloop_notanh.sh
  behavior      submit_clutter_ablation_behavior.sh
EOF
}

target="${1:-}"
[[ -n "$target" ]] || { usage; exit 2; }
shift

case "$target" in
  nonlinearity)
    launcher="$ROOT/experiments/clutter/amarel/submit_clutter_nonlinearity_ablation.sh"
    ;;
  gawf-core)
    launcher="$ROOT/experiments/clutter/amarel/submit_clutter_gawf_core_variants.sh"
    ;;
  rnn-inloop)
    launcher="$ROOT/experiments/clutter/amarel/submit_clutter_rnn_inloop_notanh.sh"
    ;;
  behavior)
    launcher="$ROOT/experiments/clutter/amarel/submit_clutter_ablation_behavior.sh"
    ;;
  *)
    usage
    exit 2
    ;;
esac

exec bash "$launcher" "$@"
