#!/usr/bin/env bash
# Classified entry point for the canonical DSW GaWF legacy no-tanh launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
exec bash "$ROOT/experiments/remote/run_dsw_clutter_gawf_legacy_notanh.sh" "$@"
