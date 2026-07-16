#!/usr/bin/env bash
# Submit chan=1 training with the six frozen chan=2 best hyperparameters, seeds 1-10.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AIM3_GRID_UTIL="experiments/generalization/clutter_best6_chan1_multiseed.py"
export AIM3_ARTIFACT_TAG="clutter_best6_chan1_10seed_ep150"
export AIM3_JOB_PREFIX="aim3-clut-c1"
export AIM3_RESULT_ROOT="clutter_best6_multiseed_40h_chan1_ep150"
exec "$SCRIPT_DIR/submit_clutter_best6_10seed.sh"
