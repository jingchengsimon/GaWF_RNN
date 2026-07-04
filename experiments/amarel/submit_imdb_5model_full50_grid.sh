#!/usr/bin/env bash
# Submit the unified IMDB five-model 50-epoch grid without early stopping.
#
# Usage:
#   bash experiments/amarel/submit_imdb_5model_full50_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

GRID_NAME="imdb_5model_full50_grid"
GRID_UTIL_REL="experiments/generalization/imdb_5model_full50_grid.py"
GRID_UTIL="$ROOT/$GRID_UTIL_REL"
RUN_SCRIPT="$SCRIPT_DIR/run_imdb_hparam_grid_array.sh"
ART_DIR="$ROOT/experiments/amarel/artifacts/$GRID_NAME"
TASK_LIST_DIR="$ART_DIR/task_lists"
mkdir -p "$ART_DIR" "$TASK_LIST_DIR"

export AIM3_NUM_WORKERS=12
export AIM3_PIN_MEMORY=1
AIM3_SETUP_CMD="source /home/js3269/enter/etc/profile.d/conda.sh && conda activate aim3_rnn"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

total_grid_tasks="$(
  GRID_UTIL_PATH="$GRID_UTIL" python - <<'PY'
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("grid", os.environ["GRID_UTIL_PATH"])
mod = importlib.util.module_from_spec(spec)
sys.modules["grid"] = mod
spec.loader.exec_module(mod)
print(mod.TOTAL_TASKS)
PY
)"
if [[ "$total_grid_tasks" != "80" ]]; then
  echo "Unexpected total_grid_tasks=$total_grid_tasks" >&2
  exit 2
fi

TASK_LIST_FILE="$TASK_LIST_DIR/tasks_$(date +%Y%m%d_%H%M%S).txt"
seq 0 "$((total_grid_tasks - 1))" > "$TASK_LIST_FILE"

export_arg="ALL"
export_arg+=",AIM3_ROOT=$ROOT"
export_arg+=",AIM3_IMDB_GRID_UTIL=$GRID_UTIL_REL"
export_arg+=",AIM3_IMDB_GRID_NAME=$GRID_NAME"
export_arg+=",AIM3_NUM_WORKERS=$AIM3_NUM_WORKERS"
export_arg+=",AIM3_PIN_MEMORY=$AIM3_PIN_MEMORY"
export_arg+=",AIM3_SETUP_CMD=$AIM3_SETUP_CMD"
export_arg+=",TASK_ID_FILE=$TASK_LIST_FILE"
export_arg+=",TASK_FILE_OFFSET=0"

job_id="$(
  sbatch --parsable \
    --job-name=aim3-imdb-full50 \
    --partition=gpu-redhat \
    --account=general \
    --gres=gpu:1 \
    --constraint=adalovelace \
    --cpus-per-task=16 \
    --mem=64G \
    --time=24:00:00 \
    --output="$ART_DIR/%A_%a.out" \
    --error="$ART_DIR/%A_%a.err" \
    --export="$export_arg" \
    --array="0-79%16" \
    "$RUN_SCRIPT"
)"

echo "job_id=$job_id"
echo "grid=$GRID_NAME"
echo "tasks=0-79%16"
echo "task_list=$TASK_LIST_FILE"
echo "resources=partition=gpu-redhat gres=gpu:1 constraint=adalovelace cpus=16 mem=64G"
echo "env=AIM3_NUM_WORKERS=12 AIM3_PIN_MEMORY=1 AIM3_SETUP_CMD=$AIM3_SETUP_CMD"
