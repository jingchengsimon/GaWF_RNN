#!/usr/bin/env bash
# Submit seed-42 100M paper PPO arrays for RedBlueDoors and MemoryS7.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p experiments/amarel/artifacts/minigrid_ppo_paper
RUNNER="$ROOT/experiments/amarel/run_minigrid_ppo_paper_array.sh"

: "${AIM3_RESULTS_PATH:?Set AIM3_RESULTS_PATH to the configured Amarel result root}"

COMMON_EXPORT="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1,SEED=42,TOTAL_TIMESTEPS=100000000"
redblue_id="$(sbatch --parsable --array=0-6 \
  --export="$COMMON_EXPORT,ENV_ID=MiniGrid-RedBlueDoors-8x8-v0" \
  "$RUNNER")"
memory_id="$(sbatch --parsable --array=0-6 \
  --export="$COMMON_EXPORT,ENV_ID=MiniGrid-MemoryS7-v0" \
  "$RUNNER")"
printf 'redblue_job_id=%s\nmemory_job_id=%s\n' "$redblue_id" "$memory_id"
