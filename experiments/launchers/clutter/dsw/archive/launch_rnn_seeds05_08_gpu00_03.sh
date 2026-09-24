#!/usr/bin/env bash
# Archival snapshot of the one-off seed5-8 acceleration launch. Do not rerun directly.

set -euo pipefail

ROOT=/mnt/workspace/sjc/aim3_gawf_rnn
ENV_ROOT=/mnt/workspace/sjc/envs/aim3_gawf_rnn_bbade07
RESULTS=/mnt/workspace/sjc/aim3_gawf_rnn_results_dsw8095
DATA=/mnt/workspace/sjc/aim3_gawf_rnn_data/stimuli
RUN_ROOT=/mnt/workspace/sjc/aim3_gawf_rnn_runs/rnn_inloop_notanh_dsw8095_v1
STATUS=$RUN_ROOT/status/formal
LOGS=$RUN_ROOT/logs
RUNNER=$ROOT/experiments/clutter/amarel/run_clutter_rnn_inloop_notanh.sh
EXPECTED_COMMIT=988ee1b0f2727189479a56e494556651b589e1ca

launch_seed() {
  local gpu="$1"
  local task_id="$2"
  local seed_tag="$3"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" AIM3_ROOT="$ROOT" AIM3_RESULTS_PATH="$RESULTS" \
    AIM3_CLUTTER_DATA_DIR="$DATA" AIM3_STATUS_DIR="$STATUS" \
    AIM3_SOURCE_COMMIT="$EXPECTED_COMMIT" \
    AIM3_SOURCE_COMMIT_FILE="$ROOT/source_commit.txt" AIM3_RUN_MODE=formal \
    AIM3_CONDA_SH=/mnt/workspace/sjc/miniconda3/etc/profile.d/conda.sh \
    AIM3_CONDA_ENV="$ENV_ROOT" AIM3_NUM_WORKERS=2 AIM3_PIN_MEMORY=1 \
    PYTHONDONTWRITEBYTECODE=1 DISABLE_TQDM=1 SLURM_ARRAY_TASK_ID="$task_id" \
    bash "$RUNNER" > "$LOGS/formal_seed${seed_tag}.out" \
    2> "$LOGS/formal_seed${seed_tag}.err" < /dev/null &
  printf '%s\n' "$!" > "$RUN_ROOT/extra_seed${seed_tag}.pid"
}

launch_seed 0 4 05
launch_seed 1 5 06
launch_seed 2 6 07
launch_seed 3 7 08
