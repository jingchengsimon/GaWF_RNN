#!/usr/bin/env bash
# Archival snapshot of the original DSW 8095 two-lane coordinator. Do not rerun directly.

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
SMOKE_STATUS=$RUN_ROOT/status/smoke/task_smoke.done
SMOKE_RESULT=$RESULTS/data/clutter/runs/preflight/clutter_rnn_inloop_notanh_40h_2epoch_seed1_v1/rnn_inloop_notanh-seed01
FORMAL_RESULT=$RESULTS/data/clutter/runs/clutter_rnn_inloop_notanh_40h_ep150_v1

[[ -s "$SMOKE_STATUS" ]] || { echo "Smoke done marker is absent" >&2; exit 1; }
shopt -s nullglob
smoke_models=("$SMOKE_RESULT"/*_model.pth)
smoke_metrics=("$SMOKE_RESULT"/*_metrics.json)
smoke_histories=("$SMOKE_RESULT"/*.pkl)
shopt -u nullglob
(( ${#smoke_models[@]} == 1 && ${#smoke_metrics[@]} == 1 \
  && ${#smoke_histories[@]} == 1 )) || {
  echo "Smoke output contract is incomplete" >&2
  exit 1
}
[[ ! -e "$FORMAL_RESULT" ]] || { echo "Formal result root already exists" >&2; exit 1; }
[[ "$(<"$ROOT/source_commit.txt")" == "$EXPECTED_COMMIT" ]] || {
  echo "Unexpected source commit" >&2
  exit 1
}

mkdir -p "$STATUS" "$LOGS"
printf 'state=running timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/formal_campaign.status"

run_seed() {
  local gpu="$1"
  local seed="$2"
  local task_id=$((seed - 1))
  local seed_tag
  printf -v seed_tag '%02d' "$seed"
  CUDA_VISIBLE_DEVICES="$gpu" \
    AIM3_ROOT="$ROOT" \
    AIM3_RESULTS_PATH="$RESULTS" \
    AIM3_CLUTTER_DATA_DIR="$DATA" \
    AIM3_STATUS_DIR="$STATUS" \
    AIM3_SOURCE_COMMIT="$EXPECTED_COMMIT" \
    AIM3_SOURCE_COMMIT_FILE="$ROOT/source_commit.txt" \
    AIM3_RUN_MODE=formal \
    AIM3_CONDA_SH=/mnt/workspace/sjc/miniconda3/etc/profile.d/conda.sh \
    AIM3_CONDA_ENV="$ENV_ROOT" \
    AIM3_NUM_WORKERS=2 AIM3_PIN_MEMORY=1 PYTHONDONTWRITEBYTECODE=1 DISABLE_TQDM=1 \
    SLURM_ARRAY_TASK_ID="$task_id" \
    bash "$RUNNER" > "$LOGS/formal_seed${seed_tag}.out" \
    2> "$LOGS/formal_seed${seed_tag}.err"
}

run_lane() {
  local gpu="$1"
  shift
  local seed
  for seed in "$@"; do
    run_seed "$gpu" "$seed"
  done
}

run_lane 4 1 3 5 7 9 & lane4=$!
run_lane 5 2 4 6 8 10 & lane5=$!
printf '4 %s\n5 %s\n' "$lane4" "$lane5" > "$RUN_ROOT/formal_lane_pids.txt"

overall=0
wait "$lane4" || overall=1
wait "$lane5" || overall=1
if (( overall == 0 )); then
  printf 'state=done timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/formal_campaign.status"
else
  printf 'state=failed timestamp=%s\n' "$(date -Is)" > "$RUN_ROOT/formal_campaign.status"
fi
exit "$overall"
