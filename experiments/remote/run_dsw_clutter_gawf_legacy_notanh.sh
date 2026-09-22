#!/usr/bin/env bash

# Launch the ten formal Clutter gawf_legacy_notanh seeds on an eight-GPU DSW node.
# Seeds 1-2 share GPU0, seeds 3-4 share GPU1, and seeds 5-10 use GPUs 2-7.

set -euo pipefail

ROOT="${AIM3_ROOT:-/mnt/workspace/sjc/aim3_gawf_rnn}"
ENV_ROOT="${AIM3_ENV_ROOT:-/mnt/workspace/sjc/envs/aim3_gawf_rnn_bbade07}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-/mnt/workspace/sjc/aim3_gawf_rnn_data/stimuli}"
RESULTS="${AIM3_RESULTS_PATH:-/mnt/workspace/sjc/aim3_gawf_rnn_results}"
RUN_ROOT="${AIM3_RUN_ROOT:-/mnt/workspace/sjc/aim3_gawf_rnn_runs/gawf_legacy_notanh_10seed_v1}"
RUN_ID="${AIM3_RUN_ID:-aim3_gawf_legacy_notanh_10seed_v1}"
SOURCE_COMMIT="988ee1b0f2727189479a56e494556651b589e1ca"
LEAF="clutter_ablate_legacy_notanh_40h_ep150_dsw5000_v1"
GPU_MAP=(0 0 1 1 2 3 4 5 6 7)

usage() {
  echo "Usage: $0 launch|run|status" >&2
}

validate() {
  [[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing Python: $ENV_ROOT/bin/python" >&2; exit 1; }
  [[ -f "$ROOT/run_task.py" ]] || { echo "Missing project: $ROOT" >&2; exit 1; }
  [[ "$(<"$ROOT/source_commit.txt")" == "$SOURCE_COMMIT" ]] || {
    echo "Unexpected source commit in $ROOT/source_commit.txt" >&2
    exit 1
  }
  local name
  for name in train validation test; do
    [[ -s "$DATA_DIR/stimulus_reg-${name}-40h-uint8.npy" ]] || {
      echo "Missing ${name} npy in $DATA_DIR" >&2
      exit 1
    }
    [[ -s "$DATA_DIR/stimulus_reg-${name}-40h-uint8.tsv" ]] || {
      echo "Missing ${name} tsv in $DATA_DIR" >&2
      exit 1
    }
  done
}

run_all() {
  validate
  mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$RESULTS"
  printf '%s\n' "$SOURCE_COMMIT" > "$RUN_ROOT/source_commit.txt"
  printf '%s\n' "${GPU_MAP[*]}" > "$RUN_ROOT/gpu_map.txt"
  export PYTHONDONTWRITEBYTECODE=1 DISABLE_TQDM=1 AIM3_NUM_WORKERS=2 AIM3_PIN_MEMORY=1
  cd "$ROOT"

  local pids=() seed gpu seed_tag suffix log status pid rc overall=0
  for seed in $(seq 1 10); do
    gpu="${GPU_MAP[$((seed - 1))]}"
    printf -v seed_tag '%02d' "$seed"
    suffix="$LEAF/gawf_legacy_notanh-seed$seed_tag"
    log="$RUN_ROOT/logs/seed${seed_tag}.log"
    status="$RUN_ROOT/status/seed${seed_tag}.status"
    printf 'state=starting seed=%s gpu=%s timestamp=%s\n' "$seed" "$gpu" "$(date -Is)" > "$status"
    CUDA_VISIBLE_DEVICES="$gpu" "$ENV_ROOT/bin/python" -B run_task.py clutter \
      --model_types gawf_legacy_notanh --hidden_sizes 256 \
      --num_layers 1 --num_epochs 150 --patience 0 \
      --lrs 0.005 --wds 0.001 --optim adamw \
      --cnn_dropout 0.0 --rnn_dropout 0.5 \
      --gawf_feedback_lr_scale 1.0 \
      --s5_num_layers 1 --s5_dropout 0.0 --s5_ssm_lr_scale 0.1 \
      --seed "$seed" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
      --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
      --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
      --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
      --checkpoint_interval_epochs 5 --auto_resume \
      --result_suffix "$suffix" > "$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    printf 'state=running seed=%s gpu=%s pid=%s timestamp=%s\n' \
      "$seed" "$gpu" "$pid" "$(date -Is)" > "$status"
    printf '%s %s %s\n' "$seed" "$gpu" "$pid"
  done
  printf '%s\n' "${pids[*]}" > "$RUN_ROOT/pids.txt"

  for seed in $(seq 1 10); do
    pid="${pids[$((seed - 1))]}"
    printf -v seed_tag '%02d' "$seed"
    status="$RUN_ROOT/status/seed${seed_tag}.status"
    if wait "$pid"; then
      rc=0
      printf 'state=done seed=%s gpu=%s pid=%s exit=%s timestamp=%s\n' \
        "$seed" "${GPU_MAP[$((seed - 1))]}" "$pid" "$rc" "$(date -Is)" > "$status"
    else
      rc=$?
      overall=1
      printf 'state=failed seed=%s gpu=%s pid=%s exit=%s timestamp=%s\n' \
        "$seed" "${GPU_MAP[$((seed - 1))]}" "$pid" "$rc" "$(date -Is)" > "$status"
    fi
  done
  printf 'state=%s timestamp=%s\n' "$([[ "$overall" -eq 0 ]] && echo done || echo failed)" \
    "$(date -Is)" > "$RUN_ROOT/campaign.status"
  exit "$overall"
}

launch() {
  validate
  command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 1; }
  if [[ -s "$RUN_ROOT/coordinator.pid" ]]; then
    local existing_pid
    existing_pid="$(<"$RUN_ROOT/coordinator.pid")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "Coordinator already exists: $existing_pid" >&2
      exit 1
    fi
  fi
  mkdir -p "$RUN_ROOT"
  nohup setsid env \
    AIM3_ROOT="$ROOT" AIM3_ENV_ROOT="$ENV_ROOT" AIM3_CLUTTER_DATA_DIR="$DATA_DIR" \
    AIM3_RESULTS_PATH="$RESULTS" AIM3_RUN_ROOT="$RUN_ROOT" AIM3_RUN_ID="$RUN_ID" \
    bash "$ROOT/experiments/remote/run_dsw_clutter_gawf_legacy_notanh.sh" run \
    > "$RUN_ROOT/coordinator.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$RUN_ROOT/coordinator.pid"
  sleep 3
  kill -0 "$(<"$RUN_ROOT/coordinator.pid")"
  "$0" status
}

status() {
  echo "run_id=$RUN_ID"
  if [[ -s "$RUN_ROOT/coordinator.pid" ]] \
    && kill -0 "$(<"$RUN_ROOT/coordinator.pid")" 2>/dev/null; then
    echo "coordinator_state=running"
  else
    echo "coordinator_state=absent"
  fi
  if [[ -d "$RUN_ROOT/status" ]]; then
    find "$RUN_ROOT/status" -maxdepth 1 -type f -name 'seed*.status' -print0 \
      | sort -z | xargs -0 -r cat
  fi
  [[ ! -f "$RUN_ROOT/campaign.status" ]] || cat "$RUN_ROOT/campaign.status"
}

case "${1:-}" in
  launch) launch ;;
  run) run_all ;;
  status) status ;;
  *) usage; exit 2 ;;
esac
