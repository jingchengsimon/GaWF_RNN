#!/usr/bin/env bash
# Evaluate GaWF feedback shuffles on the standard CM-MNIST movie with 512-frame rollouts.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ROOT PYTHON DATA_DIR CHECKPOINT_ROOT OUTPUT_ROOT RUN_ROOT GPU_CSV" >&2
  exit 2
fi

ROOT=$1
PYTHON=$2
DATA_DIR=$3
CHECKPOINT_ROOT=$4
OUTPUT_ROOT=$5
RUN_ROOT=$6
GPU_CSV=$7

[[ -d $ROOT && -x $PYTHON && -d $DATA_DIR && -d $CHECKPOINT_ROOT ]] || {
  echo "missing ROOT, PYTHON, DATA_DIR, or CHECKPOINT_ROOT" >&2
  exit 1
}
[[ ! -e $OUTPUT_ROOT && ! -e $RUN_ROOT ]] || {
  echo "refusing to overwrite an existing output or run root" >&2
  exit 1
}

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_CSV must name at least one GPU" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT"/{logs,status}
cd "$ROOT"

checkpoint_for() {
  local seed=$1 seed_tag matches
  printf -v seed_tag '%02d' "$seed"
  shopt -s nullglob
  matches=("$CHECKPOINT_ROOT/gawf_legacy_notanh-seed$seed_tag/"*_model.pth)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || {
    echo "expected one checkpoint for seed $seed, found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

run_seed() {
  local seed=$1 gpu=$2 seed_tag checkpoint unit
  printf -v seed_tag '%02d' "$seed"
  checkpoint=$(checkpoint_for "$seed")
  unit="gawf-seed$seed_tag"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -B \
    -m utils.analysis.clutter.fig2_feedback_ablation \
    --ckpt "$checkpoint" --save_dir "$OUTPUT_ROOT/$unit" \
    --data_dir "$DATA_DIR" --data_suffix 40h-uint8 \
    --conditions baseline shuffle_digit shuffle_sector shuffle_all \
    --K 10 --pre_K 10 --sequence_length 512 --batch_size 16 \
    --device cuda --seed "$seed" --exclude_window_initial_frame
  touch "$RUN_ROOT/status/$unit.done"
}

worker() {
  local lane=$1 gpu=$2 seed
  for ((seed=lane+1; seed<=10; seed+=${#GPUS[@]})); do
    run_seed "$seed" "$gpu" > "$RUN_ROOT/logs/seed$(printf '%02d' "$seed").log" 2>&1 || {
      touch "$RUN_ROOT/status/seed$(printf '%02d' "$seed").fail"
      return 1
    }
  done
}

pids=()
for lane in "${!GPUS[@]}"; do
  worker "$lane" "${GPUS[lane]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || { touch "$RUN_ROOT/status/campaign.fail"; exit 1; }
done

touch "$OUTPUT_ROOT/.complete" "$RUN_ROOT/status/campaign.done"
