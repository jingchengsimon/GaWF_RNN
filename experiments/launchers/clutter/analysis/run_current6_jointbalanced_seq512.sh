#!/usr/bin/env bash
# Evaluate a checkpoint map on the joint-balanced movie with 512-frame rollouts.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ROOT PYTHON DATA_DIR OUTPUT_ROOT GPU_CSV CHECKPOINT_MAP" >&2
  exit 2
fi

ROOT=$1
PYTHON=$2
DATA_DIR=$3
OUTPUT_ROOT=$4
GPU_CSV=$5
CHECKPOINT_MAP=$6
DATA_SUFFIX=40h-float32-jointswitch-balanced-10digit-unique

[[ -d $ROOT && -x $PYTHON && -d $DATA_DIR && -f $CHECKPOINT_MAP ]] || {
  echo "missing ROOT, PYTHON, DATA_DIR, or CHECKPOINT_MAP" >&2
  exit 1
}
[[ ! -e $OUTPUT_ROOT ]] || {
  echo "refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 1
}

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_CSV must name at least one GPU" >&2; exit 1; }
mapfile -t UNITS < <(awk -F '\t' 'NF == 3 && $1 !~ /^#/ {print}' "$CHECKPOINT_MAP")
(( ${#UNITS[@]} > 0 )) || { echo "checkpoint map is empty" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"/{test,recovery,status,logs,mplconfig}
cp "$CHECKPOINT_MAP" "$OUTPUT_ROOT/checkpoint_map.tsv"
git -C "$ROOT" rev-parse HEAD > "$OUTPUT_ROOT/source_commit.txt"
export MPLCONFIGDIR="$OUTPUT_ROOT/mplconfig"
cd "$ROOT"

run_unit() {
  local line=$1 gpu=$2 model seed checkpoint unit
  IFS=$'\t' read -r model seed checkpoint <<< "$line"
  printf -v seed '%02d' "$((10#$seed))"
  unit="$model-seed$seed"
  [[ -f $checkpoint ]] || { echo "missing checkpoint: $checkpoint" >&2; return 1; }

  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -B \
    -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
    --ckpt "$checkpoint" --model "$model" --seed "$((10#$seed))" \
    --output_dir "$OUTPUT_ROOT/test/$unit" --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" --sequence_length 512 --batch_size 16 \
    --num_workers 2 --device cuda

  mkdir -p "$OUTPUT_ROOT/recovery/$unit"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -B \
    -m utils.analysis.clutter.fig1_target_switch_recovery \
    --ckpts "$checkpoint" --save_dir "$OUTPUT_ROOT/recovery/$unit" \
    --data_dir "$DATA_DIR" --data_suffix "$DATA_SUFFIX" --sequence_length 512 \
    --window_radius 10 --batch_size 16 --device cuda --seed "$((10#$seed))" \
    --exclude_window_initial_frame
  touch "$OUTPUT_ROOT/status/$unit.done"
}

worker() {
  local lane=$1 gpu=$2 index
  for ((index=lane; index<${#UNITS[@]}; index+=${#GPUS[@]})); do
    run_unit "${UNITS[index]}" "$gpu" > "$OUTPUT_ROOT/logs/unit-$index.log" 2>&1 || {
      touch "$OUTPUT_ROOT/status/unit-$index.fail"
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
  wait "$pid" || { touch "$OUTPUT_ROOT/status/campaign.fail"; exit 1; }
done

touch "$OUTPUT_ROOT/.complete"
