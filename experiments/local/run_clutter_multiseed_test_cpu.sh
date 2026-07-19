#!/usr/bin/env bash
# Evaluate completed Clutter checkpoints in CPU shards without using training GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_DIR="${AIM3_DATA_DIR:?AIM3_DATA_DIR is required}"
CAMPAIGN_ROOT="${AIM3_CAMPAIGN_ROOT:?AIM3_CAMPAIGN_ROOT is required}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-$ROOT/results}"
CONDA_INIT="${AIM3_CONDA_INIT:?AIM3_CONDA_INIT is required}"
RUN_TAG="${AIM3_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SHARDS_PER_MODEL="${AIM3_SHARDS_PER_MODEL:-2}"
ANALYSIS_DIR="$RESULTS_ROOT/anal_data/clutter_multiseed_test"
FIGURE_DIR="$RESULTS_ROOT/anal_figs/clutter_multiseed_test"
PART_DIR="$ANALYSIS_DIR/parts_$RUN_TAG"
mkdir -p "$PART_DIR/roots" "$FIGURE_DIR"

[[ "$SHARDS_PER_MODEL" =~ ^[1-9][0-9]*$ ]] || {
  echo "AIM3_SHARDS_PER_MODEL must be a positive integer." >&2
  exit 2
}
set +u
source "$CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
cd "$ROOT"

models=(gawf rnn lstm gru mamba s5)
pids=()
for model in "${models[@]}"; do
  unit_dirs=()
  for unit_dir in "$CAMPAIGN_ROOT/$model"-seed??; do
    [[ -d "$unit_dir" ]] || continue
    compgen -G "$unit_dir/*_model.pth" >/dev/null || continue
    unit_dirs+=("$unit_dir")
  done
  [[ "${#unit_dirs[@]}" -gt 0 ]] || {
    echo "No completed $model checkpoints under $CAMPAIGN_ROOT" >&2
    exit 2
  }
  echo "model=$model completed_checkpoints=${#unit_dirs[@]}"

  for shard in $(seq 0 $((SHARDS_PER_MODEL - 1))); do
    shard_root="$PART_DIR/roots/${model}_$shard"
    mkdir -p "$shard_root"
    found=0
    for index in "${!unit_dirs[@]}"; do
      [[ $((index % SHARDS_PER_MODEL)) -eq "$shard" ]] || continue
      unit_dir="${unit_dirs[$index]}"
      ln -s "$unit_dir" "$shard_root/$(basename "$unit_dir")"
      found=$((found + 1))
    done
    [[ "$found" -gt 0 ]] || continue
    (
      export OMP_NUM_THREADS=4
      export MKL_NUM_THREADS=4
      python utils_anal/evaluate_clutter_multiseed_test.py \
        --checkpoint_root "$shard_root" \
        --data_dir "$DATA_DIR" \
        --data_suffix 40h-uint8 \
        --save_csv "$PART_DIR/${model}_$shard.csv" \
        --save_meta "$PART_DIR/${model}_$shard.json" \
        --device cpu \
        --batch_size 64 \
        --num_workers 1 \
        --chan_num 2 \
        --use_mmap \
        >"$PART_DIR/${model}_$shard.log" 2>&1
    ) &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard failed; inspect $PART_DIR/*.log" >&2
  exit 1
fi

for model in "${models[@]}"; do
  awk 'FNR == 1 && NR != 1 {next} {print}' "$PART_DIR/${model}_"*.csv \
    > "$PART_DIR/$model.csv"
done
awk 'FNR == 1 && NR != 1 {next} {print}' \
  "$PART_DIR/gawf.csv" \
  "$PART_DIR/rnn.csv" \
  "$PART_DIR/lstm.csv" \
  "$PART_DIR/gru.csv" \
  "$PART_DIR/mamba.csv" \
  "$PART_DIR/s5.csv" \
  > "$ANALYSIS_DIR/per_seed_test_accuracy.csv"

python utils_viz/clutter_multiseed_test_bars.py \
  --data_csv "$ANALYSIS_DIR/per_seed_test_accuracy.csv" \
  --save_png "$FIGURE_DIR/test_accuracy_mean_sd.png" \
  --save_summary_csv "$ANALYSIS_DIR/test_accuracy_mean_sd.csv"

echo "status=complete"
echo "parts=$PART_DIR"
echo "data=$ANALYSIS_DIR/per_seed_test_accuracy.csv"
echo "figure=$FIGURE_DIR/test_accuracy_mean_sd.png"
