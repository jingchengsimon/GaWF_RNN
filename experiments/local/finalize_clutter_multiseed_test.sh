#!/usr/bin/env bash
# Wait for six model-level test CSVs, merge them, and render the final comparison.

set -euo pipefail

ROOT="${AIM3_ROOT:?AIM3_ROOT is required}"
PART_DIR="${AIM3_PART_DIR:?AIM3_PART_DIR is required}"
RESULTS_ROOT="${AIM3_RESULTS_PATH:-$ROOT/results}"
CONDA_INIT="${AIM3_CONDA_INIT:?AIM3_CONDA_INIT is required}"
ANALYSIS_DIR="$RESULTS_ROOT/anal_index/G_behaviour/evaluate_clutter_multiseed_test/data"
FIGURE_DIR="$RESULTS_ROOT/anal_index/G_behaviour/evaluate_clutter_multiseed_test/figs"
models=(gawf rnn lstm gru mamba s5)

for _ in $(seq 1 360); do
  complete=1
  for model in "${models[@]}"; do
    [[ -s "$PART_DIR/$model.csv" ]] || complete=0
  done
  [[ "$complete" -eq 1 ]] && break
  sleep 30
done
[[ "${complete:-0}" -eq 1 ]] || {
  echo "Timed out waiting for six model CSVs in $PART_DIR" >&2
  exit 1
}

mkdir -p "$ANALYSIS_DIR" "$FIGURE_DIR"
awk 'FNR == 1 && NR != 1 {next} {print}' \
  "$PART_DIR/gawf.csv" \
  "$PART_DIR/rnn.csv" \
  "$PART_DIR/lstm.csv" \
  "$PART_DIR/gru.csv" \
  "$PART_DIR/mamba.csv" \
  "$PART_DIR/s5.csv" \
  > "$ANALYSIS_DIR/per_seed_test_accuracy.csv"

set +u
source "$CONDA_INIT"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
cd "$ROOT"
python utils_viz/clutter_multiseed_test_bars.py \
  --data_csv "$ANALYSIS_DIR/per_seed_test_accuracy.csv" \
  --save_png "$FIGURE_DIR/test_accuracy_mean_sd.png" \
  --save_summary_csv "$ANALYSIS_DIR/test_accuracy_mean_sd.csv"
