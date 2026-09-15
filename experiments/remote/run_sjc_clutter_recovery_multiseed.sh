#!/usr/bin/env bash
# Generate joint-balanced Fig1 and Supplementary 1 recovery curves for ten training seeds.

set -eo pipefail

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:-$ROOT/results}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:-$ROOT/source/clutter/stimuli}"
RUN_ROOT="$RESULTS/data/clutter/seed_search/clutter_best6_multiseed_40h_ep150"
DATA_SUFFIX="40h-float32-jointswitch-balanced-10digit-unique"
FIG1_BASE="$RESULTS/data/analysis/fig1_target_switch_recovery_jointbalanced_6model_10seed"
SUPPLE1_BASE="$RESULTS/data/analysis/supple1_feedback_ablation_recovery_jointbalanced_10seed"
FINAL="$FIG1_BASE/final"
CONDA_SH="${AIM3_CONDA_SH:-/G/anaconda3/etc/profile.d/conda.sh}"
MODELS=(gawf rnn lstm gru mamba s5)
SUPPLE1_PDF="Supple1_jointswitch_balanced_10digit_unique_sector_covered_"
SUPPLE1_PDF+="prepost10_fig_ablation_switch_recovery.pdf"

source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export PYTHONDONTWRITEBYTECODE=1

checkpoint_for() {
  local model="$1" seed="$2" matches
  shopt -s nullglob
  matches=("$RUN_ROOT/$model-seed$seed/"*_model.pth)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || {
    echo "Expected one checkpoint for $model-seed$seed, found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

run_fig1_task() {
  local task="$1" gpu="$2" model seed output checkpoint
  model="${MODELS[task / 10]}"
  printf -v seed '%02d' "$((task % 10 + 1))"
  output="$FIG1_BASE/$model-seed$seed"
  if compgen -G "$output/fg_switch_offset_acc_*.npz" >/dev/null; then
    return
  fi
  checkpoint="$(checkpoint_for "$model" "$seed")"
  CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.fig1_target_switch_recovery \
    --ckpts "$checkpoint" --save_dir "$output" --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" --window_radius 10 --batch_size 64 --device cuda \
    --seed "${seed#0}" --exclude_window_initial_frame
}

run_supple1_task() {
  local seed="$1" gpu="$2" output checkpoint
  printf -v seed '%02d' "$seed"
  output="$SUPPLE1_BASE/gawf-seed$seed"
  if [[ -f "$output/ablation_metrics.json" ]]; then
    return
  fi
  checkpoint="$(checkpoint_for gawf "$seed")"
  CUDA_VISIBLE_DEVICES="$gpu" python -m utils.analysis.clutter.fig2_feedback_ablation \
    --ckpt "$checkpoint" --save_dir "$output" --data_dir "$DATA_DIR" \
    --data_suffix "$DATA_SUFFIX" --conditions baseline clear_digit clear_sector clear_all \
    --K 10 --pre_K 10 --sequence_length 512 --batch_size 16 --device cuda \
    --seed "${seed#0}" --exclude_window_initial_frame
}

worker() {
  local gpu="$1" task seed
  for ((task=gpu; task<60; task+=2)); do
    run_fig1_task "$task" "$gpu"
  done
  for ((seed=gpu+1; seed<=10; seed+=2)); do
    run_supple1_task "$seed" "$gpu"
  done
}

mkdir -p "$FIG1_BASE" "$SUPPLE1_BASE"
worker 0 &
pid0=$!
worker 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

(( $(find "$FIG1_BASE" -mindepth 2 -maxdepth 2 -name 'fg_switch_offset_acc_*.npz' | wc -l) == 60 ))
(( $(find "$SUPPLE1_BASE" -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 10 ))
[[ ! -e "$FINAL" ]] || { echo "Refusing to overwrite final output: $FINAL" >&2; exit 1; }
mkdir -p "$FINAL" "$RESULTS/save"

python -m utils.analysis.clutter.fig1_multiseed_summary \
  --test_csv "$RESULTS/save_data/fig1/test_accuracy_summary/best_acc_test_mean_std.csv" \
  --train_data_dir "$RESULTS/save_data/fig1/validation_loss_histories" \
  --recovery_dir "$FIG1_BASE" \
  --ablation_dir "$RESULTS/save_data/fig2/gawf_shuffle_ablation" \
  --output_png "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.png" \
  --output_pdf "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf"

python -m utils.analysis.clutter.supple1_feedback_ablation \
  --data_dir "$SUPPLE1_BASE" --save_dir "$FINAL" \
  --conditions baseline clear_digit clear_sector clear_all

cp "$FINAL/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf" \
  "$RESULTS/save/Fig1_best6_multiseed_shuffle_2x4_seq512.pdf"
cp "$FINAL/fig_ablation_switch_recovery.pdf" \
  "$RESULTS/save/$SUPPLE1_PDF"
touch "$FINAL/.complete"
