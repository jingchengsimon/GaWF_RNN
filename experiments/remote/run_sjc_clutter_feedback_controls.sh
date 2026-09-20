#!/usr/bin/env bash
# Train and evaluate four parameter-matched Clutter feedback controls on two SJC GPUs.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DISABLE_TQDM=1
export AIM3_NUM_WORKERS="${AIM3_NUM_WORKERS:-2}"
export AIM3_PIN_MEMORY="${AIM3_PIN_MEMORY:-1}"

ROOT="${AIM3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS="${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
DATA_DIR="${AIM3_CLUTTER_DATA_DIR:?AIM3_CLUTTER_DATA_DIR is required}"
RUN_TAG="clutter_feedback_controls_ep150_v1"
RUN_BASE="$RESULTS/data/clutter/runs/feedback_controls/$RUN_TAG"
TEST_BASE="$RESULTS/data/analysis/feedback_controls_reset_excluded_test_10seed_v1"
ABLATION_BASE="$RESULTS/data/analysis/feedback_controls_shuffle_resetexcluded_10seed_v1"
GAWF_ABLATION="$RESULTS/data/analysis/supple1_feedback_shuffle_recovery_resetexcluded_10seed_v1"
ORIGINAL_TEST_CSV="$RESULTS/data/analysis/fig1_reset_excluded_behavior_6model_10seed_v8/final/reset_excluded_test_accuracy_10seed.csv"
SUMMARY_ROOT="$RESULTS/data/analysis/feedback_controls_formal_10seed_v1"
ARTIFACT_ROOT="$RESULTS/artifacts/clutter_feedback_controls_ep150_v1"
PROTOCOL_JSON="$ARTIFACT_ROOT/protocol_checks.json"
SANITY_JSON="$ARTIFACT_ROOT/sanity_seed1_200steps.json"
BALANCED_SUFFIX="40h-float32-jointswitch-balanced-10digit-unique"

MODELS=(gawf_additive rnn_fb gru_fb lstm_fb)
WIDTHS=(271 272 103 79)
LRS=(0.005 0.001 0.005 0.001)
WDS=(0.001 0.00001 0.001 0.001)

mkdir -p "$ARTIFACT_ROOT"
exec 9>"$ARTIFACT_ROOT/runner.lock"
flock -n 9 || { echo "Another feedback-control runner is active" >&2; exit 1; }
if [[ -f "$SUMMARY_ROOT/final/.complete" ]]; then
  echo "Feedback-control campaign is already complete: $SUMMARY_ROOT/final"
  exit 0
fi
[[ ! -e "$SUMMARY_ROOT/final" ]] || {
  echo "Incomplete summary output requires inspection: $SUMMARY_ROOT/final" >&2
  exit 1
}

for required in \
  "$DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$DATA_DIR/stimulus_reg-train-40h-uint8.tsv" \
  "$DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$DATA_DIR/stimulus_reg-validation-40h-uint8.tsv" \
  "$DATA_DIR/stimulus_reg-test-40h-uint8.npy" \
  "$DATA_DIR/stimulus_reg-test-40h-uint8.tsv" \
  "$DATA_DIR/stimulus_reg-test-$BALANCED_SUFFIX.npy" \
  "$DATA_DIR/stimulus_reg-test-$BALANCED_SUFFIX.tsv"; do
  [[ -s "$required" ]] || { echo "Missing required dataset file: $required" >&2; exit 1; }
done
[[ -f "$ORIGINAL_TEST_CSV" ]] || { echo "Missing original-model test CSV" >&2; exit 1; }
(( $(find "$GAWF_ABLATION" -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 10 )) || {
  echo "Expected ten existing GaWF shuffle-ablation results" >&2; exit 1;
}

cd "$ROOT"
if [[ ! -f "$PROTOCOL_JSON" ]]; then
  [[ ! -e "$PROTOCOL_JSON" ]] || {
    echo "Incomplete protocol-check output: $PROTOCOL_JSON" >&2; exit 1;
  }
  python -B -m experiments.clutter.feedback_control_protocol_check \
    --output "$PROTOCOL_JSON" 2>&1 | tee "$ARTIFACT_ROOT/protocol_checks.log"
fi
if [[ ! -f "$SANITY_JSON" ]]; then
  [[ ! -e "$SANITY_JSON" ]] || { echo "Incomplete sanity output: $SANITY_JSON" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=0 python -B -m experiments.clutter.feedback_control_sanity \
    --output "$SANITY_JSON" --device cuda --steps 200 --seed 1 \
    2>&1 | tee "$ARTIFACT_ROOT/sanity.log"
fi

checkpoint_for() {
  local result_dir="$1" matches
  shopt -s nullglob
  matches=("$result_dir"/*_model.pth)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || {
    echo "Expected one final checkpoint in $result_dir, found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

validate_training_outputs() {
  local result_dir="$1"
  (( $(find "$result_dir" -maxdepth 1 -name '*_model.pth' | wc -l) == 1 )) || return 1
  (( $(find "$result_dir" -maxdepth 1 -name '*_metrics.json' | wc -l) == 1 )) || return 1
  (( $(find "$result_dir" -maxdepth 1 -name '*.pkl' | wc -l) == 1 )) || return 1
}

run_unit() {
  local task="$1" gpu="$2" model_index seed model width lr wd seed_tag
  local suffix result_dir checkpoint test_dir ablation_dir log
  model_index=$((task / 10))
  seed=$((task % 10 + 1))
  model="${MODELS[model_index]}"
  width="${WIDTHS[model_index]}"
  lr="${LRS[model_index]}"
  wd="${WDS[model_index]}"
  printf -v seed_tag '%02d' "$seed"
  suffix="feedback_controls/$RUN_TAG/$model-seed$seed_tag"
  result_dir="$RUN_BASE/$model-seed$seed_tag"
  test_dir="$TEST_BASE/$model-seed$seed_tag"
  ablation_dir="$ABLATION_BASE/$model-seed$seed_tag"
  log="$ARTIFACT_ROOT/$model-seed$seed_tag.log"

  if compgen -G "$result_dir/*_model.pth" >/dev/null \
    || compgen -G "$result_dir/*_metrics.json" >/dev/null; then
    validate_training_outputs "$result_dir" || {
      echo "Incomplete final training outputs: $result_dir" >&2; return 1;
    }
  else
    CUDA_VISIBLE_DEVICES="$gpu" python -B run_task.py clutter \
      --model_types "$model" --hidden_sizes "$width" \
      --num_layers 1 --num_epochs 150 --patience 0 \
      --lrs "$lr" --wds "$wd" --optim adamw \
      --cnn_dropout 0.0 --rnn_dropout 0.5 \
      --seed "$seed" --use_acceleration --use_sector_mode --use_mmap --chan_num 2 \
      --data_dir "$DATA_DIR" --results_dir "$RESULTS" \
      --data_suffix 40h-uint8 --eval_data_suffix 40h-uint8 \
      --input_cast_mode device --frame_layout compact --shuffle_block_size -1 \
      --checkpoint_interval_epochs 5 --auto_resume --result_suffix "$suffix" \
      2>&1 | tee "$log"
  fi

  validate_training_outputs "$result_dir" || {
    echo "Training did not produce its complete output contract: $result_dir" >&2; return 1;
  }
  checkpoint="$(checkpoint_for "$result_dir")"
  if [[ ! -f "$test_dir/reset_excluded_test_accuracy.json" ]]; then
    [[ ! -e "$test_dir" ]] || { echo "Incomplete test output: $test_dir" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="$gpu" python -B -m \
      utils.analysis.clutter.fig1_reset_excluded_test_accuracy collect \
      --ckpt "$checkpoint" --model "$model" --seed "$seed" --output_dir "$test_dir" \
      --data_dir "$DATA_DIR" --data_suffix 40h-uint8 --sequence_length 32 \
      --batch_size 256 --num_workers 2 --device cuda
  fi
  if [[ ! -f "$ablation_dir/ablation_metrics.json" ]]; then
    [[ ! -e "$ablation_dir" ]] || {
      echo "Incomplete feedback-ablation output: $ablation_dir" >&2; return 1;
    }
    CUDA_VISIBLE_DEVICES="$gpu" python -B -m utils.analysis.clutter.fig2_feedback_ablation \
      --ckpt "$checkpoint" --save_dir "$ablation_dir" --data_dir "$DATA_DIR" \
      --data_suffix "$BALANCED_SUFFIX" \
      --conditions baseline shuffle_digit shuffle_sector shuffle_all \
      --K 10 --pre_K 10 --sequence_length 512 --batch_size 16 --device cuda \
      --seed "$seed" --exclude_window_initial_frame
  fi
}

worker() {
  local gpu="$1" task
  for ((task=gpu; task<40; task+=2)); do
    run_unit "$task" "$gpu"
  done
}

worker 0 &
pid0=$!
worker 1 &
pid1=$!
wait "$pid0"
wait "$pid1"

(( $(find "$RUN_BASE" -mindepth 2 -maxdepth 2 -name '*_model.pth' | wc -l) == 40 ))
(( $(find "$TEST_BASE" -mindepth 2 -maxdepth 2 -name reset_excluded_test_accuracy.json | wc -l) == 40 ))
(( $(find "$ABLATION_BASE" -mindepth 2 -maxdepth 2 -name ablation_metrics.json | wc -l) == 40 ))

FEEDBACK_TEST_CSV="$SUMMARY_ROOT/feedback_controls_reset_excluded_test_accuracy_10seed.csv"
mkdir -p "$SUMMARY_ROOT"
python -B -m utils.analysis.clutter.fig1_reset_excluded_test_accuracy aggregate \
  --data_root "$TEST_BASE" --output_csv "$FEEDBACK_TEST_CSV" \
  --models gawf_additive rnn_fb gru_fb lstm_fb

python -B -m utils.analysis.clutter.feedback_control_summary \
  --original_test_csv "$ORIGINAL_TEST_CSV" \
  --feedback_test_csv "$FEEDBACK_TEST_CSV" \
  --gawf_ablation_dir "$GAWF_ABLATION" \
  --feedback_ablation_dir "$ABLATION_BASE" \
  --output_dir "$SUMMARY_ROOT/final"
touch "$SUMMARY_ROOT/final/.complete"
