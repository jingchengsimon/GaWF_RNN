#!/usr/bin/env bash
# Run a local 2-GPU GaWF feedback-dimension comparison.
#
# Default experiment:
#   scale=40h, model=gawf, hidden=256, lr=0.005, wd=0.001
#   conditions: legacy + dz 8/16/32/64
#   feedback is enabled from the first epoch, matching full-grid GaWF training.
#
# From repo root:
#   bash experiments/local/run_gawf_dz_compare_2gpu.sh
#
# Optional environment overrides:
#   AIM3_DATA_DIR=/path/to/stimuli
#   GPU0=0 GPU1=1
#   DZ_VALUES="8 16 32 64"
#   NUM_EPOCHS=100 PATIENCE=15
#   HIDDEN_SIZE=256 LR=0.005 WD=0.001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SCALE="${SCALE:-40h}"
DATA_SUFFIX="${DATA_SUFFIX:-${SCALE}-float32}"
EVAL_DATA_SUFFIX="${EVAL_DATA_SUFFIX:-40h-uint8}"
RESULT_SUFFIX="${RESULT_SUFFIX:-gawf_dz_compare_${SCALE}_fullfb}"
LOG_DIR="${LOG_DIR:-$ROOT/experiments/local/artifacts/gawf_dz_compare_${SCALE}_fullfb}"
STATUS_DIR="$LOG_DIR/status"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DZ_VALUES="${DZ_VALUES:-8 16 32 64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LR="${LR:-0.005}"
WD="${WD:-0.001}"
CNN_DROPOUT="${CNN_DROPOUT:-0.0}"
RNN_DROPOUT="${RNN_DROPOUT:-0.5}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-42}"

usage() {
  cat <<'EOF'
Usage:
  bash experiments/local/run_gawf_dz_compare_2gpu.sh [--dry-run]

Runs GaWF legacy plus explicit dz values on two local GPUs.
Defaults use the current 40h GaWF full-grid hparams:
  hidden_size=256, lr=0.005, wd=0.001

Environment overrides:
  SCALE=40h
  DZ_VALUES="8 16 32 64"
  GPU0=0 GPU1=1
  AIM3_DATA_DIR=/path/to/stimuli
  HIDDEN_SIZE=256 LR=0.005 WD=0.001
  NUM_EPOCHS=100 PATIENCE=15
  RESULT_SUFFIX=gawf_dz_compare_40h_fullfb
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$LOG_DIR" "$STATUS_DIR"

ts() {
  date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z'
}

if [[ -n "${AIM3_DATA_DIR:-}" ]]; then
  DATA_DIR="$AIM3_DATA_DIR"
elif [[ -d "$ROOT/stimuli" ]]; then
  DATA_DIR="$ROOT/stimuli"
elif [[ -d "/G/MIMOlab/Codes/aim3_RNN/stimuli" ]]; then
  DATA_DIR="/G/MIMOlab/Codes/aim3_RNN/stimuli"
elif [[ -d "/scratch/${USER}/stimuli" ]]; then
  DATA_DIR="/scratch/${USER}/stimuli"
else
  echo "Data directory not found. Set AIM3_DATA_DIR or create $ROOT/stimuli." >&2
  exit 2
fi

CONDITIONS=(legacy)
for dz in $DZ_VALUES; do
  CONDITIONS+=("dz${dz}")
done

condition_dz_arg() {
  local condition="$1"
  if [[ "$condition" == legacy ]]; then
    return 0
  fi
  printf '%s\n' "${condition#dz}"
}

metrics_path_for_condition() {
  local condition="$1"
  local dz_suffix=""
  if [[ "$condition" != legacy ]]; then
    dz_suffix="_dz${condition#dz}"
  fi
  printf '%s\n' \
    "$ROOT/results/train_data/$RESULT_SUFFIX/$condition/gawf_sector_acc_h${HIDDEN_SIZE}_lr${LR}_wd${WD}_cdo${CNN_DROPOUT}_rdo${RNN_DROPOUT}${dz_suffix}_metrics.json"
}

run_one() {
  local gpu="$1"
  local condition="$2"
  local log_prefix="$LOG_DIR/${condition}_h${HIDDEN_SIZE}_lr${LR}_wd${WD}_fullfb"
  local done_file="$STATUS_DIR/${condition}.done"
  local fail_file="$STATUS_DIR/${condition}.fail"
  local metrics_path
  metrics_path="$(metrics_path_for_condition "$condition")"

  if [[ -f "$metrics_path" ]]; then
    echo "[$(ts)] skip condition=$condition existing metrics=$metrics_path"
    {
      echo "status=skipped_existing"
      echo "condition=$condition"
      echo "metrics_path=$metrics_path"
      echo "timestamp=$(ts)"
    } > "$done_file"
    rm -f "$fail_file"
    return 0
  fi

  cmd=(
    python train_model.py
    --model_types gawf
    --hidden_sizes "$HIDDEN_SIZE"
    --data_suffix "$DATA_SUFFIX"
    --eval_data_suffix "$EVAL_DATA_SUFFIX"
    --data_dir "$DATA_DIR"
    --lrs "$LR"
    --wds "$WD"
    --cnn_dropout "$CNN_DROPOUT"
    --rnn_dropout "$RNN_DROPOUT"
    --num_epochs "$NUM_EPOCHS"
    --patience "$PATIENCE"
    --seed "$SEED"
    --use_acceleration
    --use_sector_mode
    --result_suffix "$RESULT_SUFFIX/$condition"
  )
  if [[ "$condition" != legacy ]]; then
    cmd+=(--dz "$(condition_dz_arg "$condition")")
  fi

  echo "[$(ts)] start condition=$condition gpu=$gpu metrics=$metrics_path"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s DISABLE_TQDM=1 ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_TQDM=1 "${cmd[@]}" \
    > "${log_prefix}.out" 2> "${log_prefix}.err"
  local train_rc=$?
  set -e

  if [[ "$train_rc" -ne 0 ]]; then
    {
      echo "status=train_failed"
      echo "condition=$condition"
      echo "exit_code=$train_rc"
      echo "metrics_path=$metrics_path"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(ts)"
    } > "$fail_file"
    echo "[$(ts)] fail condition=$condition rc=$train_rc"
    return "$train_rc"
  fi

  if [[ -f "$metrics_path" ]]; then
    {
      echo "status=done"
      echo "condition=$condition"
      echo "metrics_path=$metrics_path"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(ts)"
    } > "$done_file"
    rm -f "$fail_file"
    echo "[$(ts)] done condition=$condition"
  else
    {
      echo "status=missing_metrics"
      echo "condition=$condition"
      echo "metrics_path=$metrics_path"
      echo "log_prefix=$log_prefix"
      echo "timestamp=$(ts)"
    } > "$fail_file"
    echo "[$(ts)] missing metrics condition=$condition"
    return 1
  fi
}

echo "AIM3 local 2-GPU GaWF dz comparison"
echo "root=$ROOT"
echo "data_dir=$DATA_DIR"
echo "scale=$SCALE"
echo "data_suffix=$DATA_SUFFIX"
echo "eval_data_suffix=$EVAL_DATA_SUFFIX"
echo "result_suffix=$RESULT_SUFFIX"
echo "conditions=${CONDITIONS[*]}"
echo "gpu_ids=$GPU0,$GPU1"
echo "fixed_hparams=hidden_size=$HIDDEN_SIZE,lr=$LR,wd=$WD,cnn_dropout=$CNN_DROPOUT,rnn_dropout=$RNN_DROPOUT"
echo "num_epochs=$NUM_EPOCHS patience=$PATIENCE seed=$SEED feedback=full"
echo "log_dir=$LOG_DIR"

idx=0
total="${#CONDITIONS[@]}"
while [[ "$idx" -lt "$total" ]]; do
  condition="${CONDITIONS[$idx]}"
  run_one "$GPU0" "$condition" &
  p0=$!

  next_idx=$((idx + 1))
  if [[ "$next_idx" -lt "$total" ]]; then
    next_condition="${CONDITIONS[$next_idx]}"
    run_one "$GPU1" "$next_condition" &
    p1=$!
    set +e
    wait "$p0"
    rc0=$?
    wait "$p1"
    rc1=$?
    set -e
    if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
      echo "[$(ts)] one or more conditions in pair ${condition},${next_condition} failed; continuing"
    fi
  else
    set +e
    wait "$p0"
    rc0=$?
    set -e
    if [[ "$rc0" -ne 0 ]]; then
      echo "[$(ts)] condition ${condition} failed; continuing"
    fi
  fi

  idx=$((idx + 2))
done

echo "Completed GaWF dz comparison condition list (${total} condition(s))."
echo "Logs: $LOG_DIR"
echo "Results: $ROOT/results/train_data/$RESULT_SUFFIX"
echo "Summarize with: bash experiments/local/summarize_gawf_dz_compare.sh"
