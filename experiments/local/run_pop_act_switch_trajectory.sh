#!/usr/bin/env bash
# Analyze and visualize switch-aligned transient trajectories for the six selected Clutter models.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

POP_ROOT="${POP_ROOT:-results/anal_index/D_variance_decomposition/export_pop_act/data}"
DATA_ROOT="${DATA_ROOT:-results/anal_index/F_timing/pop_act_switch_trajectory/data}"
FIG_ROOT="${FIG_ROOT:-results/anal_index/F_timing/pop_act_switch_trajectory/figs}"

RUN_TAGS=(
  gawf_sector_acc_h256_lr0.005_wd0.001_cdo0.0_rdo0.5_model
  rnn_sector_acc_h275_lr0.001_wd1e-05_cdo0.0_rdo0.5_model
  lstm_sector_acc_h80_lr0.001_wd0.001_cdo0.0_rdo0.5_model
  gru_sector_acc_h105_lr0.005_wd0.001_cdo0.0_rdo0.5_model
  mamba_sector_acc_dmodel170_lr0.001_wd0.001_cdo0.0_rdo0.5_model
  s5_sector_acc_dmodel256_state128_lr0.001_wd0.0_cdo0.0_rdo0.5_model
)

for run_tag in "${RUN_TAGS[@]}"; do
  pop_dir="$POP_ROOT/$run_tag"
  if [[ ! -f "$pop_dir/pop_act.npy" || ! -f "$pop_dir/labels.tsv" ]]; then
    echo "Missing pop_act export: $pop_dir" >&2
    exit 2
  fi
  python utils_anal/pop_act_switch_trajectory.py \
    --pop_act_dir "$pop_dir" \
    --save_dir "$DATA_ROOT" \
    --run_tag "$run_tag" \
    --pre_frames 8 \
    --post_frames 20 \
    --n_components 3
  python utils_viz/pop_act_switch_trajectory.py \
    --data_dir "$DATA_ROOT/$run_tag" \
    --save_dir "$FIG_ROOT" \
    --run_tag "$run_tag"
done
