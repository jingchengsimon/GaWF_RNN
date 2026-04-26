#!/usr/bin/env bash
# 作用：在超参文件已存在的前提下，仅对四个训练时长 scale 串行跑 Phase3（四模型/scale）并调用 plot_generalization；不执行 Phase1/2。
#   默认走 short 路径（需 phase2_final_hparams_short.json）；传 full 时需已完成 full Phase2（phase2_final_hparams.json），作图与 _ep${NUM_EPOCHS} 一致。
# 主流水线不调用本脚本，仅供本地补跑/调 epoch 等场景使用。用法见下方注释。
#
# Usage (repo root):
#   bash experiments/archive/run_local_phase3.sh
#   bash experiments/archive/run_local_phase3.sh full
#   NUM_EPOCHS=50 bash experiments/archive/run_local_phase3.sh
#   NUM_EPOCHS=50 bash experiments/archive/run_local_phase3.sh full
# 首参若为纯数字，视为 NUM_EPOCHS 且保持默认 short 配置（与旧 run_local_phase3_short 兼容）。
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
GEN="$ROOT/experiments/generalization"

PROFILE=short
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_EPOCHS="$1"
  shift
elif [[ "${1:-}" == full || "${1:-}" == short ]]; then
  PROFILE="$1"
  shift
fi
: "${NUM_EPOCHS:=100}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_EPOCHS="$1"
fi
export NUM_EPOCHS

if [[ "$PROFILE" == full ]]; then
  unset CSV_TAG 2>/dev/null || true
  export CSV_TAG=""
  HP_JSON="$GEN/artifacts/phase2_final_hparams.json"
  if [[ ! -f "$HP_JSON" ]]; then
    echo "Missing: $HP_JSON (run full Phase1+2 first, or copy hparams file)" >&2
    exit 1
  fi
  for sk in 4h 10h 20h 40h; do
    bash "$GEN/phase3_train_scale.sh" "$sk" full
  done
  python "$ROOT/utils_viz/plot_generalization.py" --csv_tag "_ep${NUM_EPOCHS}"
  echo "Done. csv_tag=_ep${NUM_EPOCHS}"
else
  HP_JSON="$GEN/artifacts/phase2_final_hparams_short.json"
  if [[ ! -f "$HP_JSON" ]]; then
    echo "Missing: $HP_JSON" >&2
    echo "Create from Phase1 short, e.g.:" >&2
    echo "  python experiments/generalization/collect_results.py emit_hparams_shared \\" >&2
    echo "    --phase1_best experiments/generalization/artifacts/phase1_best_short.json \\" >&2
    echo "    --out $HP_JSON" >&2
    exit 1
  fi
  export CSV_TAG="${CSV_TAG:-_short}"
  for sk in 4h 10h 20h 40h; do
    bash "$GEN/phase3_train_scale.sh" "$sk" short
  done
  TAG="${CSV_TAG}_ep${NUM_EPOCHS}"
  python "$ROOT/utils_viz/plot_generalization.py" --csv_tag "$TAG"
  echo "Done. csv_tag=$TAG  (phase3_summary_*${TAG}.csv under artifacts/)"
  echo "Figures: results/anal_figs/generalization/*${TAG#_}.*"
fi
