#!/usr/bin/env bash
# 作用：在四个已完成 Phase1 的 train_data 子目录上调用 collect_results，按 scale 汇总 GAWF 最优 lr/wd。
#   full 仅写 phase1_best.json；short 会写入 phase1_best_short.json 并再经 emit_hparams_shared 生成 phase2_final_hparams_short.json。
# 主流水线（run_all_scales_2gpu.sh）已内联同样逻辑，本脚本供离线/补跑时手动使用。默认 short。
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

MODE="${1:-short}"
if [[ ! "$MODE" =~ ^(short|full)$ ]]; then
  echo "Usage: $0 [short|full]   (default: short)" >&2
  exit 1
fi
CR=("$ROOT/experiments/generalization/collect_results.py")
if [[ "$MODE" == full ]]; then
  python "${CR[@]}" phase1 \
    "results/train_data/gen_phase1_gawf_4h" \
    "results/train_data/gen_phase1_gawf_10h" \
    "results/train_data/gen_phase1_gawf_20h" \
    "results/train_data/gen_phase1_gawf_40h"
else
  python "${CR[@]}" phase1 \
    "results/train_data/gen_phase1_short_gawf_4h" \
    "results/train_data/gen_phase1_short_gawf_10h" \
    "results/train_data/gen_phase1_short_gawf_20h" \
    "results/train_data/gen_phase1_short_gawf_40h" \
    --out "$ROOT/experiments/generalization/artifacts/phase1_best_short.json"
  python "${CR[@]}" emit_hparams_shared \
    --phase1_best "$ROOT/experiments/generalization/artifacts/phase1_best_short.json" \
    --out "$ROOT/experiments/generalization/artifacts/phase2_final_hparams_short.json"
fi
