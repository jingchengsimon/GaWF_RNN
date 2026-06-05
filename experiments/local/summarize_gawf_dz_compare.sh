#!/usr/bin/env bash
# Summarize local GaWF dz-comparison metrics.
#
# From repo root:
#   bash experiments/local/summarize_gawf_dz_compare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SCALE="${SCALE:-40h}"
RESULT_SUFFIX="${RESULT_SUFFIX:-gawf_dz_compare_${SCALE}_fb${FB_START_EPOCH:-50}}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/local/artifacts/gawf_dz_compare_${SCALE}_fb${FB_START_EPOCH:-50}}"
DZ_VALUES="${DZ_VALUES:-8 16 32 64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LR="${LR:-0.005}"
WD="${WD:-0.001}"
CNN_DROPOUT="${CNN_DROPOUT:-0.0}"
RNN_DROPOUT="${RNN_DROPOUT:-0.5}"
FB_START_EPOCH="${FB_START_EPOCH:-50}"

mkdir -p "$OUT_DIR"

python - <<'PY'
import csv
import json
import os

root = os.getcwd()
scale = os.environ.get("SCALE", "40h")
result_suffix = os.environ.get("RESULT_SUFFIX", f"gawf_dz_compare_{scale}_fb{os.environ.get('FB_START_EPOCH', '50')}")
out_dir = os.environ.get("OUT_DIR", os.path.join(root, "experiments", "local", "artifacts", f"gawf_dz_compare_{scale}_fb{os.environ.get('FB_START_EPOCH', '50')}"))
dz_values = os.environ.get("DZ_VALUES", "8 16 32 64").split()
hidden = os.environ.get("HIDDEN_SIZE", "256")
lr = os.environ.get("LR", "0.005")
wd = os.environ.get("WD", "0.001")
cnn_dropout = os.environ.get("CNN_DROPOUT", "0.0")
rnn_dropout = os.environ.get("RNN_DROPOUT", "0.5")
fb_start = os.environ.get("FB_START_EPOCH", "50")

conditions = ["legacy"] + [f"dz{x}" for x in dz_values]

def metrics_path(condition: str) -> str:
    dz_suffix = "" if condition == "legacy" else f"_dz{condition[2:]}"
    stem = (
        f"gawf_sector_acc_h{hidden}_lr{lr}_wd{wd}"
        f"_cdo{cnn_dropout}_rdo{rnn_dropout}{dz_suffix}_fb{fb_start}_metrics.json"
    )
    return os.path.join(root, "results", "train_data", result_suffix, condition, stem)

def get_first(m, *keys):
    for key in keys:
        if m.get(key) is not None:
            return m.get(key)
    return None

rows = []
for condition in conditions:
    path = metrics_path(condition)
    row = {
        "condition": condition,
        "scale": scale,
        "expected_feedback_dim": "" if condition == "legacy" else condition[2:],
        "metrics_path": path,
        "status": "missing",
    }
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        train_acc = get_first(m, "train_acc_at_best_val", "best_train_acc_char")
        val_acc = get_first(m, "val_acc_at_best", "best_val_acc_char")
        train_sector = get_first(m, "train_acc_sector_at_best_val_sector", "best_train_acc_pos")
        val_sector = get_first(m, "val_acc_sector_at_best", "best_val_acc_pos")
        row.update(
            {
                "status": "done",
                "feedback_dim": m.get("feedback_dim"),
                "hidden_size": m.get("hidden_size"),
                "lr": m.get("lr"),
                "weight_decay": m.get("weight_decay"),
                "num_epochs": m.get("num_epochs"),
                "actual_epochs": m.get("actual_epochs"),
                "stopped_by_patience": m.get("stopped_by_patience"),
                "best_epoch_val_acc_1based": m.get("best_epoch_val_acc_1based"),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "overfit_gap": m.get("overfit_gap"),
                "train_acc_sector": train_sector,
                "val_acc_sector": val_sector,
                "overfit_gap_sector": m.get("overfit_gap_sector"),
            }
        )
    rows.append(row)

fieldnames = [
    "condition",
    "scale",
    "status",
    "expected_feedback_dim",
    "feedback_dim",
    "hidden_size",
    "lr",
    "weight_decay",
    "num_epochs",
    "actual_epochs",
    "stopped_by_patience",
    "best_epoch_val_acc_1based",
    "train_acc",
    "val_acc",
    "overfit_gap",
    "train_acc_sector",
    "val_acc_sector",
    "overfit_gap_sector",
    "metrics_path",
]
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "gawf_dz_compare_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

json_path = os.path.join(out_dir, "gawf_dz_compare_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
print("")
for row in rows:
    if row["status"] != "done":
        print(f"{row['condition']}: missing")
    else:
        print(
            f"{row['condition']}: val={row.get('val_acc')} "
            f"sector={row.get('val_acc_sector')} epochs={row.get('actual_epochs')} "
            f"feedback_dim={row.get('feedback_dim')}"
        )
PY
