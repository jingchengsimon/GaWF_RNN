#!/usr/bin/env bash
#SBATCH --job-name=aim3-imdb-prep
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Prepare the deterministic processed IMDB tensors on a Slurm compute node.

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/run_task.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_DATA_DIR:?AIM3_DATA_DIR must be exported at submission}"
IMDB_DIR="$AIM3_DATA_DIR/imdb"
META_PATH="$IMDB_DIR/imdb_meta.json"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn

required=(
  vocab.json imdb_meta.json
  imdb_train_ids.pt imdb_train_len.pt imdb_train_label.pt
  imdb_val_ids.pt imdb_val_len.pt imdb_val_label.pt
  imdb_test_ids.pt imdb_test_len.pt imdb_test_label.pt
)

complete=1
for name in "${required[@]}"; do
  if [[ ! -s "$IMDB_DIR/$name" ]]; then
    complete=0
    break
  fi
done

if [[ "$complete" -eq 0 ]]; then
  python -B source/text/prepare_imdb_data.py \
    --data_dir "$AIM3_DATA_DIR" \
    --vocab_size 25000 \
    --min_freq 1 \
    --max_len 400 \
    --val_frac 0.1 \
    --seed 42
fi

python -B - "$META_PATH" "$IMDB_DIR" <<'PY'
import json
import os
import sys

meta_path, imdb_dir = sys.argv[1:]
with open(meta_path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
expected = {
    "vocab_size": 25000,
    "max_len": 400,
    "min_freq": 1,
    "val_frac": 0.1,
    "seed": 42,
    "n_train": 22500,
    "n_val": 2500,
    "n_test": 25000,
}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"IMDB metadata mismatch for {key}: {meta.get(key)!r} != {value!r}")
required = [
    "vocab.json",
    "imdb_meta.json",
    *[f"imdb_{split}_{kind}.pt" for split in ("train", "val", "test")
      for kind in ("ids", "len", "label")],
]
missing = [name for name in required if not os.path.isfile(os.path.join(imdb_dir, name))]
if missing:
    raise SystemExit(f"Missing processed IMDB files: {missing}")
print(json.dumps({"status": "valid", "imdb_dir": imdb_dir, **expected}, sort_keys=True))
PY
