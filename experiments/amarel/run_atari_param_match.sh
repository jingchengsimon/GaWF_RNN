#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-param-match
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Run PyTorch-based Atari parameter matching on a Slurm compute node.
# Submit scripts must never execute this workload directly on a login node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
NUM_LAYERS="${PARAM_MATCH_NUM_LAYERS:?PARAM_MATCH_NUM_LAYERS is required}"
MODELS_SPEC="${PARAM_MATCH_MODELS:?PARAM_MATCH_MODELS is required}"
OUT_DIR="${PARAM_MATCH_OUT_DIR:?PARAM_MATCH_OUT_DIR is required}"
REQUIRED_SPEC="${PARAM_MATCH_REQUIRED:-ann:$MODELS_SPEC}"
ARTIFACT_TAG="${ARTIFACT_TAG:-atari_param_match}"
STATUS_DIR="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG/status"
mkdir -p "$OUT_DIR" "$STATUS_DIR"

IFS=':' read -r -a MODELS <<< "$MODELS_SPEC"
IFS=':' read -r -a REQUIRED <<< "$REQUIRED_SPEC"

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u
export KMP_DUPLICATE_LIB_OK=TRUE

echo "[$(date -Is)] parameter match layers=$NUM_LAYERS models=${MODELS[*]} out=$OUT_DIR"
python -m experiments.atari.atari_ssm_param_match \
  --conv_out 3136 \
  --hidden_size 512 \
  --ssm_state_size 128 \
  --num_actions 6 \
  --num_layers "$NUM_LAYERS" \
  --models "${MODELS[@]}" \
  --out_dir "$OUT_DIR"

python - "$OUT_DIR/atari_param_match.json" "$NUM_LAYERS" "$REQUIRED_SPEC" <<'PY'
import json
import sys

path, num_layers, required_spec = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
required = set(required_spec.split(":"))
missing = required - set(data["matched"])
if missing:
    raise RuntimeError(f"Missing parameter matches: {sorted(missing)}")
if data.get("candidate_num_layers") != num_layers:
    raise RuntimeError(
        f"candidate_num_layers={data.get('candidate_num_layers')} != {num_layers}"
    )
if data.get("num_actions") != 6:
    raise RuntimeError(f"num_actions={data.get('num_actions')} != 6")
PY

touch "$STATUS_DIR/complete.done"
echo "[$(date -Is)] parameter match complete"
