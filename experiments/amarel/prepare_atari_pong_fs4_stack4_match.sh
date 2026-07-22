#!/usr/bin/env bash
#SBATCH --job-name=aim3-pong-fs4s4-match
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Build the only missing L1 parameter-match entry (Mamba) on a compute node,
# then merge it with the already-used strict-1-frame L1 matching table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$ROOT"

: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
BASE_MATCH_JSON="${BASE_MATCH_JSON:-$AIM3_RESULTS_PATH/atari_param_match/atari_param_match.json}"
MATCH_DIR="${MATCH_DIR:-$AIM3_RESULTS_PATH/atari_param_match_fs4_stack4_l1}"
L2_MATCH_JSON="${L2_MATCH_JSON:-$AIM3_RESULTS_PATH/atari_param_match_depth2/atari_param_match.json}"
[[ -f "$BASE_MATCH_JSON" ]] || { echo "Missing base match JSON: $BASE_MATCH_JSON" >&2; exit 2; }
[[ -f "$L2_MATCH_JSON" ]] || { echo "Missing L2 match JSON: $L2_MATCH_JSON" >&2; exit 2; }

CONDA_SH="${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
set +u
source "$CONDA_SH"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

COMPONENT_DIR="$MATCH_DIR/mamba_component"
mkdir -p "$COMPONENT_DIR"
python -m experiments.atari.atari_ssm_param_match \
  --models mamba \
  --out_dir "$COMPONENT_DIR"

python - "$BASE_MATCH_JSON" "$COMPONENT_DIR/atari_param_match.json" \
  "$MATCH_DIR/atari_param_match.json" <<'PY'
import json
import os
import sys

base_path, component_path, output_path = sys.argv[1:4]
with open(base_path, encoding="utf-8") as handle:
    base = json.load(handle)
with open(component_path, encoding="utf-8") as handle:
    component = json.load(handle)
required_base = {"rnn", "gru", "lstm", "gawf", "s5"}
if not required_base <= set(base["matched"]):
    raise RuntimeError(f"Base match JSON lacks {sorted(required_base - set(base['matched']))}")
if "mamba" not in component["matched"]:
    raise RuntimeError("Mamba parameter matching did not produce an entry")
matched = {"ann": component["matched"]["ann"]}
matched.update({model: base["matched"][model] for model in sorted(required_base)})
matched["mamba"] = component["matched"]["mamba"]
component["matched"] = matched
os.makedirs(os.path.dirname(output_path), exist_ok=True)
tmp_path = output_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as handle:
    json.dump(component, handle, indent=2)
os.replace(tmp_path, output_path)
print(f"wrote complete L1 match table: {output_path}")
PY

python - "$MATCH_DIR/atari_param_match.json" "$L2_MATCH_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
required = {"ann", "rnn", "gru", "lstm", "gawf", "s5", "mamba"}
assert required <= set(data["matched"])
assert data.get("num_actions") == 6
assert data.get("candidate_num_layers") == 1

with open(sys.argv[2], encoding="utf-8") as handle:
    l2 = json.load(handle)
required_l2 = {"ann", "rnn", "gru", "lstm", "gawf"}
assert required_l2 <= set(l2["matched"])
assert all(l2["matched"][model].get("num_layers") == 2 for model in required_l2)
PY
