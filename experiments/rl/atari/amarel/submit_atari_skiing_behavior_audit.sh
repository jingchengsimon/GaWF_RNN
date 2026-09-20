#!/usr/bin/env bash
# Submit the three-model Skiing behavior audit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
RUNNER="$SCRIPT_DIR/run_atari_skiing_behavior_audit.sh"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${AIM3_SOURCE_PATH:?AIM3_SOURCE_PATH is required}"
audit_tag="${AUDIT_TAG:-behavior_audit_4m_greedy20_seed1}"
amp_dtype="${AUDIT_AMP_DTYPE:-bfloat16}"
[[ "$audit_tag" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid AUDIT_TAG: $audit_tag" >&2; exit 2; }
[[ "$amp_dtype" == none || "$amp_dtype" == bfloat16 || "$amp_dtype" == float16 ]] || {
  echo "Invalid AUDIT_AMP_DTYPE: $amp_dtype" >&2
  exit 2
}

dry_run=0
if [[ $# -gt 0 ]]; then
  [[ $# -eq 1 && $1 == "--dry-run" ]] || {
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
  }
  dry_run=1
fi

[[ -f "$ROOT/run_task.py" ]] || { echo "Missing project root: $ROOT" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
for model in lstm gru gawf; do
  leaf="atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1"
  leaf+="_extend2mto4m_2m_${model}_seed1"
  [[ -s "$AIM3_SOURCE_PATH/$leaf/metrics.json" ]] || {
    echo "Missing source leaf: $AIM3_SOURCE_PATH/$leaf" >&2
    exit 2
  }
done

exports="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$AIM3_RESULTS_PATH"
exports+=",AIM3_SOURCE_PATH=$AIM3_SOURCE_PATH"
exports+=",AUDIT_TAG=$audit_tag,AUDIT_AMP_DTYPE=$amp_dtype"
if [[ $dry_run -eq 1 ]]; then
  printf 'sbatch --parsable --chdir=%q --export=%q %q\n' "$ROOT" "$exports" "$RUNNER"
  exit 0
fi

artifact_dir="$AIM3_RESULTS_PATH/artifacts/rl/atari/$audit_tag"
mkdir -p "$artifact_dir"
job_id="$(
  sbatch --parsable \
    --job-name="aim3-ski-audit-$amp_dtype" \
    --chdir="$ROOT" \
    --output="$artifact_dir/%A_%a.out" \
    --error="$artifact_dir/%A_%a.err" \
    --export="$exports" \
    "$RUNNER"
)"
echo "$job_id"
