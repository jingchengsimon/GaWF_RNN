#!/usr/bin/env bash
# Submit the thirty-unit GaWF core-placement campaign: two legacy-GaWF variants and the
# unmodified RNN-aligned GaWF, ten seeds each, full 30-way concurrency.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
MAX_CONCURRENT=0
while (( $# )); do
  case "$1" in
    --max-concurrent) MAX_CONCURRENT="${2:?--max-concurrent requires an integer}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || {
  echo "--max-concurrent must be a non-negative integer (0 = full 30-way concurrency)" >&2; exit 2;
}

ARRAY_SPEC="0-29"
if (( MAX_CONCURRENT > 0 )); then
  ARRAY_SPEC="0-29%${MAX_CONCURRENT}"
fi
LAUNCHER="$SCRIPT_DIR/run_clutter_gawf_core_variants.sh"
ARTIFACT_DIR="$SCRIPT_DIR/artifacts/clutter_gawf_core_variants_v1"
STATUS_DIR="$ARTIFACT_DIR/status"
GENERATED="$SCRIPT_DIR/generated"

if (( DRY_RUN )); then
  printf 'submit: array=%s max_concurrent=%s\n' "$ARRAY_SPEC" "$MAX_CONCURRENT"
  printf 'tasks 0-9   gawf_legacy_nowrap seeds 1-10 -> runs/clutter_ablate_legacy_nowrap_40h_ep150_v1\n'
  printf 'tasks 10-19 gawf_legacy_notanh seeds 1-10 -> runs/clutter_ablate_legacy_notanh_40h_ep150_v1\n'
  printf 'tasks 20-29 gawf_rnncore      seeds 1-10 -> runs/clutter_ablate_rnncore_40h_ep150_v1\n'
  printf 'test export: data/analysis/clutter_ablation_reset_excluded_test_legacy_rnncore_v1\n'
  printf 'launcher=%s logs=%s\n' "$LAUNCHER" "$ARTIFACT_DIR"
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-train-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-validation-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.npy"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$STATUS_DIR" "$GENERATED" \
  "$AIM3_RESULTS_PATH/data/analysis/clutter_ablation_reset_excluded_test_legacy_rnncore_v1"
printf 'human-authorized smoke waiver: this campaign has no preflight gate by explicit request\n' \
  > "$STATUS_DIR/smoke_waiver.txt"
SOURCE_COMMIT="${AIM3_SOURCE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)}"
printf '%s\n' "$SOURCE_COMMIT" > "$STATUS_DIR/source_commit.txt"

SBATCH_SCRIPT="$GENERATED/sbatch_clutter_gawf_core_variants_$(date +%Y%m%d_%H%M%S).sh"
{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=aim3-gawf-cores\n#SBATCH --partition=gpu\n#SBATCH --account=general\n'
  printf '#SBATCH --gres=gpu:1\n#SBATCH --constraint=adalovelace\n'
  printf '#SBATCH --cpus-per-task=16\n#SBATCH --mem=64G\n#SBATCH --time=24:00:00\n'
  printf '#SBATCH --requeue\n#SBATCH --array=%s\n' "$ARRAY_SPEC"
  printf '#SBATCH --output=%s/%%A_%%a.out\n#SBATCH --error=%s/%%A_%%a.err\n' \
    "$ARTIFACT_DIR" "$ARTIFACT_DIR"
  printf 'export AIM3_ROOT=%q\n' "$ROOT"
  printf 'export AIM3_RESULTS_PATH=%q\n' "$AIM3_RESULTS_PATH"
  printf 'export AIM3_CLUTTER_DATA_DIR=%q\n' "$AIM3_CLUTTER_DATA_DIR"
  printf 'export AIM3_STATUS_DIR=%q\n' "$STATUS_DIR"
  printf 'export AIM3_SOURCE_COMMIT=%q\n' "$SOURCE_COMMIT"
  printf 'export PYTHONDONTWRITEBYTECODE=1\n'
  printf 'exec bash %q\n' "$LAUNCHER"
} > "$SBATCH_SCRIPT"

sbatch --chdir "$ROOT" --export=ALL "$SBATCH_SCRIPT"
