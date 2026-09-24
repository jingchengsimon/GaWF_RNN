#!/usr/bin/env bash
# Submit the ablation behaviour analysis: one array element per model/seed unit, each running the
# compute-node launcher (reset-excluded test-accuracy collect when missing plus switch-recovery
# inference). Elements whose training unit has not finished simply exit without work, so the same
# submission can be repeated to sweep the remaining units.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIM3_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DRY_RUN=0
MAX_CONCURRENT=0
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --max-concurrent) MAX_CONCURRENT="${2:?}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || { echo "--max-concurrent must be an integer" >&2; exit 2; }

MODELS=(gawf_nowrap rnn_nowrap lstm_nowrap gru_nowrap mamba_nowrap s5_nowrap gawf_notanh rnn_notanh)
ARRAY_SPEC="0-39"
if (( MAX_CONCURRENT > 0 )); then
  ARRAY_SPEC="0-39%${MAX_CONCURRENT}"
fi
LAUNCHER="$SCRIPT_DIR/run_clutter_ablation_behavior.sh"
LOG_DIR="$SCRIPT_DIR/../amarel/artifacts/clutter_ablation_behavior_v1"
GENERATED="$ROOT/experiments/clutter/amarel/generated"

if (( DRY_RUN )); then
  printf 'submit: array=%s max_concurrent=%s\n' "$ARRAY_SPEC" "$MAX_CONCURRENT"
  printf 'unit map: index i -> model %s, seed i%%5+1\n' "${MODELS[*]}"
  printf 'launcher=%s logs=%s\n' "$LAUNCHER" "$LOG_DIR"
  printf 'test export: data/analysis/clutter_ablation_reset_excluded_test_10seed_v1\n'
  printf 'recovery export: data/analysis/ablation_behavior_recovery_jointswitch10digit_v1\n'
  exit 0
fi

: "${AIM3_RESULTS_PATH:?Export AIM3_RESULTS_PATH}"
: "${AIM3_CLUTTER_DATA_DIR:?Export AIM3_CLUTTER_DATA_DIR}"
for required in \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-uint8.npy" \
  "$AIM3_CLUTTER_DATA_DIR/stimulus_reg-test-40h-float32-jointswitch-balanced-10digit-unique.npy"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

mkdir -p "$LOG_DIR" "$GENERATED" "$AIM3_RESULTS_PATH/data/analysis/ablation_behavior_recovery_jointswitch10digit_v1"
SBATCH_SCRIPT="$GENERATED/sbatch_clutter_ablation_behavior_$(date +%Y%m%d_%H%M%S).sh"
{
  printf '#!/usr/bin/env bash\n#SBATCH --job-name=aim3-abl-behavior\n'
  printf '#SBATCH --partition=gpu\n'
  printf '#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=64G\n#SBATCH --time=04:00:00\n'
  printf '#SBATCH --output=%s/%%A_%%a.out\n#SBATCH --error=%s/%%A_%%a.err\n' "$LOG_DIR" "$LOG_DIR"
  printf 'set -euo pipefail\nexport AIM3_SKIP_COLLECT=1\n'
  printf 'mapfile -t MODELS <<< "%s"\n' "$(printf '%s\n' "${MODELS[@]}")"
  printf 'for MODEL in "${MODELS[@]}"; do\n'
  printf '  for SEED in 1 2 3 4 5; do\n'
  printf '    bash %q --model "$MODEL" --seed "$SEED" || echo "unit-failed: $MODEL-seed$SEED" >&2\n' "$LAUNCHER"
  printf '  done\n'
  printf 'done\n'
} > "$SBATCH_SCRIPT"

sbatch --chdir "$ROOT" --export=ALL "$SBATCH_SCRIPT"
