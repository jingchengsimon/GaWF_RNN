#!/usr/bin/env bash
# Submit fixed best-hyperparameter IMDB training for five models and seeds 1--10.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_TAG="imdb_5model_best10seed"
PREP_SCRIPT="$SCRIPT_DIR/run_prepare_imdb_data.sh"
RUN_SCRIPT="$SCRIPT_DIR/run_imdb_5model_best10seed.sh"
ART_DIR="$ROOT/experiments/text/amarel/artifacts/$RUN_TAG"
DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"
RESULTS_DIR="${AIM3_RESULTS_PATH:-/scratch/${USER}/results}"
ARRAY_SPEC="0-49%10"

echo "run_tag=$RUN_TAG"
echo "models=lstm,rnn,gru,gawf,gawf_logits"
echo "seeds=1-10"
echo "array=$ARRAY_SPEC"
echo "data_dir=$DATA_DIR"
echo "results_dir=$RESULTS_DIR"
echo "result_path=$RESULTS_DIR/data/text/runs/$RUN_TAG"
echo "resources=partition=gpu gres=gpu:1 constraint=adalovelace cpus=16 mem=64G"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry_run=1"
  echo "preflight=prepare processed IMDB data when missing"
  echo "training_dependency=afterok:<preflight_job_id> when preflight is submitted"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on an Amarel login node." >&2
  exit 1
fi

mkdir -p "$ART_DIR"
export_arg="ALL,AIM3_ROOT=$ROOT,AIM3_RESULTS_PATH=$RESULTS_DIR"
export_arg+=",AIM3_DATA_DIR=$DATA_DIR,AIM3_NUM_WORKERS=12,AIM3_PIN_MEMORY=1"

dependency_args=()
prep_job_id=""
if [[ ! -s "$DATA_DIR/imdb/imdb_meta.json" ]]; then
  prep_job_id="$(
    sbatch --parsable \
      --output="$ART_DIR/%A_prep.out" \
      --error="$ART_DIR/%A_prep.err" \
      --export="$export_arg" \
      "$PREP_SCRIPT"
  )"
  dependency_args+=("--dependency=afterok:$prep_job_id")
fi

train_job_id="$(
  sbatch --parsable \
    --output="$ART_DIR/%A_%a.out" \
    --error="$ART_DIR/%A_%a.err" \
    --export="$export_arg" \
    --array="$ARRAY_SPEC" \
    "${dependency_args[@]}" \
    "$RUN_SCRIPT"
)"

echo "prep_job_id=${prep_job_id:-not_needed}"
echo "train_job_id=$train_job_id"
echo "status=squeue -j ${prep_job_id:+$prep_job_id,}$train_job_id"
