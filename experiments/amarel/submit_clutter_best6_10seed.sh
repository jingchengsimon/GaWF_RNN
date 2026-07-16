#!/usr/bin/env bash
# Submit ten fixed-best Clutter jobs, each containing six model tasks for one seed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_clutter_best6_10seed_array.sh"
GRID_UTIL="${AIM3_GRID_UTIL:-experiments/generalization/clutter_best6_multiseed.py}"
ARTIFACT_TAG="${AIM3_ARTIFACT_TAG:-clutter_best6_10seed_ep150}"
JOB_PREFIX="${AIM3_JOB_PREFIX:-aim3-clutter}"
RESULT_ROOT="${AIM3_RESULT_ROOT:-clutter_best6_multiseed_40h_ep150}"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/$ARTIFACT_TAG"
AIM3_CONDA_INIT="${AIM3_CONDA_INIT:-/home/${USER}/enter/etc/profile.d/conda.sh}"
AIM3_DATA_DIR="${AIM3_DATA_DIR:-/scratch/${USER}/stimuli}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this launcher on an Amarel login node." >&2
  exit 1
fi
if [[ ! -f "$AIM3_CONDA_INIT" ]]; then
  echo "Conda initialization script not found: $AIM3_CONDA_INIT" >&2
  exit 2
fi
mkdir -p "$ARTIFACT_DIR/status"
submission_log="$ARTIFACT_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
job_ids=()
for seed in $(seq 1 10); do
  task_offset="$(((seed - 1) * 6))"
  job_id="$({
    sbatch --parsable \
      --job-name="$JOB_PREFIX-s$(printf '%02d' "$seed")" \
      --constraint=adalovelace \
      --cpus-per-task=16 \
      --mem=64G \
      --array="0-5" \
      --output="$ARTIFACT_DIR/%A_%a.out" \
      --error="$ARTIFACT_DIR/%A_%a.err" \
      --export=ALL,AIM3_ROOT="$ROOT",AIM3_CONDA_INIT="$AIM3_CONDA_INIT",AIM3_CONDA_ENV=aim3_rnn,AIM3_DATA_DIR="$AIM3_DATA_DIR",AIM3_NUM_WORKERS=0,AIM3_PIN_MEMORY=0,AIM3_GRID_UTIL="$GRID_UTIL",TASK_OFFSET="$task_offset" \
      "$RUN_SCRIPT"
  } | tr -d '[:space:]')"
  job_id="${job_id%%;*}"
  job_ids+=("$job_id")
  echo "seed=$seed task_offset=$task_offset job_id=$job_id" | tee -a "$submission_log"
done
job_ids_csv="$(IFS=,; echo "${job_ids[*]}")"

{
  echo "timestamp=$(date -Is)"
  echo "job_ids=$job_ids_csv"
  echo "root=$ROOT"
  echo "submission_shape=10 independent jobs x array[0-5]"
  echo "models=gawf,rnn,lstm,gru,mamba,s5"
  echo "seeds=1-10"
  echo "epochs=150"
  echo "patience=0"
  echo "data_suffix=40h-float32"
  echo "eval_data_suffix=40h-float32"
  echo "resources=partition:gpu-redhat,account:general,gpu:1,constraint:adalovelace,cpus:16,mem:64G"
  echo "dataloader=num_workers:0,pin_memory:false,mmap:true"
  echo "grid_util=$GRID_UTIL"
  echo "result_root=results/train_data/$RESULT_ROOT"
  echo "status_command=squeue -j $job_ids_csv"
} | tee -a "$submission_log"

printf '%s\n' "$job_ids_csv"
