#!/usr/bin/env bash
# Submit 60 fixed-best Clutter jobs: six models x ten independent seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

RUN_SCRIPT="$SCRIPT_DIR/run_clutter_best6_10seed_array.sh"
ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/clutter_best6_10seed_ep150"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-12}"
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
if [[ ! "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ARRAY_CONCURRENCY must be a positive integer." >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR/status"
submission_log="$ARTIFACT_DIR/submission_$(date +%Y%m%d_%H%M%S).log"
job_id="$({
  sbatch --parsable \
    --constraint=adalovelace \
    --cpus-per-task=16 \
    --mem=64G \
    --array="0-59%${ARRAY_CONCURRENCY}" \
    --export=ALL,AIM3_ROOT="$ROOT",AIM3_CONDA_INIT="$AIM3_CONDA_INIT",AIM3_CONDA_ENV=aim3_rnn,AIM3_DATA_DIR="$AIM3_DATA_DIR",AIM3_NUM_WORKERS=0,AIM3_PIN_MEMORY=0 \
    "$RUN_SCRIPT"
} | tr -d '[:space:]')"
job_id="${job_id%%;*}"

{
  echo "timestamp=$(date -Is)"
  echo "job_id=$job_id"
  echo "root=$ROOT"
  echo "array=0-59%${ARRAY_CONCURRENCY}"
  echo "models=gawf,rnn,lstm,gru,mamba,s5"
  echo "seeds=1-10"
  echo "epochs=150"
  echo "patience=0"
  echo "data_suffix=40h-float32"
  echo "eval_data_suffix=40h-float32"
  echo "resources=partition:gpu-redhat,account:general,gpu:1,constraint:adalovelace,cpus:16,mem:64G"
  echo "dataloader=num_workers:0,pin_memory:false,mmap:true"
  echo "result_root=results/train_data/clutter_best6_multiseed_40h_ep150"
  echo "status_command=squeue -j $job_id"
} | tee "$submission_log"

printf '%s\n' "$job_id"
