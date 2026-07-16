#!/usr/bin/env bash
#SBATCH --job-name=aim3-gawf-fb-bench
#SBATCH --partition=gpu-redhat
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/amarel/artifacts/gawf_feedback_benchmark/%j.out
#SBATCH --error=experiments/amarel/artifacts/gawf_feedback_benchmark/%j.err

set -euo pipefail

ROOT="${AIM3_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/train_atari_dqn.py" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
export AIM3_NUM_WORKERS=12
export AIM3_PIN_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

ARTIFACT_DIR="$ROOT/experiments/amarel/artifacts/gawf_feedback_benchmark"
RESULT_DIR="$ROOT/results/benchmarks/gawf_feedback"
mkdir -p "$ARTIFACT_DIR/status" "$RESULT_DIR"

python experiments/amarel/benchmark_gawf_feedback_acceleration.py \
  --output_dir "$RESULT_DIR"

python - "$RESULT_DIR/gawf_feedback_benchmark_validated.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
required = {"legacy_split_eager", "combined_eager", "combined_compiled"}
if set(result.get("timings", {})) != required:
    raise RuntimeError(f"Incomplete benchmark timings in {path}")
if result.get("shape") != {
    "batch_size": 8,
    "input_size": 3136,
    "hidden_size": 1577,
    "feedback_dim": 6,
}:
    raise RuntimeError(f"Unexpected benchmark shape in {path}")
if result.get("validation_passed") is not True:
    raise RuntimeError(f"Numerical validation did not pass in {path}")
PY

printf 'status=done job_id=%s result=%s timestamp=%s\n' \
  "${SLURM_JOB_ID:-manual}" \
  "$RESULT_DIR/gawf_feedback_benchmark_validated.json" "$(date -Is)" \
  > "$ARTIFACT_DIR/status/gawf_feedback_benchmark_validated.done"
