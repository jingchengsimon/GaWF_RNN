#!/usr/bin/env bash
#SBATCH --job-name=aim3-ski-audit
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${AIM3_ROOT:?AIM3_ROOT is required}"
: "${AIM3_RESULTS_PATH:?AIM3_RESULTS_PATH is required}"
: "${AIM3_SOURCE_PATH:?AIM3_SOURCE_PATH is required}"
cd "$AIM3_ROOT"

models=(lstm gru gawf)
model="${models[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"
audit_tag="${AUDIT_TAG:-behavior_audit_4m_greedy20_seed1}"
amp_dtype="${AUDIT_AMP_DTYPE:-bfloat16}"
[[ "$audit_tag" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid AUDIT_TAG: $audit_tag" >&2; exit 2; }
[[ "$amp_dtype" == none || "$amp_dtype" == bfloat16 || "$amp_dtype" == float16 ]] || {
  echo "Invalid AUDIT_AMP_DTYPE: $amp_dtype" >&2
  exit 2
}
leaf="atari_dqn_skiing_fs4_stack4_l3_full18_stallactionfix_v1"
leaf+="_extend2mto4m_2m_${model}_seed1"
source_dir="$AIM3_SOURCE_PATH/$leaf"
output_dir="$AIM3_RESULTS_PATH/data/rl/atari/5task_18action/"
output_dir+="formal_20m_4mpertask_raw_seeds/$audit_tag/$model"
metrics_path="$source_dir/metrics.json"
checkpoint="$(find "$source_dir" -maxdepth 1 -type f -name '*.pth' -print)"

[[ -s "$metrics_path" ]] || { echo "Missing source metrics: $metrics_path" >&2; exit 2; }
[[ -n "$checkpoint" && -s "$checkpoint" ]] || {
  echo "Expected one source checkpoint in $source_dir" >&2
  exit 2
}
[[ "$(printf '%s\n' "$checkpoint" | wc -l)" -eq 1 ]] || {
  echo "Expected exactly one source checkpoint in $source_dir" >&2
  exit 2
}
[[ ! -e "$output_dir/summary.json" && ! -e "$output_dir/step_trace.npz" ]] || {
  echo "Refusing to overwrite audit output: $output_dir" >&2
  exit 3
}
mkdir -p "$output_dir"

set +u
source "${AIM3_CONDA_SH:-/home/js3269/enter/etc/profile.d/conda.sh}"
conda activate "${AIM3_CONDA_ENV:-aim3_rnn}"
set -u

python -B -m utils.analysis.rl.atari.evaluate_skiing_behavior \
  --metrics_path "$metrics_path" \
  --checkpoint "$checkpoint" \
  --summary_path "$output_dir/summary.json" \
  --trace_path "$output_dir/step_trace.npz" \
  --num_episodes "${NUM_EVAL_EPISODES:-20}" \
  --eval_seed "${EVAL_SEED:-20260904}" \
  --device cuda \
  --amp_dtype "$amp_dtype"

[[ -s "$output_dir/summary.json" && -s "$output_dir/step_trace.npz" ]] || {
  echo "Behavior audit outputs are incomplete: $output_dir" >&2
  exit 4
}
touch "$output_dir/done"
echo "model=$model output=$output_dir"
