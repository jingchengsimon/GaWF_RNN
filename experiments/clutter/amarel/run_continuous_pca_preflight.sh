#!/usr/bin/env bash
#SBATCH --job-name=clutter-pca-preflight
#SBATCH --partition=main
#SBATCH --account=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
# Materialize the transferred movie on a compute node before dependent inference.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$1"; DATA_DIR="$2"; CONDA_SH="$3"
STEM="$DATA_DIR/stimulus_reg-test-40h-float32-jointswitch-balanced-10digit-unique"
test -n "${SLURM_JOB_ID:-}"
printf '%s  %s\n' c43e04782158d7bbaa553e1acddb1996ce3013b914d53fb854fe59c5c5577879 "$STEM.npy.gz" | sha256sum -c -
gzip -dc "$STEM.npy.gz" > "$STEM.npy"
test "$(stat -c %s "$STEM.npy")" = 2123366528
cd "$ROOT"
set +u
source "$CONDA_SH"
conda activate aim3_rnn
set -u
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [[ "${4:-}" == install-deps ]]; then
  python -B - <<'PY'
import importlib.util
import subprocess
import sys

packages = {
    "pytest": "pytest", "numpy": "numpy", "torch": "torch",
    "matplotlib": "matplotlib", "pandas": "pandas", "scipy": "scipy",
    "seaborn": "seaborn", "tqdm": "tqdm", "sklearn": "scikit-learn",
}
missing = [package for module, package in packages.items()
           if importlib.util.find_spec(module) is None]
print("Missing dependencies:", missing, flush=True)
if missing:
    subprocess.run([sys.executable, "-B", "-m", "pip", "install", "--no-cache-dir", *missing],
                   check=True)
PY
fi
python -B -m pytest -q experiments/tests/test_continuous_switch_pca.py
python -B -m utils.analysis.clutter.continuous_switch_pca collect --help
sha256sum "$STEM.npy" "$STEM.tsv" "$STEM.npy.gz"
