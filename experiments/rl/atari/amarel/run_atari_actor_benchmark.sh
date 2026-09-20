#!/usr/bin/env bash
# Compute-node-only validation and one-allocation actor selection benchmark.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 DISABLE_TQDM=1
: "${AIM3_ROOT:?}" "${ACTOR_BENCHMARK_RESULT:?}"
cd "$AIM3_ROOT"
source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
trap 'rc=$?; if ((rc)); then echo "launcher exit=$rc" >> "$ACTOR_BENCHMARK_RESULT/fail"; fi' EXIT
nvidia-smi > "$ACTOR_BENCHMARK_RESULT/gpu_start.txt"
lscpu > "$ACTOR_BENCHMARK_RESULT/cpu.txt"
python -B -m pytest -q -p no:cacheprovider experiments/tests/test_atari_stage_profile.py experiments/tests/test_atari_multitask_protocol.py experiments/tests/test_atari_actor_benchmark.py experiments/tests/test_atari_parallel_collection.py experiments/tests/test_amarel_submit_safety.py > "$ACTOR_BENCHMARK_RESULT/tests.log" 2>&1
python -B -m experiments.rl.atari.amarel.actor_benchmark --output "$ACTOR_BENCHMARK_RESULT" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true; wait "$child"; exit 99' USR1 TERM
wait "$child"
