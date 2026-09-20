#!/usr/bin/env bash
# One GaWF-only short profiling job; CPU/CUDA checks and smoke run on this compute node.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 DISABLE_TQDM=1
: "${AIM3_ROOT:?}" "${AIM3_RESULTS_PATH:?}" "${PROFILE_RESULT:?}"
cd "$AIM3_ROOT"
source /home/js3269/enter/etc/profile.d/conda.sh
conda activate aim3_rnn
nvidia-smi > "$PROFILE_RESULT/gpu_start.txt"
lscpu > "$PROFILE_RESULT/cpu.txt"
trap 'rc=$?; if ((rc)); then echo "$rc" > "$PROFILE_RESULT/fail"; fi' EXIT
python -B -m pytest -q -p no:cacheprovider experiments/tests/test_atari_stage_profile.py experiments/tests/test_atari_multitask_protocol.py experiments/tests/test_amarel_submit_safety.py > "$PROFILE_RESULT/tests.log" 2>&1
args=(--env_ids ALE/Pong-v5 ALE/Breakout-v5 ALE/Assault-v5 ALE/Seaquest-v5 ALE/Riverraid-v5
 --action_space_mode full18 --atari_env_protocol baseline --model_type gawf --num_layers 3
 --hidden_size 605 --feedback_mode qvalues --gawf_feedback_lr_scale 1.0 --frame_skip 4
 --frame_stack 4 --flicker_prob 0 --num_envs 1 --task_schedule transition_balanced
 --replay_sampling task_balanced --replay_backing mmap --replay_layout per_task --buffer_size 500000
 --batch_size 32 --seq_len 16 --sequences_per_batch 8 --learning_rate 1e-4
 --learning_rate_decay_step 0 --learning_rate_decay_per_task_steps 2000000
 --learning_rate_decay_scale .1 --start_epsilon 1 --end_epsilon .05 --exploration_steps 5000000
 --gamma .99 --train_frequency 4 --target_network_frequency 1000 --max_grad_norm 10
 --seed 1 --device cuda --amp_dtype bfloat16 --allow_tf32 --cudnn_benchmark --fused_optimizer
 --checkpoint_interval_steps 0 --keep_replay_on_success --record_timing --profile_stages)
python -B run_task.py atari-dqn "${args[@]}" --total_timesteps 25000 --learning_starts 100 --learning_starts_per_task 20 --save_dir "$PROFILE_RESULT/smoke" --result_suffix riverraid_c_profile_smoke > "$PROFILE_RESULT/smoke.log" 2>&1
python -B - "$PROFILE_RESULT/smoke" <<'PY'
import json,sys,math
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'metrics.json').read_text())
assert m['global_step']==25000 and math.isfinite(m['loss'])
assert Path(m['checkpoint']).is_file()
s=json.loads((p/'profile/stages.json').read_text())
assert s['phase_counts'].get('optimization',0)>0
assert (p/'profile/trace.json').stat().st_size>0
(p.parent/'smoke_accepted.json').write_text(json.dumps({'steps':25000,'retained':True})+'\n')
PY
python -B run_task.py atari-dqn "${args[@]}" --total_timesteps 200000 --learning_starts 20000 --learning_starts_per_task 20000 --save_dir "$PROFILE_RESULT/profile_run" --result_suffix riverraid_c_gawf_profile_seed1 > "$PROFILE_RESULT/profile_run.log" 2>&1
python -B - "$PROFILE_RESULT/profile_run" <<'PY'
import json,sys,math
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'metrics.json').read_text())
assert m['global_step']==200000 and math.isfinite(m['loss'])
assert m['reward_clip'] is True and m['gamma']==.99
assert m['buffer_size_per_task']==500000 and m['replay_sampling']=='task_balanced'
assert Path(m['checkpoint']).is_file()
s=json.loads((p/'profile/stages.json').read_text())
t=json.loads((p/'profile/throughput.json').read_text())
assert s['phase_counts'].get('optimization',0)>=60 and t['steps']>=10000
assert (p/'profile/trace.json').stat().st_size>0
(p.parent/'done').write_text('validated\n')
PY
