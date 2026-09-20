"""Run one allocation's GaWF C five-task actor benchmark and choose 1 or 5 actors.

Inputs: an empty exact result root. Outputs: two smokes, four fresh 300k trials,
raw GPU samples, stage traces and a deterministic recommendation.json. All outputs retained.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys

TASKS = ['ALE/Pong-v5', 'ALE/Breakout-v5', 'ALE/Assault-v5',
         'ALE/Seaquest-v5', 'ALE/Riverraid-v5']
ORDER = [(1, 1), (5, 1), (5, 2), (1, 2)]


def command(actors: int, output: Path, smoke: bool) -> list[str]:
    """Preserve C schedules, per-task replay capacity and transition-based UTD."""
    values = dict(action_space_mode='full18', atari_env_protocol='baseline', model_type='gawf',
                  num_layers=3, hidden_size=605, feedback_mode='qvalues', gawf_feedback_lr_scale=1,
                  frame_skip=4, frame_stack=4, flicker_prob=0, num_envs=actors,
                  actor_workers=0 if actors == 1 else 5, task_schedule='transition_balanced',
                  replay_sampling='task_balanced', replay_layout='per_task', replay_backing='mmap',
                  buffer_size=500000, batch_size=32, seq_len=16, sequences_per_batch=8,
                  learning_rate=1e-4, learning_rate_decay_step=0,
                  learning_rate_decay_per_task_steps=2000000, learning_rate_decay_scale=.1,
                  start_epsilon=1, end_epsilon=.05, exploration_steps=5000000,
                  gamma=.99, train_frequency=4, target_network_frequency=1000, max_grad_norm=10,
                  seed=1, device='cuda', amp_dtype='bfloat16', checkpoint_interval_steps=0,
                  total_timesteps=25000 if smoke else 300000,
                  learning_starts=100 if smoke else 20000,
                  learning_starts_per_task=20 if smoke else 20000,
                  save_dir=str(output), result_suffix=output.name)
    argv = [sys.executable, '-B', 'run_task.py', 'atari-dqn', '--env_ids', *TASKS]
    for key, value in values.items():
        argv.extend(['--' + key, str(value)])
    argv += ['--allow_tf32', '--cudnn_benchmark', '--fused_optimizer',
             '--keep_replay_on_success', '--record_timing']
    if not smoke:
        argv.append('--profile_stages')
    return argv


def validate(output: Path, actors: int, smoke: bool) -> dict:
    """Require real training, expected protocol and a substantial post-trace window."""
    metrics = json.loads((output / 'metrics.json').read_text())
    expected = dict(global_step=25000 if smoke else 300000, model_type='gawf',
                    seq_len=16, buffer_size_per_task=500000, replay_sampling='task_balanced',
                    gamma=.99, reward_clip=True, amp_dtype='bfloat16',
                    exploration_steps=5000000, learning_rate_decay_per_task_steps=2000000)
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f'{output.name}: {key}={metrics.get(key)}, expected {value}')
    if not math.isfinite(metrics['loss']) or not Path(metrics['checkpoint']).is_file():
        raise RuntimeError('Missing trained model or non-finite loss')
    if set(metrics['per_env']) != set(TASKS):
        raise RuntimeError('Five-task identity mismatch')
    if metrics['timing']['optimizer_updates'] <= 0:
        raise RuntimeError('No optimizer updates')
    row = dict(actors=actors, unit=output.name, steps=metrics['global_step'],
               optimizer_updates=metrics['timing']['optimizer_updates'],
               task_steps={k: v['environment_steps'] for k, v in metrics['per_env'].items()})
    if not smoke:
        timing = json.loads((output / 'profile/throughput.json').read_text())
        stages = json.loads((output / 'profile/stages.json').read_text())
        if timing['steps'] < 50000 or stages['phase_counts'].get('optimization', 0) < 60:
            raise RuntimeError('Insufficient post-warmup training/profile coverage')
        rate = timing['steps_per_second']
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError('Invalid throughput')
        row.update(throughput=timing, stages=stages)
    return row


def choose(rows: list[dict]) -> dict:
    """Prefer five only with >=10% median gain and a gain in each paired repeat."""
    rates = {n: [r['throughput']['steps_per_second'] for r in rows if r['actors'] == n]
             for n in (1, 5)}
    if any(len(v) != 2 for v in rates.values()):
        raise ValueError('Need two valid trials for each actor count')
    medians = {n: statistics.median(v) for n, v in rates.items()}
    paired = [rates[5][i] / rates[1][i] for i in range(2)]
    gain = medians[5] / medians[1]
    selected = 5 if gain >= 1.10 and min(paired) > 1.0 else 1
    return dict(selected_actors=selected, median_steps_per_second=medians,
                median_speedup=gain, paired_speedups=paired, trials=rows,
                projected_20m_compute_hours=20000000 / medians[selected] / 3600,
                rule='Choose 5 only if median gain >=10% and both paired repeats improve; else 1.',
                scope='Same allocation, C optimization, five tasks, 500k/task capacity, fixed UTD.',
                limits=['300k trials use partly filled replay; projection excludes queue time.',
                        'Selection concerns throughput; it does not establish long-run return parity.',
                        'Five means one fixed-task CPU process per game and one GPU learner.',
                        'No untested recommendation above five actors.'])


def run_one(root: Path, actors: int, repeat: int, smoke: bool) -> dict:
    """Run one fresh isolated process and preserve its command, logs and replay."""
    label = f'smoke_actor{actors}' if smoke else f'actor{actors}_repeat{repeat}'
    output = root / label
    if output.exists():
        raise FileExistsError(output)
    artifact = root / (label + '_artifacts')
    artifact.mkdir(exist_ok=False)
    argv = command(actors, output, smoke)
    (artifact / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
    with (artifact / 'gpu.csv').open('w') as gpu, (artifact / 'train.log').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,name,utilization.gpu,'
                                    'utilization.memory,memory.used,power.draw,clocks.sm',
                                    '--format=csv', '-l', '10'], stdout=gpu, stderr=subprocess.STDOUT)
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        def pause(signum: int, frame: object) -> None:
            process.send_signal(signal.SIGUSR1)
        previous = {s: signal.signal(s, pause) for s in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            rc = process.wait()
        finally:
            monitor.terminate()
            monitor.wait()
            for s, handler in previous.items():
                signal.signal(s, handler)
    if rc:
        (artifact / 'fail').write_text(str(rc))
        raise RuntimeError(f'{label} exit={rc}; see {artifact / "train.log"}')
    row = validate(output, actors, smoke)
    (artifact / 'done').write_text(json.dumps(row, indent=2) + '\n')
    return row


def main() -> None:
    """Execute both smokes then interleaved trials, and persist a final recommendation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output
    try:
        for actors in (1, 5):
            run_one(root, actors, 0, True)
        rows = []
        for actors, repeat in ORDER:
            rows.append(run_one(root, actors, repeat, False))
            (root / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        decision = choose(rows)
        (root / 'recommendation.json').write_text(json.dumps(decision, indent=2) + '\n')
        (root / 'done').write_text('validated\n')
    except BaseException as exc:
        (root / 'fail').write_text(str(exc) + '\n')
        raise


if __name__ == '__main__':
    main()
