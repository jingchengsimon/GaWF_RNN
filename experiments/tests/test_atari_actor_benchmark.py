"""Validate conservative actor selection and C protocol argument construction."""
from pathlib import Path

from experiments.rl.atari.amarel.actor_benchmark import choose, command


def rows(a, b):
    return [dict(actors=n, throughput={'steps_per_second': rate})
            for n, rates in [(1, a), (5, b)] for rate in rates]


def test_selection_requires_consistent_material_gain():
    assert choose(rows([30, 32], [45, 44]))['selected_actors'] == 5
    assert choose(rows([30, 32], [31, 33]))['selected_actors'] == 1
    assert choose(rows([30, 32], [50, 31]))['selected_actors'] == 1


def test_both_actor_commands_share_c_and_replay_protocol():
    for n in [1, 5]:
        args = command(n, Path('/tmp/unused'), False)
        for key, value in {'--num_envs': str(n), '--buffer_size': '500000',
                           '--exploration_steps': '5000000', '--seq_len': '16',
                           '--learning_rate_decay_per_task_steps': '2000000'}.items():
            assert args[args.index(key) + 1] == value
        assert 'ALE/Riverraid-v5' in args
        assert '--keep_replay_on_success' in args


def test_exact_benchmark_commands_parse_in_real_training_cli():
    from utils.training.train_scripts.atari_dqn import (
        build_arg_parser, _resolve_task_config, _validate_actor_workers,
    )
    for n in [1, 5]:
        args = build_arg_parser().parse_args(command(n, Path('/tmp/unused'), False)[4:])
        env_ids, _ = _resolve_task_config(args)
        _validate_actor_workers(args, env_ids)


def test_actual_five_task_async_envs_have_stable_task_identity():
    import numpy as np
    from utils.training.atari.atari_envs import (
        make_fixed_multitask_async_vector_atari_env, multitask_scheduler_states,
    )
    from utils.training.train_scripts.atari_dqn import _extract_step_env_ids
    from experiments.rl.atari.amarel.actor_benchmark import TASKS
    envs = make_fixed_multitask_async_vector_atari_env(
        tuple(TASKS), seed=1, frame_stack=4, frame_skip=4,
    )
    try:
        obs, _ = envs.reset(seed=1)
        assert len(obs) == 5 and envs.single_action_space.n == 18
        for _ in range(10):
            obs, _, _, _, infos = envs.step(np.zeros(5, dtype=np.int64))
            assert _extract_step_env_ids(infos, 5) == TASKS
        states = multitask_scheduler_states(envs)
        assert [s['task_idx'] for s in states] == list(range(5))
    finally:
        envs.close()
