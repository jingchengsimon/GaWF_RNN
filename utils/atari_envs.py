"""Lazy Gymnasium/ALE environment helpers for Atari experiments.

This module intentionally imports Gymnasium only inside factory functions so model
tests can run on machines without Atari ROMs or Gymnasium extras installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ATARI_PILOT_ENVS = (
    "ALE/Pong-v5",
    "ALE/Breakout-v5",
    "ALE/MsPacman-v5",
    "ALE/BeamRider-v5",
)


def _register_ale_envs(gym) -> None:
    """Register ALE namespaces for Gymnasium versions that require explicit setup."""
    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require ale-py, e.g. "
            "`pip install 'gymnasium[atari]' ale-py`."
        ) from exc
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)


def _frame_stack(env, gym, frame_stack: int):
    if frame_stack <= 1:
        return env
    if hasattr(gym.wrappers, "FrameStackObservation"):
        try:
            return gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        except TypeError:
            return gym.wrappers.FrameStackObservation(env, frame_stack)
    if hasattr(gym.wrappers, "FrameStack"):
        return gym.wrappers.FrameStack(env, num_stack=frame_stack)
    raise RuntimeError("Gymnasium frame stack wrapper not found")


def _flicker(env, gym, flicker_prob: float, seed: int):
    """Flickering-Atari partial observability (Hausknecht & Stone, 2015).

    At every timestep the entire screen is obscured (returned as an all-zero
    frame) with probability ``flicker_prob``, otherwise the true observation is
    passed through. This is applied on the preprocessed 84x84 frame and *before*
    frame stacking, so with ``frame_stack=1`` each single-frame observation is
    independently blanked, turning the MDP into a POMDP that requires temporal
    integration to recover the hidden game state.
    """
    if flicker_prob <= 0.0:
        return env

    import numpy as np  # local import: numpy is only needed inside the factory

    class _FlickerObservation(gym.ObservationWrapper):
        def __init__(self, env, prob: float, rng_seed: int) -> None:
            super().__init__(env)
            self.prob = float(prob)
            self._rng = np.random.default_rng(rng_seed)

        def observation(self, observation):
            if self._rng.random() < self.prob:
                return np.zeros_like(np.asarray(observation))
            return observation

    return _FlickerObservation(env, flicker_prob, seed)


def make_atari_env(
    env_id: str,
    seed: int,
    idx: int,
    frame_stack: int = 4,
    frame_skip: int = 4,
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
    full_action_space: bool = False,
) -> Callable[[], object]:
    """Return a thunk that creates one preprocessed Atari environment."""

    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

    def thunk():
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "Atari experiments require gymnasium with Atari extras, e.g. "
                "`pip install 'gymnasium[atari,accept-rom-license]'`."
            ) from exc
        _register_ale_envs(gym)

        env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=full_action_space,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if video_dir is None:
                raise ValueError("video_dir must be set when capture_video=True")
            env = gym.wrappers.RecordVideo(env, video_dir)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=frame_skip,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )
        env = _flicker(env, gym, flicker_prob, seed + idx)
        env = _frame_stack(env, gym, frame_stack)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vector_atari_env(
    env_id: str,
    seed: int,
    num_envs: int,
    frame_stack: int = 4,
    frame_skip: int = 4,
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
    full_action_space: bool = False,
) -> Any:
    """Create a synchronous vector Atari environment."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require gymnasium with Atari extras, e.g. "
            "`pip install 'gymnasium[atari,accept-rom-license]'`."
        ) from exc
    _register_ale_envs(gym)

    env_fns = [
        make_atari_env(
            env_id,
            seed,
            idx,
            frame_stack,
            frame_skip,
            flicker_prob,
            capture_video,
            video_dir,
            full_action_space,
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def make_multitask_atari_env(
    env_ids: tuple[str, ...],
    seed: int,
    idx: int,
    frame_stack: int = 1,
    frame_skip: int = 1,
    flicker_prob: float = 0.0,
    task_schedule: str = "round_robin",
) -> Callable[[], object]:
    """Return one task-blind Atari env that switches games at episode resets.

    All component games expose ALE's canonical 18-action space. The active
    ``env_id`` and integer ``task_id`` are emitted in ``info`` for metrics only;
    neither is added to the observation consumed by the agent.
    """
    if len(env_ids) < 2:
        raise ValueError("Multi-task Atari requires at least two env_ids")
    if len(set(env_ids)) != len(env_ids):
        raise ValueError("Multi-task env_ids must be unique")
    if task_schedule != "round_robin":
        raise ValueError(f"Unsupported Phase0 task_schedule: {task_schedule}")

    def thunk():
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "Atari experiments require gymnasium with Atari extras, e.g. "
                "`pip install 'gymnasium[atari,accept-rom-license]'`."
            ) from exc
        _register_ale_envs(gym)

        component_envs = [
            make_atari_env(
                env_id=env_id,
                seed=seed + task_idx * 10_000,
                idx=idx,
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                flicker_prob=flicker_prob,
                full_action_space=True,
            )()
            for task_idx, env_id in enumerate(env_ids)
        ]

        class _EpisodeSwitchAtariEnv(gym.Env):
            metadata = component_envs[0].metadata

            def __init__(self) -> None:
                super().__init__()
                self._envs = component_envs
                self._env_ids = env_ids
                self._next_task_idx = idx % len(self._envs)
                self._active_task_idx: int | None = None
                self._has_reset = [False] * len(self._envs)
                self.action_space = self._envs[0].action_space
                self.observation_space = self._envs[0].observation_space
                self.render_mode = getattr(self._envs[0], "render_mode", None)
                for env_id, env in zip(self._env_ids, self._envs):
                    if env.action_space != self.action_space:
                        raise RuntimeError(f"Action space mismatch for {env_id}")
                    if env.observation_space != self.observation_space:
                        raise RuntimeError(f"Observation space mismatch for {env_id}")
                if self.action_space.__class__.__name__ != "Discrete":
                    raise RuntimeError("Atari multi-task action space must be Discrete")
                if int(self.action_space.n) != 18:
                    raise RuntimeError(
                        "Atari multi-task Phase0 requires the canonical 18-action space"
                    )

            def _add_task_info(self, info: dict[str, Any]) -> dict[str, Any]:
                if self._active_task_idx is None:
                    raise RuntimeError("Multi-task environment has not been reset")
                enriched = dict(info)
                enriched["task_id"] = self._active_task_idx
                enriched["env_id"] = self._env_ids[self._active_task_idx]
                return enriched

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ):
                task_idx = self._next_task_idx
                self._next_task_idx = (task_idx + 1) % len(self._envs)
                self._active_task_idx = task_idx
                reset_seed = seed
                if reset_seed is None and not self._has_reset[task_idx]:
                    reset_seed = seed_value + task_idx * 10_000 + idx
                obs, info = self._envs[task_idx].reset(seed=reset_seed, options=options)
                self._has_reset[task_idx] = True
                return obs, self._add_task_info(info)

            def step(self, action: int):
                if self._active_task_idx is None:
                    raise RuntimeError("Call reset() before step()")
                obs, reward, terminated, truncated, info = self._envs[
                    self._active_task_idx
                ].step(action)
                return (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    self._add_task_info(info),
                )

            def render(self):
                if self._active_task_idx is None:
                    return None
                return self._envs[self._active_task_idx].render()

            def close(self) -> None:
                for env in self._envs:
                    env.close()

        seed_value = seed
        return _EpisodeSwitchAtariEnv()

    return thunk


def make_multitask_vector_atari_env(
    env_ids: tuple[str, ...],
    seed: int,
    num_envs: int,
    frame_stack: int = 1,
    frame_skip: int = 1,
    flicker_prob: float = 0.0,
    task_schedule: str = "round_robin",
) -> Any:
    """Create synchronous task-blind Atari envs with episode-level switching."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "Atari experiments require gymnasium with Atari extras, e.g. "
            "`pip install 'gymnasium[atari,accept-rom-license]'`."
        ) from exc
    _register_ale_envs(gym)
    env_fns = [
        make_multitask_atari_env(
            env_ids=env_ids,
            seed=seed,
            idx=idx,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            flicker_prob=flicker_prob,
            task_schedule=task_schedule,
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)
