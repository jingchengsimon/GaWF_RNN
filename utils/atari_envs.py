"""Lazy Gymnasium/ALE environment helpers for Atari experiments.

This module intentionally imports Gymnasium only inside factory functions so model
tests can run on machines without Atari ROMs or Gymnasium extras installed.
"""

from __future__ import annotations

from collections.abc import Callable

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
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
) -> Callable[[], object]:
    """Return a thunk that creates one preprocessed Atari environment."""

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
            full_action_space=False,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if video_dir is None:
                raise ValueError("video_dir must be set when capture_video=True")
            env = gym.wrappers.RecordVideo(env, video_dir)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
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
    flicker_prob: float = 0.0,
    capture_video: bool = False,
    video_dir: str | None = None,
):
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
            env_id, seed, idx, frame_stack, flicker_prob, capture_video, video_dir
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)
