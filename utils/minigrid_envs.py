"""Lazy Gymnasium/MiniGrid environment helpers for the DRQN family.

MiniGrid is a gymnasium-native suite, so this mirrors ``utils/atari_envs.py``:
factories import gymnasium/minigrid only inside the thunk so model tests run on
machines without the extras. We keep the *partial* egocentric 7x7 symbolic view
(the source of partial observability) via ``ImgObsWrapper`` and transpose it to
channel-first ``(3, 7, 7)`` uint8 to match the DRQN encoder-slot convention. No
frame stacking: the recurrent core is the only source of memory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

# A few canonical memory/navigation envs. MemoryS* is the standard memory
# benchmark whose corridor length (S7<S9<S11<S13) sets the required memory horizon.
MINIGRID_PILOT_ENVS = (
    "MiniGrid-MemoryS7-v0",
    "MiniGrid-MemoryS9-v0",
    "MiniGrid-MemoryS11-v0",
    "MiniGrid-MemoryS13-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-RedBlueDoors-6x6-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
)

MiniGridVectorBackend = Literal["sync", "async"]
MINIGRID_VECTOR_BACKENDS = ("sync", "async")


def make_minigrid_env(
    env_id: str,
    seed: int,
    idx: int,
    agent_view_size: int | None = None,
    capture_video: bool = False,
    video_dir: str | None = None,
) -> Callable[[], object]:
    """Return a thunk that creates one MiniGrid env with a channel-first symbolic view.

    ``agent_view_size`` (odd, >=3) shrinks the egocentric view; a small view (e.g. 3)
    forces the agent to rely on memory instead of seeing the whole room reactively
    (as in Toro Icarte et al., 2020 for RedBlueDoors/Memory).
    """

    def thunk():
        try:
            import gymnasium as gym
            import minigrid  # noqa: F401 - registers MiniGrid-* envs with gymnasium
            from minigrid.wrappers import ImgObsWrapper
        except ImportError as exc:
            raise ImportError(
                "MiniGrid experiments require the 'minigrid' package, e.g. "
                "`pip install minigrid`."
            ) from exc
        import numpy as np

        render_mode = "rgb_array" if capture_video else None
        make_kwargs = {"render_mode": render_mode}
        if agent_view_size is not None:
            make_kwargs["agent_view_size"] = int(agent_view_size)
        env = gym.make(env_id, **make_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if video_dir is None:
                raise ValueError("video_dir must be set when capture_video=True")
            env = gym.wrappers.RecordVideo(env, video_dir)
        env = ImgObsWrapper(env)  # keep only the 7x7x3 symbolic image (drop mission)

        class _ChannelFirst(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                h, w, c = env.observation_space.shape
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=(c, h, w), dtype=np.uint8
                )

            def observation(self, obs):
                return np.ascontiguousarray(
                    np.transpose(np.asarray(obs), (2, 0, 1)), dtype=np.uint8
                )

        env = _ChannelFirst(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def make_vector_minigrid_env(
    env_id: str,
    seed: int,
    num_envs: int,
    agent_view_size: int | None = None,
    capture_video: bool = False,
    video_dir: str | None = None,
    vector_backend: MiniGridVectorBackend = "sync",
) -> object:
    """Create a synchronous or subprocess-parallel vector MiniGrid environment."""
    if vector_backend not in MINIGRID_VECTOR_BACKENDS:
        raise ValueError(f"Unsupported MiniGrid vector backend: {vector_backend}")
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MiniGrid experiments require gymnasium and minigrid, e.g. "
            "`pip install minigrid`."
        ) from exc

    env_fns = [
        make_minigrid_env(env_id, seed, idx, agent_view_size, capture_video, video_dir)
        for idx in range(num_envs)
    ]
    if vector_backend == "async":
        # CUDA is configured before environment construction in the trainer;
        # spawn avoids inheriting an initialized CUDA runtime via fork.
        return gym.vector.AsyncVectorEnv(env_fns, context="spawn")
    return gym.vector.SyncVectorEnv(env_fns)
