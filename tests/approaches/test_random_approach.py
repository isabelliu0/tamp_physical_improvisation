"""Test random_approach.py."""

from tamp_improv.approaches.random_approach import RandomApproach
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_random_approach():
    """Tests for RandomApproach()."""

    env = Blocks2DEnv(render_mode="rgb_array")

    # Uncomment to watch a video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/blocks2d-test")

    approach = RandomApproach(env.observation_space, env.action_space, seed=123)

    obs, info = env.reset()
    action = approach.reset(obs, info)

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        action = approach.step(obs, reward, terminated, truncated, info)

    env.close()
