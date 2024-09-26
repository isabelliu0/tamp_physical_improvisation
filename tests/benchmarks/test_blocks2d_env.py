"""Tests for Blocks2DEnv()."""

import numpy as np
from gymnasium.wrappers import RecordVideo

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_blocks_2d_env():
    """Tests for Blocks2DEnv()."""

    env = Blocks2DEnv(render_mode="rgb_array")
    env = RecordVideo(env, "blocks2d-test")
    assert env.observation_space.shape == (5,)
    assert env.action_space.shape == (3,)
    obs, info = env.reset()
    assert obs.shape == (5,)
    assert np.isclose(obs[0], 0.5)
    assert np.isclose(obs[1], 1.0)
    assert not info

    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        print("Action:", action)
        obs, reward, done, terminated, info = env.step(action)
        print("Obs:", obs)
        print("Reward:", reward)

    env.close()
