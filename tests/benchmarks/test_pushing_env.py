"""Tests for pushing environment wrapper."""

import numpy as np

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


class RandomPushingPolicy(ImprovisationalPolicy[np.ndarray, np.ndarray]):
    """Random policy for testing."""

    def __init__(self, env):
        self.action_space = env.action_space

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


def test_pushing_env_basic():
    """Test core functionality of pushing environment."""
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)

    obs, _ = env.reset()
    assert env.is_target_area_blocked(obs), "Block 2 should start blocking target area"

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = np.array([0.1, 0.0, 0.0])  # Try to push right
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        assert steps <= env.max_episode_steps, "Episode should not exceed max steps"

        if terminated:
            assert not env.is_target_area_blocked(
                obs
            ), "Episode should end when area is clear"


def test_random_policy():
    """Test policy interface with random actions."""
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)
    policy = RandomPushingPolicy(env)

    obs, _ = env.reset()
    action = policy.get_action(obs)
    assert env.action_space.contains(action), "Policy should generate valid actions"
