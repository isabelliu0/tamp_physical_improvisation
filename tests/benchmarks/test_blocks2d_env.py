"""Tests for Blocks2DEnv()."""

import numpy as np
from gymnasium.wrappers import RecordVideo, TimeLimit

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_blocks_2d_env():
    """Tests for Blocks2DEnv()."""

    env = Blocks2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)
    env = RecordVideo(env, "videos/blocks2d-test")
    obs, info = env.reset()

    env.action_space.seed(123)

    # Hard-coded sequence of actions to reach the goal
    actions = [
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, 0.0, 1.0]),  # Activate gripper and pick up block
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.0, 0.0, -1.0]),  # Drop block
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()
