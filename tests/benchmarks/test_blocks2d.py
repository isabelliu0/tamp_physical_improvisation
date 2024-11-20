"""Tests for Blocks2D environment."""

import numpy as np
from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.blocks2d import Blocks2DEnvironment


def test_blocks2d_env():
    """Test basic functionality of Blocks2D environment."""
    env = Blocks2DEnvironment(include_pushing_models=False)
    base_env = env.env
    base_env = TimeLimit(base_env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # base_env = RecordVideo(base_env, "videos/blocks2d-test")

    obs, info = base_env.reset()

    base_env.action_space.seed(123)

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
        obs, reward, terminated, truncated, info = base_env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    base_env.close()
