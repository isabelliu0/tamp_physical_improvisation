"""Tests for core blocks2d environment."""

import numpy as np
from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv, GraphBlocks2DEnv


def test_graph_blocks2d_env():
    """Test basic functionality of Blocks2D environment."""
    env = GraphBlocks2DEnv(n_blocks=3, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/graph-blocks2d-test")

    obs, info = env.reset()

    env.action_space.seed(123)

    # Hard-coded sequence of actions
    actions = [
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()


def test_blocks2d_env():
    """Test basic functionality of Blocks2D environment."""
    env = Blocks2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/blocks2d-test")

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
