"""Test rl_improvisational_tamp_approach.py."""

from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from tamp_improv.approaches.rl_improvisational_tamp_approach import (
    RLBlocks2DImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_rl_blocks2d_improvisational_tamp_approach(
    policy_path: str = "trained_policies/pushing_policy",
):
    """Tests for RLBlocks2DImprovisationalTAMPApproach().

    Args:
        policy_path: Path to saved RL policy weights
    """
    # Skip if policy file doesn't exist
    if not Path(f"{policy_path}.zip").exists():
        pytest.skip(
            f"Policy file not found at {policy_path}.zip. "
            "Skipping test as this is expected in CI environment."
        )

    # Set up environment
    base_env = Blocks2DEnv(render_mode="rgb_array")
    env = cast(
        gym.Env[NDArray[np.float32], NDArray[np.float32]],
        TimeLimit(base_env, max_episode_steps=100),
    )

    # # Uncomment to watch a video.
    # from gymnasium.wrappers import RecordVideo

    # video_folder = "videos/blocks2d-rl-improvisational-tamp-test"
    # Path(video_folder).mkdir(parents=True, exist_ok=True)
    # env = RecordVideo(env, video_folder)

    # Create approach with trained policy
    approach = RLBlocks2DImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        policy_path=policy_path,
    )

    # Run episode
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward: float = 0.0
    for step in range(100):  # should terminate earlier
        action = approach.step(obs, 0, False, False, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        print(f"Step {step + 1}: Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    env.close()  # type: ignore[no-untyped-call]
