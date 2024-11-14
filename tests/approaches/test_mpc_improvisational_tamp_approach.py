"""Test mpc_improvisational_tamp_approach.py."""

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from tamp_improv.approaches.mpc_improvisational_tamp_approach import (
    MPCBlocks2DImprovisationalTAMPApproach,
    PredictiveSamplingConfig,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_mpc_blocks2d_improvisational_tamp_approach():
    """Tests for MPCBlocks2DImprovisationalTAMPApproach()."""
    # Set up environment
    base_env = Blocks2DEnv(render_mode="rgb_array")
    env = cast(
        gym.Env[NDArray[np.float32], NDArray[np.float32]],
        TimeLimit(base_env, max_episode_steps=100),
    )

    # # Uncomment to watch a video.
    # from gymnasium.wrappers import RecordVideo
    # from pathlib import Path

    # video_folder = "videos/blocks2d-mpc-improvisational-tamp-test"
    # Path(video_folder).mkdir(parents=True, exist_ok=True)
    # env = RecordVideo(env, video_folder)

    # Create config for predictive sampling
    config = PredictiveSamplingConfig(
        num_rollouts=100, horizon=35, num_control_points=5, noise_scale=1.0
    )

    # Create approach with MPC policy
    approach = MPCBlocks2DImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        config=config,
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
