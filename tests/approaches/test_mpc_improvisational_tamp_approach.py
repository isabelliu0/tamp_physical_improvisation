"""Test running MPC blocks2d improvisational TAMP approach online."""

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from tamp_improv.approaches.mpc_improvisational_tamp_approach import (
    MPCBlocks2DConfig,
    MPCBlocks2DImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_mpc_blocks2d_improvisational_tamp_approach():
    """Tests for MPCBlocks2DImprovisationalTAMPApproach()."""
    base_env = Blocks2DEnv(render_mode="rgb_array")
    env = cast(
        gym.Env[NDArray[np.float32], NDArray[np.float32]],
        TimeLimit(base_env, max_episode_steps=100),
    )

    # # Uncomment to watch a video
    # from pathlib import Path

    # from gymnasium.wrappers import RecordVideo

    # video_folder = "videos/blocks2d-mpc-improvisational-tamp-test"
    # Path(video_folder).mkdir(parents=True, exist_ok=True)
    # env = RecordVideo(env, video_folder)

    # Create MPC configuration
    config = MPCBlocks2DConfig(
        horizon=20,
        num_rollouts=100,
        noise_scale=1.0,
        num_control_points=10,
        dt=0.5,
        warm_start=True,
    )

    # Create approach
    approach = MPCBlocks2DImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        config=config,
    )

    # Run episode
    obs, info = env.reset(seed=456)  # Different seed than approach
    approach.reset(obs, info)

    total_reward = 0.0
    episode_data = []

    for step in range(100):
        print(f"\n{'='*50}\nStep {step + 1}")
        print(f"Current robot position: {obs[0:2]}")
        print(f"Current block 2 position: {obs[6:8]}")

        # Get distance to block 2 before action
        pre_distance = info.get("distance_to_block2", None)
        print(f"Distance to block 2: {pre_distance}")

        # Get and apply action
        action = approach.step(obs, 0, False, False, info)
        print(f"\nExecuting action: {action}")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Record step data
        post_distance = info.get("distance_to_block2", None)

        step_data = {
            "step": step,
            "action": action,
            "reward": reward,
            "pre_distance": pre_distance,
            "post_distance": post_distance,
            "terminated": terminated,
            "truncated": truncated,
        }
        episode_data.append(step_data)

        print("\nStep Results:")
        print(f"Reward: {reward}")
        print(f"New distance: {post_distance}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")

            # Print episode summary
            print("\nEpisode Summary:")
            print(f"Total steps: {step + 1}")
            print(f"Final reward: {total_reward}")
            print(f"Initial distance: {episode_data[0]['pre_distance']}")
            print(f"Final distance: {episode_data[-1]['post_distance']}")
            break
    else:
        print("\nEpisode didn't finish within 100 steps")

    env.close()
