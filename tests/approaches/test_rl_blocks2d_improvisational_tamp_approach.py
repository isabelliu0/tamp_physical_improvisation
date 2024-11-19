"""Test rl_blocks2d_improvisational_tamp_approach.py with online training."""

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from tamp_improv.approaches.rl_blocks2d_improvisational_tamp_approach import (
    RLBlocks2DImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_rl_blocks2d_improvisational_tamp_approach(
    save_path: str = "trained_policies/pushing_policy_online",
    train_timesteps: int = 100_000,
    num_test_episodes: int = 5,
):
    """Tests RLBlocks2DImprovisationalTAMPApproach with online training.

    Args:
        save_path: Path to save/load policy weights
        train_timesteps: Number of timesteps for training
        num_test_episodes: Number of test episodes to run
    """
    # Set up environment
    base_env = Blocks2DEnv(render_mode="rgb_array")
    env = cast(
        gym.Env[NDArray[np.float32], NDArray[np.float32]],
        TimeLimit(base_env, max_episode_steps=100),
    )

    # # Create video recording if desired
    # from pathlib import Path
    # video_folder = "videos/blocks2d-rl-improvisational-tamp-test-online"
    # Path(video_folder).mkdir(parents=True, exist_ok=True)

    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, video_folder, episode_trigger=lambda x: x % 10 == 0)

    # Create approach - don't load existing policy since we're training online
    approach = RLBlocks2DImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        policy_path=save_path,
        train_online=True,  # Enable online training
        train_timesteps=train_timesteps,
    )

    print("\n=== Running Test Episodes ===")

    success_count = 0
    total_steps = []
    total_rewards = []

    for episode in range(num_test_episodes):
        print(f"\nEpisode {episode + 1}/{num_test_episodes}")

        # Reset environment and approach
        obs, info = env.reset(seed=episode)
        approach.reset(obs, info)

        # Run episode
        total_reward: float = 0.0
        step_count = 0

        for step in range(100):  # should terminate earlier
            action = approach.step(obs, 0, False, False, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            print(f"Step {step + 1}: Action: {action}, Reward: {reward}")

            if terminated or truncated:
                print(f"Episode finished after {step + 1} steps")
                print(f"Total reward: {total_reward}")
                if float(reward) > 0.0:  # Assuming positive reward indicates success
                    success_count += 1
                break
        else:
            print("Episode didn't finish within 100 steps")

        total_steps.append(step_count)
        total_rewards.append(total_reward)

    # Print summary statistics
    print("\n=== Test Results ===")
    print(f"Success rate: {success_count/num_test_episodes:.2%}")
    print(f"Average episode steps: {np.mean(total_steps):.1f}")
    print(f"Average episode reward: {np.mean(total_rewards):.2f}")

    # Save final policy
    if hasattr(approach, "save_policy"):
        approach.save_policy(save_path)
        print(f"\nSaved trained policy to {save_path}")

    env.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    test_rl_blocks2d_improvisational_tamp_approach(
        train_timesteps=1_000_000, num_test_episodes=5
    )
