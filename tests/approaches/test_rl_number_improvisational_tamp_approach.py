"""Test rl_number_improvisational_tamp_approach.py."""

from pathlib import Path
from typing import cast

import gymnasium as gym
import pytest
from gymnasium.wrappers import TimeLimit

from tamp_improv.approaches.rl_number_improvisational_tamp_approach import (
    RLNumberImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.number_env import NumberEnv


def test_rl_number_improvisational_tamp_approach(
    policy_path: str = "trained_policies/number_policy",
):
    """Tests for RLNumberImprovisationalTAMPApproach().

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
    base_env = NumberEnv()
    env = cast(
        gym.Env[int, int],
        TimeLimit(base_env, max_episode_steps=10),
    )

    # Create approach with trained policy
    approach = RLNumberImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        policy_path=policy_path,
    )

    # Run episode
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward: float = 0.0
    for step in range(10):  # should terminate earlier
        action = approach.step(obs, 0, False, False, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        print(f"Step {step + 1}: State: {obs}, Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 10 steps")

    env.close()  # type: ignore[no-untyped-call]
