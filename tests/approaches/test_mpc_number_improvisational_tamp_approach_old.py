"""Test mpc_number_improvisational_tamp_approach.py."""

import gymnasium as gym

from tamp_improv.approaches.mpc_improvisational_policy import PredictiveSamplingConfig
from tamp_improv.approaches.mpc_number_improvisational_tamp_approach import (
    MPCNumberImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.number_env_old import NumberEnv


def test_mpc_number_improvisational_tamp_approach():
    """Tests for MPCNumberImprovisationalTAMPApproach()."""
    # Set up environment
    env = gym.wrappers.TimeLimit(NumberEnv(), max_episode_steps=10)

    # Create config for predictive sampling
    config = PredictiveSamplingConfig(
        num_rollouts=20,
        horizon=10,
        num_control_points=3,
        noise_scale=0.5,  # Adjusted for discrete actions
    )

    # Create approach with MPC policy
    approach = MPCNumberImprovisationalTAMPApproach(
        env.observation_space,
        env.action_space,
        seed=123,
        config=config,
    )

    # Run episode
    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward = 0.0
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

    env.close()
