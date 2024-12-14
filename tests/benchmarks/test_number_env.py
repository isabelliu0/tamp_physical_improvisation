"""Tests for core number environment."""

from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.number_env import NumberEnv


def test_number_env():
    """Test basic number environment functionality."""
    env = NumberEnv()
    env = TimeLimit(env, max_episode_steps=10)
    obs, info = env.reset()

    # Hard-coded sequence of actions to reach the goal
    actions = [
        [1, 0],  # Move from state 0 to 1
        [1, 0],  # Move from state 1 to 2
    ]

    print("Initial state:", obs)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, State: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()
