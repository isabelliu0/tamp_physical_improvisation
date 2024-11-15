"""Tests for SimpleTransitionEnv()."""

from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.number_env import NumberEnv


def test_simple_transition_env():
    """Tests for SimpleTransitionEnv."""
    env = NumberEnv()
    env = TimeLimit(env, max_episode_steps=100)
    obs, info = env.reset()

    env.action_space.seed(123)

    # Hard-coded sequence of actions to reach the goal
    actions = [
        1,  # Move from state 0 to 1
        1,  # Move from state 1 to 2
        1,  # Move from state 2 to 3
    ]

    print("Initial state:", obs)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, State: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()
