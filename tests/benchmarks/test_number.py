"""Tests for Number environment."""

from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.number import NumberEnvironment


def test_number_env():
    """Test basic functionality of Number environment."""
    env = NumberEnvironment(switch_off_improvisational_models=False)
    base_env = env.env
    base_env = TimeLimit(base_env, max_episode_steps=10)
    obs, info = base_env.reset()

    base_env.action_space.seed(123)

    # Hard-coded sequence of actions to reach the goal
    actions = [
        1,  # Move from state 0 to 1
        1,  # Move from state 1 to 2
    ]

    print("Initial state:", obs)
    for action in actions:
        obs, reward, terminated, truncated, info = base_env.step(action)
        print(f"Action: {action}, State: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    base_env.close()
