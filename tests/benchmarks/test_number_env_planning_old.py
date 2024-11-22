"""Tests for NumberEnv with TaskThenMotionPlanner."""

from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.number_env_old import NumberEnv
from tamp_improv.number_planning import create_number_planning_models


def test_number_env_with_planner():
    """Tests for number env planning."""
    env = NumberEnv()
    env = TimeLimit(env, max_episode_steps=10)

    types, predicates, perceiver, operators, skills = create_number_planning_models(
        switch_off_improvisational_models=False,
    )

    planner = TaskThenMotionPlanner(
        types, predicates, perceiver, operators, skills, planner_id="pyperplan"
    )

    obs, info = env.reset()
    print("Initial observation:", obs)

    objects, atoms, goal = perceiver.reset(obs, info)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs, info)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print("Current problem:")
        print(planner._current_problem)  # pylint: disable=protected-access
        print("Current domain:")
        print(planner._domain)  # pylint: disable=protected-access
        raise

    total_reward = 0
    for step in range(10):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, State: {obs}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 10 steps")

    env.close()
