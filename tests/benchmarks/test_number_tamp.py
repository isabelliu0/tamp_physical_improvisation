"""Tests for Number environment with TAMP."""

from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.number import BaseNumberTAMPSystem
from tamp_improv.benchmarks.number_env import NumberEnv


def test_number_tamp_system():
    """Test Number environment with TAMP planner."""
    # Create TAMP system
    tamp_system = BaseNumberTAMPSystem.create_default()

    # Create environment with time limit
    env = NumberEnv()
    env = TimeLimit(env, max_episode_steps=10)

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.components.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        planner_id="pyperplan",
    )

    obs, info = env.reset()
    print("Initial observation:", obs)

    objects, atoms, goal = tamp_system.components.perceiver.reset(obs, info)
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
    for step in range(10):
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
