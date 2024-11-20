"""Tests for Number environment with TAMP."""

from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.number import NumberEnvironment


def test_number_env_with_planner():
    """Test Number environment with TAMP planner."""
    env = NumberEnvironment(switch_off_improvisational_models=False)
    base_env = env.env
    base_env = TimeLimit(base_env, max_episode_steps=10)

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=env.types,
        predicates=env.predicates,
        perceiver=env.components.perceiver,
        operators=env.operators,
        skills=env.skills,
        planner_id="pyperplan",
    )

    obs, info = base_env.reset()
    print("Initial observation:", obs)

    objects, atoms, goal = env.components.perceiver.reset(obs, info)
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
        obs, reward, terminated, truncated, _ = base_env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, State: {obs}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 10 steps")

    base_env.close()
