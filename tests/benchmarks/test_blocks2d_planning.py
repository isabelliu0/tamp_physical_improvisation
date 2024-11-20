"""Tests for Blocks2D environment with TAMP."""

from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.blocks2d import Blocks2DEnvironment


def test_blocks2d_env_with_planner():
    """Test Blocks2D environment with TAMP planner."""
    env = Blocks2DEnvironment(include_pushing_models=True)
    base_env = env.env
    base_env = TimeLimit(base_env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # base_env = RecordVideo(base_env, "videos/blocks2d-planning-test")

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=env.types,
        predicates=env.predicates,
        perceiver=env.perceiver,
        operators=env.operators,
        skills=env.skills,
        planner_id="pyperplan",
    )

    obs, info = base_env.reset()
    objects, atoms, goal = env.perceiver.reset(obs, info)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs, info)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print(
            "Current problem:",
            planner._current_problem,  # pylint: disable=protected-access
        )
        print("Current domain:", planner._domain)  # pylint: disable=protected-access
        raise

    total_reward = 0
    for step in range(100):
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = base_env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Obs: {obs}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    base_env.close()
