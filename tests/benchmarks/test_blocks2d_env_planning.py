"""Tests for Blocks2DEnv with TaskThenMotionPlanner."""

from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.blocks2d_planning import (
    Blocks2DPerceiver,
    operators,
    predicates,
    skills,
    types,
)


def test_blocks_2d_env_with_planner():
    """Tests for block2d planning."""
    env = Blocks2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/blocks2d-planning-test-2")

    perceiver = Blocks2DPerceiver(env.unwrapped)

    planner = TaskThenMotionPlanner(
        types, predicates, perceiver, operators, skills, planner_id="pyperplan"
    )

    obs, _ = env.reset()
    print("Initial observation:", obs)

    objects, atoms, goal = perceiver.reset(obs)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print("Current problem:")
        print(planner._current_problem)  # pylint: disable=protected-access
        print("Current domain:")
        print(planner._domain)  # pylint: disable=protected-access
        raise

    total_reward = 0
    for step in range(100):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Obs: {obs}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    env.close()
