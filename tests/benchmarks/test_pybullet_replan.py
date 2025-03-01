"""Test Clear and Place replanning behavior."""

import time

import numpy as np
import pytest
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlaceSceneDescription,
)
from pybullet_blocks.planning_models.action import get_active_operators_and_skills
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    ClearAndPlacePyBulletBlocksPerceiver,
)
from task_then_motion_planning.planning import TaskThenMotionPlanner


@pytest.mark.skip(reason="for debugging use only")
def test_replan():
    """Test replanning behavior with specific observation."""
    # Create environment
    scene_description = ClearAndPlaceSceneDescription(
        num_obstacle_blocks=3,
        stack_blocks=True,
        robot_max_joint_delta=0.1,
    )

    # Create both env and sim like in original test
    env = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description,
        use_gui=True,
    )
    sim = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )

    # Create components for planning
    perceiver = ClearAndPlacePyBulletBlocksPerceiver(sim)
    operators, skill_types = get_active_operators_and_skills(False)
    skills = {s(sim, max_motion_planning_time=1.0) for s in skill_types}

    # Create planner
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    )

    # The observation at replanning time
    obs = np.array(
        [
            1.00000000e00,
            5.83175652e-01,
            3.08012688e-02,
            1.05573158e-01,
            -1.71591001e-01,
            5.85455622e-01,
            2.22821466e-01,
            7.60360992e-01,
            0.00000000e00,
            0.00000000e00,
            1.00000000e00,
            6.24083958e-01,
            5.61019213e-02,
            1.19680503e-01,
            -1.54332868e-01,
            5.89235766e-01,
            2.45583750e-01,
            7.54102911e-01,
            1.00000000e00,
            0.00000000e00,
            1.00000000e00,
            6.02183780e-01,
            1.63817541e-01,
            9.95943905e-02,
            5.58437883e-01,
            8.29527911e-01,
            -4.81639167e-03,
            -2.71631185e-03,
            2.00000000e00,
            0.00000000e00,
            1.00000000e00,
            3.79715741e-01,
            2.50915647e-01,
            1.31794140e-01,
            3.49683751e-02,
            -1.27801325e-01,
            4.98037621e-01,
            8.56972906e-01,
            1.90000000e01,
            1.00000000e00,
            1.00000000e00,
            5.00000000e-01,
            0.00000000e00,
            7.60000000e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.00000000e00,
            0.00000000e00,
            7.49191062e-01,
            8.12193130e-02,
            -1.03522521e-01,
            -2.69474700e00,
            4.25908762e-01,
            3.02862195e00,
            1.53268713e00,
            4.00000000e-02,
            4.00000000e-02,
            1.00000000e00,
            -1.78739429e-04,
            2.14964151e-04,
            3.77617776e-04,
            7.06966817e-01,
            7.07246721e-01,
            -5.84696318e-05,
            -1.75990244e-05,
        ],
        dtype=np.float32,
    )

    # Set environments to this state
    print("\nResetting environments to replanning state...")
    _ = env.reset_from_state(obs)
    obs_sim, info_sim = sim.reset_from_state(obs)

    print("\nInitial state:")
    print(f"Env grasp transform: {env.current_grasp_transform}")
    print(f"Env held object: {env.current_held_object_id}")
    print(f"Sim grasp transform: {sim.current_grasp_transform}")
    print(f"Sim held object: {sim.current_held_object_id}")

    # Reset planner and get first action
    print("\nResetting planner...")
    planner.reset(obs_sim, info_sim)

    # Keep stepping until termination or failure
    max_steps = 100
    step = 0
    terminated = False

    while not terminated and step < max_steps:
        print(f"\nTaking step {step}...")
        try:
            action = planner.step(obs_sim)
            print(f"Planned action: {action}")

            # Verify action validity
            valid = env.action_space.contains(action)
            print(f"Action valid: {valid}")

            # Take step
            obs_sim, reward, terminated, _, _ = env.step(action)
            print(f"Step result - Reward: {reward}, Terminated: {terminated}")

            time.sleep(0.5)  # Longer pause to observe each action

            step += 1

        except AssertionError as e:
            print(f"Step failed with assertion: {e}")
            raise
        except Exception as e:
            print(f"Step failed with error: {e}")
            raise

    if terminated:
        print("\nSuccessfully placed target block in target area!")
    else:
        print("\nFailed to complete task within max steps")

    env.close()
    sim.close()
