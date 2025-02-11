"""Test Clear and Place replanning behavior."""

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


@pytest.mark.skip(reason="Debugging...")
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
        scene_description=scene_description, use_gui=True
    )
    sim = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )

    # Create components for planning
    perceiver = ClearAndPlacePyBulletBlocksPerceiver(sim)
    operators, skill_types = get_active_operators_and_skills(False)
    skills = {s(sim, max_motion_planning_time=0.1) for s in skill_types}

    # Create planner
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    )

    # Initialize with the actual observation (output from a previous RL run)
    obs = np.array(
        [
            1.00000000e00,
            4.87802652e-01,
            -9.51681790e-02,
            9.99346085e-02,
            1.81027854e-04,
            -8.93723069e-04,
            -5.15129257e-02,
            9.98671912e-01,
            0.00000000e00,
            0.00000000e00,
            1.00000000e00,
            2.57998598e-01,
            1.69431620e-01,
            9.91156035e-02,
            -8.73056694e-03,
            1.36011607e-02,
            -6.07724173e-01,
            7.93983700e-01,
            1.00000000e00,
            0.00000000e00,
            1.00000000e00,
            1.78284843e-01,
            -1.86425833e-01,
            1.96720811e-01,
            -1.07112299e-02,
            3.70161906e-02,
            2.35650675e-01,
            9.71073545e-01,
            2.00000000e00,
            0.00000000e00,
            1.00000000e00,
            5.12331188e-01,
            1.36681378e-01,
            -3.82549083e-03,
            2.48252585e-02,
            1.75887559e-01,
            2.08088758e-01,
            9.61845280e-01,
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
            5.49191059e-01,
            7.31403304e-01,
            -3.03522524e-01,
            -1.94500147e00,
            3.58768494e-01,
            2.26084297e00,
            1.92987544e00,
            4.00000000e-02,
            4.00000000e-02,
            1.00000000e00,
            -1.78806484e-04,
            2.14934349e-04,
            3.77610326e-04,
            7.06966817e-01,
            7.07246721e-01,
            -5.84676454e-05,
            -1.76021258e-05,
        ],
        dtype=np.float32,
    )

    # Set environments to this state
    print("\nResetting environments to replanning state...")
    obs_env, _ = env.reset_from_state(obs)
    obs_sim, info_sim = sim.reset_from_state(obs)

    print("\nInitial state:")
    print(f"Env grasp transform: {env.current_grasp_transform}")
    print(f"Env held object: {env.current_held_object_id}")
    print(f"Sim grasp transform: {sim.current_grasp_transform}")
    print(f"Sim held object: {sim.current_held_object_id}")

    # Reset planner and get first action
    print("\nResetting planner...")
    planner.reset(obs_sim, info_sim)

    print("\nTaking first planned action...")
    try:
        action = planner.step(obs_sim)
        print(f"Planned action: {action}")

        # Verify action validity
        valid = env.action_space.contains(action)
        print(f"Action valid: {valid}")
        print(f"Action space: {env.action_space}")

        # Take step
        obs_env, reward, done, _, _ = env.step(action)
        print("Step successful")

        import time

        time.sleep(1.0)  # Pause to observe result

    except AssertionError as e:
        print(f"Step failed with assertion: {e}")
        raise
    except Exception as e:
        print(f"Step failed with error: {e}")
        raise

    env.close()
    sim.close()
