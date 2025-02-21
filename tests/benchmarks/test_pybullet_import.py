"""Test pybullet base environment directly from import."""

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


@pytest.mark.skip(reason="TAMP plan would fail without improvisational policies.")
def test_pybullet_import():
    """Test base functionality matching original test."""
    scene_description = ClearAndPlaceSceneDescription(
        num_obstacle_blocks=3,
        stack_blocks=True,
    )

    # Create both env and sim
    env = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )
    sim = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )

    max_motion_planning_time = 0.1

    # Create components directly
    perceiver = ClearAndPlacePyBulletBlocksPerceiver(sim)
    operators, skill_types = get_active_operators_and_skills(False)
    skills = {
        s(sim, max_motion_planning_time=max_motion_planning_time) for s in skill_types
    }

    # Create planner with components
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    )

    # Run episode
    obs, info = env.reset(seed=124)
    planner.reset(obs, info)

    for step in range(10000):
        action = planner.step(obs)
        print(f"Step {step}: {action}")
        obs, _, done, _, _ = env.step(action)
        if done:
            print(f"Goal reached in {step + 1} steps!")
            break
    else:
        assert False, "Goal not reached"

    env.close()
