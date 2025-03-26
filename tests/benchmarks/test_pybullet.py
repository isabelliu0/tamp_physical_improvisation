"""Test pybullet clear and place environment."""

from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


def test_pybullet():
    """Test base TAMP functionality of pybullet environment."""
    # Create system
    system = ClearAndPlaceTAMPSystem.create_default(seed=124, render_mode=None)

    # Create planner
    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    # Run episode
    obs, info = system.env.reset(seed=124)
    planner.reset(obs, info)

    for step in range(10000):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)
        if done:
            print(f"Goal reached in {step + 1} steps with pushing models!")
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    system.env.close()
