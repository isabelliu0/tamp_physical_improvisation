"""Test base random approach."""

import pytest

from tamp_improv.approaches.random import RandomApproach
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


def run_episode(system, approach, max_steps: int):
    """Run single episode with approach."""
    obs, info = system.reset()
    step_result = approach.reset(obs, info)

    # Process first step
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    if terminated or truncated:
        return 1

    # Process remaining steps
    for step in range(1, max_steps):
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        if terminated or truncated:
            return step + 1
    return max_steps


@pytest.mark.parametrize(
    "system_cls,max_steps",
    [(Blocks2DTAMPSystem, 100), (NumberTAMPSystem, 10), (ClearAndPlaceTAMPSystem, 200)],
)
def test_random_approach(system_cls, max_steps):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)
    approach = RandomApproach(system, seed=42)

    steps = run_episode(system, approach, max_steps)
    assert steps <= max_steps
