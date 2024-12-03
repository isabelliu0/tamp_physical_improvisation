"""Test base random approach."""

import pytest

from tamp_improv.approaches.random import RandomApproach
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem


def run_episode(system, approach, max_steps: int):
    """Run single episode with approach."""
    obs, info = system.reset()
    action = approach.reset(obs, info)

    for step in range(max_steps):
        obs, reward, terminated, truncated, info = system.env.step(action)
        action = approach.step(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    return step + 1


@pytest.mark.parametrize(
    "system_cls,max_steps",
    [
        (Blocks2DTAMPSystem, 100),
        (NumberTAMPSystem, 10),
    ],
)
def test_random_approach(system_cls, max_steps):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)
    approach = RandomApproach(system, seed=42)

    steps = run_episode(system, approach, max_steps)
    assert steps <= max_steps
