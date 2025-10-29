"""Test planning graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.policies.pushing_pybullet import (
    PybulletPushingPolicy,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower import ObstacleTowerTAMPSystem


@pytest.mark.parametrize(
    "system_cls,policy_cls,env_name",
    [
        (Obstacle2DTAMPSystem, PushingPolicy, "obstacle2d"),
        (ObstacleTowerTAMPSystem, PybulletPushingPolicy, "pybullet"),
        (ClutteredDrawerTAMPSystem, PybulletPushingPolicy, "cluttered_drawer"),
        (CleanupTableTAMPSystem, PybulletPushingPolicy, "cleanup_table"),
    ],
)
def test_planning_graph_visualization(system_cls, policy_cls, env_name):
    """Test building the planning graphs."""
    system = system_cls.create_default(seed=42)
    approach = ImprovisationalTAMPApproach(
        system=system,
        policy=policy_cls(seed=42),
        seed=42,
    )

    print("Resetting system and approach...")
    obs, info = system.reset()
    objects, init_atoms, goal_atoms = system.perceiver.reset(obs, info)

    output_dir = Path("results/planning_graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    approach._goal = goal_atoms  # pylint: disable=protected-access
    graph = approach._create_planning_graph(  # pylint: disable=protected-access
        objects, init_atoms
    )

    return graph
