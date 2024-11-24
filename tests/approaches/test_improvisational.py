"""Test improvisational TAMP approaches."""

from pathlib import Path

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    collect_training_data,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem


@pytest.fixture
def test_config():
    """Test configuration."""
    return TrainingConfig(
        seed=42, num_episodes=2, max_steps=50, collect_episodes=2, render=False
    )


# @pytest.mark.parametrize("system_cls", [Blocks2DTAMPSystem, NumberTAMPSystem])
# def test_collect_training_data(system_cls, test_config):
#     """Test training data collection."""
#     system = system_cls.create_default(seed=42)
#     policy = MPCPolicy(seed=42)  # Use MPC just for collection
#     approach = ImprovisationalTAMPApproach(system, policy, seed=42)

#     train_data = collect_training_data(system, approach, test_config)

#     assert len(train_data.states) > 0
#     assert len(train_data.operators) == len(train_data.states)
#     assert len(train_data.preconditions) == len(train_data.states)


@pytest.mark.parametrize(
    "system_cls,config",
    [
        (
            Blocks2DTAMPSystem,
            MPCConfig(
                num_rollouts=100, horizon=35, num_control_points=5, noise_scale=1.0
            ),
        ),
        (
            NumberTAMPSystem,
            MPCConfig(
                num_rollouts=20,
                horizon=10,
                num_control_points=3,
                noise_scale=0.5,
            ),
        ),
    ],
)
def test_mpc_approach(system_cls, config, test_config):
    """Test MPC improvisational approach."""
    system = system_cls.create_default(seed=42)
    policy = MPCPolicy(seed=42, config=config)
    approach = ImprovisationalTAMPApproach(system, policy, seed=42)

    metrics = train_and_evaluate(system, type(policy), test_config)

    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Avg episode length: {metrics.avg_episode_length:.2f}")


# @pytest.mark.parametrize("system_cls,policy_path", [
#     (Blocks2DTAMPSystem, "test_policies/blocks2d_policy"),
#     (NumberTAMPSystem, "test_policies/number_policy"),
# ])
# def test_rl_approach(system_cls, policy_path, tmp_path, test_config):
#     """Test RL improvisational approach."""
#     system = system_cls.create_default(seed=42)

#     # Test training
#     policy = RLPolicy(seed=42)
#     approach = ImprovisationalTAMPApproach(system, policy, seed=42)

#     test_config.save_dir = str(tmp_path)
#     metrics = train_and_evaluate(system, type(policy), test_config)

#     print(f"Success rate: {metrics.success_rate:.2%}")
#     print(f"Avg episode length: {metrics.avg_episode_length:.2f}")

#     # Test loading and execution
#     policy_file = tmp_path / f"{system_cls.__name__.lower()}_policy.zip"
#     assert policy_file.exists()

#     loaded_policy = RLPolicy(seed=42)
#     loaded_policy.load(str(policy_file))
#     loaded_approach = ImprovisationalTAMPApproach(system, loaded_policy, seed=42)

#     metrics = train_and_evaluate(system, type(loaded_policy), test_config)
#     print(f"Success rate (loaded): {metrics.success_rate:.2%}")
