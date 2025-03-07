"""Test the framework using a hard-coded pushing policy."""

import pytest

from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem


@pytest.fixture
def test_config():
    """Test configuration."""
    return TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=True,
    )


# pylint: disable=redefined-outer-name
def test_framework_with_hardcoded_policy(test_config):
    """Test the framework using a hard-coded pushing policy."""
    print("\n=== Testing Framework with Hard-coded Pushing Policy ===")

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=42, render_mode="rgb_array" if test_config.render else None
    )

    # Create policy
    def policy_factory(seed: int) -> PushingPolicy:
        return PushingPolicy(seed=seed)

    # Run test
    metrics = train_and_evaluate(
        system, policy_factory, test_config, policy_name="PushingPolicy"
    )

    print("\nResults:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
