"""Test the framework using a hard-coded pushing policy."""

from typing import Union

import pytest

from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.policies.pushing_pybullet import (
    PybulletPushingPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


@pytest.fixture(scope="function", name="training_config")
def _get_training_config():
    """Test configuration."""
    return TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=100,
        render=True,
    )


# @pytest.mark.skip("Debugging...")
@pytest.mark.parametrize(
    "system_cls,policy_cls,env_name",
    [
        # (Blocks2DTAMPSystem, PushingPolicy, "blocks2d"),
        (ClearAndPlaceTAMPSystem, PybulletPushingPolicy, "pybullet"),
    ],
)
def test_framework_with_hardcoded_policy(
    system_cls, policy_cls, env_name, training_config
):
    """Test the framework using a hard-coded pushing policy."""
    print(f"\n=== Testing {env_name} Framework with Hard-coded Pushing Policy ===")

    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if training_config.render else None
    )

    def policy_factory(seed: int) -> Union[PushingPolicy, PybulletPushingPolicy]:
        return policy_cls(seed=seed)

    metrics = train_and_evaluate(
        system, policy_factory, training_config, policy_name="PushingPolicy"
    )

    print("\nResults:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
