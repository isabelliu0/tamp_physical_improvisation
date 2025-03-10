"""Test improvisational TAMP approaches."""

from pathlib import Path

import pytest

from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


@pytest.fixture(scope="function", name="base_config")
def _get_base_config():
    """Test configuration."""
    return TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=True,
    )


@pytest.mark.skip("Training for new framework is not yet implemented")
@pytest.mark.parametrize(
    "system_cls,mpc_config",
    [
        (
            ClearAndPlaceTAMPSystem,
            MPCConfig(
                num_rollouts=50, horizon=10, num_control_points=10, noise_scale=0.10
            ),  # small num_rollouts for faster unit testing (temporary)
        ),
        (
            Blocks2DTAMPSystem,
            MPCConfig(
                num_rollouts=100, horizon=20, num_control_points=10, noise_scale=0.25
            ),
        ),
        (
            NumberTAMPSystem,
            MPCConfig(
                num_rollouts=20,
                horizon=5,
                num_control_points=2,
                noise_scale=0.5,
            ),
        ),
    ],
)
def test_mpc_approach(system_cls, mpc_config, base_config):
    """Test MPC improvisational approach."""
    print(f"\n=== Testing MPC on {system_cls.__name__} ===")
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if base_config.render else None
    )

    def policy_factory(seed: int) -> MPCPolicy:
        return MPCPolicy(seed=seed, config=mpc_config)

    metrics = train_and_evaluate(
        system, policy_factory, base_config, policy_name="MPCPolicy"
    )

    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Avg episode length: {metrics.avg_episode_length:.2f}")


@pytest.mark.skip("Training for new framework is not yet implemented")
@pytest.mark.parametrize(
    "system_cls",
    [
        ClearAndPlaceTAMPSystem,
        Blocks2DTAMPSystem,
        NumberTAMPSystem,
    ],
)
def test_rl_approach(system_cls, base_config):
    """Test RL improvisational approach."""
    policy_dir = Path("trained_policies")
    policy_dir.mkdir(exist_ok=True)

    # Create RL-specific policy config
    rl_policy_config = RLConfig(
        learning_rate=1e-4, batch_size=32, n_epochs=5, gamma=0.99
    )

    # Create training config
    rl_config = TrainingConfig(
        # Existing settings
        seed=base_config.seed,
        num_episodes=base_config.num_episodes,
        max_steps=base_config.max_steps,
        render=base_config.render,
        # RL-specific settings
        collect_episodes=100,
        episodes_per_scenario=1,
        force_collect=False,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data",
        save_dir="trained_policies",
    )

    print(f"\n=== Testing RL Initial Training on {system_cls.__name__} ===")
    # Test training from scratch
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if rl_config.render else None
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_policy_config)

    metrics = train_and_evaluate(
        system, policy_factory, rl_config, policy_name="RLPolicy"
    )

    print("\nRL Initial Training Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    # Test loading and execution
    policy_file = policy_dir / f"{system_cls.__name__.lower()}_RLPolicy.zip"
    if not policy_file.exists():
        pytest.skip(f"Policy file not found at {policy_file}")

    print(f"\n=== Testing RL Loaded Policy on {system_cls.__name__} ===")
    # Create new system for loaded policy
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if rl_config.render else None
    )

    def loaded_policy_factory(seed: int) -> RLPolicy:
        # Create and initialize policy with the system
        policy: RLPolicy = RLPolicy(seed=seed)
        policy.load(str(policy_file))
        return policy

    loaded_metrics = train_and_evaluate(
        system,
        loaded_policy_factory,
        rl_config,
        policy_name="RLPolicy_Loaded",
    )

    print("\nRL Loaded Policy Results:")
    print(f"Success Rate: {loaded_metrics.success_rate:.2%}")
    print(f"Average Episode Length: {loaded_metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {loaded_metrics.avg_reward:.2f}")
