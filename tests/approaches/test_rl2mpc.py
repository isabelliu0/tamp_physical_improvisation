"""Test RL2MPC improvisational approach."""

from pathlib import Path

import pytest

from tamp_improv.approaches.improvisational.policies.base import ActType, ObsType
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.policies.rl2mpc import (
    RL2MPCConfig,
    RL2MPCPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem


@pytest.mark.parametrize(
    "system_cls,config",
    [
        (
            Blocks2DTAMPSystem,
            RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=1e-4,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=100,
                    horizon=35,
                    num_control_points=5,
                    noise_scale=1.0,
                ),
                reward_threshold=-5.0,
                window_size=50,
            ),
        ),
        (
            NumberTAMPSystem,
            RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=1e-4,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=20,
                    horizon=10,
                    num_control_points=3,
                    noise_scale=0.5,
                ),
                reward_threshold=-0.5,
                window_size=5,
            ),
        ),
    ],
)
def test_rl2mpc_approach(system_cls, config):
    """Test RL2MPC improvisational approach."""
    policy_dir = Path("trained_policies")
    policy_dir.mkdir(exist_ok=True)

    # Create training config
    train_config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=False,
        collect_episodes=50,
        episodes_per_scenario=5,
        force_collect=True,
        record_training=False,
        training_record_interval=50,
        training_data_dir="training_data",
        save_dir="trained_policies",
    )

    print("\n=== Testing RL2MPC Initial Training ===")
    # Test training from scratch
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if train_config.render else None
    )

    def policy_factory(seed: int) -> RL2MPCPolicy:
        return RL2MPCPolicy(seed=seed, config=config)

    metrics = train_and_evaluate(
        system, policy_factory, train_config, policy_name="RL2MPC"
    )

    print("\nRL2MPC Initial Training Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    # Test loading and execution of trained policy
    policy_file = policy_dir / f"{system_cls.__name__.lower()}_RL2MPC_rl.zip"
    if not policy_file.exists():
        pytest.skip(f"Policy file not found at {policy_file}")

    print("\n=== Testing RL2MPC Loaded Policy ===")
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if train_config.render else None
    )

    def loaded_policy_factory(seed: int) -> RL2MPCPolicy[ObsType, ActType]:
        policy: RL2MPCPolicy[ObsType, ActType] = RL2MPCPolicy(seed=seed, config=config)
        policy.load(str(policy_file))
        return policy

    loaded_metrics = train_and_evaluate(
        system,
        loaded_policy_factory,
        train_config,
        policy_name="RL2MPC_Loaded",
    )

    print("\nRL2MPC Loaded Policy Results:")
    print(f"Success Rate: {loaded_metrics.success_rate:.2%}")
    print(f"Average Episode Length: {loaded_metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {loaded_metrics.avg_reward:.2f}")
