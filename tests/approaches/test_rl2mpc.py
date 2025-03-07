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
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


@pytest.mark.skip("Training for new framework is not yet implemented")
@pytest.mark.parametrize(
    "system_cls,config",
    [
        (
            ClearAndPlaceTAMPSystem,
            RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=1e-4,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=100,
                    horizon=50,
                    num_control_points=10,
                    noise_scale=0.10,
                ),
                reward_threshold=-90.0,
                window_size=20,
            ),
        ),
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
                    horizon=20,
                    num_control_points=10,
                    noise_scale=0.05,
                ),
                reward_threshold=-30.0,
                window_size=10,
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
        render=True,
        collect_episodes=100,
        episodes_per_scenario=5,
        force_collect=False,
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


@pytest.mark.skip("Training for new framework is not yet implemented")
@pytest.mark.parametrize(
    "system_cls,config",
    [
        (
            ClearAndPlaceTAMPSystem,
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
                    num_control_points=10,
                    noise_scale=0.10,
                ),  # small num_rollouts/horizon for faster unit testing
                reward_threshold=-30.0,
                window_size=1,  # Small window to quickly switch to MPC
            ),
        ),
    ],
)
def test_pretrained_rl_with_mpc(system_cls, config):
    """Test loading pre-trained RL policy and using MPC immediately."""
    # Create minimal config - no training/collection needed
    eval_config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=100,
        render=True,
        collect_episodes=0,
        force_collect=False,
        episodes_per_scenario=0,
    )

    print(f"\n=== Testing Pre-trained RL + MPC on {system_cls.__name__} ===")

    # Create system
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if eval_config.render else None
    )

    class PreTrainedRL2MPCPolicy(RL2MPCPolicy[ObsType, ActType]):
        """Modified RL2MPC policy that skips training."""

        @property
        def requires_training(self) -> bool:
            return False

    def policy_factory(seed: int) -> PreTrainedRL2MPCPolicy:
        policy: PreTrainedRL2MPCPolicy = PreTrainedRL2MPCPolicy(
            seed=seed, config=config
        )

        rl_policy_file = (
            Path("trained_policies") / f"{system_cls.__name__}_RLPolicy.zip"
        )
        if not rl_policy_file.exists():
            pytest.skip(f"Pre-trained RL policy not found at {rl_policy_file}")

        print(f"\nLoading pre-trained RL policy from: {rl_policy_file}")
        policy.rl_policy.load(str(rl_policy_file))

        # Force immediate switch to MPC
        policy._threshold_reached = True  # pylint: disable=protected-access

        return policy

    # Run evaluation
    metrics = train_and_evaluate(
        system,
        policy_factory,
        eval_config,
        policy_name="PretrainedRL_with_MPC",
    )

    print("\nPre-trained RL + MPC Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
