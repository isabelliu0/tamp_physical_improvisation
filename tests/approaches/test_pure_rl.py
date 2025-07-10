"""Test script for pure RL baseline."""

import pytest
import torch

from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.policies.sac_her import (
    SACHERConfig,
    SACHERPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_pure_rl,
    train_and_evaluate_sac_her,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def test_pure_rl_obstacle2d():
    """Test pure RL approach on Obstacle2D TAMP system."""
    print("\n=== Testing Pure RL on Obstacle2D ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=100,
        render=False,
        record_training=False,
        training_record_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-0.1,
        success_reward=10.0,
        action_scale=1.0,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device=config.device,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate_pure_rl(
        system,
        policy_factory,
        config,
        policy_name="PureRLPolicy",
    )
    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


def test_sac_her_obstacle2d():
    """Test SAC+HER baseline on Obstacle2D TAMP system."""
    print("\n=== Testing SAC+HER Baseline on Obstacle2D ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=100,
        render=False,
        record_training=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-0.1,
        success_reward=10.0,
    )

    sac_her_config = SACHERConfig(
        learning_rate=3e-4,
        batch_size=32,
        device=config.device,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> SACHERPolicy:
        return SACHERPolicy(seed=seed, config=sac_her_config)

    metrics = train_and_evaluate_sac_her(
        system,
        policy_factory,
        config,
        policy_name="SACHERPolicy",
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.parametrize(
    "system_cls",
    [GraphObstacleTowerTAMPSystem, ClutteredDrawerTAMPSystem],
)
def test_pure_rl_pybullet(system_cls):
    """Test pure RL baseline on PyBullet TAMP systems."""
    print(f"\n=== Testing Pure RL Baseline on {system_cls.__name__} ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        render=True,
        record_training=False,
        training_record_interval=200,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-1.0,
        success_reward=100.0,
        action_scale=0.015,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device=config.device,
    )

    system = system_cls.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate_pure_rl(
        system,
        policy_factory,
        config,
        policy_name="PureRLPolicy",
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.parametrize(
    "system_cls,max_atom_size",
    [
        (GraphObstacleTowerTAMPSystem, 42),
        (ClutteredDrawerTAMPSystem, 72),
    ],
)
def test_sac_her_pybullet(system_cls, max_atom_size):
    """Test SAC+HER baseline on PyBullet TAMP systems."""
    print(f"\n=== Testing SAC+HER Baseline on {system_cls.__name__} ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        render=False,
        record_training=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-1.0,
        success_reward=100.0,
        action_scale=0.015,
    )

    sac_her_config = SACHERConfig(
        learning_rate=3e-4,
        batch_size=32,
        device=config.device,
    )

    system = system_cls.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> SACHERPolicy:
        return SACHERPolicy(seed=seed, config=sac_her_config)

    metrics = train_and_evaluate_sac_her(
        system,
        policy_factory,
        config,
        policy_name="SACHERPolicy",
        max_atom_size=max_atom_size,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics
