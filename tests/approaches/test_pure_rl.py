"""Test script for pure RL baseline."""

import torch

from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_pure_rl,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


def test_pure_rl_blocks2d():
    """Test pure RL approach on Blocks2D TAMP system."""
    print("\n=== Testing Pure RL on Blocks2D ===")
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

    system = Blocks2DTAMPSystem.create_default(
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


def test_pure_rl_pybullet():
    """Test pure RL approach on PyBullet ClearAndPlace TAMP system."""
    print("\n=== Testing Pure RL on PyBullet ClearAndPLace ===")
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

    system = ClearAndPlaceTAMPSystem.create_default(
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
