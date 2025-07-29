"""Test script for hierarchical RL approach."""

import pytest
import torch

from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_hierarchical_rl,
)
from tamp_improv.benchmarks.hierarchical_wrapper import HierarchicalRLWrapper
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def test_hierarchical_rl_obstacle2d():
    """Test hierarchical RL approach on Obstacle2D TAMP system."""
    print("\n=== Testing Hierarchical RL on Obstacle2D ===")

    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=100,
        render=True,
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

    metrics = train_and_evaluate_hierarchical_rl(
        system,
        policy_factory,
        config,
        policy_name="HierarchicalRLPolicy_multi_step",
        single_step_skills=False,
        max_skill_steps=20,
        skill_failure_penalty=-1.0,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


def test_hierarchical_rl_setup():
    """Test that hierarchical RL setup works correctly without full
    training."""
    print("\n=== Testing Hierarchical RL Setup ===")

    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=10,  # Very short test
        render=False,
        record_training=False,
        training_record_interval=100,
        device="cpu",  # Force CPU for faster test
        step_penalty=-0.1,
        success_reward=10.0,
        action_scale=1.0,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode=None,
    )

    print("Testing hierarchical RL initialization...")

    wrapper = HierarchicalRLWrapper(
        tamp_system=system,
        max_episode_steps=config.max_steps,
        max_skill_steps=10,
        step_penalty=config.step_penalty,
        achievement_bonus=config.success_reward,
        action_scale=config.action_scale,
        skill_failure_penalty=-1.0,
        single_step_skills=True,
    )

    print(
        f"Created hierarchical wrapper with {len(wrapper.ground_skill_operators)} skills"
    )
    print(f"✓ Action space: {wrapper.action_space}")
    print(f"✓ Observation space: {wrapper.observation_space}")

    # Test environment reset
    obs, info = wrapper.reset(seed=42)
    print(f"✓ Environment reset successful, obs shape: {obs.shape}")

    # Test a few random actions
    for i in range(3):
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        print(
            f"✓ Step {i+1}: Action type: {info.get('action_type', 'unknown')}, Reward: {reward:.2f}"  # pylint: disable=line-too-long
        )

        if terminated or truncated:
            break

    wrapper.close()
    print("✓ Hierarchical RL setup test completed successfully")

    return True


@pytest.mark.skip(reason="Takes too long to run")
@pytest.mark.parametrize(
    "system_cls",
    [GraphObstacleTowerTAMPSystem, ClutteredDrawerTAMPSystem, CleanupTableTAMPSystem],
)
def test_hierarchical_rl_pybullet(system_cls):
    """Test hierarchical RL approach on PyBullet TAMP systems."""
    print(f"\n=== Testing Hierarchical RL on {system_cls.__name__} ===")

    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        render=True,
        record_training=True,
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

    metrics = train_and_evaluate_hierarchical_rl(
        system,
        policy_factory,
        config,
        policy_name="HierarchicalRLPolicy_multi_step",
        single_step_skills=False,
        max_skill_steps=200,
        skill_failure_penalty=-1.0,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics
