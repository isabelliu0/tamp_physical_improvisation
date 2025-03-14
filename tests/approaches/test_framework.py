"""Test the framework using a hard-coded pushing policy."""

from pathlib import Path

import pytest
import torch

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem


@pytest.fixture(scope="function", name="training_config")
def _get_training_config():
    """Test configuration."""
    return TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=True,
    )


def test_framework_with_hardcoded_policy(training_config):
    """Test the framework using a hard-coded pushing policy."""
    print("\n=== Testing Framework with Hard-coded Pushing Policy ===")

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=42, render_mode="rgb_array" if training_config.render else None
    )

    # Create policy
    def policy_factory(seed: int) -> PushingPolicy:
        return PushingPolicy(seed=seed)

    # Run test
    metrics = train_and_evaluate(
        system, policy_factory, training_config, policy_name="PushingPolicy"
    )

    print("\nResults:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")


def test_framework_integration(
    force_collect: bool = False,
    render: bool = True,
):
    """Test the integrated framework with RL policy."""
    print("\n=== Testing New Framework Integration ===")

    # Create RL-specific policy config
    rl_policy_config = RLConfig(
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create training config
    config = TrainingConfig(
        # General settings
        seed=42,
        num_episodes=5,
        max_steps=100,
        # Collection settings
        collect_episodes=5,
        episodes_per_scenario=5,
        force_collect=force_collect,
        # Save/Load settings
        save_dir="trained_policies",
        training_data_dir="training_data",
        # Visualization settings
        render=render,
        record_training=True,
        training_record_interval=50,
        # Device settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Policy-specific settings
        policy_config=rl_policy_config.__dict__,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create policy
    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_policy_config)

    # Create approach for data collection
    policy = policy_factory(config.seed)
    approach = ImprovisationalTAMPApproach(system, policy, seed=config.seed)

    # Step 1: Collect training data
    print("\n=== Step 1: Collecting Training Data ===")
    training_data = collect_graph_based_training_data(
        system=system,
        approach=approach,
        config=config.__dict__,
        max_shortcuts_per_graph=3,
        min_shortcut_distance=2,
    )

    # Report on training data
    print(f"\nCollected {len(training_data.states)} training examples")

    # Save the training data
    data_path = Path(config.training_data_dir) / system.name
    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving training data to {data_path}")
    training_data.save(data_path)

    # # Step 2: Train the policy
    # print("\n=== Step 2: Training RL Policy ===")
    # # Configure environment for training
    # if hasattr(system.wrapped_env, "configure_training"):
    #     system.wrapped_env.configure_training(training_data)

    # # Train the policy
    # policy.train(system.wrapped_env, training_data)

    # # Save the trained policy
    # save_path = Path(config.save_dir) / f"{system.name}_RLPolicy"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # print(f"Saving policy to {save_path}")
    # policy.save(str(save_path))

    # # Step 3: Evaluate the policy
    # print("\n=== Step 3: Evaluating RL Policy ===")

    # # Run full training and evaluation
    # metrics = train_and_evaluate(
    #     system, policy_factory, config, policy_name="RLPolicy"
    # )

    # # Print results
    # print("\nTraining and Evaluation Results:")
    # print(f"Success Rate: {metrics.success_rate:.2%}")
    # print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    # print(f"Average Reward: {metrics.avg_reward:.2f}")

    # # Step 4: Test the loaded policy
    # print("\n=== Step 4: Testing Loaded RL Policy ===")

    # def loaded_policy_factory(seed: int) -> RLPolicy:
    #     policy = RLPolicy(seed=seed, config=rl_policy_config)
    #     policy_path = Path(config.save_dir) / f"{system.name}_RLPolicy"
    #     policy.load(str(policy_path))
    #     return policy

    # loaded_metrics = train_and_evaluate(
    #     system, loaded_policy_factory, config, policy_name="RLPolicy_Loaded"
    # )

    # print("\nLoaded Policy Results:")
    # print(f"Success Rate: {loaded_metrics.success_rate:.2%}")
    # print(f"Average Episode Length: {loaded_metrics.avg_episode_length:.2f}")
    # print(f"Average Reward: {loaded_metrics.avg_reward:.2f}")

    # # Step 5: Check approach shortcut statistics
    # print("\n=== Step 5: Checking Shortcut Statistics ===")
    # print("Running one episode to collect shortcut statistics:")
    # policy = loaded_policy_factory(config.seed)
    # approach = ImprovisationalTAMPApproach(system, policy, seed=config.seed)

    # obs, info = system.reset()
    # approach.reset(obs, info)

    # for _ in range(config.max_steps):
    #     step_result = approach.step(obs, 0.0, False, False, info)
    #     obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    #     if terminated or truncated:
    #         break

    # shortcut_stats = approach.get_shortcut_stats()
    # print("\nShortcut Usage Statistics:")
    # print(f"Shortcuts used: {shortcut_stats['shortcuts_used']}")
    # print(f"Shortcut success count: {shortcut_stats['shortcut_success_count']}")
    # print(f"Shortcut failure count: {shortcut_stats['shortcut_failure_count']}")
    # print(f"Shortcut success rate: {shortcut_stats['shortcut_success_rate']:.2%}")

    # return metrics, loaded_metrics
