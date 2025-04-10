"""Test script for graph-based training data collection."""

from pathlib import Path

import pytest
import torch

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.policies.rl2mpc import (
    RL2MPCConfig,
    RL2MPCPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem


def test_graph_training_collection():
    """Test collecting graph-based training data."""
    print("\n=== Testing Graph-Based Training Data Collection ===")

    config = {
        "seed": 42,
        "num_episodes": 3,
        "max_steps": 50,
        "render": True,
        "collect_episodes": 1,
        "force_collect": True,
        "training_data_dir": "training_data/graph_training_data",
    }

    print("\n1. Creating system...")
    system = Blocks2DTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if config["render"] else None
    )

    print("\n2. Creating approach...")
    policy = PushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])

    print("\n3. Collecting training data...")
    train_data, _ = collect_graph_based_training_data(
        system,
        approach,
        config,
        target_specific_shortcuts=True,
    )

    print("\n=== Training Data Statistics ===")
    print(f"Collected {len(train_data)} training examples")

    # Print details of each collected shortcut
    for i in range(len(train_data)):
        print(f"\nShortcut Example {i+1}:")
        print("Source atoms:")
        for atom in sorted(train_data.current_atoms[i], key=str):
            print(f"  - {atom}")

        print("Target preimage:")
        for atom in sorted(train_data.preimages[i], key=str):
            print(f"  - {atom}")

    save_dir = Path(config["training_data_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "blocks2d_shortcuts.pkl"
    train_data.save(save_path)
    print(f"\nSaved training data to {save_path}")

    return train_data


def test_random_rollout_shortcut_selection():
    """Test collecting training data with random rollout shortcut selection."""
    print("\n=== Testing Random Rollout Shortcut Selection ===")

    config = {
        "seed": 42,
        "num_episodes": 1,
        "max_steps": 50,
        "render": False,
        "collect_episodes": 1,
        "force_collect": True,
        "training_data_dir": "training_data/random_rollout_shortcuts",
    }

    system = Blocks2DTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if config["render"] else None
    )
    policy = PushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])
    train_data, _ = collect_graph_based_training_data(
        system,
        approach,
        config,
        target_specific_shortcuts=False,
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )

    print("\n=== Training Data Statistics ===")
    print(f"Collected {len(train_data)} training examples")

    # Print details of each collected shortcut
    for i in range(len(train_data)):
        print(f"\nShortcut Example {i+1}:")
        print("Source atoms:")
        for atom in sorted(train_data.current_atoms[i], key=str):
            print(f"  - {atom}")

        print("Target preimage:")
        for atom in sorted(train_data.preimages[i], key=str):
            print(f"  - {atom}")

    save_dir = Path(config["training_data_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "blocks2d_shortcuts.pkl"
    train_data.save(save_path)
    print(f"\nSaved training data to {save_path}")

    return train_data


@pytest.mark.parametrize("use_context_wrapper", [True, False])
def test_graph_rl_pipeline(use_context_wrapper):
    """Test the full graph-based RL training and evaluation pipeline."""
    print("\n=== Testing Graph-Based RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=2,
        episodes_per_scenario=100,
        force_collect=True,
        render=True,
        record_training=False,
        training_record_interval=250,
        training_data_dir=f"training_data/graph_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/graph_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=32,
        max_preimage_size=12,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    print("\n2. Training and evaluating policy...")

    # Define policy factory
    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    # Train and evaluate with graph-based collection
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"GraphRL{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.parametrize("use_context_wrapper", [True, False])
def test_multi_rl_pipeline(use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=2,
        episodes_per_scenario=100,
        force_collect=True,
        render=True,
        record_training=False,
        training_record_interval=250,
        training_data_dir=f"training_data/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=32,
        max_preimage_size=12,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    print("\n2. Training and evaluating policy...")

    # Define policy factory
    def policy_factory(seed: int) -> MultiRLPolicy:
        return MultiRLPolicy(seed=seed, config=rl_config)

    # Train and evaluate with graph-based collection
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"MultiRL{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
        use_random_rollouts=False,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


def test_multi_rl_rollouts(
    use_context_wrapper=False,
    use_random_rollouts=True,
    num_rollouts_per_node=1000,
    max_steps_per_rollout=100,
    shortcut_success_threshold=1,
):
    """Test multi-policy RL with random rollout shortcut selection."""
    print("\n=== Testing Multi-Policy RL with Random Rollout Selection ===")

    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=200,
        collect_episodes=1,
        episodes_per_scenario=200,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=25,
        training_data_dir="training_data/multi_rl_rollouts",
        save_dir="trained_policies/multi_rl_rollouts",
        batch_size=32,
        max_preimage_size=12,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Define policy factory
    def policy_factory(seed: int) -> MultiRLPolicy:
        return MultiRLPolicy(seed=seed, config=rl_config)

    # Use the enhanced training function with direct parameters
    metrics = train_and_evaluate(
        system=system,
        policy_factory=policy_factory,
        config=config,
        policy_name="MultiRL_Rollouts",
        use_context_wrapper=use_context_wrapper,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.skip()
def test_graph_mpc_pipeline(use_context_wrapper=True):
    """Test MPC policy with specific target shortcuts."""
    print("\n=== Testing MPC Policy with Specific Target Shortcuts ===")

    # Create config
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        render=True,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Configure MPC policy with specific shortcuts
    target_shortcuts = [(0, 3), (1, 4)]

    def policy_factory(seed: int) -> MPCPolicy:
        policy: MPCPolicy = MPCPolicy(
            seed=seed,
            config=MPCConfig(
                num_rollouts=50,
                horizon=20,
                num_control_points=10,
                noise_scale=0.25,
            ),
        )

        # Add target shortcuts
        for source_id, target_id in target_shortcuts:
            policy.add_target_shortcut(source_id, target_id)

        return policy

    # Run evaluation
    print("\nRunning evaluation with target shortcuts...")
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"GraphMPC{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    return metrics


@pytest.mark.skip()
def test_graph_rl2mpc_pipeline(use_context_wrapper=True):
    """Test RL2MPC policy with the training and evaluation pipeline."""
    print("\n=== Testing RL2MPC with Training Pipeline ===")

    # Create training config
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        render=True,
        collect_episodes=2,
        episodes_per_scenario=100,
        force_collect=False,
        record_training=False,
        training_record_interval=25,
        training_data_dir=f"training_data/graph_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/graph_rl{'_context' if use_context_wrapper else ''}",
        batch_size=32,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create RL2MPC policy
    def policy_factory(seed: int) -> RL2MPCPolicy:
        return RL2MPCPolicy(
            seed=seed,
            config=RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=3e-4,
                    batch_size=32,
                    n_epochs=10,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=50,
                    horizon=20,
                    num_control_points=10,
                    noise_scale=0.05,
                ),
                reward_threshold=-15.0,
                window_size=10,
            ),
        )

    # Run train_and_evaluate
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"GraphRL2MPC{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    return metrics
