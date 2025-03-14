"""Test script for graph-based training data collection."""

from pathlib import Path

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


def test_graph_training_collection(force_collect=True, render=True):
    """Test collecting graph-based training data."""
    print("\n=== Testing Graph-Based Training Data Collection ===")

    config = {
        "seed": 42,
        "num_episodes": 3,
        "max_steps": 50,
        "collect_episodes": 1,
        "force_collect": force_collect,
        "training_data_dir": "training_data/graph_training_data",
    }

    print("\n1. Creating system...")
    system = Blocks2DTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if render else None
    )

    print("\n2. Creating approach...")
    policy = PushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])

    print("\n3. Collecting training data...")
    train_data = collect_graph_based_training_data(
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


def test_graph_rl_pipeline(force_collect=False, render=True):
    """Test the full graph-based RL training and evaluation pipeline."""
    print("\n=== Testing Graph-Based RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=3,
        max_steps=50,
        collect_episodes=2,
        episodes_per_scenario=50,
        force_collect=force_collect,
        render=render,
        record_training=True,
        training_record_interval=25,
        training_data_dir="training_data/graph_rl",
        save_dir="trained_policies/graph_rl",
        batch_size=32,
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
        seed=config.seed, render_mode="rgb_array" if render else None
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
        policy_name="GraphRL",
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics
