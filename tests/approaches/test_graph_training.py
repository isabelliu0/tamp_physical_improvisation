"""Test script for graph-based training data collection."""

from pathlib import Path

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem


def test_graph_training_collection(force_collect=True, render=True):
    """Test collecting graph-based training data."""
    print("\n=== Testing Graph-Based Training Data Collection ===")

    config = {
        "seed": 42,
        "num_episodes": 3,
        "max_steps": 50,
        "collect_episodes": 5,
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
