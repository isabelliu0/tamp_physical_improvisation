"""Test script for graph-based training data collection."""

from pathlib import Path

import pytest
import torch

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.pushing_pybullet import (
    PybulletPushingPolicy,
)
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower import ObstacleTowerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


@pytest.mark.skip("Takes too long to run.")
def test_pybullet_graph_training_collection():
    """Test collecting graph-based training data."""
    print("\n=== Testing Graph-Based Training Data Collection ===")

    config = {
        "seed": 42,
        "num_episodes": 1,
        "max_steps": 300,
        "max_training_steps_per_shortcut": 100,
        "render": True,
        "collect_episodes": 1,
        "force_collect": True,
        "training_data_dir": "training_data/graph_training_data",
    }

    print("\n1. Creating system...")
    system = ObstacleTowerTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if config["render"] else None
    )

    print("\n2. Creating approach...")
    policy = PybulletPushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])

    print("\n3. Collecting training data...")
    train_data, _ = collect_graph_based_training_data(
        system,
        approach,
        config,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
        action_scale=0.015,
    )

    print("\n=== Training Data Statistics ===")
    print(f"Collected {len(train_data)} training examples")

    # Print details of each collected shortcut
    for i in range(len(train_data)):
        print(f"\nShortcut Example {i+1}:")
        print("Source atoms:")
        for atom in sorted(train_data.current_atoms[i], key=str):
            print(f"  - {atom}")

        print("Target atoms:")
        for atom in sorted(train_data.goal_atoms[i], key=str):
            print(f"  - {atom}")

    save_dir = Path(config["training_data_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pybullet_all_shortcuts_2"
    train_data.save(save_path)
    print(f"\nSaved training data to {save_path}")

    return train_data


@pytest.mark.skip("Takes too long to run.")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(GraphObstacle2DTAMPSystem, False), (Obstacle2DTAMPSystem, False)],
)
def test_multi_rl_obstacle2d_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        max_training_steps_per_shortcut=50,
        collect_episodes=3,
        episodes_per_scenario=1000,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=125,
        training_data_dir=f"training_data/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=32,
        max_atom_size=14,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
        n_blocks=2, seed=config.seed, render_mode="rgb_array" if config.render else None
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
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.parametrize(
    "system_cls", [GraphObstacle2DTAMPSystem, Obstacle2DTAMPSystem]
)
def test_multi_rl_obstacle2d_loaded(system_cls):
    """Test MultiRL on Obstacle2D with loaded policies."""
    policy_dir = Path("trained_policies/multi_rl")
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=True,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir="training_data",
        save_dir="trained_policies",
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n=== Testing MultiRL Loaded Policies on {system_cls.__name__} ===")
    system = system_cls.create_default(
        n_blocks=3, seed=42, render_mode="rgb_array" if config.render else None
    )

    policy_name = "MultiRL"
    policy_path = policy_dir / f"{system_cls.__name__}_{policy_name}"

    if not policy_path.exists():
        pytest.skip(f"Policy directory not found at {policy_path}")

    def multi_policy_factory(seed: int) -> MultiRLPolicy:
        policy: MultiRLPolicy = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        multi_policy_factory,
        config,
        policy_name=f"{policy_name}_Loaded",
        use_context_wrapper=False,
        select_random_goal=False,
    )

    print("\nMultiRL Loaded Policies Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")


@pytest.mark.skip("Takes too long to run")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(GraphObstacleTowerTAMPSystem, False)],
)
def test_multi_rl_pybullet_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        max_training_steps_per_shortcut=100,
        collect_episodes=1,
        episodes_per_scenario=3000,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/graph_training_data",
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=16,
        max_atom_size=14,
        action_scale=0.015,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
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
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


def test_multi_rl_pybullet_loaded(system_cls=GraphObstacleTowerTAMPSystem):
    """Test MultiRL on Pybullet ObstacleTower with loaded policies."""
    policy_dir = Path("trained_policies/multi_rl")
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        render=True,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir="training_data/graph_training_data",
        save_dir="trained_policies/multi_rl",
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n=== Testing MultiRL Loaded Policies on {system_cls.__name__} ===")
    system = system_cls.create_default(
        seed=42,
        render_mode="rgb_array" if config.render else None,
        num_obstacle_blocks=3,
    )

    policy_name = "MultiRL"
    policy_path = policy_dir / f"{system_cls.__name__}_{policy_name}"

    if not policy_path.exists():
        pytest.skip(f"Policy directory not found at {policy_path}")

    def multi_policy_factory(seed: int) -> MultiRLPolicy:
        policy: MultiRLPolicy = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        multi_policy_factory,
        config,
        policy_name=f"{policy_name}_Loaded",
        use_context_wrapper=False,
        select_random_goal=False,
    )

    print("\nMultiRL Loaded Policies Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")


@pytest.mark.skip("Takes too long to run.")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(ClutteredDrawerTAMPSystem, False)],
)
def test_multi_rl_cluttered_drawer_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        max_training_steps_per_shortcut=100,
        collect_episodes=1,
        episodes_per_scenario=1500,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/graph_training_data",
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=16,
        max_atom_size=14,
        action_scale=0.005,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
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
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


def test_multi_rl_cluttered_drawer_loaded(system_cls=ClutteredDrawerTAMPSystem):
    """Test MultiRL on Pybullet ClutteredDrawer with loaded policies."""
    policy_dir = Path("trained_policies/multi_rl")
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        render=True,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir="training_data/graph_training_data",
        save_dir="trained_policies/multi_rl",
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n=== Testing MultiRL Loaded Policies on {system_cls.__name__} ===")
    system = system_cls.create_default(
        seed=42, render_mode="rgb_array" if config.render else None
    )

    policy_name = "MultiRL"
    policy_path = policy_dir / f"{system_cls.__name__}_{policy_name}"

    if not policy_path.exists():
        pytest.skip(f"Policy directory not found at {policy_path}")

    def multi_policy_factory(seed: int) -> MultiRLPolicy:
        policy: MultiRLPolicy = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        multi_policy_factory,
        config,
        policy_name=f"{policy_name}_Loaded",
        use_context_wrapper=False,
        select_random_goal=False,
    )

    print("\nMultiRL Loaded Policies Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
